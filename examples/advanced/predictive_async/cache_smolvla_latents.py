#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cache pinned SmolVLA frame-level inputs for offline future-latent learning.

This producer is intentionally limited to the frozen train/validation/test
episode splits.  It encodes each frame once and writes one safetensors shard per
episode; future-latent pairs are views over two frame rows in the same shard.
The cache is mechanism evidence, not a task-capability result or a deployable
controller.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from future_latent_cache import CACHE_SCHEMA_VERSION, MAX_PREDICTION_DELAY, validate_episode_cache
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import Tensor

import lerobot.policies  # noqa: F401 - registers policy/config subclasses
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.rollout.inference.oracle_evaluation import remap_checkpoint_action_stats
from lerobot.utils.constants import (
    ACTION,
    HF_LEROBOT_HUB_CACHE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.feature_utils import dataset_to_policy_features

DATASET_REPO_ID = "lerobot/svla_so100_pickplace"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
CHECKPOINT_REPO_ID = "lerobot/smolvla_base"
CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"

DATASET_CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.wrist",
)
POLICY_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
)
VIDEO_BACKEND = "pyav"
SPLIT_MANIFEST_PATH = Path(__file__).with_name("m3_episode_split_seed0.json")
PREPROCESSOR_STATE_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
ACTION_STATS_SOURCE_KEY = "so100.buffer.action"
TOKEN_SCALING_CONVENTION = "native_post_sqrt_hidden_dim"

EXPECTED_CAMERA_COUNT = 2
EXPECTED_TOKEN_COUNT = 64
EXPECTED_TOKEN_DIM = 960
EXPECTED_TOKEN_DTYPE = torch.bfloat16
EXPECTED_STATE_DIM = 32
EXPECTED_ACTION_DIM = 6


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve all three pinned Hub snapshots from local caches only.",
    )
    parser.add_argument(
        "--max-episodes",
        type=_positive_int,
        help="Cache only the first N authorized episodes (bounded smoke only).",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=_positive_int,
        help="Cache only the first N frames of each episode (bounded smoke only).",
    )
    args = parser.parse_args()
    if args.split == "test" and (args.max_episodes is not None or args.max_frames_per_episode is not None):
        parser.error("--split test does not permit --max-episodes or --max-frames-per-episode")
    return args


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _resolve_snapshots(*, local_files_only: bool) -> tuple[Path, Path, Path]:
    dataset_snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            local_files_only=local_files_only,
        )
    )
    checkpoint_snapshot = Path(
        snapshot_download(
            repo_id=CHECKPOINT_REPO_ID,
            revision=CHECKPOINT_REVISION,
            local_files_only=local_files_only,
        )
    )
    vlm_snapshot = Path(
        snapshot_download(
            repo_id=VLM_REPO_ID,
            revision=VLM_REVISION,
            local_files_only=local_files_only,
        )
    )

    pinned = (
        ("dataset", dataset_snapshot, DATASET_REVISION),
        ("checkpoint", checkpoint_snapshot, CHECKPOINT_REVISION),
        ("VLM", vlm_snapshot, VLM_REVISION),
    )
    for name, snapshot, expected_revision in pinned:
        if snapshot.name != expected_revision:
            raise RuntimeError(f"Pinned {name} resolved to {snapshot.name!r}, expected {expected_revision!r}")
    return dataset_snapshot, checkpoint_snapshot, vlm_snapshot


def _load_split(split: str) -> tuple[tuple[int, ...], int]:
    with SPLIT_MANIFEST_PATH.open(encoding="utf-8") as stream:
        split_manifest = json.load(stream)

    dataset = split_manifest.get("dataset", {})
    if dataset.get("repo_id") != DATASET_REPO_ID or dataset.get("revision") != DATASET_REVISION:
        raise ValueError("The frozen episode split does not match the pinned B1 dataset revision")
    split_entry = split_manifest["splits"][split]
    episode_ids = tuple(int(value) for value in split_entry["episode_ids"])
    episode_count = int(split_entry["episode_count"])
    if len(episode_ids) != episode_count:
        raise ValueError(
            f"Frozen {split} split declares {episode_count} episodes but lists {len(episode_ids)}"
        )
    return episode_ids, int(split_entry["frame_count"])


def _configure_policy(
    metadata: LeRobotDatasetMetadata,
    *,
    checkpoint_snapshot: Path,
    vlm_snapshot: Path,
    device: torch.device,
):
    config = PreTrainedConfig.from_pretrained(checkpoint_snapshot, local_files_only=True)
    if config.type != "smolvla":
        raise ValueError(f"Future-latent caching requires SmolVLA, got {config.type!r}")

    dataset_features = dataset_to_policy_features(metadata.features)
    required_features = (ACTION, OBS_STATE, *DATASET_CAMERA_KEYS)
    missing_features = [key for key in required_features if key not in dataset_features]
    if missing_features:
        raise KeyError(f"Pinned dataset is missing required features: {missing_features}")

    config.input_features = {
        OBS_STATE: dataset_features[OBS_STATE],
        POLICY_CAMERA_KEYS[0]: dataset_features[DATASET_CAMERA_KEYS[0]],
        POLICY_CAMERA_KEYS[1]: dataset_features[DATASET_CAMERA_KEYS[1]],
    }
    config.output_features = {ACTION: dataset_features[ACTION]}
    config.device = str(device)
    config.pretrained_path = checkpoint_snapshot
    config.pretrained_revision = CHECKPOINT_REVISION
    config.vlm_model_name = str(vlm_snapshot)
    config.compile_model = False

    if tuple(config.image_features) != POLICY_CAMERA_KEYS:
        raise ValueError(
            f"Adapted policy camera order is {tuple(config.image_features)!r}, expected {POLICY_CAMERA_KEYS!r}"
        )
    if config.action_feature is None or tuple(config.action_feature.shape) != (EXPECTED_ACTION_DIM,):
        shape = None if config.action_feature is None else tuple(config.action_feature.shape)
        raise ValueError(f"Pinned policy action shape is {shape}, expected ({EXPECTED_ACTION_DIM},)")
    if config.adapt_to_pi_aloha:
        raise ValueError("Pinned SO100 cache contract does not permit PI-Aloha state/action adaptation")

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        checkpoint_snapshot,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    policy.requires_grad_(False)
    return policy


def _build_preprocessor(
    config,
    *,
    checkpoint_snapshot: Path,
    vlm_snapshot: Path,
    device: torch.device,
):
    rename_map = dict(zip(DATASET_CAMERA_KEYS, POLICY_CAMERA_KEYS, strict=True))
    processor_state = load_file(str(checkpoint_snapshot / PREPROCESSOR_STATE_FILE), device="cpu")
    action_stats = remap_checkpoint_action_stats(
        processor_state,
        source_key=ACTION_STATS_SOURCE_KEY,
        action_dim=EXPECTED_ACTION_DIM,
    )
    preprocessor_overrides = {
        "rename_observations_processor": {"rename_map": rename_map},
        "tokenizer_processor": {"tokenizer_name": str(vlm_snapshot)},
        "device_processor": {"device": str(device)},
        "normalizer_processor": {"stats": action_stats},
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint_snapshot),
        preprocessor_overrides=preprocessor_overrides,
    )
    return preprocessor, rename_map


def _episode_row_indices(
    dataset: LeRobotDataset,
    episode_ids: tuple[int, ...],
    *,
    max_frames_per_episode: int | None,
) -> dict[int, tuple[int, ...]]:
    rows_by_episode: dict[int, list[tuple[int, int]]] = {episode_id: [] for episode_id in episode_ids}
    raw_episode_indices = dataset.hf_dataset["episode_index"]
    raw_frame_indices = dataset.hf_dataset["frame_index"]
    for row_index, (episode_index, frame_index) in enumerate(
        zip(raw_episode_indices, raw_frame_indices, strict=True)
    ):
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        if episode_index not in rows_by_episode:
            raise ValueError(f"Filtered dataset unexpectedly contains episode {episode_index}")
        if max_frames_per_episode is None or frame_index < max_frames_per_episode:
            rows_by_episode[episode_index].append((frame_index, row_index))

    result: dict[int, tuple[int, ...]] = {}
    for episode_id, indexed_rows in rows_by_episode.items():
        indexed_rows.sort()
        frame_indices = [frame_index for frame_index, _ in indexed_rows]
        if frame_indices != list(range(len(frame_indices))):
            raise ValueError(
                f"Episode {episode_id} frame indices are not the contiguous prefix 0..{len(frame_indices) - 1}"
            )
        if not indexed_rows:
            raise ValueError(f"Episode {episode_id} contains no cacheable frames")
        result[episode_id] = tuple(row_index for _, row_index in indexed_rows)
    return result


def _cpu_contiguous(tensor: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().to(device="cpu").contiguous()


def _extract_episode(
    dataset: LeRobotDataset,
    row_indices: tuple[int, ...],
    *,
    episode_index: int,
    policy,
    preprocessor,
    batch_size: int,
) -> dict[str, Tensor]:
    chunks: dict[str, list[Tensor]] = {}
    with torch.inference_mode():
        for batch_start in range(0, len(row_indices), batch_size):
            batch_rows = row_indices[batch_start : batch_start + batch_size]
            raw_samples = [dataset[row_index] for row_index in batch_rows]
            raw_episode_indices = torch.stack(
                [torch.as_tensor(sample["episode_index"], dtype=torch.int64) for sample in raw_samples]
            ).reshape(-1)
            if not torch.all(raw_episode_indices == episode_index):
                raise ValueError(f"Episode {episode_index} batch contains another episode")
            metadata_tensors = {
                "dataset_indices": torch.stack(
                    [torch.as_tensor(sample["index"], dtype=torch.int64) for sample in raw_samples]
                ).reshape(-1),
                "frame_indices": torch.stack(
                    [torch.as_tensor(sample["frame_index"], dtype=torch.int64) for sample in raw_samples]
                ).reshape(-1),
            }

            processed_samples = [preprocessor(sample) for sample in raw_samples]
            processed_keys = (
                *POLICY_CAMERA_KEYS,
                OBS_STATE,
                ACTION,
                OBS_LANGUAGE_TOKENS,
                OBS_LANGUAGE_ATTENTION_MASK,
            )
            processed = {
                key: torch.cat([sample[key] for sample in processed_samples], dim=0) for key in processed_keys
            }
            images, image_masks = policy.prepare_images(processed)
            image_tokens, image_token_masks = policy.model.encode_image_tokens(images, image_masks)
            states = policy.prepare_state(processed)
            actions = processed[ACTION]
            language_tokens = processed[OBS_LANGUAGE_TOKENS]
            language_attention_mask = processed[OBS_LANGUAGE_ATTENTION_MASK]

            batch_tensors = {
                **metadata_tensors,
                "states": _cpu_contiguous(states, dtype=torch.float32),
                "actions": _cpu_contiguous(actions, dtype=torch.float32),
                "language_tokens": _cpu_contiguous(language_tokens, dtype=torch.int64),
                "language_attention_mask": _cpu_contiguous(language_attention_mask, dtype=torch.bool),
            }
            for camera_index, (tokens, masks) in enumerate(zip(image_tokens, image_token_masks, strict=True)):
                batch_tensors[f"image_tokens_{camera_index}"] = _cpu_contiguous(tokens)
                batch_tensors[f"image_token_masks_{camera_index}"] = _cpu_contiguous(masks, dtype=torch.bool)

            for key, value in batch_tensors.items():
                chunks.setdefault(key, []).append(value)

    return {key: torch.cat(values, dim=0).contiguous() for key, values in chunks.items()}


def _validate_pinned_tensor_contract(tensors: dict[str, Tensor], *, episode_index: int) -> None:
    frame_count = tensors["frame_indices"].shape[0]
    expected_frame_indices = torch.arange(frame_count, dtype=torch.int64)
    if not torch.equal(tensors["frame_indices"], expected_frame_indices):
        raise ValueError(f"Episode {episode_index} frame indices are not exactly ascending 0..T-1")
    if tensors["dataset_indices"].shape != (frame_count,):
        raise ValueError(f"Episode {episode_index} dataset_indices must have shape [T]")
    if frame_count > 1 and not torch.all(tensors["dataset_indices"][1:] > tensors["dataset_indices"][:-1]):
        raise ValueError(f"Episode {episode_index} dataset indices are not strictly ascending")

    expected_shapes = {
        "states": (frame_count, EXPECTED_STATE_DIM),
        "actions": (frame_count, EXPECTED_ACTION_DIM),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(tensors[key].shape) != expected_shape:
            raise ValueError(
                f"Episode {episode_index} {key} shape is {tuple(tensors[key].shape)}, "
                f"expected {expected_shape}"
            )
        if tensors[key].dtype is not torch.float32:
            raise TypeError(f"Episode {episode_index} {key} must have dtype float32")
        if not torch.isfinite(tensors[key]).all():
            raise ValueError(f"Episode {episode_index} {key} contains non-finite values")

    if tensors["language_tokens"].ndim != 2 or tensors["language_tokens"].shape[0] != frame_count:
        raise ValueError(f"Episode {episode_index} language_tokens must have shape [T,L]")
    if tensors["language_tokens"].dtype is not torch.int64:
        raise TypeError(f"Episode {episode_index} language_tokens must have dtype int64")
    language_masks = tensors["language_attention_mask"]
    if tuple(language_masks.shape) != tuple(tensors["language_tokens"].shape):
        raise ValueError(f"Episode {episode_index} language token/mask shapes disagree")
    if language_masks.dtype is not torch.bool:
        raise TypeError(f"Episode {episode_index} language_attention_mask must have dtype bool")

    for camera_index in range(EXPECTED_CAMERA_COUNT):
        token_key = f"image_tokens_{camera_index}"
        mask_key = f"image_token_masks_{camera_index}"
        tokens = tensors[token_key]
        masks = tensors[mask_key]
        expected_token_shape = (frame_count, EXPECTED_TOKEN_COUNT, EXPECTED_TOKEN_DIM)
        if tuple(tokens.shape) != expected_token_shape:
            raise ValueError(
                f"Episode {episode_index} {token_key} shape is {tuple(tokens.shape)}, "
                f"expected {expected_token_shape}"
            )
        if tokens.dtype is not EXPECTED_TOKEN_DTYPE:
            raise TypeError(
                f"Episode {episode_index} {token_key} dtype is {tokens.dtype}, "
                f"expected {EXPECTED_TOKEN_DTYPE}"
            )
        if tuple(masks.shape) != (frame_count, EXPECTED_TOKEN_COUNT) or masks.dtype is not torch.bool:
            raise ValueError(
                f"Episode {episode_index} {mask_key} must be bool [{frame_count},{EXPECTED_TOKEN_COUNT}]"
            )
        if not torch.isfinite(tokens[masks]).all():
            raise ValueError(f"Episode {episode_index} {token_key} contains non-finite valid tokens")

    for key, tensor in tensors.items():
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"Episode {episode_index} {key} must be contiguous CPU storage")


def _tensor_metadata(tensors: dict[str, Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "storage_device": "cpu",
        }
        for key, tensor in sorted(tensors.items())
    }


def _save_and_reload_exact(tensors: dict[str, Tensor], shard_path: Path) -> dict[str, Tensor]:
    if shard_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache shard {shard_path}")
    save_file(tensors, str(shard_path))
    reloaded = load_file(str(shard_path), device="cpu")
    if reloaded.keys() != tensors.keys() or any(
        not torch.equal(reloaded[key], value) for key, value in tensors.items()
    ):
        raise RuntimeError(f"Safetensors reload differs from the written shard {shard_path}")
    return reloaded


def _git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to record the cache producer commit")
    return subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _device_manifest(device: torch.device) -> dict[str, Any]:
    manifest: dict[str, Any] = {"type": device.type, "requested": str(device)}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        manifest.update({"index": index, "name": torch.cuda.get_device_name(index)})
    return manifest


def _software_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "lerobot": _package_version("lerobot"),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "safetensors": _package_version("safetensors"),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is false")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory {args.output_dir}")

    authoritative_episode_ids, authoritative_frame_count = _load_split(args.split)
    selected_episode_ids = authoritative_episode_ids
    if args.max_episodes is not None:
        selected_episode_ids = selected_episode_ids[: args.max_episodes]
    complete_split = args.max_episodes is None and args.max_frames_per_episode is None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset_snapshot, checkpoint_snapshot, vlm_snapshot = _resolve_snapshots(
        local_files_only=args.local_files_only
    )
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        revision=DATASET_REVISION,
    )
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        revision=DATASET_REVISION,
        episodes=list(selected_episode_ids),
        video_backend=VIDEO_BACKEND,
    )
    row_indices_by_episode = _episode_row_indices(
        dataset,
        selected_episode_ids,
        max_frames_per_episode=args.max_frames_per_episode,
    )
    if complete_split and len(dataset) != authoritative_frame_count:
        raise ValueError(
            f"Frozen {args.split} split has {len(dataset)} frames, expected {authoritative_frame_count}"
        )

    policy = _configure_policy(
        metadata,
        checkpoint_snapshot=checkpoint_snapshot,
        vlm_snapshot=vlm_snapshot,
        device=device,
    )
    preprocessor, rename_map = _build_preprocessor(
        policy.config,
        checkpoint_snapshot=checkpoint_snapshot,
        vlm_snapshot=vlm_snapshot,
        device=device,
    )

    args.output_dir.mkdir(parents=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir()

    episode_entries: list[dict[str, Any]] = []
    frame_count = 0
    valid_pair_count_by_delay = {str(delay): 0 for delay in range(1, MAX_PREDICTION_DELAY + 1)}
    for episode_number, episode_index in enumerate(selected_episode_ids, start=1):
        tensors = _extract_episode(
            dataset,
            row_indices_by_episode[episode_index],
            episode_index=episode_index,
            policy=policy,
            preprocessor=preprocessor,
            batch_size=args.batch_size,
        )
        _validate_pinned_tensor_contract(tensors, episode_index=episode_index)
        episode_frame_count = int(tensors["frame_indices"].shape[0])
        relative_shard = Path("episodes") / f"episode_{episode_index:06d}.safetensors"
        reloaded = _save_and_reload_exact(tensors, args.output_dir / relative_shard)
        episode_entries.append(
            {
                "episode_index": episode_index,
                "frame_count": episode_frame_count,
                "shard": relative_shard.as_posix(),
                "tensor_metadata": _tensor_metadata(reloaded),
            }
        )
        frame_count += episode_frame_count
        for delay in range(1, MAX_PREDICTION_DELAY + 1):
            valid_pair_count_by_delay[str(delay)] += max(episode_frame_count - delay, 0)
        logging.info(
            "Cached episode %d (%d/%d): %d frames",
            episode_index,
            episode_number,
            len(selected_episode_ids),
            episode_frame_count,
        )

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "classification": "offline_future_latent_cache_not_task_capability",
        "producer": {
            "git_sha": _git_sha(),
            "command": [sys.executable, *sys.argv],
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "inputs": {
            "dataset": {
                "repo_id": DATASET_REPO_ID,
                "requested_revision": DATASET_REVISION,
                "resolved_revision": dataset_snapshot.name,
                "fps": metadata.fps,
            },
            "checkpoint": {
                "repo_id": CHECKPOINT_REPO_ID,
                "requested_revision": CHECKPOINT_REVISION,
                "resolved_revision": checkpoint_snapshot.name,
            },
            "vlm": {
                "repo_id": VLM_REPO_ID,
                "requested_revision": VLM_REVISION,
                "resolved_revision": vlm_snapshot.name,
            },
        },
        "split": args.split,
        "authoritative_episode_ids": list(authoritative_episode_ids),
        "cached_episode_ids": list(selected_episode_ids),
        "complete_split": complete_split,
        "fps": int(metadata.fps),
        "episode_count": len(episode_entries),
        "frame_count": frame_count,
        "valid_pair_count_by_delay": valid_pair_count_by_delay,
        "episodes": episode_entries,
        "camera_mapping": rename_map,
        "policy_camera_order": list(policy.config.image_features),
        "token_scaling_convention": TOKEN_SCALING_CONVENTION,
        "storage_device": "cpu",
        "semantics": {
            "state": "model_ready_normalized_and_padded",
            "action": "normalized_policy_output_original_action_dim",
            "processor_config_source": f"{CHECKPOINT_REPO_ID}@{CHECKPOINT_REVISION}",
        },
        "processor_provenance": {
            "preprocessor_config": "policy_preprocessor.json",
            "preprocessor_state_file": PREPROCESSOR_STATE_FILE,
            "action_stats": {
                "source_repo_id": CHECKPOINT_REPO_ID,
                "source_revision": CHECKPOINT_REVISION,
                "source_key": ACTION_STATS_SOURCE_KEY,
                "target_key": ACTION,
                "dataset_stats_used": False,
            },
        },
        "extraction_device": _device_manifest(device),
        "software_versions": _software_versions(),
        "local_files_only": args.local_files_only,
        "validation": {
            "all_valid_float_values_finite": True,
            "reload_exact_tensor_equality": True,
        },
    }

    for episode_entry in episode_entries:
        tensors = load_file(str(args.output_dir / episode_entry["shard"]), device="cpu")
        validate_episode_cache(manifest, episode_entry, tensors)
    _write_manifest(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "split": args.split,
                "complete_split": complete_split,
                "episode_count": len(episode_entries),
                "frame_count": frame_count,
                "valid_pair_count_by_delay": valid_pair_count_by_delay,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
