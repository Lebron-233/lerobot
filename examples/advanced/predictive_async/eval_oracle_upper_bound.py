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

"""Evaluate the offline Oracle future-vision upper bound for predictive async.

The default protocol is the one approved in GitHub Issue #1: one task, 128
episode-stratified anchors, a common cohort valid through delay 20, and delays
1..8 plus 12, 16, and 20. Each anchor uses one fixed noise tensor across every
delay; the current, Oracle-visual, and full-future-teacher paths receive
independent clones.

This is a mechanism/sensitivity upper bound. It is not a task-capability or
robot-success evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import Tensor

import lerobot.policies  # noqa: F401 - registers policy/config subclasses
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.rollout.inference.oracle_context import encode_oracle_future_tokens
from lerobot.rollout.inference.oracle_evaluation import (
    ActionTriplet,
    OracleAnchorCandidate,
    OracleEvaluationRecord,
    aggregate_by_delay,
    make_evaluation_record,
    run_with_shared_noise,
    select_common_anchor_ids,
    slice_temporal_sample,
)
from lerobot.utils.constants import ACTION, HF_LEROBOT_HUB_CACHE, OBS_STATE
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
APPROVED_PRIMARY_DELAYS = tuple(range(1, 9))
APPROVED_DIAGNOSTIC_DELAYS = (12, 16, 20)
APPROVED_DELAYS = APPROVED_PRIMARY_DELAYS + APPROVED_DIAGNOSTIC_DELAYS
VIDEO_BACKEND = "pyav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle_upper_bound"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--delays", type=int, nargs="+", default=list(APPROVED_DELAYS))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve every pinned Hub snapshot from the local cache without network access.",
    )
    return parser.parse_args()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _clone_mapping(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, Tensor) else deepcopy(value) for key, value in batch.items()
    }


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
    return dataset_snapshot, checkpoint_snapshot, vlm_snapshot


def _build_anchor_candidates(metadata: LeRobotDatasetMetadata) -> tuple[OracleAnchorCandidate, ...]:
    candidates: list[OracleAnchorCandidate] = []
    for episode in metadata.episodes:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        episode_start = int(episode["dataset_from_index"])
        candidates.extend(
            OracleAnchorCandidate(
                anchor_id=episode_start + frame_index,
                episode_index=episode_index,
                frame_index=frame_index,
                episode_length=episode_length,
            )
            for frame_index in range(episode_length)
        )
    return tuple(candidates)


def _configure_policy(
    metadata: LeRobotDatasetMetadata,
    *,
    checkpoint_snapshot: Path,
    checkpoint_revision: str,
    vlm_snapshot: Path,
    device: str,
):
    config = PreTrainedConfig.from_pretrained(
        checkpoint_snapshot,
        local_files_only=True,
    )
    if config.type != "smolvla":
        raise ValueError(f"Oracle future-vision evaluation requires SmolVLA, got {config.type!r}")

    dataset_features = dataset_to_policy_features(metadata.features)
    required_features = (ACTION, OBS_STATE, *DATASET_CAMERA_KEYS)
    missing_features = [key for key in required_features if key not in dataset_features]
    if missing_features:
        raise KeyError(f"Approved dataset is missing required features: {missing_features}")

    original_checkpoint_cameras = tuple(config.image_features)
    config.input_features = {
        OBS_STATE: dataset_features[OBS_STATE],
        POLICY_CAMERA_KEYS[0]: dataset_features[DATASET_CAMERA_KEYS[0]],
        POLICY_CAMERA_KEYS[1]: dataset_features[DATASET_CAMERA_KEYS[1]],
    }
    config.output_features = {ACTION: dataset_features[ACTION]}
    config.device = device
    config.pretrained_path = checkpoint_snapshot
    config.pretrained_revision = checkpoint_revision
    config.vlm_model_name = str(vlm_snapshot)
    config.compile_model = False

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        checkpoint_snapshot,
        config=config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    return policy, original_checkpoint_cameras


def _build_processors(config, *, checkpoint_snapshot: Path, vlm_snapshot: Path, device: str):
    rename_map = dict(zip(DATASET_CAMERA_KEYS, POLICY_CAMERA_KEYS, strict=True))
    preprocessor_overrides = {
        "rename_observations_processor": {"rename_map": rename_map},
        "tokenizer_processor": {"tokenizer_name": str(vlm_snapshot)},
        "device_processor": {"device": device},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint_snapshot),
        preprocessor_overrides=preprocessor_overrides,
    )
    return preprocessor, postprocessor, rename_map, preprocessor_overrides


def _postprocess_triplet(postprocessor, actions: ActionTriplet) -> ActionTriplet:
    return ActionTriplet(
        current=postprocessor(actions.current.clone()),
        oracle_visual=postprocessor(actions.oracle_visual.clone()),
        full_future_teacher=postprocessor(actions.full_future_teacher.clone()),
    )


def _git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _software_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "lerobot": _package_version("lerobot"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "av": _package_version("av"),
    }


def _device_manifest(device: torch.device) -> dict[str, Any]:
    manifest: dict[str, Any] = {"requested": str(device), "type": device.type}
    if device.type == "cuda":
        manifest.update(
            {
                "index": device.index if device.index is not None else torch.cuda.current_device(),
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return manifest


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    delays = tuple(sorted(args.delays))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but torch.cuda.is_available() is false")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "per_sample.jsonl"
    summary_path = args.output_dir / "summary.json"
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation artifact in {args.output_dir.resolve()}"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started_at = datetime.now(UTC)
    started_perf = time.perf_counter()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_snapshot, checkpoint_snapshot, vlm_snapshot = _resolve_snapshots(
        local_files_only=args.local_files_only
    )
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        revision=DATASET_REVISION,
    )
    if metadata.total_tasks != 1:
        raise ValueError(f"Approved protocol requires exactly one task, found {metadata.total_tasks}")

    candidates = _build_anchor_candidates(metadata)
    selected_anchor_ids = select_common_anchor_ids(
        candidates,
        delays=delays,
        count=args.anchor_count,
        seed=args.seed,
    )
    candidates_by_id = {candidate.anchor_id: candidate for candidate in candidates}
    selected_candidates = tuple(candidates_by_id[anchor_id] for anchor_id in selected_anchor_ids)
    max_delay = max(delays)
    valid_candidate_count = sum(
        candidate.frame_index + max_delay < candidate.episode_length for candidate in candidates
    )

    temporal_keys = (*DATASET_CAMERA_KEYS, OBS_STATE)
    temporal_offsets = [step / metadata.fps for step in range(max_delay + 1)]
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        revision=DATASET_REVISION,
        delta_timestamps={key: list(temporal_offsets) for key in temporal_keys},
        video_backend=VIDEO_BACKEND,
    )

    policy, original_checkpoint_cameras = _configure_policy(
        metadata,
        checkpoint_snapshot=checkpoint_snapshot,
        checkpoint_revision=CHECKPOINT_REVISION,
        vlm_snapshot=vlm_snapshot,
        device=str(device),
    )
    preprocessor, postprocessor, rename_map, preprocessor_overrides = _build_processors(
        policy.config,
        checkpoint_snapshot=checkpoint_snapshot,
        vlm_snapshot=vlm_snapshot,
        device=str(device),
    )
    # Model construction initializes modules before the checkpoint overwrites them and
    # therefore advances the RNG. Re-seed here so the recorded seed directly determines
    # the per-anchor evaluation noise sequence.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logging.info(
        "Running %d anchors x %d delays (%d paired records)",
        len(selected_candidates),
        len(delays),
        len(selected_candidates) * len(delays),
    )
    records: list[OracleEvaluationRecord] = []
    task: str | None = None
    with raw_path.open("x", encoding="utf-8") as raw_stream:
        for anchor_number, candidate in enumerate(selected_candidates, start=1):
            temporal_sample = dataset[candidate.anchor_id]
            sample_task = temporal_sample.get("task")
            if not isinstance(sample_task, str):
                raise TypeError(f"anchor_id={candidate.anchor_id} has non-string task {sample_task!r}")
            if task is None:
                task = sample_task
            elif sample_task != task:
                raise ValueError(
                    f"Approved one-task cohort drifted at anchor_id={candidate.anchor_id}: {sample_task!r}"
                )

            policy.reset()
            current_observation = slice_temporal_sample(
                temporal_sample,
                temporal_keys=temporal_keys,
                step=0,
            )
            current_batch = preprocessor(current_observation)
            temporal_batch = preprocessor(_clone_mapping(temporal_sample))
            noise = policy.model.sample_noise(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                device=device,
            )

            for delay in delays:
                future_observation = slice_temporal_sample(
                    temporal_sample,
                    temporal_keys=temporal_keys,
                    step=delay,
                )
                future_batch = preprocessor(future_observation)
                oracle_kwargs = encode_oracle_future_tokens(
                    policy,
                    temporal_batch,
                    delay_steps=delay,
                )
                policy_output = run_with_shared_noise(
                    noise,
                    current=lambda path_noise, batch=current_batch: policy.predict_action_chunk(
                        _clone_mapping(batch), noise=path_noise
                    ),
                    oracle_visual=lambda path_noise, batch=current_batch, kwargs=oracle_kwargs: (
                        policy.predict_action_chunk(_clone_mapping(batch), noise=path_noise, **kwargs)
                    ),
                    full_future_teacher=lambda path_noise, batch=future_batch: policy.predict_action_chunk(
                        _clone_mapping(batch), noise=path_noise
                    ),
                )
                post_policy = _postprocess_triplet(postprocessor, policy_output)
                record = make_evaluation_record(
                    anchor_id=candidate.anchor_id,
                    episode_index=candidate.episode_index,
                    frame_index=candidate.frame_index,
                    delay_steps=delay,
                    policy_output=policy_output,
                    post_policy=post_policy,
                )
                records.append(record)
                raw_stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            raw_stream.flush()
            logging.info(
                "Completed anchor %d/%d (episode=%d frame=%d anchor_id=%d)",
                anchor_number,
                len(selected_candidates),
                candidate.episode_index,
                candidate.frame_index,
                candidate.anchor_id,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_summary = aggregate_by_delay(records)
    completed_at = datetime.now(UTC)
    completed_perf = time.perf_counter()
    original_camera_set = set(original_checkpoint_cameras)
    mapped_checkpoint_cameras = set(POLICY_CAMERA_KEYS)
    episode_counts = Counter(candidate.episode_index for candidate in selected_candidates)

    manifest = {
        "classification": "mechanism_sensitivity_upper_bound_not_task_capability",
        "git_sha": _git_sha(),
        "command": [sys.executable, *sys.argv],
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_s": completed_perf - started_perf,
        "inputs": {
            "dataset": {
                "repo_id": DATASET_REPO_ID,
                "requested_revision": DATASET_REVISION,
                "resolved_revision": dataset_snapshot.name,
                "snapshot_path": str(dataset_snapshot.resolve()),
                "fps": metadata.fps,
                "total_episodes": metadata.total_episodes,
                "total_frames": metadata.total_frames,
                "video_backend": VIDEO_BACKEND,
            },
            "checkpoint": {
                "repo_id": CHECKPOINT_REPO_ID,
                "requested_revision": CHECKPOINT_REVISION,
                "resolved_revision": checkpoint_snapshot.name,
                "snapshot_path": str(checkpoint_snapshot.resolve()),
            },
            "vlm": {
                "repo_id": VLM_REPO_ID,
                "requested_revision": VLM_REVISION,
                "resolved_revision": vlm_snapshot.name,
                "snapshot_path": str(vlm_snapshot.resolve()),
            },
        },
        "protocol": {
            "seed": args.seed,
            "delays": list(delays),
            "primary_delays": [delay for delay in delays if delay in APPROVED_PRIMARY_DELAYS],
            "diagnostic_delays": [delay for delay in delays if delay in APPROVED_DIAGNOSTIC_DELAYS],
            "approved_full_protocol": args.anchor_count == 128 and delays == APPROVED_DELAYS,
            "task": task,
            "anchor_count": len(selected_candidates),
            "anchor_ids": [candidate.anchor_id for candidate in selected_candidates],
            "anchors": [
                {
                    "anchor_id": candidate.anchor_id,
                    "episode_index": candidate.episode_index,
                    "frame_index": candidate.frame_index,
                }
                for candidate in selected_candidates
            ],
            "selected_anchors_by_episode": {
                str(episode_index): count for episode_index, count in sorted(episode_counts.items())
            },
            "candidate_count": len(candidates),
            "valid_candidate_count_through_max_delay": valid_candidate_count,
            "boundary_filtered_count": len(candidates) - valid_candidate_count,
            "per_delay_sample_count": len(selected_candidates),
            "total_record_count": len(records),
            "noise_contract": (
                "one noise per anchor, reused across delays; three independent clones per paired record"
            ),
        },
        "camera_adaptation": {
            "dataset_to_policy": rename_map,
            "policy_camera_order": list(policy.config.image_features),
            "checkpoint_camera_order_before_adaptation": list(original_checkpoint_cameras),
            "unmapped_checkpoint_cameras": sorted(original_camera_set - mapped_checkpoint_cameras),
        },
        "processor_provenance": {
            "source_checkpoint": CHECKPOINT_REPO_ID,
            "source_revision": CHECKPOINT_REVISION,
            "preprocessor_config": "policy_preprocessor.json",
            "postprocessor_config": "policy_postprocessor.json",
            "preprocessor_overrides": preprocessor_overrides,
            "post_policy_definition": (
                "the same checkpoint postprocessor applied to all three policy-output chunks"
            ),
        },
        "device": _device_manifest(device),
        "software_versions": _software_versions(),
        "local_files_only": args.local_files_only,
        "artifacts": {
            "per_sample_jsonl": str(raw_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }
    payload = {"manifest": manifest, "summary": evaluation_summary.to_dict()}
    _write_json(summary_path, payload)
    print(
        json.dumps(
            {
                "raw_jsonl": str(raw_path.resolve()),
                "summary_json": str(summary_path.resolve()),
                "records": len(records),
                "duration_s": manifest["duration_s"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
