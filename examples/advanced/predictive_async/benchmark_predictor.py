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

"""Benchmark the frozen future-latent predictor against one full SmolVLA chunk."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor

import lerobot.policies  # noqa: F401 - registers policy/config subclasses
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.rollout.inference.oracle_evaluation import remap_checkpoint_action_stats
from lerobot.utils.constants import ACTION, HF_LEROBOT_HUB_CACHE, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

if __package__:
    from .future_latent_evaluation import load_frozen_best_predictor, select_val_anchor_pairs
else:
    from future_latent_evaluation import load_frozen_best_predictor, select_val_anchor_pairs


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
VAL_EPISODE_IDS = (5, 7, 14, 39, 49)
TEST_EPISODE_IDS = (15, 29, 31, 33, 41)
PREDICTION_DELAYS = tuple(range(1, 9))
ANCHOR_COUNT = 128
ANCHOR_SEED = 0
EXPECTED_FIRST_ANCHOR_ID = 5560
EXPECTED_FIRST_ANCHOR_EPISODE = 14
EXPECTED_FIRST_ANCHOR_FRAME = 116
BENCHMARK_DELAY = 1
WARMUP_COUNT = 50
MEASUREMENT_COUNT = 200
EFFICIENCY_TARGET_RATIO = 0.05
VIDEO_BACKEND = "pyav"
POSTPROCESSOR_STATE_FILE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
POSTPROCESSOR_STATS_SOURCE_KEY = "so100.buffer.action"


@dataclass(frozen=True, slots=True)
class PredictorBenchmarkInputs:
    current_image_tokens: tuple[Tensor, ...]
    current_image_token_masks: tuple[Tensor, ...]
    committed_actions: Tensor
    committed_mask: Tensor
    current_state: Tensor
    delay_steps: Tensor


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    samples_ms: tuple[float, ...]
    p50_ms: float
    p90_ms: float
    peak_extra_allocated_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples_ms),
            "samples_ms": list(self.samples_ms),
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "peak_extra_allocated_bytes": self.peak_extra_allocated_bytes,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def latency_percentiles(samples_ms: Sequence[float]) -> dict[str, float]:
    values = sorted(float(value) for value in samples_ms)
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("latency samples must be non-empty, finite, and non-negative")

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"p50_ms": percentile(0.5), "p90_ms": percentile(0.9)}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_cuda_latency(
    call: Callable[[], Any],
    *,
    device: torch.device,
    warmup_count: int = WARMUP_COUNT,
    measurement_count: int = MEASUREMENT_COUNT,
    measure_peak_memory: bool = False,
) -> LatencyMeasurement:
    """Measure independent synchronized wall-clock calls; CPU support is for focused tests."""
    if warmup_count < 0 or measurement_count <= 0:
        raise ValueError("warmup_count must be non-negative and measurement_count must be positive")
    if measure_peak_memory and device.type != "cuda":
        raise ValueError("peak allocated memory is only available for a CUDA benchmark")

    with torch.inference_mode():
        for _ in range(warmup_count):
            _synchronize(device)
            output = call()
            _synchronize(device)
            del output

        _synchronize(device)
        baseline_allocated = None
        if measure_peak_memory:
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        samples_ms: list[float] = []
        for _ in range(measurement_count):
            _synchronize(device)
            started_at = time.perf_counter()
            output = call()
            _synchronize(device)
            samples_ms.append((time.perf_counter() - started_at) * 1_000.0)
            del output

        peak_extra_allocated_bytes = None
        if baseline_allocated is not None:
            peak_extra_allocated_bytes = max(
                0,
                torch.cuda.max_memory_allocated(device) - baseline_allocated,
            )

    percentiles = latency_percentiles(samples_ms)
    return LatencyMeasurement(
        samples_ms=tuple(samples_ms),
        p50_ms=percentiles["p50_ms"],
        p90_ms=percentiles["p90_ms"],
        peak_extra_allocated_bytes=peak_extra_allocated_bytes,
    )


def call_public_predictor(predictor: Any, inputs: PredictorBenchmarkInputs) -> Any:
    """Use the normal module call, which executes the public Phase A forward and its host checks."""
    return predictor(
        inputs.current_image_tokens,
        inputs.current_image_token_masks,
        inputs.committed_actions,
        inputs.committed_mask,
        inputs.current_state,
        inputs.delay_steps,
    )


def call_public_policy_chunk(policy: Any, batch: dict[str, Any], noise: Tensor) -> Tensor:
    """Use the public normal-RGB policy path without profiling or token overrides."""
    return policy.predict_action_chunk(batch, noise=noise)


def _batched_predictor_inputs(pair: Any, device: torch.device) -> PredictorBenchmarkInputs:
    def batched(tensor: Tensor) -> Tensor:
        return tensor.unsqueeze(0).to(device=device)

    return PredictorBenchmarkInputs(
        current_image_tokens=tuple(batched(tensor) for tensor in pair.current_image_tokens),
        current_image_token_masks=tuple(batched(tensor) for tensor in pair.current_image_token_masks),
        committed_actions=batched(pair.committed_actions),
        committed_mask=batched(pair.committed_mask),
        current_state=batched(pair.current_state),
        delay_steps=batched(pair.delay_steps),
    )


def _predictor_input_spec(inputs: PredictorBenchmarkInputs) -> dict[str, Any]:
    def spec(tensor: Tensor) -> dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "device": str(tensor.device),
        }

    return {
        "current_image_tokens": [spec(tensor) for tensor in inputs.current_image_tokens],
        "current_image_token_masks": [spec(tensor) for tensor in inputs.current_image_token_masks],
        "committed_actions": spec(inputs.committed_actions),
        "committed_mask": spec(inputs.committed_mask),
        "current_state": spec(inputs.current_state),
        "delay_steps": spec(inputs.delay_steps),
    }


def _episode_container_paths(metadata: LeRobotDatasetMetadata, episode_ids: Sequence[int]) -> set[str]:
    paths = {str(metadata.get_data_file_path(episode_index)) for episode_index in episode_ids}
    paths.update(
        str(metadata.get_video_file_path(episode_index, video_key))
        for video_key in metadata.video_keys
        for episode_index in episode_ids
    )
    return paths


def _resolve_baseline_snapshots(*, local_files_only: bool) -> tuple[Path, Path, Path]:
    metadata_snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            allow_patterns="meta/**",
            local_files_only=local_files_only,
        )
    )
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=metadata_snapshot,
        revision=DATASET_REVISION,
    )
    val_paths = {"meta/**", *_episode_container_paths(metadata, VAL_EPISODE_IDS)}
    dataset_snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            allow_patterns=sorted(val_paths),
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


def _configure_baseline_policy(
    metadata: LeRobotDatasetMetadata,
    *,
    checkpoint_snapshot: Path,
    vlm_snapshot: Path,
    device: torch.device,
) -> Any:
    config = PreTrainedConfig.from_pretrained(checkpoint_snapshot, local_files_only=True)
    if config.type != "smolvla":
        raise ValueError(f"latency baseline requires SmolVLA, got {config.type!r}")

    dataset_features = dataset_to_policy_features(metadata.features)
    required_features = (ACTION, OBS_STATE, *DATASET_CAMERA_KEYS)
    missing_features = [key for key in required_features if key not in dataset_features]
    if missing_features:
        raise KeyError(f"pinned dataset is missing required features: {missing_features}")

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
            f"adapted policy camera order is {tuple(config.image_features)!r}, expected {POLICY_CAMERA_KEYS!r}"
        )
    if config.action_feature is None or tuple(config.action_feature.shape) != (6,):
        shape = None if config.action_feature is None else tuple(config.action_feature.shape)
        raise ValueError(f"pinned policy action shape is {shape}, expected (6,)")
    if config.adapt_to_pi_aloha:
        raise ValueError("pinned SO100 baseline does not permit PI-Aloha adaptation")
    if config.rtc_config is not None and config.rtc_config.enabled:
        raise ValueError("latency baseline must use the normal non-RTC action-chunk path")

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


def _build_baseline_preprocessor(
    config: Any,
    *,
    checkpoint_snapshot: Path,
    vlm_snapshot: Path,
    device: torch.device,
) -> Any:
    rename_map = dict(zip(DATASET_CAMERA_KEYS, POLICY_CAMERA_KEYS, strict=True))
    action_feature = config.action_feature
    if action_feature is None:
        raise ValueError("SmolVLA checkpoint config is missing its action output feature")
    processor_state = load_file(str(checkpoint_snapshot / POSTPROCESSOR_STATE_FILE), device="cpu")
    checkpoint_action_stats = remap_checkpoint_action_stats(
        processor_state,
        source_key=POSTPROCESSOR_STATS_SOURCE_KEY,
        action_dim=action_feature.shape[0],
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint_snapshot),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(vlm_snapshot)},
            "device_processor": {"device": str(device)},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {"stats": checkpoint_action_stats},
        },
    )
    return preprocessor


def _filtered_val_row(manifest: dict[str, Any], *, episode_index: int, frame_index: int) -> int:
    row = 0
    for entry in manifest["episodes"]:
        if int(entry["episode_index"]) == episode_index:
            if not 0 <= frame_index < int(entry["frame_count"]):
                raise ValueError(f"frame {frame_index} is outside val episode {episode_index}")
            return row + frame_index
        row += int(entry["frame_count"])
    raise ValueError(f"episode {episode_index} is not present in the val cache")


def _as_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, Tensor) else int(value)


def _prepare_baseline_call(
    manifest: dict[str, Any],
    anchor: Any,
    *,
    device: torch.device,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any], Tensor, dict[str, Any]]:
    dataset_snapshot, checkpoint_snapshot, vlm_snapshot = _resolve_baseline_snapshots(
        local_files_only=local_files_only
    )
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        revision=DATASET_REVISION,
    )
    shared_test_containers = sorted(
        _episode_container_paths(metadata, VAL_EPISODE_IDS)
        & _episode_container_paths(metadata, TEST_EPISODE_IDS)
    )
    policy = _configure_baseline_policy(
        metadata,
        checkpoint_snapshot=checkpoint_snapshot,
        vlm_snapshot=vlm_snapshot,
        device=device,
    )
    preprocessor = _build_baseline_preprocessor(
        policy.config,
        checkpoint_snapshot=checkpoint_snapshot,
        vlm_snapshot=vlm_snapshot,
        device=device,
    )

    episode_ids = tuple(int(value) for value in manifest["cached_episode_ids"])
    if episode_ids != VAL_EPISODE_IDS:
        raise ValueError(f"benchmark requires frozen val episodes {VAL_EPISODE_IDS}, got {episode_ids}")
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        episodes=list(episode_ids),
        revision=DATASET_REVISION,
        video_backend=VIDEO_BACKEND,
    )
    row = _filtered_val_row(
        manifest,
        episode_index=anchor.episode_index,
        frame_index=anchor.frame_index,
    )
    raw_observation = dataset[row]
    actual_episode = _as_int(raw_observation["episode_index"])
    actual_frame = _as_int(raw_observation["frame_index"])
    actual_anchor_id = _as_int(raw_observation["index"])
    if (actual_anchor_id, actual_episode, actual_frame) != (
        anchor.anchor_id,
        anchor.episode_index,
        anchor.frame_index,
    ):
        raise ValueError("filtered raw val observation does not match the frozen cache anchor")

    processed_batch = preprocessor(raw_observation)
    policy.reset()
    torch.manual_seed(ANCHOR_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(ANCHOR_SEED)
    fixed_noise = policy.model.sample_noise(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        device=device,
    )
    _synchronize(device)
    baseline_provenance = {
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "snapshot": str(dataset_snapshot),
            "episodes_loaded": list(episode_ids),
            "episode_isolation": {
                "authoritative_boundary": "logical_episode_and_sample",
                "excluded_test_episode_ids": list(TEST_EPISODE_IDS),
                "test_rows_frames_pairs_metrics_selected": False,
                "shared_physical_containers_include_test_episode_bytes": True,
                "shared_physical_containers": shared_test_containers,
            },
        },
        "checkpoint": {
            "repo_id": CHECKPOINT_REPO_ID,
            "revision": CHECKPOINT_REVISION,
            "snapshot": str(checkpoint_snapshot),
        },
        "vlm": {
            "repo_id": VLM_REPO_ID,
            "revision": VLM_REVISION,
            "snapshot": str(vlm_snapshot),
        },
        "preprocessor_outside_timed_region": True,
        "disk_and_video_decode_outside_timed_region": True,
        "fixed_noise_shape": list(fixed_noise.shape),
        "fixed_noise_dtype": str(fixed_noise.dtype).removeprefix("torch."),
    }
    return policy, processed_batch, fixed_noise, baseline_provenance


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_sha() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _device_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {"requested": str(device), "type": device.type}
    if device.type == "cuda":
        record.update(
            {
                "index": device.index if device.index is not None else torch.cuda.current_device(),
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return record


def build_benchmark_summary(
    *,
    anchor_record: dict[str, int],
    predictor_record: dict[str, Any],
    baseline_record: dict[str, Any],
    device_record: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    predictor_p90 = float(predictor_record["p90_ms"])
    baseline_p90 = float(baseline_record["p90_ms"])
    if not math.isfinite(predictor_p90) or not math.isfinite(baseline_p90) or baseline_p90 <= 0:
        raise ValueError("benchmark P90 values must be finite and baseline P90 must be positive")
    ratio = predictor_p90 / baseline_p90
    return {
        "schema_version": 1,
        "classification": "offline_latency_characterization_not_task_capability",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "anchor": anchor_record,
        "device": device_record,
        "protocol": {
            "warmup_count": WARMUP_COUNT,
            "measurement_count": MEASUREMENT_COUNT,
            "timer": "time.perf_counter",
            "cuda_synchronize_before_and_after_each_call": True,
            "predictor_public_forward": True,
            "predictor_input_host_check_included": True,
            "predictor_output_host_check_included": True,
            "baseline_public_predict_action_chunk": True,
            "baseline_normal_rgb_path": True,
            "baseline_fixed_flow_noise": True,
        },
        "predictor": predictor_record,
        "baseline_policy_chunk": baseline_record,
        "predictor_to_baseline_p90_ratio": ratio,
        "efficiency_target_ratio": EFFICIENCY_TARGET_RATIO,
        "host_checks_included": True,
        "m5_efficiency_eligible": device_record.get("type") == "cuda" and ratio < EFFICIENCY_TARGET_RATIO,
        "test_data_read": False,
        "test_data_read_definition": "no test row, frame, sample, pair, anchor, or metric selected",
        "eligible_for_test": False,
        "risk_thresholds": None,
        "provenance": provenance,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_json.exists():
        raise FileExistsError(f"output JSON already exists: {args.output_json}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    frozen = load_frozen_best_predictor(
        args.predictor_checkpoint,
        val_cache=args.val_cache,
        device=device,
    )
    anchors = select_val_anchor_pairs(frozen.val_dataset, count=ANCHOR_COUNT, seed=ANCHOR_SEED)
    if len(anchors) != ANCHOR_COUNT:
        raise RuntimeError(f"anchor selection returned {len(anchors)} anchors, expected {ANCHOR_COUNT}")
    anchor = anchors[0]
    if (anchor.anchor_id, anchor.episode_index, anchor.frame_index) != (
        EXPECTED_FIRST_ANCHOR_ID,
        EXPECTED_FIRST_ANCHOR_EPISODE,
        EXPECTED_FIRST_ANCHOR_FRAME,
    ):
        raise RuntimeError("seed-0 val anchor cohort does not start with the frozen benchmark anchor")
    if len(anchor.pairs) != len(PREDICTION_DELAYS):
        raise RuntimeError("benchmark anchor must contain one pair for every delay 1..8")
    pair = anchor.pairs[0]
    delay_steps = _as_int(pair.delay_steps)
    if delay_steps != BENCHMARK_DELAY:
        raise RuntimeError(f"first benchmark pair has delay {delay_steps}, expected {BENCHMARK_DELAY}")

    predictor_inputs = _batched_predictor_inputs(pair, device)
    _synchronize(device)
    predictor_measurement = measure_cuda_latency(
        lambda: call_public_predictor(frozen.predictor, predictor_inputs),
        device=device,
        measure_peak_memory=device.type == "cuda",
    )
    parameter_count = sum(parameter.numel() for parameter in frozen.predictor.parameters())
    predictor_record = {
        **predictor_measurement.to_dict(),
        "parameter_count": parameter_count,
        "input_spec": _predictor_input_spec(predictor_inputs),
        "path": "LightweightFutureLatentPredictor.public_forward",
        "host_checks_included": True,
    }

    manifest = frozen.val_dataset.manifest
    policy, processed_batch, fixed_noise, baseline_provenance = _prepare_baseline_call(
        manifest,
        anchor,
        device=device,
        local_files_only=args.local_files_only,
    )
    baseline_measurement = measure_cuda_latency(
        lambda: call_public_policy_chunk(policy, processed_batch, fixed_noise),
        device=device,
    )
    baseline_record = {
        **baseline_measurement.to_dict(),
        "path": "SmolVLAPolicy.public_predict_action_chunk.normal_rgb_current_observation",
        "image_encoder_included": True,
        "prefix_prefill_included": True,
        "action_expert_included": True,
        "fixed_flow_noise": True,
    }

    checkpoint = frozen.checkpoint
    anchor_record = {
        "anchor_id": int(anchor.anchor_id),
        "episode_index": int(anchor.episode_index),
        "frame_index": int(anchor.frame_index),
        "future_frame_index": int(pair.future_frame_index),
        "delay_steps": delay_steps,
        "selection_count": ANCHOR_COUNT,
        "selection_seed": ANCHOR_SEED,
    }
    provenance = {
        "benchmark_git_sha": _git_sha(),
        "predictor_checkpoint": {
            "path": str(args.predictor_checkpoint.resolve()),
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "trainer_git_sha": checkpoint["trainer_git_sha"],
            "cache_producer_sha": checkpoint["cache_producer_sha"],
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
        },
        "val_cache": {
            "path": str(args.val_cache.resolve()),
            "split": manifest["split"],
            "complete_split": manifest["complete_split"],
            "producer_git_sha": manifest["producer"]["git_sha"],
            "inputs": manifest["inputs"],
            "episode_ids": manifest["cached_episode_ids"],
        },
        "baseline": baseline_provenance,
        "software_versions": {
            "python": platform.python_version(),
            "lerobot": _package_version("lerobot"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    summary = build_benchmark_summary(
        anchor_record=anchor_record,
        predictor_record=predictor_record,
        baseline_record=baseline_record,
        device_record=_device_record(device),
        provenance=provenance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
