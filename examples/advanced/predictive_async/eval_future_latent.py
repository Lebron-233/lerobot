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

"""Run the frozen B3.2 validation-only future-latent characterization."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

if __package__:
    from .eval_oracle_upper_bound import (
        CHECKPOINT_REPO_ID,
        CHECKPOINT_REVISION,
        DATASET_REPO_ID,
        DATASET_REVISION,
        VLM_REPO_ID,
        VLM_REVISION,
        _build_processors,
        _configure_policy,
    )
    from .future_latent_evaluation import (
        EXPECTED_CACHE_PRODUCER_SHA,
        EXPECTED_TRAINER_SHA,
        aggregate_action_records,
        compute_risk_calibration,
        evaluate_four_path_pair,
        evaluate_latent_risk,
        frozen_postprocessor_provenance,
        load_frozen_best_predictor,
        select_val_anchor_pairs,
    )
else:
    from eval_oracle_upper_bound import (
        CHECKPOINT_REPO_ID,
        CHECKPOINT_REVISION,
        DATASET_REPO_ID,
        DATASET_REVISION,
        VLM_REPO_ID,
        VLM_REVISION,
        _build_processors,
        _configure_policy,
    )
    from future_latent_evaluation import (
        EXPECTED_CACHE_PRODUCER_SHA,
        EXPECTED_TRAINER_SHA,
        aggregate_action_records,
        compute_risk_calibration,
        evaluate_four_path_pair,
        evaluate_latent_risk,
        frozen_postprocessor_provenance,
        load_frozen_best_predictor,
        select_val_anchor_pairs,
    )

SEED = 0
ANCHOR_COUNT = 128
DELAYS = tuple(range(1, 9))
LATENT_BATCH_SIZE = 16
TEST_EPISODE_IDS = (15, 29, 31, 33, 41)
SHARED_DATASET_CONTAINERS = (
    "data/chunk-000/file-000.parquet",
    "videos/observation.images.top/chunk-000/file-000.mp4",
    "videos/observation.images.wrist/chunk-000/file-000.mp4",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve pinned Hub inputs from the local cache without network access.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def evaluation_protocol_markers() -> dict[str, bool | None]:
    """Return the fixed val-only markers written into every B3.2 summary."""
    return {
        "protocol_complete": True,
        "test_data_read": False,
        "eligible_for_test": False,
        "risk_thresholds": None,
    }


def _git_sha() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_evaluator_snapshots(*, local_files_only: bool) -> tuple[Path, Path, Path]:
    """Resolve dataset metadata plus the two complete frozen model snapshots."""
    dataset_snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=HF_LEROBOT_HUB_CACHE,
            allow_patterns="meta/**",
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path: Path, records: tuple[Any, ...]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            json.dump(record.to_dict(), stream, sort_keys=True, allow_nan=False)
            stream.write("\n")


def _relative_difference(actual: float, expected: float) -> float:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise RuntimeError("latent reproduction metrics must be finite")
    if expected == 0.0:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


def _compare_latent_reproduction(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, float] = {}
    metric_names = ("smoothl1", "mse", "cosine")
    if actual["total_record_count"] != expected["sample_count"]:
        raise RuntimeError("full-val latent record count did not reproduce B3.1")
    actual_by_delay = {entry["delay_steps"]: entry for entry in actual["per_delay"]}
    for delay in DELAYS:
        actual_delay = actual_by_delay[delay]
        expected_delay = expected["per_delay"][str(delay)]
        if actual_delay["sample_count"] != expected_delay["sample_count"]:
            raise RuntimeError(f"delay {delay} sample count did not reproduce B3.1")
        for path in ("identity", "predicted"):
            for metric in metric_names:
                name = f"delay_{delay}.{path}.{metric}"
                comparisons[name] = _relative_difference(
                    float(actual_delay[path][metric]), float(expected_delay[path][metric])
                )
        comparisons[f"delay_{delay}.risk_smoothl1"] = _relative_difference(
            float(actual_delay["risk_smoothl1"]), float(expected_delay["risk_smoothl1"])
        )

    expected_macro = expected["macro"]
    for path in ("identity", "predicted"):
        actual_macro = actual[f"macro_{path}"]
        for metric in metric_names:
            name = f"macro.{path}.{metric}"
            comparisons[name] = _relative_difference(
                float(actual_macro[metric]), float(expected_macro[f"{path}_{metric}"])
            )
    comparisons["macro.risk_smoothl1"] = _relative_difference(
        float(actual["macro_risk_smoothl1"]), float(expected_macro["risk_smoothl1"])
    )
    maximum = max(comparisons.values(), default=0.0)
    if maximum > 1e-6:
        worst = max(comparisons, key=comparisons.__getitem__)
        raise RuntimeError(
            "independent full-val metrics did not reproduce B3.1 within relative tolerance: "
            f"{worst}={maximum:.9g}"
        )
    return {
        "relative_tolerance": 1e-6,
        "maximum_relative_difference": maximum,
        "passed": True,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started_at = datetime.now(UTC)
    implementation_sha = _git_sha()
    frozen = load_frozen_best_predictor(
        args.predictor_checkpoint,
        val_cache=args.val_cache,
        device=device,
    )

    latent_path = args.output_dir / "latent_risk_per_pair.jsonl"
    action_path = args.output_dir / "action_per_sample.jsonl"
    summary_path = args.output_dir / "summary.json"

    logging.info("Evaluating all 14,396 canonical validation latent pairs")
    latent_evaluation = evaluate_latent_risk(
        frozen.predictor,
        frozen.val_dataset,
        device=device,
        batch_size=LATENT_BATCH_SIZE,
    )
    latent_summary = latent_evaluation.summary.to_dict()
    reproduction = _compare_latent_reproduction(
        latent_summary,
        frozen.checkpoint["best_val_metrics"],
    )
    risk_summary = compute_risk_calibration(latent_evaluation.records)

    dataset_snapshot, checkpoint_snapshot, vlm_snapshot = _resolve_evaluator_snapshots(
        local_files_only=args.local_files_only
    )
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_snapshot,
        revision=DATASET_REVISION,
    )
    policy, _ = _configure_policy(
        metadata,
        checkpoint_snapshot=checkpoint_snapshot,
        checkpoint_revision=CHECKPOINT_REVISION,
        vlm_snapshot=vlm_snapshot,
        device=str(device),
    )
    _, postprocessor, _, _ = _build_processors(
        policy.config,
        checkpoint_snapshot=checkpoint_snapshot,
        vlm_snapshot=vlm_snapshot,
        device=str(device),
    )

    # Loading the frozen modules advances RNG state. This re-seed makes the fixed
    # protocol seed directly determine the one shared flow-noise draw per anchor.
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    anchors = select_val_anchor_pairs(frozen.val_dataset, count=ANCHOR_COUNT, seed=SEED)
    action_records = []
    logging.info("Running %d validation anchors x %d delays", ANCHOR_COUNT, len(DELAYS))
    with torch.inference_mode():
        for anchor_number, anchor in enumerate(anchors, start=1):
            policy.reset()
            noise = policy.model.sample_noise(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                device=device,
            )
            for pair in anchor.pairs:
                action_records.append(
                    evaluate_four_path_pair(
                        policy,
                        frozen.predictor,
                        pair,
                        noise,
                        postprocessor,
                        anchor_id=anchor.anchor_id,
                    )
                )
            logging.info(
                "Completed action anchor %d/%d (episode=%d frame=%d)",
                anchor_number,
                len(anchors),
                anchor.episode_index,
                anchor.frame_index,
            )
    action_records_tuple = tuple(action_records)
    if len(action_records_tuple) != ANCHOR_COUNT * len(DELAYS):
        raise RuntimeError("same-noise action evaluation did not produce exactly 1,024 records")
    action_summary = aggregate_action_records(action_records_tuple)

    checkpoint = frozen.checkpoint
    summary = {
        "schema_version": 1,
        "classification": "offline_future_latent_validation_only_not_task_capability",
        "implementation_git_sha": implementation_sha,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        **evaluation_protocol_markers(),
        "trainer_git_sha": EXPECTED_TRAINER_SHA,
        "cache_producer_sha": EXPECTED_CACHE_PRODUCER_SHA,
        "frozen_predictor": {
            "path": str(args.predictor_checkpoint.resolve()),
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "predictor_config": checkpoint["predictor_config"],
            "strict_reload": True,
        },
        "val_cache": {
            "path": str(args.val_cache.resolve()),
            "split": frozen.val_dataset.manifest["split"],
            "complete_split": frozen.val_dataset.manifest["complete_split"],
            "episode_ids": frozen.val_dataset.manifest["cached_episode_ids"],
            "frame_count": frozen.val_dataset.manifest["frame_count"],
            "pair_count_by_delay": frozen.val_dataset.manifest["valid_pair_count_by_delay"],
        },
        "fixed_protocol": {
            "seed": SEED,
            "delays": list(DELAYS),
            "anchor_count": ANCHOR_COUNT,
            "same_noise": "one draw per anchor reused across all delays; independent clone per path",
            "four_paths_use_cached_token_override": True,
            "future_state_rollout": False,
        },
        "test_isolation": {
            "authoritative_boundary": "logical_episode_and_sample",
            "excluded_test_episode_ids": list(TEST_EPISODE_IDS),
            "test_rows_frames_samples_pairs_anchors_metrics_selected": False,
            "action_evaluator_raw_dataset_access": "metadata only",
            "shared_physical_containers_include_test_episode_bytes": True,
            "shared_physical_containers": list(SHARED_DATASET_CONTAINERS),
        },
        "latent_risk_record_count": len(latent_evaluation.records),
        "latent": latent_summary,
        "b31_reproduction": reproduction,
        "risk_calibration": risk_summary.to_dict(),
        "action_record_count": len(action_records_tuple),
        "action": action_summary.to_dict(),
        "postprocessor_provenance": frozen_postprocessor_provenance(),
        "post_policy_interpretation": "checkpoint postprocessor secondary table; not physical action",
        "pinned_inputs": {
            "dataset": {"repo_id": DATASET_REPO_ID, "revision": DATASET_REVISION},
            "checkpoint": {"repo_id": CHECKPOINT_REPO_ID, "revision": CHECKPOINT_REVISION},
            "vlm": {"repo_id": VLM_REPO_ID, "revision": VLM_REVISION},
        },
    }
    args.output_dir.mkdir(parents=True)
    _write_jsonl(latent_path, latent_evaluation.records)
    _write_jsonl(action_path, action_records_tuple)
    _write_json(summary_path, summary)
    logging.info("Wrote B3.2 validation-only outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
