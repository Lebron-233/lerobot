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

"""Run the frozen one-shot B4 held-out future-latent evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Mapping, Sequence
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
        FROZEN_ANCHOR_COUNT,
        FROZEN_DELAYS,
        FROZEN_SEED,
        FROZEN_TEST_EPISODES,
        FROZEN_TEST_FRAME_COUNT,
        FROZEN_TEST_PAIR_COUNT,
        aggregate_action_records,
        compute_risk_calibration,
        evaluate_four_path_pair,
        evaluate_test_latent_risk,
        frozen_postprocessor_provenance,
        load_frozen_test_predictor,
        select_test_anchor_pairs,
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
        FROZEN_ANCHOR_COUNT,
        FROZEN_DELAYS,
        FROZEN_SEED,
        FROZEN_TEST_EPISODES,
        FROZEN_TEST_FRAME_COUNT,
        FROZEN_TEST_PAIR_COUNT,
        aggregate_action_records,
        compute_risk_calibration,
        evaluate_four_path_pair,
        evaluate_test_latent_risk,
        frozen_postprocessor_provenance,
        load_frozen_test_predictor,
        select_test_anchor_pairs,
    )

LATENT_BATCH_SIZE = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-cache", type=Path, required=True)
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
    """Return the frozen markers written only after a complete held-out run."""
    return {
        "test_protocol_complete": True,
        "test_data_read": True,
        "used_for_checkpoint_or_hyperparameter_selection": False,
        "risk_thresholds": None,
        "online_or_m5_authorized": False,
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
    """Resolve metadata plus the two pinned model snapshots without dataset samples."""
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


def _write_jsonl(path: Path, records: Sequence[Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            json.dump(record.to_dict(), stream, sort_keys=True, allow_nan=False)
            stream.write("\n")


def summarize_heldout_gates(
    latent_summary: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    risk_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen held-out gates without allowing risk to change the cohort."""
    per_delay = latent_summary["per_delay"]
    if [entry["delay_steps"] for entry in per_delay] != list(FROZEN_DELAYS):
        raise ValueError("held-out latent summary must contain canonical d=1..8 results")
    improved_delays = tuple(
        entry["delay_steps"]
        for entry in per_delay
        if entry["predicted"]["smoothl1"] < entry["identity"]["smoothl1"]
    )
    latent_macro_improved = (
        latent_summary["macro_predicted"]["smoothl1"] < latent_summary["macro_identity"]["smoothl1"]
    )
    latent_passed = latent_macro_improved and len(improved_delays) >= 7
    action_passed = bool(action_summary["test_protocol_prerequisites_met"])
    risk_passed = bool(risk_summary["risk_gating_eligible"])
    heldout_passed = latent_passed and action_passed
    return {
        "latent": {
            "macro_smoothl1_improved": latent_macro_improved,
            "improved_delay_count": len(improved_delays),
            "improved_delays": list(improved_delays),
            "passed": latent_passed,
        },
        "action": {"passed": action_passed},
        "risk_diagnostic": {
            "generalization_eligible": risk_passed,
            "alters_latent_or_action_gate": False,
        },
        "m3_heldout_passed": heldout_passed,
        "m5_planning_review_eligible": heldout_passed,
    }


def build_summary(
    *,
    implementation_sha: str,
    started_at: datetime,
    completed_at: datetime,
    command: Sequence[str],
    predictor_checkpoint: Path,
    checkpoint: Mapping[str, Any],
    test_cache: Path,
    test_manifest: Mapping[str, Any],
    latent_summary: Mapping[str, Any],
    latent_record_count: int,
    risk_summary: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    action_record_count: int,
) -> dict[str, Any]:
    """Build the frozen, independently recomputable B4 held-out summary."""
    return {
        "schema_version": 1,
        "classification": "offline_future_latent_heldout_test_not_task_capability",
        "implementation_git_sha": implementation_sha,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "command": list(command),
        **evaluation_protocol_markers(),
        "trainer_git_sha": EXPECTED_TRAINER_SHA,
        "training_cache_producer_sha": EXPECTED_CACHE_PRODUCER_SHA,
        "frozen_predictor": {
            "path": str(predictor_checkpoint.resolve()),
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "predictor_config": checkpoint["predictor_config"],
            "strict_reload": True,
        },
        "test_cache": {
            "path": str(test_cache.resolve()),
            "producer_git_sha": test_manifest["producer"]["git_sha"],
            "split": test_manifest["split"],
            "complete_split": test_manifest["complete_split"],
            "episode_ids": test_manifest["cached_episode_ids"],
            "frame_count": test_manifest["frame_count"],
            "pair_count_by_delay": test_manifest["valid_pair_count_by_delay"],
        },
        "fixed_protocol": {
            "test_episode_ids": list(FROZEN_TEST_EPISODES),
            "test_frame_count": FROZEN_TEST_FRAME_COUNT,
            "latent_pair_count": FROZEN_TEST_PAIR_COUNT,
            "seed": FROZEN_SEED,
            "delays": list(FROZEN_DELAYS),
            "anchor_count": FROZEN_ANCHOR_COUNT,
            "anchor_selection": "select_common_anchor_ids_episode_stratified",
            "same_noise": "one draw per anchor reused across all delays; independent clone per path",
            "current": "(Z_t,s_t)",
            "oracle_visual": "(Z_t+d,s_t)",
            "predicted_visual": "(Z_t+DeltaZ,s_t)",
            "full_future_teacher": "(Z_t+d,s_t+d)",
            "four_paths_use_cached_token_override": True,
            "future_state_rollout": False,
            "language_rollout": False,
            "fallback": None,
            "residual_scale": None,
        },
        "latent_risk_record_count": latent_record_count,
        "latent": dict(latent_summary),
        "risk_calibration": dict(risk_summary),
        "action_record_count": action_record_count,
        "action": dict(action_summary),
        "heldout_gates": summarize_heldout_gates(latent_summary, action_summary, risk_summary),
        "postprocessor_provenance": frozen_postprocessor_provenance(),
        "post_policy_interpretation": "checkpoint postprocessor secondary table; not physical action",
        "pinned_inputs": {
            "dataset": {"repo_id": DATASET_REPO_ID, "revision": DATASET_REVISION},
            "checkpoint": {"repo_id": CHECKPOINT_REPO_ID, "revision": CHECKPOINT_REVISION},
            "vlm": {"repo_id": VLM_REPO_ID, "revision": VLM_REVISION},
        },
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
    frozen = load_frozen_test_predictor(
        args.predictor_checkpoint,
        test_cache=args.test_cache,
        expected_test_cache_producer_sha=implementation_sha,
        device=device,
    )

    latent_path = args.output_dir / "latent_risk_per_pair.jsonl"
    action_path = args.output_dir / "action_per_sample.jsonl"
    summary_path = args.output_dir / "summary.json"

    logging.info("Evaluating all %d canonical held-out latent pairs", FROZEN_TEST_PAIR_COUNT)
    latent_evaluation = evaluate_test_latent_risk(
        frozen.predictor,
        frozen.test_dataset,
        device=device,
        batch_size=LATENT_BATCH_SIZE,
    )
    latent_summary = latent_evaluation.summary.to_dict()
    risk_summary = compute_risk_calibration(latent_evaluation.records).to_dict()

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

    torch.manual_seed(FROZEN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(FROZEN_SEED)

    anchors = select_test_anchor_pairs(
        frozen.test_dataset,
        count=FROZEN_ANCHOR_COUNT,
        seed=FROZEN_SEED,
    )
    action_records = []
    logging.info("Running %d held-out anchors x %d delays", FROZEN_ANCHOR_COUNT, len(FROZEN_DELAYS))
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
                "Completed held-out action anchor %d/%d (episode=%d frame=%d)",
                anchor_number,
                len(anchors),
                anchor.episode_index,
                anchor.frame_index,
            )
    action_records_tuple = tuple(action_records)
    if len(action_records_tuple) != FROZEN_ANCHOR_COUNT * len(FROZEN_DELAYS):
        raise RuntimeError("same-noise held-out action evaluation did not produce exactly 1,024 records")
    action_summary = aggregate_action_records(action_records_tuple).to_dict()

    summary = build_summary(
        implementation_sha=implementation_sha,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        command=[sys.executable, *sys.argv],
        predictor_checkpoint=args.predictor_checkpoint,
        checkpoint=frozen.checkpoint,
        test_cache=args.test_cache,
        test_manifest=frozen.test_dataset.manifest,
        latent_summary=latent_summary,
        latent_record_count=len(latent_evaluation.records),
        risk_summary=risk_summary,
        action_summary=action_summary,
        action_record_count=len(action_records_tuple),
    )

    args.output_dir.mkdir(parents=True)
    _write_jsonl(latent_path, latent_evaluation.records)
    _write_jsonl(action_path, action_records_tuple)
    _write_json(summary_path, summary)
    logging.info("Wrote frozen held-out outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
