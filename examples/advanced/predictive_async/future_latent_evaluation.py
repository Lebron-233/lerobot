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

"""Validation-only characterization helpers for the frozen future-latent predictor."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor
from lerobot.rollout.inference.oracle_evaluation import OracleAnchorCandidate, select_common_anchor_ids
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

if __package__:
    from .future_latent_cache import FutureLatentPair
    from .future_latent_training import (
        FutureLatentCacheDataset,
        collate_future_latent_pairs,
        compute_future_latent_objective,
        compute_identity_baseline_metrics,
        forward_predictor,
        move_future_latent_batch,
    )
else:
    from future_latent_cache import FutureLatentPair
    from future_latent_training import (
        FutureLatentCacheDataset,
        collate_future_latent_pairs,
        compute_future_latent_objective,
        compute_identity_baseline_metrics,
        forward_predictor,
        move_future_latent_batch,
    )


EXPECTED_TRAINER_SHA = "9e618076f617751c297d92626ad422dbbf30c03b"
EXPECTED_CACHE_PRODUCER_SHA = "eff8be608c899d0841ad5967d80d5d726cbe4394"
EXPECTED_BEST_EPOCH = 29
EXPECTED_BEST_GLOBAL_STEP = 57_750

FROZEN_DELAYS = tuple(range(1, 9))
FROZEN_ANCHOR_COUNT = 128
FROZEN_SEED = 0
FROZEN_VAL_EPISODES = (5, 7, 14, 39, 49)
FROZEN_VAL_PAIR_COUNT = 14_396

DATASET_REPO_ID = "lerobot/svla_so100_pickplace"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
CHECKPOINT_REPO_ID = "lerobot/smolvla_base"
CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
POLICY_CAMERA_ORDER = (
    "observation.images.camera1",
    "observation.images.camera2",
)
POSTPROCESSOR_STATE_FILE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
POSTPROCESSOR_STATS_SOURCE_KEY = "so100.buffer.action"

_EXPECTED_SPLIT_COUNTS = {
    "train": {
        "episode_count": 40,
        "frame_count": 15_576,
        "valid_pair_count_by_delay": {
            "1": 15_536,
            "2": 15_496,
            "3": 15_456,
            "4": 15_416,
            "5": 15_376,
            "6": 15_336,
            "7": 15_296,
            "8": 15_256,
        },
    },
    "val": {
        "episode_count": 5,
        "frame_count": 1_822,
        "valid_pair_count_by_delay": {
            "1": 1_817,
            "2": 1_812,
            "3": 1_807,
            "4": 1_802,
            "5": 1_797,
            "6": 1_792,
            "7": 1_787,
            "8": 1_782,
        },
    },
}


@dataclass(frozen=True, slots=True)
class FrozenPredictor:
    predictor: LightweightFutureLatentPredictor
    checkpoint: dict[str, Any]
    val_dataset: FutureLatentCacheDataset


@dataclass(frozen=True, slots=True)
class LatentMetrics:
    smoothl1: float
    mse: float
    cosine: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatentRiskRecord:
    episode_index: int
    frame_index: int
    future_frame_index: int
    delay_steps: int
    predicted_risk: float
    actual_mse: float
    predicted: LatentMetrics
    identity: LatentMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatentDelaySummary:
    delay_steps: int
    sample_count: int
    predicted: LatentMetrics
    identity: LatentMetrics
    risk_smoothl1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatentRiskSummary:
    total_record_count: int
    per_delay: tuple[LatentDelaySummary, ...]
    macro_predicted: LatentMetrics
    macro_identity: LatentMetrics
    macro_risk_smoothl1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LatentRiskEvaluation:
    records: tuple[LatentRiskRecord, ...]
    summary: LatentRiskSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RiskBin:
    index: int
    count: int
    mean_predicted_risk: float
    mean_actual_mse: float
    min_predicted_risk: float
    max_predicted_risk: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskCalibrationSummary:
    record_count: int
    overall_spearman: float | None
    per_delay_spearman: dict[str, float | None]
    bins: tuple[RiskBin, ...]
    bottom_quintile_count: int
    top_quintile_count: int
    bottom_quintile_actual_mse_mean: float
    top_quintile_actual_mse_mean: float
    top_bottom_actual_mse_ratio: float | None
    top_bottom_ratio_defined: bool
    risk_gating_eligible: bool
    risk_thresholds: None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValAnchor:
    anchor_id: int
    episode_index: int
    frame_index: int
    pairs: tuple[FutureLatentPair, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "future_frame_indices": [pair.future_frame_index for pair in self.pairs],
            "delays": [int(pair.delay_steps.item()) for pair in self.pairs],
        }


@dataclass(frozen=True, slots=True)
class FourPathActions:
    current: Tensor
    oracle_visual: Tensor
    predicted_visual: Tensor
    full_future_teacher: Tensor


@dataclass(frozen=True, slots=True)
class ActionError:
    l1: float
    rmse: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionSpaceErrors:
    current_vs_teacher: ActionError
    oracle_visual_vs_teacher: ActionError
    predicted_visual_vs_teacher: ActionError

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FourPathActionRecord:
    anchor_id: int
    episode_index: int
    frame_index: int
    future_frame_index: int
    delay_steps: int
    policy_output: ActionSpaceErrors
    post_policy: ActionSpaceErrors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CaptureMetric:
    current: float
    oracle_visual: float
    predicted_visual: float
    oracle_gap: float
    capture: float | None
    capture_defined: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionSpaceAggregate:
    l1: CaptureMetric
    rmse: CaptureMetric

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionDelaySummary:
    delay_steps: int
    sample_count: int
    policy_output: ActionSpaceAggregate
    post_policy: ActionSpaceAggregate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionEvaluationSummary:
    anchor_ids: tuple[int, ...]
    total_record_count: int
    per_delay: tuple[ActionDelaySummary, ...]
    macro: ActionDelaySummary
    positive_policy_l1_capture_delay_count: int
    policy_l1_nonpositive_oracle_gap_delays: tuple[int, ...]
    max_policy_l1_regression: float | None
    max_policy_rmse_regression: float | None
    test_protocol_prerequisites_met: bool
    eligible_for_test: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a mean of an empty sequence")
    return math.fsum(values) / len(values)


def _manifest_cache_provenance(cache_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(cache_dir.resolve()),
        "split": manifest["split"],
        "complete_split": manifest["complete_split"],
        "producer_git_sha": manifest["producer"]["git_sha"],
        "inputs": manifest["inputs"],
        "policy_camera_order": manifest["policy_camera_order"],
        "episode_count": manifest["episode_count"],
        "frame_count": manifest["frame_count"],
        "valid_pair_count_by_delay": manifest["valid_pair_count_by_delay"],
        "semantics": manifest["semantics"],
        "token_scaling_convention": manifest["token_scaling_convention"],
    }


def _pathless_cache_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key != "path"}


def _validate_frozen_cache_provenance(provenance: Any, *, split: str) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"checkpoint {split} cache provenance must be an object")
    expected_counts = _EXPECTED_SPLIT_COUNTS[split]
    _require_equal(provenance.get("split"), split, name=f"{split} cache split")
    _require_equal(provenance.get("complete_split"), True, name=f"{split} cache completeness")
    _require_equal(
        provenance.get("producer_git_sha"),
        EXPECTED_CACHE_PRODUCER_SHA,
        name=f"{split} cache producer SHA",
    )
    _require_equal(
        provenance.get("policy_camera_order"),
        list(POLICY_CAMERA_ORDER),
        name=f"{split} cache camera order",
    )
    for field in ("episode_count", "frame_count", "valid_pair_count_by_delay"):
        _require_equal(
            provenance.get(field),
            expected_counts[field],
            name=f"{split} cache {field}",
        )

    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"checkpoint {split} cache must record pinned inputs")
    expected_inputs = {
        "dataset": (DATASET_REPO_ID, DATASET_REVISION),
        "checkpoint": (CHECKPOINT_REPO_ID, CHECKPOINT_REVISION),
        "vlm": (VLM_REPO_ID, VLM_REVISION),
    }
    for input_name, (repo_id, revision) in expected_inputs.items():
        entry = inputs.get(input_name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"checkpoint {split} cache is missing {input_name} provenance")
        _require_equal(entry.get("repo_id"), repo_id, name=f"{split} {input_name} repo_id")
        _require_equal(
            entry.get("requested_revision"), revision, name=f"{split} {input_name} requested revision"
        )
        _require_equal(
            entry.get("resolved_revision"), revision, name=f"{split} {input_name} resolved revision"
        )

    _require_equal(
        provenance.get("token_scaling_convention"),
        "native_post_sqrt_hidden_dim",
        name=f"{split} token scaling convention",
    )
    semantics = provenance.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError(f"checkpoint {split} cache must record state/action semantics")
    _require_equal(
        semantics.get("state"), "model_ready_normalized_and_padded", name=f"{split} state semantics"
    )
    _require_equal(
        semantics.get("action"),
        "normalized_policy_output_original_action_dim",
        name=f"{split} action semantics",
    )
    _require_equal(
        semantics.get("processor_config_source"),
        f"{CHECKPOINT_REPO_ID}@{CHECKPOINT_REVISION}",
        name=f"{split} processor config source",
    )


def _production_predictor_config() -> FutureLatentConfig:
    return FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, enabled=True)


def load_frozen_best_predictor(
    checkpoint_path: Path,
    *,
    val_cache: Path | FutureLatentCacheDataset,
    device: torch.device | str,
) -> FrozenPredictor:
    """Validate and load the one frozen B3.1 best predictor and its complete val cache."""
    val_dataset = (
        val_cache
        if isinstance(val_cache, FutureLatentCacheDataset)
        else FutureLatentCacheDataset(Path(val_cache), expected_split="val")
    )
    if val_dataset.expected_split != "val" or len(val_dataset) != FROZEN_VAL_PAIR_COUNT:
        raise ValueError("frozen predictor evaluation requires the complete 14,396-pair val cache")

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("frozen predictor checkpoint must contain a dictionary")
    _require_equal(payload.get("checkpoint_kind"), "best", name="checkpoint kind")
    _require_equal(payload.get("trainer_git_sha"), EXPECTED_TRAINER_SHA, name="trainer SHA")
    _require_equal(payload.get("cache_producer_sha"), EXPECTED_CACHE_PRODUCER_SHA, name="cache producer SHA")
    _require_equal(payload.get("epoch"), EXPECTED_BEST_EPOCH, name="best epoch")
    _require_equal(payload.get("global_step"), EXPECTED_BEST_GLOBAL_STEP, name="best global step")
    if "optimizer_state_dict" in payload or "rng_state" in payload:
        raise ValueError("best checkpoint must not contain optimizer or RNG state")

    expected_config = asdict(_production_predictor_config())
    _require_equal(payload.get("predictor_config"), expected_config, name="predictor config")
    train_config = payload.get("train_config")
    if not isinstance(train_config, Mapping):
        raise ValueError("best checkpoint is missing its train config")
    expected_train_fields = {
        "schema_version": 1,
        "classification": "offline_future_latent_predictor_training_not_task_capability",
        "run_kind": "train_val",
        "protocol_complete": True,
        "eligible_for_checkpoint_selection": True,
        "eligible_for_test": False,
        "trainer_git_sha": EXPECTED_TRAINER_SHA,
        "cache_producer_sha": EXPECTED_CACHE_PRODUCER_SHA,
        "future_latent_config": expected_config,
        "device": "cuda",
        "micro_batch_size": 16,
        "effective_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "num_workers": 0,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-4,
        },
        "scheduler": None,
        "grad_clip_norm": 1.0,
        "lambda_cos": 0.1,
        "lambda_risk": 0.1,
        "max_epochs": 30,
        "seed": 0,
        "max_optimizer_steps": None,
        "max_val_pairs_per_delay": None,
        "resume_from": None,
    }
    for field, expected in expected_train_fields.items():
        _require_equal(train_config.get(field), expected, name=f"train config {field}")

    cache_provenance = payload.get("cache_provenance")
    if not isinstance(cache_provenance, Mapping):
        raise ValueError("best checkpoint is missing cache provenance")
    for split in ("train", "val"):
        provenance = cache_provenance.get(split)
        _validate_frozen_cache_provenance(provenance, split=split)
        configured = train_config.get(f"{split}_cache")
        if not isinstance(configured, Mapping):
            raise ValueError(f"train config is missing {split} cache provenance")
        _require_equal(
            _pathless_cache_identity(configured),
            _pathless_cache_identity(provenance),
            name=f"{split} cache provenance in checkpoint and train config",
        )

    current_val_provenance = _manifest_cache_provenance(val_dataset.cache_dir, val_dataset.manifest)
    _require_equal(
        _pathless_cache_identity(cache_provenance["val"]),
        _pathless_cache_identity(current_val_provenance),
        name="checkpoint and requested val cache identity",
    )
    best_metrics = payload.get("best_val_metrics")
    if not isinstance(best_metrics, Mapping):
        raise ValueError("best checkpoint is missing best validation metrics")
    _require_equal(best_metrics.get("epoch"), EXPECTED_BEST_EPOCH, name="best metrics epoch")
    _require_equal(
        best_metrics.get("global_step"), EXPECTED_BEST_GLOBAL_STEP, name="best metrics global step"
    )
    selection_state = payload.get("selection_state")
    selected_best = getattr(selection_state, "best", None)
    _require_equal(getattr(selected_best, "epoch", None), EXPECTED_BEST_EPOCH, name="selection best epoch")

    predictor = _production_predictor_config()
    model = LightweightFutureLatentPredictor(predictor)
    model.load_state_dict(payload["predictor_state_dict"], strict=True)
    model.to(device=torch.device(device), dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    return FrozenPredictor(predictor=model, checkpoint=payload, val_dataset=val_dataset)


def frozen_postprocessor_provenance() -> dict[str, Any]:
    """Return the approved secondary action-space transform provenance."""
    return {
        "source_repo_id": CHECKPOINT_REPO_ID,
        "source_revision": CHECKPOINT_REVISION,
        "config": "policy_postprocessor.json",
        "state_file": POSTPROCESSOR_STATE_FILE,
        "source_key": POSTPROCESSOR_STATS_SOURCE_KEY,
        "target_key": "action",
        "dataset_stats_used": False,
        "definition": "the same checkpoint postprocessor applied independently to all four chunks",
    }


def aggregate_latent_risk_records(records: Sequence[LatentRiskRecord]) -> LatentRiskSummary:
    """Aggregate raw latent records by delay, retaining identity and predicted metrics."""
    if not records:
        raise ValueError("latent risk records must not be empty")
    by_delay: dict[int, list[LatentRiskRecord]] = defaultdict(list)
    seen: set[tuple[int, int, int]] = set()
    for record in records:
        key = (record.episode_index, record.frame_index, record.delay_steps)
        if key in seen:
            raise ValueError(f"duplicate latent record for episode/frame/delay {key}")
        seen.add(key)
        values = (
            record.predicted_risk,
            record.actual_mse,
            record.predicted.smoothl1,
            record.predicted.mse,
            record.predicted.cosine,
            record.identity.smoothl1,
            record.identity.mse,
            record.identity.cosine,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"latent record {key} contains a non-finite metric")
        by_delay[record.delay_steps].append(record)

    def metric_mean(delay_records: Sequence[LatentRiskRecord], field: str) -> LatentMetrics:
        metrics = [getattr(record, field) for record in delay_records]
        return LatentMetrics(
            smoothl1=_mean([metric.smoothl1 for metric in metrics]),
            mse=_mean([metric.mse for metric in metrics]),
            cosine=_mean([metric.cosine for metric in metrics]),
        )

    def risk_smoothl1(record: LatentRiskRecord) -> float:
        difference = abs(record.predicted_risk - record.actual_mse)
        return 0.5 * difference * difference if difference < 1.0 else difference - 0.5

    per_delay = tuple(
        LatentDelaySummary(
            delay_steps=delay,
            sample_count=len(by_delay[delay]),
            predicted=metric_mean(by_delay[delay], "predicted"),
            identity=metric_mean(by_delay[delay], "identity"),
            risk_smoothl1=_mean([risk_smoothl1(record) for record in by_delay[delay]]),
        )
        for delay in sorted(by_delay)
    )
    macro_predicted = LatentMetrics(
        smoothl1=_mean([entry.predicted.smoothl1 for entry in per_delay]),
        mse=_mean([entry.predicted.mse for entry in per_delay]),
        cosine=_mean([entry.predicted.cosine for entry in per_delay]),
    )
    macro_identity = LatentMetrics(
        smoothl1=_mean([entry.identity.smoothl1 for entry in per_delay]),
        mse=_mean([entry.identity.mse for entry in per_delay]),
        cosine=_mean([entry.identity.cosine for entry in per_delay]),
    )
    return LatentRiskSummary(
        total_record_count=len(records),
        per_delay=per_delay,
        macro_predicted=macro_predicted,
        macro_identity=macro_identity,
        macro_risk_smoothl1=_mean([entry.risk_smoothl1 for entry in per_delay]),
    )


@torch.inference_mode()
def evaluate_latent_risk(
    predictor: LightweightFutureLatentPredictor,
    dataset: FutureLatentCacheDataset,
    *,
    device: torch.device | str,
    batch_size: int = 16,
) -> LatentRiskEvaluation:
    """Evaluate every canonical pair in the frozen complete validation cache."""
    if dataset.expected_split != "val" or len(dataset) != FROZEN_VAL_PAIR_COUNT:
        raise ValueError("latent/risk evaluation requires the complete 14,396-pair val cache")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device)
    predictor.eval()
    records: list[LatentRiskRecord] = []
    for start in range(0, len(dataset), batch_size):
        pairs = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = move_future_latent_batch(collate_future_latent_pairs(pairs), target_device)
        prediction = forward_predictor(predictor, batch)
        objective = compute_future_latent_objective(
            prediction,
            batch,
            lambda_cos=0.1,
            lambda_risk=0.1,
        )
        identity = compute_identity_baseline_metrics(batch)

        predicted_risk = prediction.predicted_error.detach().float().cpu().tolist()
        actual_mse = objective.risk_target.detach().float().cpu().tolist()
        predicted_smoothl1 = objective.per_sample_smoothl1.detach().float().cpu().tolist()
        predicted_mse = objective.per_sample_mse.detach().float().cpu().tolist()
        predicted_cosine = objective.per_sample_cosine.detach().float().cpu().tolist()
        identity_smoothl1 = identity.smoothl1.detach().float().cpu().tolist()
        identity_mse = identity.mse.detach().float().cpu().tolist()
        identity_cosine = identity.cosine.detach().float().cpu().tolist()
        for offset, pair in enumerate(pairs):
            records.append(
                LatentRiskRecord(
                    episode_index=pair.episode_index,
                    frame_index=pair.frame_index,
                    future_frame_index=pair.future_frame_index,
                    delay_steps=int(pair.delay_steps.item()),
                    predicted_risk=predicted_risk[offset],
                    actual_mse=actual_mse[offset],
                    predicted=LatentMetrics(
                        smoothl1=predicted_smoothl1[offset],
                        mse=predicted_mse[offset],
                        cosine=predicted_cosine[offset],
                    ),
                    identity=LatentMetrics(
                        smoothl1=identity_smoothl1[offset],
                        mse=identity_mse[offset],
                        cosine=identity_cosine[offset],
                    ),
                )
            )

    summary = aggregate_latent_risk_records(records)
    _require_equal(
        tuple(entry.delay_steps for entry in summary.per_delay),
        FROZEN_DELAYS,
        name="latent evaluation delays",
    )
    _require_equal(summary.total_record_count, FROZEN_VAL_PAIR_COUNT, name="latent record count")
    expected_counts = _EXPECTED_SPLIT_COUNTS["val"]["valid_pair_count_by_delay"]
    _require_equal(
        {str(entry.delay_steps): entry.sample_count for entry in summary.per_delay},
        expected_counts,
        name="latent records by delay",
    )
    return LatentRiskEvaluation(records=tuple(records), summary=summary)


def _tie_aware_ranks(values: Sequence[float]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[indexed[position][0]] = average_rank
        start = stop
    return tuple(ranks)


def tie_aware_spearman(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    """Return Spearman correlation with average ranks, or ``None`` for a constant side."""
    if len(predicted) != len(actual) or not predicted:
        raise ValueError("predicted and actual must have the same non-zero length")
    if not all(math.isfinite(value) for value in (*predicted, *actual)):
        raise ValueError("Spearman inputs must be finite")
    predicted_ranks = _tie_aware_ranks(predicted)
    actual_ranks = _tie_aware_ranks(actual)
    predicted_mean = _mean(predicted_ranks)
    actual_mean = _mean(actual_ranks)
    predicted_centered = [value - predicted_mean for value in predicted_ranks]
    actual_centered = [value - actual_mean for value in actual_ranks]
    predicted_square_sum = math.fsum(value * value for value in predicted_centered)
    actual_square_sum = math.fsum(value * value for value in actual_centered)
    if predicted_square_sum == 0.0 or actual_square_sum == 0.0:
        return None
    covariance = math.fsum(
        predicted_value * actual_value
        for predicted_value, actual_value in zip(predicted_centered, actual_centered, strict=True)
    )
    correlation = covariance / math.sqrt(predicted_square_sum * actual_square_sum)
    return min(1.0, max(-1.0, correlation))


def _equal_count_groups(
    records: Sequence[LatentRiskRecord], group_count: int
) -> tuple[tuple[LatentRiskRecord, ...], ...]:
    if group_count <= 0 or len(records) < group_count:
        raise ValueError("equal-count grouping requires at least one record per group")
    ordered = sorted(enumerate(records), key=lambda item: (item[1].predicted_risk, item[0]))
    base_size, remainder = divmod(len(ordered), group_count)
    groups: list[tuple[LatentRiskRecord, ...]] = []
    cursor = 0
    for group_index in range(group_count):
        size = base_size + (group_index < remainder)
        groups.append(tuple(record for _, record in ordered[cursor : cursor + size]))
        cursor += size
    return tuple(groups)


def equal_count_bins(records: Sequence[LatentRiskRecord], *, bin_count: int = 10) -> tuple[RiskBin, ...]:
    """Build deterministic equal-count bins ordered by predicted risk."""
    groups = _equal_count_groups(records, bin_count)
    return tuple(
        RiskBin(
            index=index,
            count=len(group),
            mean_predicted_risk=_mean([record.predicted_risk for record in group]),
            mean_actual_mse=_mean([record.actual_mse for record in group]),
            min_predicted_risk=min(record.predicted_risk for record in group),
            max_predicted_risk=max(record.predicted_risk for record in group),
        )
        for index, group in enumerate(groups, start=1)
    )


def top_bottom_quintile_ratio(records: Sequence[LatentRiskRecord]) -> float | None:
    """Return top-risk over bottom-risk actual-MSE mean, or ``None`` at zero denominator."""
    bottom, top = _risk_quintile_tails(records)
    bottom_mean = _mean([record.actual_mse for record in bottom])
    top_mean = _mean([record.actual_mse for record in top])
    return None if bottom_mean == 0.0 else top_mean / bottom_mean


def _risk_quintile_tails(
    records: Sequence[LatentRiskRecord],
) -> tuple[tuple[LatentRiskRecord, ...], tuple[LatentRiskRecord, ...]]:
    tail_count = len(records) // 5
    if tail_count == 0:
        raise ValueError("risk quintiles require at least five records")
    ordered = tuple(
        record
        for _, record in sorted(
            enumerate(records),
            key=lambda item: (item[1].predicted_risk, item[0]),
        )
    )
    return ordered[:tail_count], ordered[-tail_count:]


def compute_risk_calibration(records: Sequence[LatentRiskRecord]) -> RiskCalibrationSummary:
    """Compute the frozen rank, calibration-bin, and top/bottom risk diagnostics."""
    if not records:
        raise ValueError("risk calibration requires latent records")
    predicted = [record.predicted_risk for record in records]
    actual = [record.actual_mse for record in records]
    overall = tie_aware_spearman(predicted, actual)
    by_delay: dict[int, list[LatentRiskRecord]] = defaultdict(list)
    for record in records:
        by_delay[record.delay_steps].append(record)
    per_delay = {
        str(delay): tie_aware_spearman(
            [record.predicted_risk for record in by_delay[delay]],
            [record.actual_mse for record in by_delay[delay]],
        )
        for delay in sorted(by_delay)
    }
    bins = equal_count_bins(records, bin_count=10)
    bottom, top = _risk_quintile_tails(records)
    bottom_actual = _mean([record.actual_mse for record in bottom])
    top_actual = _mean([record.actual_mse for record in top])
    ratio = top_bottom_quintile_ratio(records)
    eligible = overall is not None and ratio is not None and overall >= 0.30 and ratio >= 1.5
    return RiskCalibrationSummary(
        record_count=len(records),
        overall_spearman=overall,
        per_delay_spearman=per_delay,
        bins=bins,
        bottom_quintile_count=len(bottom),
        top_quintile_count=len(top),
        bottom_quintile_actual_mse_mean=bottom_actual,
        top_quintile_actual_mse_mean=top_actual,
        top_bottom_actual_mse_ratio=ratio,
        top_bottom_ratio_defined=ratio is not None,
        risk_gating_eligible=eligible,
    )


def select_val_anchor_pairs(
    dataset: FutureLatentCacheDataset,
    *,
    count: int = FROZEN_ANCHOR_COUNT,
    seed: int = FROZEN_SEED,
) -> tuple[ValAnchor, ...]:
    """Select a deterministic common-anchor cohort entirely from a val cache.

    The returned pairs are eager and ordered by ``d=1..8``.  In particular, this
    function never constructs the candidate pool from the original dataset, whose
    non-validation episodes are outside the B3.2 data boundary.
    """
    if dataset.expected_split != "val":
        raise ValueError("action characterization anchors must come from a val cache")

    candidates: list[OracleAnchorCandidate] = []
    candidate_by_id: dict[int, OracleAnchorCandidate] = {}
    pair_index_by_key: dict[tuple[int, int, int], int] = {}
    for pair_index, spec in enumerate(dataset.pair_specs):
        key = (spec.episode_position, spec.frame_offset, spec.delay_steps)
        pair_index_by_key[key] = pair_index

    for episode_position, episode_entry in enumerate(dataset.manifest["episodes"]):
        frame_count = int(episode_entry["frame_count"])
        episode_index = int(episode_entry["episode_index"])
        _, tensors = dataset._load_episode(episode_position)
        for frame_offset in range(frame_count):
            candidate = OracleAnchorCandidate(
                anchor_id=int(tensors["dataset_indices"][frame_offset].item()),
                episode_index=episode_index,
                frame_index=int(tensors["frame_indices"][frame_offset].item()),
                episode_length=frame_count,
            )
            candidates.append(candidate)
            candidate_by_id[candidate.anchor_id] = candidate

    anchor_ids = select_common_anchor_ids(
        candidates,
        delays=FROZEN_DELAYS,
        count=count,
        seed=seed,
    )
    anchors: list[ValAnchor] = []
    episode_position_by_index = {
        int(entry["episode_index"]): position for position, entry in enumerate(dataset.manifest["episodes"])
    }
    for anchor_id in anchor_ids:
        candidate = candidate_by_id[anchor_id]
        episode_position = episode_position_by_index[candidate.episode_index]
        pairs = tuple(
            dataset[pair_index_by_key[(episode_position, candidate.frame_index, delay_steps)]]
            for delay_steps in FROZEN_DELAYS
        )
        for pair in pairs:
            validate_pair_language_equality(pair)
        anchors.append(
            ValAnchor(
                anchor_id=anchor_id,
                episode_index=candidate.episode_index,
                frame_index=candidate.frame_index,
                pairs=pairs,
            )
        )
    return tuple(anchors)


def validate_pair_language_equality(pair: FutureLatentPair) -> None:
    """Enforce the approved one-task current/future language invariant."""
    tokens_equal = torch.equal(pair.current_language_tokens, pair.future_language_tokens)
    masks_equal = torch.equal(
        pair.current_language_attention_mask,
        pair.future_language_attention_mask,
    )
    if not tokens_equal or not masks_equal:
        raise ValueError(
            "current/future language differs for "
            f"episode={pair.episode_index}, frame={pair.frame_index}, "
            f"delay={int(pair.delay_steps.item())}; language rollout is not authorized"
        )


def _validate_four_path_actions(actions: FourPathActions) -> None:
    named_actions = {
        "current": actions.current,
        "oracle_visual": actions.oracle_visual,
        "predicted_visual": actions.predicted_visual,
        "full_future_teacher": actions.full_future_teacher,
    }
    expected_shape = tuple(actions.full_future_teacher.shape)
    for name, action in named_actions.items():
        if not isinstance(action, Tensor):
            raise TypeError(f"{name} action chunk must be a Tensor")
        if not action.is_floating_point():
            raise TypeError(f"{name} action chunk must have a floating dtype, got {action.dtype}")
        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"all four action chunks must have shape {expected_shape}, "
                f"but {name} has {tuple(action.shape)}"
            )
        if not bool(torch.isfinite(action).all().item()):
            raise ValueError(f"{name} action chunk contains non-finite values")


def run_four_path_same_noise(
    noise: Tensor,
    *,
    current: Callable[[Tensor], Tensor],
    oracle_visual: Callable[[Tensor], Tensor],
    predicted_visual: Callable[[Tensor], Tensor],
    full_future_teacher: Callable[[Tensor], Tensor],
) -> FourPathActions:
    """Run four action paths with independent clones of one caller-owned noise tensor."""
    if not isinstance(noise, Tensor):
        raise TypeError("noise must be a Tensor")
    actions = FourPathActions(
        current=current(noise.clone()),
        oracle_visual=oracle_visual(noise.clone()),
        predicted_visual=predicted_visual(noise.clone()),
        full_future_teacher=full_future_teacher(noise.clone()),
    )
    _validate_four_path_actions(actions)
    return actions


def postprocess_four_path_actions(
    postprocessor: Callable[[Tensor], Tensor], actions: FourPathActions
) -> FourPathActions:
    """Apply the same checkpoint postprocessor independently to all four chunks."""
    _validate_four_path_actions(actions)
    outputs = FourPathActions(
        current=postprocessor(actions.current.clone()),
        oracle_visual=postprocessor(actions.oracle_visual.clone()),
        predicted_visual=postprocessor(actions.predicted_visual.clone()),
        full_future_teacher=postprocessor(actions.full_future_teacher.clone()),
    )
    _validate_four_path_actions(outputs)
    return outputs


def _action_error(prediction: Tensor, teacher: Tensor) -> ActionError:
    delta = prediction.detach().to(device="cpu", dtype=torch.float64) - teacher.detach().to(
        device="cpu", dtype=torch.float64
    )
    return ActionError(
        l1=delta.abs().mean().item(),
        rmse=delta.square().mean().sqrt().item(),
    )


def _action_space_errors(actions: FourPathActions) -> ActionSpaceErrors:
    _validate_four_path_actions(actions)
    teacher = actions.full_future_teacher
    return ActionSpaceErrors(
        current_vs_teacher=_action_error(actions.current, teacher),
        oracle_visual_vs_teacher=_action_error(actions.oracle_visual, teacher),
        predicted_visual_vs_teacher=_action_error(actions.predicted_visual, teacher),
    )


def _clone_policy_batch(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.clone() for key, value in batch.items()}


@torch.inference_mode()
def evaluate_four_path_pair(
    policy: Any,
    predictor: LightweightFutureLatentPredictor,
    pair: FutureLatentPair,
    noise: Tensor,
    postprocessor: Callable[[Tensor], Tensor],
    *,
    anchor_id: int,
) -> FourPathActionRecord:
    """Evaluate one cached ``t -> t+d`` pair through the approved four paths."""
    validate_pair_language_equality(pair)
    if noise.ndim != 3 or noise.shape[0] != 1:
        raise ValueError(f"one-anchor flow noise must have shape [1,T,A], got {tuple(noise.shape)}")
    device = noise.device
    predictor_batch = move_future_latent_batch(collate_future_latent_pairs([pair]), device)
    prediction = forward_predictor(predictor, predictor_batch)
    predicted_tokens = tuple(
        (current_tokens.float() + delta_tokens.float()).to(dtype=current_tokens.dtype)
        for current_tokens, delta_tokens in zip(
            predictor_batch.current_image_tokens,
            prediction.delta_tokens,
            strict=True,
        )
    )

    current_batch = {
        OBS_STATE: pair.current_state.unsqueeze(0).to(device=device),
        OBS_LANGUAGE_TOKENS: pair.current_language_tokens.unsqueeze(0).to(device=device),
        OBS_LANGUAGE_ATTENTION_MASK: pair.current_language_attention_mask.unsqueeze(0).to(device=device),
    }
    future_batch = {
        OBS_STATE: pair.future_state.unsqueeze(0).to(device=device),
        OBS_LANGUAGE_TOKENS: pair.future_language_tokens.unsqueeze(0).to(device=device),
        OBS_LANGUAGE_ATTENTION_MASK: pair.future_language_attention_mask.unsqueeze(0).to(device=device),
    }

    def action_path(
        batch: Mapping[str, Tensor],
        image_tokens: tuple[Tensor, ...],
        image_token_masks: tuple[Tensor, ...],
    ) -> Callable[[Tensor], Tensor]:
        return lambda path_noise: policy.predict_action_chunk(
            _clone_policy_batch(batch),
            noise=path_noise,
            future_image_tokens=image_tokens,
            future_image_token_masks=image_token_masks,
        )

    policy_output = run_four_path_same_noise(
        noise,
        current=action_path(
            current_batch,
            predictor_batch.current_image_tokens,
            predictor_batch.current_image_token_masks,
        ),
        oracle_visual=action_path(
            current_batch,
            predictor_batch.target_image_tokens,
            predictor_batch.target_image_token_masks,
        ),
        predicted_visual=action_path(
            current_batch,
            predicted_tokens,
            predictor_batch.current_image_token_masks,
        ),
        full_future_teacher=action_path(
            future_batch,
            predictor_batch.target_image_tokens,
            predictor_batch.target_image_token_masks,
        ),
    )
    post_policy = postprocess_four_path_actions(postprocessor, policy_output)
    return FourPathActionRecord(
        anchor_id=anchor_id,
        episode_index=pair.episode_index,
        frame_index=pair.frame_index,
        future_frame_index=pair.future_frame_index,
        delay_steps=int(pair.delay_steps.item()),
        policy_output=_action_space_errors(policy_output),
        post_policy=_action_space_errors(post_policy),
    )


def _capture_metric(records: Sequence[FourPathActionRecord], space: str, metric: str) -> CaptureMetric:
    errors = [getattr(record, space) for record in records]
    current = _mean([getattr(error.current_vs_teacher, metric) for error in errors])
    oracle = _mean([getattr(error.oracle_visual_vs_teacher, metric) for error in errors])
    predicted = _mean([getattr(error.predicted_visual_vs_teacher, metric) for error in errors])
    oracle_gap = current - oracle
    capture_defined = oracle_gap > 0.0
    return CaptureMetric(
        current=current,
        oracle_visual=oracle,
        predicted_visual=predicted,
        oracle_gap=oracle_gap,
        capture=(current - predicted) / oracle_gap if capture_defined else None,
        capture_defined=capture_defined,
    )


def _aggregate_action_group(
    records: Sequence[FourPathActionRecord], *, delay_steps: int
) -> ActionDelaySummary:
    return ActionDelaySummary(
        delay_steps=delay_steps,
        sample_count=len(records),
        policy_output=ActionSpaceAggregate(
            l1=_capture_metric(records, "policy_output", "l1"),
            rmse=_capture_metric(records, "policy_output", "rmse"),
        ),
        post_policy=ActionSpaceAggregate(
            l1=_capture_metric(records, "post_policy", "l1"),
            rmse=_capture_metric(records, "post_policy", "rmse"),
        ),
    )


def _relative_regression(predicted: float, current: float) -> float | None:
    if current == 0.0:
        return 0.0 if predicted == 0.0 else None
    return (predicted - current) / current


def aggregate_action_records(records: Sequence[FourPathActionRecord]) -> ActionEvaluationSummary:
    """Aggregate paired four-path errors and apply the frozen val action gate."""
    if not records:
        raise ValueError("action records must not be empty")

    by_delay: dict[int, list[FourPathActionRecord]] = defaultdict(list)
    anchor_metadata: dict[int, tuple[int, int]] = {}
    seen: set[tuple[int, int]] = set()
    for record in records:
        key = (record.anchor_id, record.delay_steps)
        if key in seen:
            raise ValueError(f"duplicate action record for anchor/delay {key}")
        seen.add(key)
        metadata = (record.episode_index, record.frame_index)
        previous_metadata = anchor_metadata.setdefault(record.anchor_id, metadata)
        if previous_metadata != metadata:
            raise ValueError(f"anchor_id={record.anchor_id} has inconsistent episode/frame provenance")
        metric_values = (
            record.policy_output.current_vs_teacher.l1,
            record.policy_output.current_vs_teacher.rmse,
            record.policy_output.oracle_visual_vs_teacher.l1,
            record.policy_output.oracle_visual_vs_teacher.rmse,
            record.policy_output.predicted_visual_vs_teacher.l1,
            record.policy_output.predicted_visual_vs_teacher.rmse,
            record.post_policy.current_vs_teacher.l1,
            record.post_policy.current_vs_teacher.rmse,
            record.post_policy.oracle_visual_vs_teacher.l1,
            record.post_policy.oracle_visual_vs_teacher.rmse,
            record.post_policy.predicted_visual_vs_teacher.l1,
            record.post_policy.predicted_visual_vs_teacher.rmse,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in metric_values):
            raise ValueError(f"action record {key} contains an invalid error metric")
        by_delay[record.delay_steps].append(record)

    anchor_ids = tuple(sorted(anchor_metadata))
    expected_anchor_set = set(anchor_ids)
    for delay_steps, delay_records in by_delay.items():
        delay_anchor_set = {record.anchor_id for record in delay_records}
        if delay_anchor_set != expected_anchor_set:
            raise ValueError(f"delay={delay_steps} does not contain the common anchor cohort")

    per_delay = tuple(
        _aggregate_action_group(by_delay[delay_steps], delay_steps=delay_steps)
        for delay_steps in sorted(by_delay)
    )
    macro = _aggregate_action_group(records, delay_steps=0)
    positive_capture_count = sum(
        entry.policy_output.l1.capture_defined and entry.policy_output.l1.capture > 0.0 for entry in per_delay
    )
    nonpositive_primary_gaps = tuple(
        entry.delay_steps for entry in per_delay if entry.policy_output.l1.oracle_gap <= 0.0
    )
    l1_regressions = tuple(
        _relative_regression(
            entry.policy_output.l1.predicted_visual,
            entry.policy_output.l1.current,
        )
        for entry in per_delay
    )
    rmse_regressions = tuple(
        _relative_regression(
            entry.policy_output.rmse.predicted_visual,
            entry.policy_output.rmse.current,
        )
        for entry in per_delay
    )
    max_l1_regression = None if any(value is None for value in l1_regressions) else max(l1_regressions)
    max_rmse_regression = None if any(value is None for value in rmse_regressions) else max(rmse_regressions)
    exact_protocol_shape = (
        tuple(entry.delay_steps for entry in per_delay) == FROZEN_DELAYS
        and len(anchor_ids) == FROZEN_ANCHOR_COUNT
        and len(records) == FROZEN_ANCHOR_COUNT * len(FROZEN_DELAYS)
    )
    no_large_regression = all(
        value is not None and value <= 0.10 for value in (*l1_regressions, *rmse_regressions)
    )
    prerequisites_met = (
        exact_protocol_shape
        and not nonpositive_primary_gaps
        and macro.policy_output.l1.predicted_visual < macro.policy_output.l1.current
        and macro.policy_output.rmse.predicted_visual < macro.policy_output.rmse.current
        and positive_capture_count >= 6
        and no_large_regression
    )
    return ActionEvaluationSummary(
        anchor_ids=anchor_ids,
        total_record_count=len(records),
        per_delay=per_delay,
        macro=macro,
        positive_policy_l1_capture_delay_count=positive_capture_count,
        policy_l1_nonpositive_oracle_gap_delays=nonpositive_primary_gaps,
        max_policy_l1_regression=max_l1_regression,
        max_policy_rmse_regression=max_rmse_regression,
        test_protocol_prerequisites_met=prerequisites_met,
    )
