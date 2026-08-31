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

"""Pure helpers for the offline SmolVLA Oracle upper-bound evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor


class OracleEvaluationError(ValueError):
    """Raised when an Oracle evaluation input violates the paired-data contract."""


@dataclass(frozen=True)
class OracleAnchorCandidate:
    """An episode-relative frame eligible for deterministic anchor selection."""

    anchor_id: int
    episode_index: int
    frame_index: int
    episode_length: int


@dataclass(frozen=True)
class ActionTriplet:
    """The three paired action chunks compared by the Oracle evaluation."""

    current: Tensor
    oracle_visual: Tensor
    full_future_teacher: Tensor


@dataclass(frozen=True)
class PairedActionErrors:
    """Per-chunk errors against the full-future teacher in one action space."""

    current_vs_teacher_l1: float
    current_vs_teacher_l2_rmse: float
    oracle_visual_vs_teacher_l1: float
    oracle_visual_vs_teacher_l2_rmse: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class OracleEvaluationRecord:
    """One anchor/delay result in policy-output and post-policy action spaces."""

    anchor_id: int
    episode_index: int
    frame_index: int
    delay_steps: int
    policy_output: PairedActionErrors
    post_policy: PairedActionErrors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSpaceSummary:
    """Ratio-of-means summary for one action space and one delay bucket."""

    current_vs_teacher_l1_mean: float
    current_vs_teacher_l2_rmse_mean: float
    oracle_visual_vs_teacher_l1_mean: float
    oracle_visual_vs_teacher_l2_rmse_mean: float
    relative_l1_error_reduction: float | None
    relative_l2_error_reduction: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class DelaySummary:
    """Aggregate result for a single delay bucket."""

    delay_steps: int
    sample_count: int
    policy_output: ActionSpaceSummary
    post_policy: ActionSpaceSummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleEvaluationSummary:
    """Aggregate result with the common-cohort invariant already checked."""

    anchor_ids: tuple[int, ...]
    delays: tuple[DelaySummary, ...]
    total_record_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_ids": list(self.anchor_ids),
            "delays": [delay.to_dict() for delay in self.delays],
            "total_record_count": self.total_record_count,
        }


def _validated_delays(delays: Sequence[int]) -> tuple[int, ...]:
    if not delays:
        raise OracleEvaluationError("delays must contain at least one future horizon")
    if any(not isinstance(delay, int) or isinstance(delay, bool) for delay in delays):
        raise TypeError("delays must contain integer step offsets")
    if any(delay < 0 for delay in delays):
        raise OracleEvaluationError(f"delays must be >= 0, got {list(delays)}")
    if len(set(delays)) != len(delays):
        raise OracleEvaluationError(f"delays must be unique, got {list(delays)}")
    return tuple(sorted(delays))


def slice_temporal_sample(
    sample: Mapping[str, Any],
    *,
    temporal_keys: Sequence[str],
    step: int,
) -> dict[str, Any]:
    """Select one horizon from an unbatched temporal LeRobot sample.

    Every temporal key must have the same leading horizon and matching one-dimensional
    boolean ``*_is_pad`` metadata. The selected horizon must be inside the source episode.
    Returned tensors are cloned, corresponding padding keys are removed, and the input
    mapping is never mutated. This makes the result safe to pass independently through a
    policy preprocessor whose policy-side adaptation may mutate tensors in place.
    """
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if step < 0:
        raise OracleEvaluationError(f"step must be >= 0, got {step}")
    if not temporal_keys:
        raise OracleEvaluationError("temporal_keys must contain at least one camera or state feature")
    if len(set(temporal_keys)) != len(temporal_keys):
        raise OracleEvaluationError(f"temporal_keys must be unique, got {list(temporal_keys)}")

    horizon: int | None = None
    padding_keys: set[str] = set()
    for key in temporal_keys:
        if key not in sample:
            raise KeyError(f"Temporal sample is missing required feature {key!r}")
        value = sample[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Temporal feature {key!r} must be a torch tensor")
        if value.ndim < 1:
            raise OracleEvaluationError(
                f"Temporal feature {key!r} must have a leading horizon dimension, got rank {value.ndim}"
            )

        key_horizon = value.shape[0]
        if horizon is None:
            horizon = key_horizon
        elif key_horizon != horizon:
            raise OracleEvaluationError(
                f"Temporal features disagree on horizon: expected {horizon}, but {key!r} has {key_horizon}"
            )

        pad_key = f"{key}_is_pad"
        padding_keys.add(pad_key)
        if pad_key not in sample:
            raise KeyError(f"Temporal sample is missing episode-boundary metadata {pad_key!r}")
        padding = sample[pad_key]
        if not isinstance(padding, Tensor):
            raise TypeError(f"Episode-boundary metadata {pad_key!r} must be a torch tensor")
        if padding.ndim != 1:
            raise OracleEvaluationError(
                f"Episode-boundary metadata {pad_key!r} must have shape [T], got {tuple(padding.shape)}"
            )
        if padding.dtype is not torch.bool:
            raise TypeError(f"Episode-boundary metadata {pad_key!r} must have dtype torch.bool")
        if padding.shape[0] != key_horizon:
            raise OracleEvaluationError(
                f"{key!r} and {pad_key!r} disagree on horizon: {key_horizon} versus {padding.shape[0]}"
            )

    assert horizon is not None
    if step >= horizon:
        raise OracleEvaluationError(f"step={step} is outside temporal horizon={horizon}")

    for key in temporal_keys:
        pad_key = f"{key}_is_pad"
        if bool(sample[pad_key][step].item()):
            raise OracleEvaluationError(f"Temporal feature {key!r} at step={step} is episode padding")

    sliced: dict[str, Any] = {}
    temporal_key_set = set(temporal_keys)
    for key, value in sample.items():
        if key in padding_keys:
            continue
        if key in temporal_key_set:
            sliced[key] = value[step].clone()
        elif isinstance(value, Tensor):
            sliced[key] = value.clone()
        else:
            sliced[key] = deepcopy(value)
    return sliced


def select_common_anchor_ids(
    candidates: Sequence[OracleAnchorCandidate],
    *,
    delays: Sequence[int],
    count: int = 128,
    seed: int = 0,
) -> tuple[int, ...]:
    """Select one deterministic episode-stratified cohort valid for every delay.

    Candidates contain only episode metadata, so callers can construct them without
    decoding camera frames. Selection first requires ``frame_index + max(delays)`` to
    remain inside the episode, then shuffles deterministically and draws round-robin
    across episodes. A short common pool is an explicit error; the requested count is
    never silently reduced.
    """
    validated_delays = _validated_delays(delays)
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(f"count must be an integer, got {type(count).__name__}")
    if count <= 0:
        raise OracleEvaluationError(f"count must be > 0, got {count}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    seen_anchor_ids: set[int] = set()
    valid_by_episode: dict[int, list[OracleAnchorCandidate]] = defaultdict(list)
    max_delay = validated_delays[-1]
    for candidate in candidates:
        if candidate.anchor_id in seen_anchor_ids:
            raise OracleEvaluationError(f"Duplicate anchor_id {candidate.anchor_id}")
        seen_anchor_ids.add(candidate.anchor_id)
        if candidate.episode_index < 0:
            raise OracleEvaluationError(
                f"anchor_id={candidate.anchor_id} has negative episode_index={candidate.episode_index}"
            )
        if candidate.episode_length <= 0:
            raise OracleEvaluationError(
                f"anchor_id={candidate.anchor_id} has invalid episode_length={candidate.episode_length}"
            )
        if not 0 <= candidate.frame_index < candidate.episode_length:
            raise OracleEvaluationError(
                f"anchor_id={candidate.anchor_id} has frame_index={candidate.frame_index} outside "
                f"episode_length={candidate.episode_length}"
            )
        if candidate.frame_index + max_delay < candidate.episode_length:
            valid_by_episode[candidate.episode_index].append(candidate)

    valid_count = sum(len(group) for group in valid_by_episode.values())
    if valid_count < count:
        raise OracleEvaluationError(
            f"Common anchor pool valid through max delay {max_delay} has {valid_count} candidates; "
            f"required {count}"
        )

    rng = random.Random(seed)
    episode_order = sorted(valid_by_episode)
    rng.shuffle(episode_order)
    for episode_index in episode_order:
        valid_by_episode[episode_index].sort(key=lambda candidate: candidate.anchor_id)
        rng.shuffle(valid_by_episode[episode_index])

    selected: list[int] = []
    offsets = dict.fromkeys(episode_order, 0)
    while len(selected) < count:
        for episode_index in episode_order:
            offset = offsets[episode_index]
            episode_candidates = valid_by_episode[episode_index]
            if offset >= len(episode_candidates):
                continue
            selected.append(episode_candidates[offset].anchor_id)
            offsets[episode_index] = offset + 1
            if len(selected) == count:
                break
    return tuple(selected)


def run_with_shared_noise(
    noise: Tensor,
    *,
    current: Callable[[Tensor], Tensor],
    oracle_visual: Callable[[Tensor], Tensor],
    full_future_teacher: Callable[[Tensor], Tensor],
) -> ActionTriplet:
    """Run all three prediction paths with independent clones of one noise tensor."""
    if not isinstance(noise, Tensor):
        raise TypeError(f"noise must be a torch tensor, got {type(noise).__name__}")
    outputs = ActionTriplet(
        current=current(noise.clone()),
        oracle_visual=oracle_visual(noise.clone()),
        full_future_teacher=full_future_teacher(noise.clone()),
    )
    _validate_action_triplet(outputs)
    return outputs


def remap_checkpoint_action_stats(
    checkpoint_stats: Mapping[str, Tensor],
    *,
    source_key: str,
    action_dim: int,
) -> dict[str, dict[str, Tensor]]:
    """Map one explicit checkpoint embodiment's mean/std statistics to ``action``.

    Some multi-embodiment checkpoints store processor state under keys such as
    ``so100.buffer.action.mean`` while the generic unnormalizer looks up the literal
    feature key ``action``. The offline evaluator must choose the approved embodiment
    explicitly rather than silently falling back to another embodiment or dataset stats.
    """
    if not source_key:
        raise OracleEvaluationError("source_key must identify one checkpoint action-stat entry")
    if not isinstance(action_dim, int) or isinstance(action_dim, bool):
        raise TypeError(f"action_dim must be an integer, got {type(action_dim).__name__}")
    if action_dim <= 0:
        raise OracleEvaluationError(f"action_dim must be > 0, got {action_dim}")

    remapped: dict[str, Tensor] = {}
    for stat_name in ("mean", "std"):
        checkpoint_key = f"{source_key}.{stat_name}"
        if checkpoint_key not in checkpoint_stats:
            raise OracleEvaluationError(
                f"Checkpoint processor state is missing required statistic {checkpoint_key!r}"
            )
        value = checkpoint_stats[checkpoint_key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Checkpoint statistic {checkpoint_key!r} must be a torch tensor")
        if tuple(value.shape) != (action_dim,):
            raise OracleEvaluationError(
                f"Checkpoint statistic {checkpoint_key!r} must have shape ({action_dim},), "
                f"got {tuple(value.shape)}"
            )
        remapped[stat_name] = value.clone()
    return {"action": remapped}


def postprocess_action_triplet(
    postprocessor: Callable[[Tensor], Tensor], actions: ActionTriplet
) -> ActionTriplet:
    """Apply one postprocessor independently without mutating policy-output chunks."""
    _validate_action_triplet(actions)
    outputs = ActionTriplet(
        current=postprocessor(actions.current.clone()),
        oracle_visual=postprocessor(actions.oracle_visual.clone()),
        full_future_teacher=postprocessor(actions.full_future_teacher.clone()),
    )
    _validate_action_triplet(outputs)
    return outputs


def _validate_action_triplet(actions: ActionTriplet) -> None:
    named_actions = {
        "current": actions.current,
        "oracle_visual": actions.oracle_visual,
        "full_future_teacher": actions.full_future_teacher,
    }
    for name, action in named_actions.items():
        if not isinstance(action, Tensor):
            raise TypeError(f"{name} action chunk must be a torch tensor")
        if not action.is_floating_point():
            raise TypeError(f"{name} action chunk must have a floating dtype, got {action.dtype}")
        if not bool(torch.isfinite(action).all().item()):
            raise OracleEvaluationError(f"{name} action chunk contains non-finite values")

    expected_shape = tuple(actions.full_future_teacher.shape)
    for name in ("current", "oracle_visual"):
        action = named_actions[name]
        if tuple(action.shape) != expected_shape:
            raise OracleEvaluationError(
                f"Paired action chunks must have identical shapes; teacher has {expected_shape}, "
                f"but {name} has {tuple(action.shape)}"
            )


def compute_paired_action_errors(actions: ActionTriplet) -> PairedActionErrors:
    """Compute per-chunk mean absolute error and RMSE against the teacher."""
    _validate_action_triplet(actions)

    teacher = actions.full_future_teacher.detach().to(device="cpu", dtype=torch.float64)

    def errors(prediction: Tensor) -> tuple[float, float]:
        delta = prediction.detach().to(device="cpu", dtype=torch.float64) - teacher
        l1 = delta.abs().mean().item()
        l2_rmse = delta.square().mean().sqrt().item()
        return l1, l2_rmse

    current_l1, current_l2 = errors(actions.current)
    oracle_l1, oracle_l2 = errors(actions.oracle_visual)
    return PairedActionErrors(
        current_vs_teacher_l1=current_l1,
        current_vs_teacher_l2_rmse=current_l2,
        oracle_visual_vs_teacher_l1=oracle_l1,
        oracle_visual_vs_teacher_l2_rmse=oracle_l2,
    )


def make_evaluation_record(
    *,
    anchor_id: int,
    episode_index: int,
    frame_index: int,
    delay_steps: int,
    policy_output: ActionTriplet,
    post_policy: ActionTriplet,
) -> OracleEvaluationRecord:
    """Build one raw paired record in both required action spaces."""
    if delay_steps < 0:
        raise OracleEvaluationError(f"delay_steps must be >= 0, got {delay_steps}")
    return OracleEvaluationRecord(
        anchor_id=anchor_id,
        episode_index=episode_index,
        frame_index=frame_index,
        delay_steps=delay_steps,
        policy_output=compute_paired_action_errors(policy_output),
        post_policy=compute_paired_action_errors(post_policy),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _relative_reduction(current_mean: float, oracle_mean: float) -> float | None:
    if current_mean == 0.0:
        return None
    return (current_mean - oracle_mean) / current_mean


def _summarize_action_space(errors: Sequence[PairedActionErrors]) -> ActionSpaceSummary:
    current_l1 = _mean([error.current_vs_teacher_l1 for error in errors])
    current_l2 = _mean([error.current_vs_teacher_l2_rmse for error in errors])
    oracle_l1 = _mean([error.oracle_visual_vs_teacher_l1 for error in errors])
    oracle_l2 = _mean([error.oracle_visual_vs_teacher_l2_rmse for error in errors])
    return ActionSpaceSummary(
        current_vs_teacher_l1_mean=current_l1,
        current_vs_teacher_l2_rmse_mean=current_l2,
        oracle_visual_vs_teacher_l1_mean=oracle_l1,
        oracle_visual_vs_teacher_l2_rmse_mean=oracle_l2,
        relative_l1_error_reduction=_relative_reduction(current_l1, oracle_l1),
        relative_l2_error_reduction=_relative_reduction(current_l2, oracle_l2),
    )


def aggregate_by_delay(records: Sequence[OracleEvaluationRecord]) -> OracleEvaluationSummary:
    """Aggregate paired records while enforcing one identical anchor cohort per delay."""
    if not records:
        raise OracleEvaluationError("records must contain at least one Oracle evaluation result")

    by_delay: dict[int, list[OracleEvaluationRecord]] = defaultdict(list)
    anchor_metadata: dict[int, tuple[int, int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for record in records:
        pair = (record.delay_steps, record.anchor_id)
        if pair in seen_pairs:
            raise OracleEvaluationError(
                f"Duplicate Oracle evaluation record for delay={record.delay_steps}, "
                f"anchor_id={record.anchor_id}"
            )
        seen_pairs.add(pair)

        metadata = (record.episode_index, record.frame_index)
        previous_metadata = anchor_metadata.setdefault(record.anchor_id, metadata)
        if previous_metadata != metadata:
            raise OracleEvaluationError(
                f"anchor_id={record.anchor_id} has inconsistent episode/frame metadata: "
                f"{previous_metadata} versus {metadata}"
            )
        by_delay[record.delay_steps].append(record)

    sorted_delays = sorted(by_delay)
    common_anchor_ids = tuple(sorted(record.anchor_id for record in by_delay[sorted_delays[0]]))
    common_anchor_set = set(common_anchor_ids)
    for delay in sorted_delays[1:]:
        delay_anchor_set = {record.anchor_id for record in by_delay[delay]}
        if delay_anchor_set != common_anchor_set:
            missing = sorted(common_anchor_set - delay_anchor_set)
            extra = sorted(delay_anchor_set - common_anchor_set)
            raise OracleEvaluationError(
                f"Delay {delay} does not use the common anchor cohort; missing={missing}, extra={extra}"
            )

    summaries: list[DelaySummary] = []
    for delay in sorted_delays:
        delay_records = by_delay[delay]
        summaries.append(
            DelaySummary(
                delay_steps=delay,
                sample_count=len(delay_records),
                policy_output=_summarize_action_space([record.policy_output for record in delay_records]),
                post_policy=_summarize_action_space([record.post_policy for record in delay_records]),
            )
        )

    return OracleEvaluationSummary(
        anchor_ids=common_anchor_ids,
        delays=tuple(summaries),
        total_record_count=len(records),
    )
