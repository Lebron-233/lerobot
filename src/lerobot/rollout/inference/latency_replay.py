# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Deterministic latency planning and replay without robot hardware."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from lerobot.policies.rtc.latency_tracker import LatencyTracker


def latency_to_steps(latency_s: float, fps: float) -> int:
    """Convert seconds to control steps without float32 boundary inflation.

    ``LatencyTracker.percentile`` currently returns a value computed through a
    float32 array. A mathematically exact one-step latency can therefore arrive
    here as ``1.00000001`` steps and would incorrectly ceil to two. Only values
    within float32-scale tolerance of an integer are snapped to that integer;
    all meaningful fractional steps still round upward.
    """
    steps = latency_s * fps
    nearest_step = round(steps)
    if math.isclose(steps, nearest_step, rel_tol=1e-7, abs_tol=1e-9):
        return int(nearest_step)
    return math.ceil(steps)


@dataclass(frozen=True)
class DelayPlan:
    """One delay estimate computed only from previously observed latencies."""

    estimated_latency_s: float
    raw_required_delay_steps: int
    planned_delay_steps: int
    available_after_guard_steps: int
    prediction_cap_exceeded: bool


@dataclass(frozen=True)
class LatencyReplayItem:
    """Planning and observed takeover offsets for one replayed request."""

    index: int
    latency_s: float
    estimated_latency_s: float | None
    raw_required_delay_steps: int | None
    planned_delay_steps: int | None
    prediction_cap_exceeded: bool | None
    actual_delay_steps: int
    available_actions: int
    available_after_guard_steps: int
    late_steps: int | None
    early_steps: int | None
    underflow_steps: int


@dataclass(frozen=True)
class LatencyReplaySummary:
    """Aggregate statistics for a latency replay."""

    sample_count: int
    calibrated_sample_count: int
    latency_window_sample_count: int
    latency_p50_s: float
    latency_p90_s: float
    late_chunk_count: int
    late_chunk_rate: float | None
    prediction_cap_exceeded_count: int
    underflow_count: int
    underflow_steps: int
    mean_actual_delay_steps: float
    mean_planned_delay_steps: float | None


@dataclass(frozen=True)
class LatencyReplayResult:
    """Per-request records and their aggregate summary."""

    items: tuple[LatencyReplayItem, ...]
    summary: LatencyReplaySummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _validate_planner_inputs(
    *,
    fps: float,
    latency_quantile: float,
    delay_safety_margin_steps: int,
    min_prediction_delay: int,
    max_prediction_delay: int,
    available_actions: int,
    committed_guard_steps: int,
) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and > 0, got {fps}")
    if not 0.0 <= latency_quantile <= 1.0:
        raise ValueError(f"latency_quantile must be in [0, 1], got {latency_quantile}")
    if delay_safety_margin_steps < 0:
        raise ValueError(f"delay_safety_margin_steps must be >= 0, got {delay_safety_margin_steps}")
    if min_prediction_delay < 0:
        raise ValueError(f"min_prediction_delay must be >= 0, got {min_prediction_delay}")
    if max_prediction_delay < min_prediction_delay:
        raise ValueError(
            "max_prediction_delay must be >= min_prediction_delay, got "
            f"{max_prediction_delay} < {min_prediction_delay}"
        )
    if available_actions < 0:
        raise ValueError(f"available_actions must be >= 0, got {available_actions}")
    if committed_guard_steps < 0:
        raise ValueError(f"committed_guard_steps must be >= 0, got {committed_guard_steps}")


def compute_delay_plan(
    latency_tracker: LatencyTracker,
    *,
    fps: float,
    latency_quantile: float,
    delay_safety_margin_steps: int,
    min_prediction_delay: int,
    max_prediction_delay: int,
    available_actions: int,
    committed_guard_steps: int,
) -> DelayPlan | None:
    """Plan a takeover offset from prior latency samples.

    ``None`` is returned when the tracker is empty. Availability is the final
    authority: when fewer actions remain than ``min_prediction_delay``, the
    plan is reduced below that configured minimum rather than claiming actions
    that the queue cannot commit.
    """
    _validate_planner_inputs(
        fps=fps,
        latency_quantile=latency_quantile,
        delay_safety_margin_steps=delay_safety_margin_steps,
        min_prediction_delay=min_prediction_delay,
        max_prediction_delay=max_prediction_delay,
        available_actions=available_actions,
        committed_guard_steps=committed_guard_steps,
    )
    if len(latency_tracker) == 0:
        return None

    estimated_latency_s = latency_tracker.percentile(latency_quantile)
    if estimated_latency_s is None:
        return None

    raw_required_delay_steps = latency_to_steps(estimated_latency_s, fps) + delay_safety_margin_steps
    available_after_guard_steps = max(0, available_actions - committed_guard_steps)
    effective_max_delay = min(max_prediction_delay, available_after_guard_steps)
    planned_delay_steps = min(max(raw_required_delay_steps, min_prediction_delay), effective_max_delay)
    return DelayPlan(
        estimated_latency_s=estimated_latency_s,
        raw_required_delay_steps=raw_required_delay_steps,
        planned_delay_steps=planned_delay_steps,
        available_after_guard_steps=available_after_guard_steps,
        prediction_cap_exceeded=raw_required_delay_steps > max_prediction_delay,
    )


def _resolve_available_actions(available_actions: int | Iterable[int], sample_count: int) -> tuple[int, ...]:
    if isinstance(available_actions, int):
        values = (available_actions,) * sample_count
    else:
        values = tuple(int(value) for value in available_actions)
        if len(values) != sample_count:
            raise ValueError(
                "available_actions must be one integer or contain one value per latency sample "
                f"({len(values)} values for {sample_count} samples)"
            )
    if any(value < 0 for value in values):
        raise ValueError(f"available_actions values must be >= 0, got {values}")
    return values


def replay_latencies(
    latencies_s: Iterable[float],
    *,
    fps: float,
    latency_quantile: float = 0.9,
    latency_window: int = 50,
    delay_safety_margin_steps: int = 1,
    min_prediction_delay: int = 0,
    max_prediction_delay: int = 8,
    available_actions: int | Iterable[int] = 30,
    committed_guard_steps: int = 2,
) -> LatencyReplayResult:
    """Replay observed latencies through the quantile delay planner.

    Each plan is made before adding that request's latency to the tracker, so
    the first item is an uncalibrated cold start and has ``planned_delay_steps``
    set to ``None``. ``actual_delay_steps`` is the observed takeover offset
    ``ceil(latency * fps)``. Underflow counts the part of that offset beyond the
    actions available when the request began.
    """
    latencies = tuple(float(latency) for latency in latencies_s)
    if not latencies:
        raise ValueError("latencies_s must contain at least one sample")
    if latency_window <= 0:
        raise ValueError(f"latency_window must be > 0, got {latency_window}")
    for latency in latencies:
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"latency samples must be finite and >= 0, got {latency}")

    available_per_sample = _resolve_available_actions(available_actions, len(latencies))
    tracker = LatencyTracker(maxlen=latency_window)
    items: list[LatencyReplayItem] = []

    for index, (latency_s, available) in enumerate(zip(latencies, available_per_sample, strict=True)):
        plan = compute_delay_plan(
            tracker,
            fps=fps,
            latency_quantile=latency_quantile,
            delay_safety_margin_steps=delay_safety_margin_steps,
            min_prediction_delay=min_prediction_delay,
            max_prediction_delay=max_prediction_delay,
            available_actions=available,
            committed_guard_steps=committed_guard_steps,
        )
        actual_delay_steps = latency_to_steps(latency_s, fps)
        if plan is None:
            late_steps = None
            early_steps = None
            estimated_latency_s = None
            raw_required_delay_steps = None
            planned_delay_steps = None
            prediction_cap_exceeded = None
            available_after_guard_steps = max(0, available - committed_guard_steps)
        else:
            late_steps = max(0, actual_delay_steps - plan.planned_delay_steps)
            early_steps = max(0, plan.planned_delay_steps - actual_delay_steps)
            estimated_latency_s = plan.estimated_latency_s
            raw_required_delay_steps = plan.raw_required_delay_steps
            planned_delay_steps = plan.planned_delay_steps
            prediction_cap_exceeded = plan.prediction_cap_exceeded
            available_after_guard_steps = plan.available_after_guard_steps

        items.append(
            LatencyReplayItem(
                index=index,
                latency_s=latency_s,
                estimated_latency_s=estimated_latency_s,
                raw_required_delay_steps=raw_required_delay_steps,
                planned_delay_steps=planned_delay_steps,
                prediction_cap_exceeded=prediction_cap_exceeded,
                actual_delay_steps=actual_delay_steps,
                available_actions=available,
                available_after_guard_steps=available_after_guard_steps,
                late_steps=late_steps,
                early_steps=early_steps,
                underflow_steps=max(0, actual_delay_steps - available),
            )
        )
        tracker.add(latency_s)

    calibrated_items = [item for item in items if item.planned_delay_steps is not None]
    late_chunk_count = sum(bool(item.late_steps) for item in calibrated_items)
    prediction_cap_exceeded_count = sum(item.prediction_cap_exceeded is True for item in calibrated_items)
    underflow_items = [item for item in items if item.underflow_steps > 0]
    planned_steps = [
        int(item.planned_delay_steps) for item in calibrated_items if item.planned_delay_steps is not None
    ]
    summary = LatencyReplaySummary(
        sample_count=len(items),
        calibrated_sample_count=len(calibrated_items),
        latency_window_sample_count=len(tracker),
        latency_p50_s=float(tracker.percentile(0.5)),
        latency_p90_s=float(tracker.percentile(0.9)),
        late_chunk_count=late_chunk_count,
        late_chunk_rate=(late_chunk_count / len(calibrated_items) if calibrated_items else None),
        prediction_cap_exceeded_count=prediction_cap_exceeded_count,
        underflow_count=len(underflow_items),
        underflow_steps=sum(item.underflow_steps for item in underflow_items),
        mean_actual_delay_steps=sum(item.actual_delay_steps for item in items) / len(items),
        mean_planned_delay_steps=(sum(planned_steps) / len(planned_steps) if planned_steps else None),
    )
    return LatencyReplayResult(items=tuple(items), summary=summary)
