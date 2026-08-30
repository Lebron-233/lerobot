#!/usr/bin/env python

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

"""Thread-safe scheduled action queue for predictive asynchronous inference.

Unlike :class:`~lerobot.policies.rtc.action_queue.ActionQueue`, a scheduled
queue never installs a newly generated chunk merely because it arrived.  A
future-context request first freezes a committed prefix and an absolute
``takeover_index``.  Its result is staged until that index, then atomically
becomes active in :meth:`ScheduledActionQueue.get_with_task`.

The queue owns one lock covering its active chunk, in-flight plan, staged
chunk, epochs, and engine-lifetime action index.  Successful gets are the only
operation that advance that index; an underflow does not consume an index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

import torch
from torch import Tensor


class InstallOutcome(StrEnum):
    """Result of trying to install a bootstrap action chunk."""

    INSTALLED = "installed"
    BUSY = "busy"
    STALE = "stale"


class PlanOutcome(StrEnum):
    """Result of trying to create a takeover plan."""

    CREATED = "created"
    PLAN_IN_FLIGHT = "plan_in_flight"
    INSUFFICIENT_ACTIONS = "insufficient_actions"
    STALE = "stale"


class StageOutcome(StrEnum):
    """Disposition of a completed predictive inference request."""

    STAGED_EARLY = "staged_early"
    STAGED_ON_TIME = "staged_on_time"
    DEADLINE_MISS = "deadline_miss"
    STALE = "stale"
    ALREADY_STAGED = "already_staged"


class CancelOutcome(StrEnum):
    """Result of trying to cancel an in-flight plan."""

    CANCELLED = "cancelled"
    STALE = "stale"


class GetOutcome(StrEnum):
    """How a control-loop get was served."""

    ACTION = "action"
    TAKEOVER = "takeover"
    UNDERFLOW = "underflow"


@dataclass(frozen=True, slots=True)
class TakeoverPlan:
    """Immutable metadata and committed-prefix snapshot for one request.

    ``committed_policy_actions`` and ``committed_post_policy_actions`` are
    independently padded to ``max_prediction_delay`` along the time dimension.
    Their feature dimensions may differ.  ``committed_mask`` identifies the
    actual ``planned_delay_steps`` prefix; padded rows are false.
    """

    request_id: int
    next_action_index: int
    planned_delay_steps: int
    takeover_index: int
    committed_policy_actions: Tensor
    committed_post_policy_actions: Tensor
    committed_mask: Tensor
    reset_epoch: int
    task_epoch: int
    task: str


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Outcome and queue position for a bootstrap installation."""

    outcome: InstallOutcome
    next_action_index: int


@dataclass(frozen=True, slots=True)
class PlanCreationResult:
    """Outcome of plan creation, with a private worker copy when successful."""

    outcome: PlanOutcome
    plan: TakeoverPlan | None
    available_steps: int
    required_steps: int


@dataclass(frozen=True, slots=True)
class StageResult:
    """Outcome of staging a completed chunk."""

    outcome: StageOutcome
    request_id: int
    next_action_index: int
    takeover_index: int | None
    late_steps: int


@dataclass(frozen=True, slots=True)
class CancelResult:
    """Outcome of cancelling a plan."""

    outcome: CancelOutcome
    request_id: int


@dataclass(frozen=True, slots=True)
class GetResult:
    """One scheduled action, or an explicit underflow result.

    ``action_index`` is the absolute index of the returned action.  On
    underflow it is the still-unconsumed ``next_action_index``.
    """

    policy_action: Tensor | None
    post_policy_action: Tensor | None
    task: str | None
    action_index: int
    outcome: GetOutcome


@dataclass(slots=True)
class _ActionChunk:
    policy_actions: Tensor
    post_policy_actions: Tensor
    task: str
    offset: int = 0


@dataclass(slots=True)
class _StagedChunk:
    policy_actions: Tensor
    post_policy_actions: Tensor
    task: str


def _validate_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _validate_chunk(policy_actions: Tensor, post_policy_actions: Tensor) -> None:
    if not isinstance(policy_actions, Tensor) or not isinstance(post_policy_actions, Tensor):
        raise TypeError("policy_actions and post_policy_actions must be torch.Tensor instances")
    if policy_actions.ndim != 2 or post_policy_actions.ndim != 2:
        raise ValueError(
            "policy_actions and post_policy_actions must both have shape [T, A], got "
            f"{tuple(policy_actions.shape)} and {tuple(post_policy_actions.shape)}"
        )
    if policy_actions.shape[0] != post_policy_actions.shape[0]:
        raise ValueError(
            "policy_actions and post_policy_actions must have the same time dimension, got "
            f"{policy_actions.shape[0]} and {post_policy_actions.shape[0]}"
        )
    if policy_actions.shape[0] == 0:
        raise ValueError("action chunks must contain at least one timestep")


def _clone_plan(plan: TakeoverPlan) -> TakeoverPlan:
    return TakeoverPlan(
        request_id=plan.request_id,
        next_action_index=plan.next_action_index,
        planned_delay_steps=plan.planned_delay_steps,
        takeover_index=plan.takeover_index,
        committed_policy_actions=plan.committed_policy_actions.clone(),
        committed_post_policy_actions=plan.committed_post_policy_actions.clone(),
        committed_mask=plan.committed_mask.clone(),
        reset_epoch=plan.reset_epoch,
        task_epoch=plan.task_epoch,
        task=plan.task,
    )


class ScheduledActionQueue:
    """Manage committed, staged, and active chunks under one queue lock.

    Args:
        reset_epoch: Initial reset generation owned by the inference engine.
        task_epoch: Initial task generation owned by the inference engine.

    A queue supports a single in-flight :class:`TakeoverPlan`.  A task change
    invalidates the plan and staged result while preserving the active chunk
    and its original task provenance.  A reset additionally clears the active
    chunk.  Neither operation rewinds ``next_action_index``.
    """

    def __init__(self, *, reset_epoch: int = 0, task_epoch: int = 0) -> None:
        _validate_non_negative_int(reset_epoch, "reset_epoch")
        _validate_non_negative_int(task_epoch, "task_epoch")

        self._lock = Lock()
        self._active: _ActionChunk | None = None
        self._plan: TakeoverPlan | None = None
        self._staged: _StagedChunk | None = None
        self._next_action_index = 0
        self._last_request_id = -1
        self._reset_epoch = reset_epoch
        self._task_epoch = task_epoch

    @property
    def next_action_index(self) -> int:
        """Absolute index of the next successful get in this queue lifetime."""
        with self._lock:
            return self._next_action_index

    @property
    def reset_epoch(self) -> int:
        with self._lock:
            return self._reset_epoch

    @property
    def task_epoch(self) -> int:
        with self._lock:
            return self._task_epoch

    def install_active_chunk(
        self,
        policy_actions: Tensor,
        post_policy_actions: Tensor,
        *,
        task: str,
        reset_epoch: int,
        task_epoch: int,
    ) -> InstallResult:
        """Install a bootstrap/fallback chunk only when no work would be replaced."""
        _validate_chunk(policy_actions, post_policy_actions)
        if not isinstance(task, str):
            raise TypeError(f"task must be a string, got {type(task).__name__}")

        with self._lock:
            if reset_epoch != self._reset_epoch or task_epoch != self._task_epoch:
                return InstallResult(InstallOutcome.STALE, self._next_action_index)
            if self._remaining_steps_locked() or self._plan is not None or self._staged is not None:
                return InstallResult(InstallOutcome.BUSY, self._next_action_index)

            self._active = _ActionChunk(
                policy_actions=policy_actions.clone(),
                post_policy_actions=post_policy_actions.clone(),
                task=task,
            )
            return InstallResult(InstallOutcome.INSTALLED, self._next_action_index)

    def create_takeover_plan(
        self,
        *,
        request_id: int,
        planned_delay_steps: int,
        max_prediction_delay: int,
        committed_guard_steps: int,
        reset_epoch: int,
        task_epoch: int,
        task: str,
    ) -> PlanCreationResult:
        """Atomically freeze a padded committed prefix and takeover index.

        The caller must already have handled prediction-cap fallback:
        ``planned_delay_steps`` may not exceed ``max_prediction_delay``.  The
        active chunk must cover both the committed prefix and requested guard.
        """
        _validate_non_negative_int(request_id, "request_id")
        _validate_non_negative_int(planned_delay_steps, "planned_delay_steps")
        _validate_non_negative_int(max_prediction_delay, "max_prediction_delay")
        _validate_non_negative_int(committed_guard_steps, "committed_guard_steps")
        if planned_delay_steps > max_prediction_delay:
            raise ValueError(
                "planned_delay_steps exceeds max_prediction_delay; prediction-cap fallback "
                "must be resolved before creating a takeover plan"
            )
        if not isinstance(task, str):
            raise TypeError(f"task must be a string, got {type(task).__name__}")

        required_steps = planned_delay_steps + committed_guard_steps
        with self._lock:
            available_steps = self._remaining_steps_locked()
            if reset_epoch != self._reset_epoch or task_epoch != self._task_epoch:
                return PlanCreationResult(PlanOutcome.STALE, None, available_steps, required_steps)
            if self._plan is not None:
                return PlanCreationResult(
                    PlanOutcome.PLAN_IN_FLIGHT,
                    None,
                    available_steps,
                    required_steps,
                )
            if request_id <= self._last_request_id:
                raise ValueError(
                    f"request_id must increase monotonically; previous={self._last_request_id}, "
                    f"received={request_id}"
                )
            if available_steps < required_steps:
                return PlanCreationResult(
                    PlanOutcome.INSUFFICIENT_ACTIONS,
                    None,
                    available_steps,
                    required_steps,
                )

            if self._active is None:
                # ``available_steps >= required_steps`` only permits this when both are zero.
                # A zero-delay plan still needs action feature dimensions for static padding.
                return PlanCreationResult(
                    PlanOutcome.INSUFFICIENT_ACTIONS,
                    None,
                    available_steps,
                    required_steps,
                )

            start = self._active.offset
            stop = start + planned_delay_steps
            policy_prefix = self._active.policy_actions[start:stop]
            post_policy_prefix = self._active.post_policy_actions[start:stop]
            padded_policy = torch.zeros(
                (max_prediction_delay, self._active.policy_actions.shape[1]),
                dtype=self._active.policy_actions.dtype,
                device=self._active.policy_actions.device,
            )
            padded_post_policy = torch.zeros(
                (max_prediction_delay, self._active.post_policy_actions.shape[1]),
                dtype=self._active.post_policy_actions.dtype,
                device=self._active.post_policy_actions.device,
            )
            padded_policy[:planned_delay_steps].copy_(policy_prefix)
            padded_post_policy[:planned_delay_steps].copy_(post_policy_prefix)
            committed_mask = torch.zeros(
                max_prediction_delay,
                dtype=torch.bool,
                device=self._active.policy_actions.device,
            )
            committed_mask[:planned_delay_steps] = True

            internal_plan = TakeoverPlan(
                request_id=request_id,
                next_action_index=self._next_action_index,
                planned_delay_steps=planned_delay_steps,
                takeover_index=self._next_action_index + planned_delay_steps,
                committed_policy_actions=padded_policy,
                committed_post_policy_actions=padded_post_policy,
                committed_mask=committed_mask,
                reset_epoch=reset_epoch,
                task_epoch=task_epoch,
                task=task,
            )
            self._plan = internal_plan
            self._last_request_id = request_id
            return PlanCreationResult(
                PlanOutcome.CREATED,
                _clone_plan(internal_plan),
                available_steps,
                required_steps,
            )

    def stage_chunk(
        self,
        policy_actions: Tensor,
        post_policy_actions: Tensor,
        *,
        request_id: int,
        reset_epoch: int,
        task_epoch: int,
        task: str,
    ) -> StageResult:
        """Stage a matching result, or explicitly reject stale/late work.

        Any ``late_steps > 0`` is a deadline miss in the MVP.  The predictive
        chunk is discarded in full; its prefix is never skipped as if those
        actions had already executed.
        """
        _validate_chunk(policy_actions, post_policy_actions)

        with self._lock:
            plan = self._plan
            if plan is None or not self._matches_plan_locked(
                plan,
                request_id=request_id,
                reset_epoch=reset_epoch,
                task_epoch=task_epoch,
                task=task,
            ):
                return StageResult(
                    StageOutcome.STALE,
                    request_id,
                    self._next_action_index,
                    None if plan is None else plan.takeover_index,
                    0,
                )
            if self._staged is not None:
                return StageResult(
                    StageOutcome.ALREADY_STAGED,
                    request_id,
                    self._next_action_index,
                    plan.takeover_index,
                    max(0, self._next_action_index - plan.takeover_index),
                )

            late_steps = max(0, self._next_action_index - plan.takeover_index)
            if late_steps:
                # Clear only the plan proven to match this result.  A stale result
                # above never clears a newer request's plan or staged chunk.
                self._plan = None
                self._staged = None
                return StageResult(
                    StageOutcome.DEADLINE_MISS,
                    request_id,
                    self._next_action_index,
                    plan.takeover_index,
                    late_steps,
                )

            self._staged = _StagedChunk(
                policy_actions=policy_actions.clone(),
                post_policy_actions=post_policy_actions.clone(),
                task=plan.task,
            )
            outcome = (
                StageOutcome.STAGED_ON_TIME
                if self._next_action_index == plan.takeover_index
                else StageOutcome.STAGED_EARLY
            )
            return StageResult(
                outcome,
                request_id,
                self._next_action_index,
                plan.takeover_index,
                0,
            )

    def cancel_plan(
        self,
        *,
        request_id: int,
        reset_epoch: int,
        task_epoch: int,
    ) -> CancelResult:
        """Cancel only the exactly matching plan and its staged result."""
        with self._lock:
            plan = self._plan
            if (
                plan is None
                or plan.request_id != request_id
                or plan.reset_epoch != reset_epoch
                or plan.task_epoch != task_epoch
            ):
                return CancelResult(CancelOutcome.STALE, request_id)
            self._plan = None
            self._staged = None
            return CancelResult(CancelOutcome.CANCELLED, request_id)

    def invalidate_task(self, task_epoch: int) -> bool:
        """Invalidate plan/staged state for a newer task, preserving active actions.

        Returns true only when the epoch advanced.  Repeating or observing an
        older epoch is a no-op: task updates may race a control-thread snapshot,
        and stale invalidation must never clear a newer plan.
        """
        _validate_non_negative_int(task_epoch, "task_epoch")
        with self._lock:
            if task_epoch <= self._task_epoch:
                return False
            self._task_epoch = task_epoch
            self._plan = None
            self._staged = None
            return True

    def reset(self, reset_epoch: int, *, task_epoch: int | None = None) -> None:
        """Clear all action state for a newer reset without rewinding its index."""
        _validate_non_negative_int(reset_epoch, "reset_epoch")
        if task_epoch is not None:
            _validate_non_negative_int(task_epoch, "task_epoch")
        with self._lock:
            if reset_epoch <= self._reset_epoch:
                raise ValueError(
                    f"reset_epoch must increase; current={self._reset_epoch}, received={reset_epoch}"
                )
            self._reset_epoch = reset_epoch
            if task_epoch is not None and task_epoch > self._task_epoch:
                self._task_epoch = task_epoch
            self._active = None
            self._plan = None
            self._staged = None

    def get(self) -> Tensor | None:
        """Return the next post-policy action, or ``None`` on underflow."""
        return self.get_with_task().post_policy_action

    def get_with_task(self) -> GetResult:
        """Atomically take over when due, then return one action and provenance."""
        with self._lock:
            took_over = False
            if (
                self._plan is not None
                and self._staged is not None
                and self._next_action_index == self._plan.takeover_index
            ):
                self._active = _ActionChunk(
                    policy_actions=self._staged.policy_actions,
                    post_policy_actions=self._staged.post_policy_actions,
                    task=self._staged.task,
                )
                self._plan = None
                self._staged = None
                took_over = True

            if self._active is None or self._active.offset >= self._active.policy_actions.shape[0]:
                return GetResult(
                    policy_action=None,
                    post_policy_action=None,
                    task=None,
                    action_index=self._next_action_index,
                    outcome=GetOutcome.UNDERFLOW,
                )

            offset = self._active.offset
            result = GetResult(
                policy_action=self._active.policy_actions[offset].clone(),
                post_policy_action=self._active.post_policy_actions[offset].clone(),
                task=self._active.task,
                action_index=self._next_action_index,
                outcome=GetOutcome.TAKEOVER if took_over else GetOutcome.ACTION,
            )
            self._active.offset += 1
            self._next_action_index += 1
            return result

    def qsize(self) -> int:
        """Number of remaining active actions (staged actions are not active yet)."""
        with self._lock:
            return self._remaining_steps_locked()

    def available_steps(self) -> int:
        """Alias for the active committed capacity used by the delay planner."""
        return self.qsize()

    def empty(self) -> bool:
        return self.qsize() == 0

    def get_action_index(self) -> int:
        """Compatibility spelling for :attr:`next_action_index`."""
        return self.next_action_index

    def peek_policy_actions(self, n: int) -> Tensor | None:
        """Clone up to ``n`` unconsumed active actions in policy space."""
        _validate_non_negative_int(n, "n")
        with self._lock:
            if self._active is None:
                return None
            start = self._active.offset
            return self._active.policy_actions[start : start + n].clone()

    def peek_post_policy_actions(self, n: int) -> Tensor | None:
        """Clone up to ``n`` unconsumed active actions after policy postprocessing."""
        _validate_non_negative_int(n, "n")
        with self._lock:
            if self._active is None:
                return None
            start = self._active.offset
            return self._active.post_policy_actions[start : start + n].clone()

    def peek_original(self, n: int) -> Tensor | None:
        """Return policy-space actions using the terminology of ``ActionQueue``."""
        return self.peek_policy_actions(n)

    def peek_processed(self, n: int) -> Tensor | None:
        """Return post-policy actions using the terminology of ``ActionQueue``."""
        return self.peek_post_policy_actions(n)

    def plan_snapshot(self) -> TakeoverPlan | None:
        """Return a clone of the current plan for diagnostics and tests."""
        with self._lock:
            return None if self._plan is None else _clone_plan(self._plan)

    def has_staged_chunk(self) -> bool:
        with self._lock:
            return self._staged is not None

    def _remaining_steps_locked(self) -> int:
        if self._active is None:
            return 0
        return max(0, self._active.policy_actions.shape[0] - self._active.offset)

    @staticmethod
    def _matches_plan_locked(
        plan: TakeoverPlan,
        *,
        request_id: int,
        reset_epoch: int,
        task_epoch: int,
        task: str,
    ) -> bool:
        return (
            plan.request_id == request_id
            and plan.reset_epoch == reset_epoch
            and plan.task_epoch == task_epoch
            and plan.task == task
        )


__all__ = [
    "CancelOutcome",
    "CancelResult",
    "GetOutcome",
    "GetResult",
    "InstallOutcome",
    "InstallResult",
    "PlanCreationResult",
    "PlanOutcome",
    "ScheduledActionQueue",
    "StageOutcome",
    "StageResult",
    "TakeoverPlan",
]
