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

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from lerobot.policies.rtc.scheduled_action_queue import (
    CancelOutcome,
    GetOutcome,
    InstallOutcome,
    PlanOutcome,
    ScheduledActionQueue,
    StageOutcome,
    StageResult,
)


def _actions(steps: int, dim: int, *, start: float = 0.0) -> torch.Tensor:
    return torch.arange(start, start + steps * dim, dtype=torch.float32).reshape(steps, dim)


def _install(
    queue: ScheduledActionQueue,
    *,
    steps: int = 8,
    task: str = "pick",
    reset_epoch: int = 0,
    task_epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy = _actions(steps, 2, start=10)
    post_policy = _actions(steps, 3, start=100)
    result = queue.install_active_chunk(
        policy,
        post_policy,
        task=task,
        reset_epoch=reset_epoch,
        task_epoch=task_epoch,
    )
    assert result.outcome is InstallOutcome.INSTALLED
    return policy, post_policy


def _plan(
    queue: ScheduledActionQueue,
    *,
    request_id: int = 0,
    delay: int = 2,
    max_delay: int = 4,
    guard: int = 1,
    reset_epoch: int = 0,
    task_epoch: int = 0,
    task: str = "pick",
):
    result = queue.create_takeover_plan(
        request_id=request_id,
        planned_delay_steps=delay,
        max_prediction_delay=max_delay,
        committed_guard_steps=guard,
        reset_epoch=reset_epoch,
        task_epoch=task_epoch,
        task=task,
    )
    assert result.outcome is PlanOutcome.CREATED
    assert result.plan is not None
    return result.plan


def _stage(
    queue: ScheduledActionQueue,
    *,
    request_id: int = 0,
    reset_epoch: int = 0,
    task_epoch: int = 0,
    task: str = "pick",
) -> tuple[torch.Tensor, torch.Tensor, StageResult]:
    policy = _actions(4, 4, start=1_000)
    post_policy = _actions(4, 5, start=2_000)
    result = queue.stage_chunk(
        policy,
        post_policy,
        request_id=request_id,
        reset_epoch=reset_epoch,
        task_epoch=task_epoch,
        task=task,
    )
    return policy, post_policy, result


def test_bootstrap_get_tracks_absolute_index_and_underflow() -> None:
    queue = ScheduledActionQueue()
    policy, post_policy = _install(queue, steps=2)

    first = queue.get_with_task()
    second = queue.get_with_task()
    underflow = queue.get_with_task()
    repeated_underflow = queue.get_with_task()

    assert first.outcome is GetOutcome.ACTION
    assert first.action_index == 0
    assert first.task == "pick"
    torch.testing.assert_close(first.policy_action, policy[0])
    torch.testing.assert_close(first.post_policy_action, post_policy[0])
    assert second.action_index == 1
    assert underflow.outcome is GetOutcome.UNDERFLOW
    assert underflow.policy_action is None
    assert underflow.post_policy_action is None
    assert underflow.task is None
    assert underflow.action_index == 2
    assert repeated_underflow.action_index == 2
    assert queue.next_action_index == 2


def test_install_and_get_are_private_clones() -> None:
    queue = ScheduledActionQueue()
    policy, post_policy = _install(queue, steps=2)
    expected_policy = policy.clone()
    expected_post_policy = post_policy.clone()

    policy.fill_(-1)
    post_policy.fill_(-2)
    peeked_policy = queue.peek_policy_actions(2)
    peeked_post_policy = queue.peek_post_policy_actions(2)
    assert peeked_policy is not None
    assert peeked_post_policy is not None
    peeked_policy.fill_(-3)
    peeked_post_policy.fill_(-4)

    result = queue.get_with_task()
    torch.testing.assert_close(result.policy_action, expected_policy[0])
    torch.testing.assert_close(result.post_policy_action, expected_post_policy[0])
    assert result.policy_action is not None
    assert result.post_policy_action is not None
    result.policy_action.fill_(-5)
    result.post_policy_action.fill_(-6)
    torch.testing.assert_close(queue.peek_original(1), expected_policy[1:2])
    torch.testing.assert_close(queue.peek_processed(1), expected_post_policy[1:2])


def test_policy_and_post_policy_feature_dims_may_differ_but_time_must_match() -> None:
    queue = ScheduledActionQueue()
    result = queue.install_active_chunk(
        torch.zeros(3, 2),
        torch.zeros(3, 7),
        task="pick",
        reset_epoch=0,
        task_epoch=0,
    )
    assert result.outcome is InstallOutcome.INSTALLED

    with pytest.raises(ValueError, match="same time dimension"):
        ScheduledActionQueue().install_active_chunk(
            torch.zeros(3, 2),
            torch.zeros(2, 7),
            task="pick",
            reset_epoch=0,
            task_epoch=0,
        )


def test_takeover_plan_freezes_padded_private_committed_prefix() -> None:
    queue = ScheduledActionQueue()
    policy, post_policy = _install(queue)
    queue.get_with_task()

    plan = _plan(queue, request_id=4, delay=3, max_delay=5, guard=2)
    assert plan.next_action_index == 1
    assert plan.takeover_index == 4
    assert plan.planned_delay_steps == 3
    assert plan.committed_policy_actions.shape == (5, 2)
    assert plan.committed_post_policy_actions.shape == (5, 3)
    assert plan.committed_mask.tolist() == [True, True, True, False, False]
    torch.testing.assert_close(plan.committed_policy_actions[:3], policy[1:4])
    torch.testing.assert_close(plan.committed_post_policy_actions[:3], post_policy[1:4])
    torch.testing.assert_close(plan.committed_policy_actions[3:], torch.zeros(2, 2))
    torch.testing.assert_close(plan.committed_post_policy_actions[3:], torch.zeros(2, 3))

    policy.fill_(-1)
    post_policy.fill_(-1)
    plan.committed_policy_actions.fill_(-2)
    plan.committed_post_policy_actions.fill_(-2)
    plan.committed_mask.fill_(False)
    private_snapshot = queue.plan_snapshot()
    assert private_snapshot is not None
    assert private_snapshot.committed_mask.tolist() == [True, True, True, False, False]
    assert torch.all(private_snapshot.committed_policy_actions[:3] >= 0)
    assert torch.all(private_snapshot.committed_post_policy_actions[:3] >= 0)


def test_plan_creation_enforces_single_in_flight_and_available_guard() -> None:
    queue = ScheduledActionQueue()
    _install(queue, steps=4)
    _plan(queue, request_id=1, delay=2, max_delay=4, guard=2)

    second = queue.create_takeover_plan(
        request_id=2,
        planned_delay_steps=1,
        max_prediction_delay=4,
        committed_guard_steps=1,
        reset_epoch=0,
        task_epoch=0,
        task="pick",
    )
    assert second.outcome is PlanOutcome.PLAN_IN_FLIGHT

    assert queue.cancel_plan(request_id=1, reset_epoch=0, task_epoch=0).outcome is CancelOutcome.CANCELLED
    insufficient = queue.create_takeover_plan(
        request_id=2,
        planned_delay_steps=3,
        max_prediction_delay=4,
        committed_guard_steps=2,
        reset_epoch=0,
        task_epoch=0,
        task="pick",
    )
    assert insufficient.outcome is PlanOutcome.INSUFFICIENT_ACTIONS
    assert insufficient.available_steps == 4
    assert insufficient.required_steps == 5


def test_plan_rejects_silent_prediction_cap_clamping() -> None:
    queue = ScheduledActionQueue()
    _install(queue)

    with pytest.raises(ValueError, match="prediction-cap fallback"):
        queue.create_takeover_plan(
            request_id=0,
            planned_delay_steps=5,
            max_prediction_delay=4,
            committed_guard_steps=0,
            reset_epoch=0,
            task_epoch=0,
            task="pick",
        )


def test_early_result_waits_for_exact_takeover_index() -> None:
    queue = ScheduledActionQueue()
    old_policy, old_post_policy = _install(queue)
    plan = _plan(queue, delay=2)
    new_policy, new_post_policy, staged = _stage(queue)
    assert staged.outcome is StageOutcome.STAGED_EARLY
    assert queue.has_staged_chunk()

    before_0 = queue.get_with_task()
    before_1 = queue.get_with_task()
    takeover = queue.get_with_task()
    after = queue.get_with_task()

    assert [before_0.action_index, before_1.action_index, takeover.action_index] == [0, 1, 2]
    assert before_0.outcome is GetOutcome.ACTION
    assert before_1.outcome is GetOutcome.ACTION
    torch.testing.assert_close(before_0.post_policy_action, old_post_policy[0])
    torch.testing.assert_close(before_1.policy_action, old_policy[1])
    assert takeover.outcome is GetOutcome.TAKEOVER
    assert takeover.action_index == plan.takeover_index
    torch.testing.assert_close(takeover.policy_action, new_policy[0])
    torch.testing.assert_close(takeover.post_policy_action, new_post_policy[0])
    assert after.outcome is GetOutcome.ACTION
    torch.testing.assert_close(after.post_policy_action, new_post_policy[1])


def test_exactly_on_time_stage_switches_on_next_get() -> None:
    queue = ScheduledActionQueue()
    old_policy, _ = _install(queue)
    _plan(queue, delay=2)
    torch.testing.assert_close(queue.get_with_task().policy_action, old_policy[0])
    torch.testing.assert_close(queue.get_with_task().policy_action, old_policy[1])

    new_policy, _, staged = _stage(queue)
    assert staged.outcome is StageOutcome.STAGED_ON_TIME
    takeover = queue.get_with_task()
    assert takeover.outcome is GetOutcome.TAKEOVER
    assert takeover.action_index == 2
    torch.testing.assert_close(takeover.policy_action, new_policy[0])


@pytest.mark.parametrize("late_steps", [1, 3])
def test_any_late_result_is_dropped_without_skipping_its_prefix(late_steps: int) -> None:
    queue = ScheduledActionQueue()
    old_policy, _ = _install(queue, steps=7)
    plan = _plan(queue, delay=1, guard=2)

    first = queue.get_with_task()
    old_after_takeover = None
    for _ in range(late_steps):
        old_after_takeover = queue.get_with_task()
    assert first.action_index == 0
    assert old_after_takeover is not None
    assert old_after_takeover.action_index == plan.takeover_index + late_steps - 1
    torch.testing.assert_close(old_after_takeover.policy_action, old_policy[late_steps])

    new_policy, _, result = _stage(queue)
    assert result.outcome is StageOutcome.DEADLINE_MISS
    assert result.late_steps == late_steps
    assert queue.plan_snapshot() is None
    assert not queue.has_staged_chunk()

    next_result = queue.get_with_task()
    assert next_result.outcome is GetOutcome.ACTION
    torch.testing.assert_close(next_result.policy_action, old_policy[late_steps + 1])
    assert not torch.equal(next_result.policy_action, new_policy[late_steps])


def test_late_result_with_exhausted_active_explicitly_underflows() -> None:
    queue = ScheduledActionQueue()
    _install(queue, steps=2)
    _plan(queue, delay=1, guard=1)
    queue.get_with_task()
    queue.get_with_task()

    _, _, stage_result = _stage(queue)
    assert stage_result.outcome is StageOutcome.DEADLINE_MISS
    assert stage_result.late_steps == 1
    underflow = queue.get_with_task()
    assert underflow.outcome is GetOutcome.UNDERFLOW
    assert underflow.action_index == 2
    assert queue.next_action_index == 2


def test_staged_chunk_is_a_private_clone() -> None:
    queue = ScheduledActionQueue()
    _install(queue)
    _plan(queue, delay=1)
    new_policy, new_post_policy, result = _stage(queue)
    assert result.outcome is StageOutcome.STAGED_EARLY
    expected_policy = new_policy[0].clone()
    expected_post_policy = new_post_policy[0].clone()
    new_policy.fill_(-1)
    new_post_policy.fill_(-1)

    queue.get_with_task()
    takeover = queue.get_with_task()
    torch.testing.assert_close(takeover.policy_action, expected_policy)
    torch.testing.assert_close(takeover.post_policy_action, expected_post_policy)


def test_duplicate_stage_does_not_replace_first_staged_chunk() -> None:
    queue = ScheduledActionQueue()
    _install(queue)
    _plan(queue, delay=1)
    first_policy, _, first = _stage(queue)
    assert first.outcome is StageOutcome.STAGED_EARLY

    replacement = torch.full((4, 4), -99.0)
    duplicate = queue.stage_chunk(
        replacement,
        torch.full((4, 5), -99.0),
        request_id=0,
        reset_epoch=0,
        task_epoch=0,
        task="pick",
    )
    assert duplicate.outcome is StageOutcome.ALREADY_STAGED
    queue.get_with_task()
    takeover = queue.get_with_task()
    torch.testing.assert_close(takeover.policy_action, first_policy[0])


def test_stale_result_cannot_clear_a_newer_plan_across_a_b_a_tasks() -> None:
    queue = ScheduledActionQueue()
    _install(queue, steps=10, task="A")
    old_plan = _plan(queue, request_id=1, delay=1, task="A")

    assert queue.invalidate_task(1)
    plan_b = _plan(queue, request_id=2, delay=1, task_epoch=1, task="B")
    stale_policy = torch.zeros(2, 2)
    stale_post_policy = torch.zeros(2, 3)
    stale_b = queue.stage_chunk(
        stale_policy,
        stale_post_policy,
        request_id=old_plan.request_id,
        reset_epoch=old_plan.reset_epoch,
        task_epoch=old_plan.task_epoch,
        task=old_plan.task,
    )
    assert stale_b.outcome is StageOutcome.STALE
    assert queue.plan_snapshot().request_id == plan_b.request_id

    assert queue.invalidate_task(2)
    new_plan_a = _plan(queue, request_id=3, delay=1, task_epoch=2, task="A")
    stale_a = queue.stage_chunk(
        stale_policy,
        stale_post_policy,
        request_id=old_plan.request_id,
        reset_epoch=old_plan.reset_epoch,
        task_epoch=old_plan.task_epoch,
        task=old_plan.task,
    )
    assert stale_a.outcome is StageOutcome.STALE
    assert queue.plan_snapshot().request_id == new_plan_a.request_id


def test_task_change_invalidates_plan_and_stage_but_preserves_active_provenance() -> None:
    queue = ScheduledActionQueue()
    old_policy, _ = _install(queue, task="old task")
    _plan(queue, delay=2, task="old task")
    _, _, staged = _stage(queue, task="old task")
    assert staged.outcome is StageOutcome.STAGED_EARLY

    assert queue.invalidate_task(1)
    assert queue.plan_snapshot() is None
    assert not queue.has_staged_chunk()
    result = queue.get_with_task()
    assert result.task == "old task"
    torch.testing.assert_close(result.policy_action, old_policy[0])
    assert queue.invalidate_task(1) is False


def test_stale_task_invalidation_and_reset_snapshot_never_rewind_task_epoch() -> None:
    queue = ScheduledActionQueue()
    _install(queue, task="A")
    assert queue.invalidate_task(2)

    assert queue.invalidate_task(1) is False
    assert queue.task_epoch == 2

    queue.reset(1, task_epoch=1)
    assert queue.task_epoch == 2
    assert queue.qsize() == 0


def test_reset_clears_all_state_without_reusing_action_indexes() -> None:
    queue = ScheduledActionQueue()
    _install(queue)
    queue.get_with_task()
    queue.get_with_task()
    _plan(queue, request_id=1, delay=1)
    _stage(queue, request_id=1)

    queue.reset(1)
    assert queue.qsize() == 0
    assert queue.plan_snapshot() is None
    assert not queue.has_staged_chunk()
    assert queue.next_action_index == 2
    underflow = queue.get_with_task()
    assert underflow.outcome is GetOutcome.UNDERFLOW
    assert underflow.action_index == 2

    _install(queue, steps=4, reset_epoch=1)
    _plan(queue, request_id=2, delay=1, reset_epoch=1)
    stale = queue.stage_chunk(
        torch.zeros(2, 2),
        torch.zeros(2, 3),
        request_id=1,
        reset_epoch=0,
        task_epoch=0,
        task="pick",
    )
    assert stale.outcome is StageOutcome.STALE
    assert queue.plan_snapshot().request_id == 2


def test_cancel_only_clears_the_matching_plan() -> None:
    queue = ScheduledActionQueue()
    _install(queue)
    plan = _plan(queue, request_id=3)
    _stage(queue, request_id=3)

    stale = queue.cancel_plan(request_id=2, reset_epoch=0, task_epoch=0)
    assert stale.outcome is CancelOutcome.STALE
    assert queue.plan_snapshot().request_id == plan.request_id
    assert queue.has_staged_chunk()

    cancelled = queue.cancel_plan(request_id=3, reset_epoch=0, task_epoch=0)
    assert cancelled.outcome is CancelOutcome.CANCELLED
    assert queue.plan_snapshot() is None
    assert not queue.has_staged_chunk()


def test_bootstrap_rejects_stale_or_busy_install_without_replacing_active() -> None:
    queue = ScheduledActionQueue()
    old_policy, _ = _install(queue, steps=2)
    stale = queue.install_active_chunk(
        torch.full((2, 2), -1.0),
        torch.full((2, 3), -1.0),
        task="pick",
        reset_epoch=1,
        task_epoch=0,
    )
    busy = queue.install_active_chunk(
        torch.full((2, 2), -2.0),
        torch.full((2, 3), -2.0),
        task="pick",
        reset_epoch=0,
        task_epoch=0,
    )
    assert stale.outcome is InstallOutcome.STALE
    assert busy.outcome is InstallOutcome.BUSY
    torch.testing.assert_close(queue.get_with_task().policy_action, old_policy[0])


def test_request_id_stays_monotonic_across_reset() -> None:
    queue = ScheduledActionQueue()
    _install(queue)
    _plan(queue, request_id=5)
    queue.reset(1)
    _install(queue, reset_epoch=1)

    with pytest.raises(ValueError, match="request_id must increase monotonically"):
        queue.create_takeover_plan(
            request_id=5,
            planned_delay_steps=1,
            max_prediction_delay=4,
            committed_guard_steps=1,
            reset_epoch=1,
            task_epoch=0,
            task="pick",
        )


def test_concurrent_gets_return_each_absolute_index_once() -> None:
    queue = ScheduledActionQueue()
    _install(queue, steps=100)

    def drain() -> list[int]:
        indexes: list[int] = []
        while True:
            result = queue.get_with_task()
            if result.outcome is GetOutcome.UNDERFLOW:
                return indexes
            indexes.append(result.action_index)

    with ThreadPoolExecutor(max_workers=4) as executor:
        indexes = [
            index for worker_indexes in executor.map(lambda _: drain(), range(4)) for index in worker_indexes
        ]

    assert sorted(indexes) == list(range(100))
    assert len(set(indexes)) == 100
    assert queue.next_action_index == 100


@pytest.mark.parametrize(
    ("policy_shape", "post_policy_shape", "message"),
    [
        ((2,), (2, 3), "shape \\[T, A\\]"),
        ((0, 2), (0, 3), "at least one timestep"),
    ],
)
def test_chunk_shape_validation(policy_shape, post_policy_shape, message) -> None:
    queue = ScheduledActionQueue()
    with pytest.raises(ValueError, match=message):
        queue.install_active_chunk(
            torch.zeros(policy_shape),
            torch.zeros(post_policy_shape),
            task="pick",
            reset_epoch=0,
            task_epoch=0,
        )
