"""Exact simulated age is causal and does not claim concurrent worker execution."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from eval_so101_controlled_delay import create_plan, install_or_stage  # noqa: E402

from lerobot.policies.rtc.scheduled_action_queue import GetOutcome, ScheduledActionQueue  # noqa: E402


def test_all_seven_committed_actions_execute_before_new_result_is_visible():
    queue = ScheduledActionQueue(reset_epoch=0, task_epoch=0)
    old = torch.arange(50).float()[:, None].expand(50, 6).clone()
    install_or_stage(queue, old, old * 2, None, 0)
    for step in range(20):
        assert queue.get_with_task().action_index == step
    plan = create_plan(queue, 1)
    assert plan.takeover_index == 27
    torch.testing.assert_close(plan.committed_policy_actions[:7], old[20:27], rtol=0, atol=0)
    new = torch.full((50, 6), 100.0)
    install_or_stage(queue, new, new * 2, plan, 1)
    for index in range(20, 27):
        get = queue.get_with_task()
        assert get.outcome == GetOutcome.ACTION and get.action_index == index
        torch.testing.assert_close(get.post_policy_action, old[index] * 2, rtol=0, atol=0)
    get = queue.get_with_task()
    assert get.outcome == GetOutcome.TAKEOVER and get.action_index == 27
    torch.testing.assert_close(get.post_policy_action, new[0] * 2, rtol=0, atol=0)
