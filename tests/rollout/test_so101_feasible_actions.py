import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from leisaac_so101_contract import action_to_radians
from leisaac_so101_matched import motor_to_physical
from so101_feasible_actions import FeasibleActionProjector, SO101FeasibleAsyncEngine

from lerobot.policies.rtc.scheduled_action_queue import ScheduledActionQueue


def test_projection_preserves_in_range_values_exactly_and_rejects_nan():
    projector = FeasibleActionProjector(torch.zeros(6), torch.ones(6))
    action = torch.tensor([[1.23, -25, 3, 4, 5, 50.0]])
    assert torch.equal(projector(action), action)
    assert projector.records[-1]["projected_components"] == 0
    action[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        projector(action)


def test_actual_queue_prefix_matches_projected_execution_not_raw_policy():
    mean = torch.tensor([2, -5, 4, 6, 8, 30.0])
    std = torch.tensor([10, 20, 15, 5, 10, 12.0])
    projector = FeasibleActionProjector(mean, std)
    raw_native = torch.tensor([120, -102, 105, 100.04, 30, -0.2])
    raw = ((raw_native - mean) / std).reshape(1, 1, 6).repeat(1, 50, 1)
    context = SimpleNamespace(
        action_projector=projector,
        _postprocessor=lambda normalized: motor_to_physical(normalized * std + mean),
    )
    metrics = {}
    normalized, physical = SO101FeasibleAsyncEngine._prepare_queue_actions(context, raw, metrics)
    assert metrics["action_projection"]["projected_components"] == 250
    for row in physical:
        action_to_radians(row.tolist())
    queue = ScheduledActionQueue(reset_epoch=0, task_epoch=0)
    queue.install_active_chunk(normalized, physical, task="fixture", reset_epoch=0, task_epoch=0)
    plan = queue.create_takeover_plan(
        request_id=1,
        planned_delay_steps=8,
        max_prediction_delay=8,
        committed_guard_steps=2,
        reset_epoch=0,
        task_epoch=0,
        task="fixture",
    ).plan
    torch.testing.assert_close(
        motor_to_physical(plan.committed_policy_actions * std + mean),
        plan.committed_post_policy_actions,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(plan.committed_policy_actions[0], raw[0, 0])
