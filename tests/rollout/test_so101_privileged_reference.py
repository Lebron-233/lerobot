"""Privileged information is confined to explicitly labeled target-time references."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from eval_so101_controlled_delay import create_plan, install_or_stage  # noqa: E402
from run_so101_privileged_reference import (  # noqa: E402
    decoder_context,
    paired_summary,
    prefix_geometry_differences,
    stage_at_target,
)

from lerobot.policies.rtc.scheduled_action_queue import GetOutcome, ScheduledActionQueue  # noqa: E402


def test_privileged_generation_only_after_seven_old_actions():
    queue = ScheduledActionQueue(reset_epoch=0, task_epoch=0)
    old = torch.arange(50).float()[:, None].expand(50, 6).clone()
    install_or_stage(queue, old, old, None, 0)
    for _ in range(20):
        queue.get_with_task()
    plan = create_plan(queue, 1)
    with pytest.raises(ValueError, match="exact target"):
        stage_at_target(queue, old, old, plan)
    for i in range(20, 27):
        assert queue.get_with_task().post_policy_action[0] == i
    new = torch.ones_like(old) * 999
    stage_at_target(queue, new, new, plan)
    get = queue.get_with_task()
    assert get.outcome == GetOutcome.TAKEOVER and get.action_index == 27
    assert get.post_policy_action[0] == 999


def test_future_state_is_privileged_only_for_joint_reference():
    request = {"tokens": "past", "masks": "mask", "predicted_state": "causal_prediction"}
    assert decoder_context("state_only", request)[2] == "causal_prediction"
    with pytest.raises(ValueError, match="cannot consume"):
        decoder_context("state_only", request, "future", "mask", "actual_future_state")
    assert (
        decoder_context("oracle_visual", request, "future", "mask", "actual_future_state")[2]
        == "causal_prediction"
    )
    assert (
        decoder_context("oracle_joint", request, "future", "mask", "actual_future_state")[2]
        == "actual_future_state"
    )


def test_privileged_gain_is_never_a_learned_gain():
    arms = ["state_only", "oracle_visual", "oracle_joint"]
    rows = [
        {
            "block": block,
            "arm": arm,
            "success": True,
            "technical_valid": True,
            "restricted_time_s": 80 if arm == "state_only" else 40,
            "initial_geometry_match": True,
            "bootstrap_matches": True,
        }
        for block in range(8)
        for arm in arms
    ]
    report = paired_summary(rows, arms)
    assert report["privileged_visual_reference_positive"]
    assert not report["learned_task_benefit"] and not report["realtime_qualified"]
    rows[0]["initial_geometry_match"] = False
    assert not paired_summary(rows, arms)["privileged_visual_reference_positive"]


def test_shared_commands_do_not_hide_pre_takeover_state_divergence(tmp_path):
    rows = [{"block": 0, "arm": arm} for arm in ("state_only", "oracle_visual")]
    for row in rows:
        folder = tmp_path / ("block00_" + row["arm"])
        folder.mkdir()
        shifted = row["arm"] == "oracle_visual"
        tick = {"state": [float(shifted)] * 6, "objects_before": {"orange": [0.01 * shifted, 0, 0]}}
        (folder / "ticks.jsonl").write_text(json.dumps(tick) + "\n")
    differences = prefix_geometry_differences(tmp_path, rows)
    assert differences[0]["state_transport_abs_max_by_coordinate"] == [1.0] * 6
    assert differences[0]["object_position_abs_max_m"] == pytest.approx(0.01)
