"""The closed audit must reject future-data leakage disguised as causal input."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from audit_so101_privileged_reference import information_schedule  # noqa: E402


def fixture(arm="oracle_visual", count=30):
    ticks = [
        {
            "tick": t,
            "sim_step": t,
            "dispatch": "completed",
            "normalized_action": [float(t)] * 6,
            "queue_outcome": "takeover" if t == 27 else "action",
        }
        for t in range(count)
    ]
    events = [
        {
            "request_id": 1,
            "request_step": 20,
            "takeover_index": 27,
            "noise_seed": 3610 * 100000 + 20,
            "state_forecaster_calls": 1,
            "visual_forecaster_calls": 0,
            "privileged": arm.startswith("oracle"),
            "generated_step": 27 if arm.startswith("oracle") else 20,
            "visual_observation_step": 27 if arm.startswith("oracle") else 20,
            "state_observation_step": 20,
            "outcome": "staged_on_time" if arm.startswith("oracle") else "staged_early",
            "committed_prefix": [[float(t)] * 6 for t in range(20, 27)],
        }
    ]
    return ticks, events


def test_actual_target_observation_cannot_be_labeled_a_state_only_prediction():
    ticks, events = fixture("state_only")
    assert information_schedule(ticks, events, "state_only", 3610)["takeovers"] == 1
    invalid = copy.deepcopy(events)
    invalid[0]["visual_observation_step"] = 27
    with pytest.raises(ValueError, match="time boundary"):
        information_schedule(ticks, invalid, "state_only", 3610)


def test_privileged_calls_after_early_stop_are_not_invented():
    ticks, events = fixture(count=25)
    events[0].update(generated_step=None, outcome="censored_before_future_observation")
    report = information_schedule(ticks, events, "oracle_visual", 3610)
    assert report["privileged_generations"] == 0 and report["actual_old_prefix_actions"] == 5
    events[0]["generated_step"] = 27
    with pytest.raises(ValueError, match="invented"):
        information_schedule(ticks, events, "oracle_visual", 3610)
