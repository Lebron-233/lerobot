"""A real prefix mismatch must not pass an otherwise favorable task summary."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from audit_so101_controlled_delay import audit_schedule  # noqa: E402


def fixture(n=30):
    ticks = [
        {
            "tick": i,
            "sim_step": i,
            "queue_action_index": i,
            "sim_time_s": i / 30,
            "dispatch": "completed",
            "normalized_action": [float(i)] * 6,
            "queue_outcome": "takeover" if i == 27 else "action",
        }
        for i in range(n)
    ]
    events = [
        {
            "request_id": i,
            "request_step": step,
            "noise_seed": 3410 * 100000 + step,
            "physics_paused_during_inference": True,
            "simulated_delay_steps": 7 if i else 0,
            "takeover_index": 27 if i else None,
            "state_predictor_calls": i,
            "visual_predictor_calls": i,
            "outcome": "staged_for_future_index" if i else "installed",
        }
        for i, step in enumerate((0, 20))
    ]
    events[1]["committed_normalized_prefix"] = [[float(i)] * 6 for i in range(20, 27)]
    return ticks, events


class ControlledAuditTests(unittest.TestCase):
    def test_seven_actual_old_steps_and_takeover(self):
        ticks, events = fixture()
        result = audit_schedule(ticks, events, "joint", 3410)
        self.assertEqual(result["executed_old_prefix_actions"], 7)
        self.assertEqual(result["takeovers"], 1)
        self.assertEqual(result["state_calls"], 1)

    def test_wrong_executed_prefix_is_rejected(self):
        ticks, events = fixture()
        ticks[24]["normalized_action"][0] += 1
        with self.assertRaisesRegex(ValueError, "old committed action"):
            audit_schedule(ticks, events, "joint", 3410)

    def test_early_task_stop_is_not_an_executed_takeover(self):
        ticks, events = fixture(24)
        result = audit_schedule(ticks, events, "joint", 3410)
        self.assertEqual(result["takeovers"], 0)
        self.assertEqual(result["executed_old_prefix_actions"], 4)
        self.assertEqual(result["unused_prefix_rows_at_task_stop"], 3)

    def test_state_only_must_not_count_a_visual_forecast(self):
        ticks, events = fixture()
        with self.assertRaisesRegex(ValueError, "Component-call"):
            audit_schedule(ticks, events, "state_only", 3410)
        changed = copy.deepcopy(events)
        changed[1]["visual_predictor_calls"] = 0
        self.assertEqual(audit_schedule(ticks, changed, "state_only", 3410)["visual_calls"], 0)


if __name__ == "__main__":
    unittest.main()
