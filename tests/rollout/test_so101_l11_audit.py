"""Independent endpoint accounting, runnable without importing the model runtime."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from audit_so101_l11_results import outcome_evidence


class EndpointAuditTest(unittest.TestCase):
    def fixture(self):
        witness = {
            "native_success": False,
            "plate_position": [0.0, 0.0, 0.0],
            "joint_positions_radians": [0.0] * 6,
            "oranges": {"Orange001": {"position": [0.04, 0.0, 0.03], "linear_velocity": [0.0] * 3}},
        }
        rows = [
            {
                "dispatch": "completed",
                "tick": i,
                "terminated": False,
                "task_transition_after_action": copy.deepcopy(witness),
            }
            for i in range(10)
        ]
        rows[0]["task_diagnostics_before_action"] = {
            "object_positions_world": {"Plate": [0.0] * 3, "Orange001": [0.2, 0.0, 0.03]}
        }
        result = {
            "status": "task_subgoal_reached",
            "success": None,
            "subprocess_returncode": 0,
            "metrics_closed": True,
        }
        return rows, result

    def test_subgoal_preserves_native_null_and_exposes_actual_object_motion(self):
        rows, result = self.fixture()
        evidence = outcome_evidence(rows, result)
        self.assertTrue(evidence["first_placement_success"])
        self.assertIsNone(evidence["native_success"])
        self.assertEqual(evidence["restricted_placement_time_s"], 10 / 30)
        self.assertAlmostEqual(evidence["object_movement_to_endpoint_m"]["Orange001"], 0.16)
        self.assertTrue(evidence["endpoint_is_not_a_contact_release_test"])

    def test_technical_failure_retains_failure_cost_even_after_region_occupancy(self):
        rows, result = self.fixture()
        result["status"] = "technical_failure"
        evidence = outcome_evidence(rows, result)
        self.assertFalse(evidence["first_placement_success"])
        self.assertEqual(evidence["restricted_placement_time_s"], 120.0)

    def test_missing_pre_reset_witness_is_not_a_valid_success(self):
        rows, result = self.fixture()
        rows[-1]["task_transition_after_action"] = None
        with self.assertRaises(ValueError):
            outcome_evidence(rows, result)


if __name__ == "__main__":
    unittest.main()
