import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from audit_so101_action_benefit import summarize_cases
from run_so101_task_comparison import comparison_summary


def test_task_gate_requires_both_halves_and_no_added_technical_failures():
    rows = []
    for seed in range(20261101, 20261113):
        for mode in ("sync", "identity", "predicted"):
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "technical_valid": True,
                    "first_placement_success": True,
                    "restricted_placement_time_s": 40 if mode == "predicted" else 60,
                    "initial_state": [0],
                    "initial_objects": {"a": [0]},
                    "initial_camera_poses": {"a": [0]},
                }
            )
    assert comparison_summary(rows)["stable_task_benefit_gate"]
    rows[-1]["technical_valid"] = False
    rows[-1]["first_placement_success"] = False
    rows[-1]["restricted_placement_time_s"] = 120
    assert not comparison_summary(rows)["stable_task_benefit_gate"]


def test_action_gate_uses_episodes_and_predeclared_case_completeness():
    rows = [
        {"episode": e, "identity_l1_25": 1.0, "predicted_l1_25": 0.8} for e in range(6) for _ in range(36)
    ]
    assert summarize_cases(rows, [])["stable_action_gate"]
    assert not summarize_cases(rows[:-1], [{"missing": True}])["stable_action_gate"]
    for row in rows:
        row["identity_l1_25"] = 0.0
    assert not summarize_cases(rows, [])["stable_action_gate"]
