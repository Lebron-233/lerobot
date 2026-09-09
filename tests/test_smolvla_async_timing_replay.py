"""CPU tests for the fixed trace driver; preserve and expose the existing late-chunk contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from replay_smolvla_async_timing import VirtualControl, queue_boundary_cases  # noqa: E402


def test_ticks_keep_advancing_and_planner_uses_only_completed_history():
    replay = VirtualControl([0.25] * 4 + [0.08] * 8, "synthetic A")
    result = replay.run()
    assert result["status"] == "completed", result["exception"]
    assert result["trace_requests_completed"] == 12
    assert result["absolute_indices_unique"]
    assert result["no_action_ticks"]
    assert result["wall_ticks"] > result["actions_consumed"]
    assert all(r["current_duration_revealed_after_planning"] for r in result["records"])
    assert any(r["kind"] == "planned" for r in result["records"])


def test_boundary_probe_preserves_whole_discard_for_every_late_result():
    cases = queue_boundary_cases()
    assert all(case["passed"] for case in cases)
    by_name = {case["case"]: case for case in cases}
    for late, old_action in ((1, 4), (2, 5), (3, 6)):
        case = by_name[f"late_{late}_whole_discard"]
        assert case["late_steps"] == late
        assert case["actual_outcome"] == "deadline_miss"
        assert case["actual_next_action"] == case["expected_next_action"] == old_action
        assert case["matching_plan_cleared"] is True
        assert case["no_staged_chunk"] is True
