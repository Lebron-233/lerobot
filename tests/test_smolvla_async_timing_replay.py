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


def test_boundary_probe_reports_the_real_missing_crop_contract():
    cases = queue_boundary_cases()
    failed = [case for case in cases if not case["passed"]]
    assert len(failed) == 1
    assert failed[0]["case"] == "slightly_late_prefix_crop"
    assert failed[0]["late_steps"] == 1
    assert failed[0]["actual_outcome"] == "deadline_miss"
    assert failed[0]["actual_next_action"] == 4
    assert failed[0]["expected_next_action"] == 101
