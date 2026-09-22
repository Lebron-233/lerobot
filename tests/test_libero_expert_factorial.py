"""CPU tests for 2x2 scientific contrast and complete denominators."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from audit_libero_expert_factorial import reduce_rows  # noqa: E402
from libero_expert_factorial import LABELS, MODES  # noqa: E402


def rows(d=4.0):
    values = dict(zip(LABELS, (10.0, 8.0, 7.0, 6.0, d), strict=True))
    return [
        {
            "checkpoint": label,
            "mode": mode,
            "case": i,
            "trajectory_id": str(i // 4),
            **{
                space: dict.fromkeys(
                    ("block", "suffix", "first", "short"),
                    values[label] + (4 if mode == "mismatched_prefix" else 0),
                )
                for space in ("normalized", "command")
            },
        }
        for label in LABELS
        for mode in MODES
        for i in range(16)
    ]


def test_true_increment_passes_and_preserves_four_trajectory_denominator():
    result = reduce_rows(rows())
    assert result["expert_adaptation_followup_supported"]
    assert result["prefix_training_interaction"]["first"]["difference_in_gains"] == 1
    assert len(result["per_trajectory_gain_vs_projection_prefix"]["first"]) == 4


def test_general_adaptation_is_not_prefix_specific_increment():
    result = reduce_rows(rows(d=5))
    assert result["development_gates"]["better_than_projection_prefix"]
    assert not result["development_gates"]["positive_interaction_both_metrics"]
    assert not result["expert_adaptation_followup_supported"]


def test_missing_case_is_rejected():
    with pytest.raises(ValueError, match="population"):
        reduce_rows(rows()[:-1])


def test_first_action_or_command_regression_cannot_hide_in_block_mean():
    values = rows()
    for v in values:
        if v["checkpoint"] == "expert_prefix" and v["mode"] == "correct_prefix":
            v["normalized"]["first"] = 8.5
            v["command"]["short"] = 12
    result = reduce_rows(values)
    assert not result["development_gates"]["better_than_projection_prefix"]
    assert not result["development_gates"]["command_not_worse_than_projection_prefix"]
