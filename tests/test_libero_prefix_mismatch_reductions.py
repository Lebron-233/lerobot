"""Check comparison sign, ties, and episode weights before observing F-PFX1."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_prefix_mismatch as audit  # noqa: E402


@pytest.mark.parametrize("control,treatment,expected", [
    (0.2, 0.1, "better"), (0.1, 0.2, "worse"), (0.1, 0.1, "tie"),
    (0.1, 0.1 - 1e-8, "tie"),
])
def test_direction_is_control_minus_treatment_with_frozen_tolerance(control, treatment, expected):
    assert audit.direction(control, treatment) == expected


def test_episode_macro_does_not_upweight_larger_groups():
    rows = [{"key": [0, 48, 3], "latent": 2, "row0": 2, "chunk": 2}]
    rows += [{"key": [1, 48, r], "latent": 8, "row0": 8, "chunk": 8} for r in (3, 4, 5)]
    assert audit.summary(rows)["row0"]["episode_macro"] == 5
    assert audit.summary(rows)["row0"]["sample_mean"] == 6.5


def test_paired_comparison_rejects_different_denominators():
    with pytest.raises(ValueError, match="denominators"):
        audit.comparison([{"key": [0, 48, 3]}], [{"key": [0, 48, 4]}])
