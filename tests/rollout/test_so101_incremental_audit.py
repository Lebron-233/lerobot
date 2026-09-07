"""Keep paired-noise grids complete rather than reporting selected cases."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from audit_so101_incremental_vision import check_case_grid  # noqa: E402


def test_two_noise_draws_per_complete_query_are_required():
    rows = [{"episode": 6, "query_index": 0, "noise_index": n} for n in (0, 1)]
    check_case_grid(rows, {6: 1})
    with pytest.raises(ValueError, match="denominator"):
        check_case_grid(rows[:1], {6: 1})
    with pytest.raises(ValueError, match="denominator"):
        check_case_grid(rows + rows[:1], {6: 1})
