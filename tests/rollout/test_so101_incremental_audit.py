"""Keep paired-noise grids complete rather than reporting selected cases."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from audit_so101_incremental_vision import check_case_grid  # noqa: E402
from train_so101_incremental_vision import training_batch  # noqa: E402


def test_two_noise_draws_per_complete_query_are_required():
    rows = [{"episode": 6, "query_index": 0, "noise_index": n} for n in (0, 1)]
    check_case_grid(rows, {6: 1})
    with pytest.raises(ValueError, match="denominator"):
        check_case_grid(rows[:1], {6: 1})
    with pytest.raises(ValueError, match="denominator"):
        check_case_grid(rows + rows[:1], {6: 1})


def test_balanced_gradient_batches_never_select_validation_episodes():
    groups = {e: [{"episode": e, "query_index": 0, "noise_index": 0}] for e in range(6)}
    rng = random.Random(3500)
    for _ in range(10):
        batch = training_batch(groups, rng, "balanced4")
        assert len(batch) == len({row["episode"] for row in batch}) == 4
        assert all(row["episode"] < 6 for row in batch)
