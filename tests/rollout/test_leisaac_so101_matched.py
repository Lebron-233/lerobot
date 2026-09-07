"""New candidate coordinates and processor boundary, not task outcomes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from leisaac_so101_contract import action_to_radians
from leisaac_so101_matched import (
    MotorStatePreprocessor,
    PhysicalActionPostprocessor,
    motor_to_physical,
    physical_to_motor,
)


@pytest.mark.parametrize("shape", [(6,), (1, 6), (1, 50, 6)])
def test_motor_units_match_official_affine_formula_without_clipping(shape):
    motor = torch.tensor([90.0, -100.0, 100.0, 0.0, 50.0, 75.0]).expand(shape)
    physical = motor_to_physical(motor)
    assert physical.reshape(-1, 6)[0].tolist() == pytest.approx([99, -100, 90, 0, 80, 75])
    torch.testing.assert_close(physical_to_motor(physical), motor)
    assert motor.reshape(-1, 6)[0, 0] == 90  # no input mutation


def test_candidate_postprocessing_remains_separate_from_policy_space():
    class SavedPost:
        steps = ()

        def __call__(self, x):
            return x * 10 + 20

        def reset(self):
            self.reset_called = True

    normalized = torch.zeros(1, 6)
    saved = SavedPost()
    post = PhysicalActionPostprocessor(saved)
    physical = post(normalized)
    torch.testing.assert_close(physical_to_motor(physical), torch.full_like(normalized, 20))
    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
    post.reset()
    assert saved.reset_called


def test_native_state_coordinates_precede_saved_mean_std_normalizer():
    seen = []

    class SavedPre:
        steps = ()

        def __call__(self, batch):
            seen.append(batch["observation.state"].clone())
            return {**batch, "observation.state": (batch["observation.state"] - 3) / 2}

    pre = MotorStatePreprocessor(SavedPre())
    raw = torch.tensor([[99.0, -100.0, 90.0, 0.0, 80.0, 75.0]])
    result = pre({"observation.state": raw})
    torch.testing.assert_close(seen[0], torch.tensor([[90.0, -100.0, 100.0, 0.0, 50.0, 75.0]]))
    torch.testing.assert_close(result["observation.state"], (seen[0] - 3) / 2)
    assert raw[0, 0] == 99


def test_infeasible_motor_target_is_not_hidden_by_new_mapping():
    physical = motor_to_physical(torch.tensor([0.0, 105, 0, 0, 0, 50]))
    with pytest.raises(ValueError, match="Out-of-range shoulder_lift"):
        action_to_radians(physical.tolist())
    assert physical_to_motor(physical)[1] > 100
