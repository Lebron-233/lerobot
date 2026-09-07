"""No noise schedule or subgoal may be confused with native task success."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from run_so101_noise_reference import outcomes  # noqa: E402
from run_so101_privileged_reference import generate  # noqa: E402
from so101_noise_coupling import ActionIndexedNoise  # noqa: E402


def test_overlap_and_bootstrap_are_exact_but_windows_are_not_mutable_aliases():
    schedule = ActionIndexedNoise(3710, device="cpu", limit=100)
    expected = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(371000000))
    assert torch.equal(schedule.window(0), expected)
    assert torch.equal(schedule(20, 27)[:, 27:], schedule(47, 54)[:, :23])
    schedule.window(0).zero_()
    assert torch.equal(schedule.window(0), expected)
    with pytest.raises(ValueError):
        schedule(20, 20)
    with pytest.raises(ValueError):
        schedule.window(101)


def test_exact_noise_reaches_frozen_decoder_without_global_rng_draws():
    seen = []
    policy = SimpleNamespace(
        predict_action_chunk=lambda batch, **kwargs: seen.append(kwargs["noise"]) or torch.zeros(1, 50, 6)
    )
    noise = torch.randn(1, 50, 32)
    before = torch.random.get_rng_state().clone()
    generate(policy, lambda x: x, lambda x: x, {}, (), (), torch.zeros(1, 32), 42, explicit_noise=noise)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(seen[0], noise) and seen[0].data_ptr() != noise.data_ptr()


def test_subgoal_is_not_a_native_three_orange_success_and_later_failure_is_retained():
    witness = {
        "native_success": False,
        "plate_position": [0, 0, 0],
        "oranges": {"one": {"position": [0, 0, 0], "linear_velocity": [0, 0, 0]}},
    }
    ticks = [
        {"tick": i, "dispatch": "completed", "terminated": False, "task_transition_after_action": witness}
        for i in range(20)
    ]
    result = outcomes(ticks, {"status": "terminal", "native_success": False})
    assert result["subgoal_success"] and not result["native_success"]
    assert result["native_time_s"] == 120 and result["first_occupancy_observed_step"] == 9
    failure = outcomes(ticks, {"status": "technical_failure"})
    assert not failure["subgoal_success"] and failure["subgoal_time_s"] == 120
    assert failure["first_occupancy_observed_step"] == 9
