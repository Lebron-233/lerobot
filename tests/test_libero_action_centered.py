"""F-ACR1 algebra, gradients and fixed decision gate on synthetic CPU tensors."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_action_centered as audit  # noqa: E402
import libero_action_centered as study  # noqa: E402


def test_null_action_cancels_before_base_addition_exactly():
    base = (torch.tensor([1.0, 3.0]),)
    delta = (torch.tensor([1e10, -1e10]),)
    assert torch.equal(study.compose(base, delta, delta, "centered")[0], base[0])
    assert not torch.equal(base[0] + delta[0] - delta[0], base[0])


def test_ordinary_branch_does_not_subtract_zero_action():
    base, action, zero = (torch.tensor([1.0]),), (torch.tensor([2.0]),), (torch.tensor([5.0]),)
    assert study.compose(base, action, zero, "ordinary")[0].item() == 3
    assert study.compose(base, action, zero, "centered")[0].item() == -2


def test_centering_keeps_action_gradient_but_cancels_shared_bias():
    weight = torch.tensor(2.0, requires_grad=True)
    bias = torch.tensor(7.0, requires_grad=True)
    base = (torch.tensor(3.0),)
    output = study.compose(base, (weight * 5 + bias,), (bias,), "centered")[0]
    output.backward()
    assert weight.grad.item() == 5 and bias.grad.item() == 0


def test_unknown_arm_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        study.compose((), (), (), "other")


@pytest.mark.parametrize("failed", ["base", "mismatch", "ordinary", "episodes", "chunk", "initial", None])
def test_candidate_requires_every_frozen_condition(failed):
    metrics = {"centered": {"true": {"chunk": {"episode_macro": 1.0}}},
               "base": {"chunk": {"episode_macro": 2.0}}, "identity": {"chunk": {"episode_macro": 3.0}}}
    contrasts = {n: {"macro_direction": "better", "episode_directions": {"better": 3}} for n in
                 ("base_minus_centered", "mismatch_minus_centered", "ordinary_minus_centered")}
    if failed in ("base", "mismatch", "ordinary"):
        contrasts[f"{failed}_minus_centered"]["macro_direction"] = "tie"
    if failed == "episodes":
        contrasts["base_minus_centered"]["episode_directions"]["better"] = 2
    if failed == "chunk":
        metrics["base"]["chunk"]["episode_macro"] = 0.5
    assert audit.candidate_gate(metrics, contrasts, 0 if failed == "initial" else 36) == (failed is None)


def test_formal_budget_accounts_for_centered_extra_forward():
    assert study.LIMITS["decoder"] == 88 + 2 * (72 + 3 * 16 + 88 + 88 + 85)
    assert study.LIMITS["predictor_forwards"] == (2 + 1) * (72 + 3 * 16 + 88 + 88 + 85)
