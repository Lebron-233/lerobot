"""Analytic CPU regressions for the frozen-denoiser RTC input Jacobian.

No scientific weights, scientific samples, GPU, or robot environment.
"""

import pytest
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def processor():
    return RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.ONES,
                                  execution_horizon=1, max_guidance_weight=10.0))


@pytest.mark.parametrize("squeezed", [False, True])
@pytest.mark.parametrize("matrix", [[[2.0, 0.0], [0.0, 2.0]], [[0.4, 0.7], [-0.2, 0.1]]])
def test_frozen_linear_denoiser_matches_analytic_vjp(matrix, squeezed):
    # x_clean=x-t*v(x); correction=(I-t*A)^T*((prefix-x_clean)*weights).
    a = torch.tensor(matrix, dtype=torch.float64)
    x = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)
    target = torch.tensor([[[3.0, -1.0]]], dtype=torch.float64)
    if squeezed:
        x, target = x[0], target[0]
    t = 0.5
    velocity = x @ a.T
    err = target - (x - t * velocity)
    expected = velocity - 2.0 * (err @ (torch.eye(2, dtype=x.dtype) - t * a))
    with torch.no_grad():
        actual = processor().denoise_step(x, target, 1, t, lambda v: v @ a.T)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not x.requires_grad and not a.requires_grad


def test_no_prefix_is_exact_base_and_no_input_grad_mutation():
    x = torch.tensor([[[0.5, -0.2]]], dtype=torch.float64)
    with torch.no_grad():
        expected = torch.sin(x)
        actual = processor().denoise_step(x, None, 0, 0.5, torch.sin)
    assert torch.equal(actual, expected)
    assert not x.requires_grad


def test_nonlinear_frozen_denoiser_uses_derivative():
    x = torch.tensor([[[0.2, -0.7]]], dtype=torch.float64)
    prefix = torch.tensor([[[0.9, 0.1]]], dtype=torch.float64)
    t = 0.5
    velocity = x.sin()
    expected = velocity - 2.0 * (1 - t * x.cos()) * (prefix - (x - t * velocity))
    with torch.no_grad():
        actual = processor().denoise_step(x, prefix, 1, t, torch.sin)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_frozen_parameter_gradient_is_not_accumulated():
    layer = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64).requires_grad_(False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2, dtype=torch.float64) * 2)
        x = torch.ones(1, 1, 2, dtype=torch.float64)
        actual = processor().denoise_step(x, torch.zeros_like(x), 1, 0.5, layer)
    torch.testing.assert_close(actual, 2 * x, rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in layer.parameters())


def test_padding_is_not_an_observed_zero_action_target():
    x = torch.ones(1, 3, 32, dtype=torch.float64)
    prefix = torch.zeros(1, 1, 7, dtype=torch.float64)
    with torch.no_grad():
        actual = processor().denoise_step(x, prefix, 1, 0.5, torch.zeros_like)
    expected = torch.zeros_like(x)
    expected[:, 0, :7] = 2.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_horizon_ignores_all_prefix_values():
    x = torch.ones(1, 3, 7, dtype=torch.float64)
    with torch.no_grad():
        actual = processor().denoise_step(x, x * 4, 0, 0.5, torch.sin, execution_horizon=0)
    torch.testing.assert_close(actual, x.sin(), rtol=0, atol=0)
