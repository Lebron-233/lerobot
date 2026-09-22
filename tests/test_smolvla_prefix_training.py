"""CPU unit/integration tests; tiny Transformer only, no pretrained weights."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from smolvla_prefix_training import (  # noqa: E402
    build_prefix_flow,
    embed_suffix_per_action,
    episode_action_window,
    prefix_training_forward,
    suffix_loss,
)

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching  # noqa: E402


class TinyExpert(nn.Module):
    expert_hidden_size = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, *, attention_mask, inputs_embeds, **kwargs):
        prefix, suffix = inputs_embeds
        x = torch.cat([prefix, suffix], dim=1)
        y = functional.scaled_dot_product_attention(
            x[:, None], x[:, None], x[:, None], attn_mask=attention_mask[:, None], dropout_p=0.0
        )[:, 0]
        y = self.proj(x + y)
        return (y[:, : prefix.shape[1]], y[:, prefix.shape[1] :]), None


class TinyModel(nn.Module):
    embed_suffix = VLAFlowMatching.embed_suffix
    forward = VLAFlowMatching.forward

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(min_period=0.004, max_period=4.0, chunk_size=50)
        self.action_in_proj = nn.Linear(32, 8)
        self.action_time_mlp_in = nn.Linear(16, 8)
        self.action_time_mlp_out = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 32)
        self.vlm_with_expert = TinyExpert()
        self.context = nn.Parameter(torch.randn(1, 2, 8))

    def embed_prefix(self, images, masks, tokens, lang_masks, *, state):
        b = len(state)
        return self.context.expand(b, -1, -1), torch.ones(b, 2, dtype=torch.bool), torch.zeros(b, 2)


@pytest.fixture
def inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = TinyModel()
        actions, noise = torch.randn(2, 50, 32), torch.randn(2, 50, 32)
    actions[..., 7:] = 0
    return model, actions, noise, torch.tensor([0.25, 0.75])


def run(model, actions, noise, time, c, valid=None):
    return prefix_training_forward(
        model, None, None, None, None, torch.zeros(2, 32), actions, noise, time, c, valid
    )


def test_scalar_reference_and_uniform_per_action_embeddings_are_exact(inputs):
    m, a, _, t = inputs
    expected = m.embed_suffix(a, t)
    for times in (t, t[:, None].expand(2, 50)):
        actual = embed_suffix_per_action(m, a, times)
        assert all(torch.equal(x, y) for x, y in zip(expected, actual, strict=True))


def test_c0_full_original_training_loss_and_parameter_gradients_exact(inputs):
    m, a, n, t = inputs
    expected = m(None, None, None, None, torch.zeros(2, 32), a, n, t)
    grads = torch.autograd.grad(expected[..., :7].mean(), tuple(m.parameters()))
    got = run(m, a, n, t, torch.tensor([0, 0]))
    assert torch.equal(expected, got["elementwise"])
    assert torch.equal(expected[..., :7].mean(), got["loss"])
    actual = torch.autograd.grad(got["loss"], tuple(m.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(grads, actual, strict=True))


@pytest.mark.parametrize("c", [0, 1, 3, 8, 49])
def test_clean_prefix_time_and_valid_denominator(inputs, c):
    _, a, n, t = inputs
    f = build_prefix_flow(a, n, t, torch.tensor([c, c]))
    assert torch.equal(f.noisy_actions[:, :c], a[:, :c])
    assert not f.flow_times[:, :c].count_nonzero()
    assert torch.equal(f.flow_times[:, c:], t[:, None].expand(2, 50 - c))
    assert f.loss_mask.sum() == 2 * (50 - c) * 7
    v = torch.zeros_like(a, requires_grad=True)
    g = torch.autograd.grad(suffix_loss(v, f), v)[0]
    assert not g[~f.loss_mask].count_nonzero()
    assert g[f.loss_mask].abs().sum() > 0


def test_mixed_prefixes_terminal_padding_and_model_backward(inputs):
    m, a, n, t = inputs
    valid = torch.arange(50)[None] < torch.tensor([8, 12])[:, None]
    c = torch.tensor([3, 8])
    got = run(m, a, n, t, c, valid)
    assert got["flow"].loss_mask.sum() == (5 + 4) * 7
    grads = torch.autograd.grad(got["loss"], tuple(m.parameters()))
    assert all(torch.isfinite(x).all() for x in grads)
    assert all(x.abs().sum() > 0 for x in grads)
    # Padded future rows are not valid keys and must not affect valid predictions.
    other = a.clone()
    other[~valid] = 1000
    changed = run(m, other, n, t, c, valid)
    assert torch.equal(got["velocity"][valid], changed["velocity"][valid])
    assert torch.equal(got["loss"], changed["loss"])


def test_prefix_is_context_not_a_direct_loss_target(inputs):
    m, a, n, t = inputs
    c = torch.tensor([3, 8])
    out = run(m, a, n, t, c)
    other = a.clone()
    other[:, 0, :7] += 1
    got = run(m, other, n, t, c)
    assert not torch.equal(out["velocity"][:, 8:, :7], got["velocity"][:, 8:, :7])
    assert not out["flow"].loss_mask[:, :3].any()


@pytest.mark.parametrize("bad", [-1, 50, 51])
def test_invalid_counts_rejected(inputs, bad):
    _, a, n, t = inputs
    with pytest.raises(ValueError, match="suffix"):
        build_prefix_flow(a, n, t, torch.tensor([bad, 3]))


def test_empty_suffix_holes_and_nonfinite_rejected(inputs):
    _, a, n, t = inputs
    valid = torch.arange(50)[None].expand(2, 50) < 3
    with pytest.raises(ValueError, match="suffix"):
        build_prefix_flow(a, n, t, torch.tensor([3, 3]), valid)
    valid = torch.ones(2, 50, dtype=torch.bool)
    valid[0, 2] = False
    with pytest.raises(ValueError, match="contiguous"):
        build_prefix_flow(a, n, t, torch.tensor([0, 0]), valid)
    a[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        build_prefix_flow(a, n, t, torch.tensor([0, 0]))


def test_counts_must_be_integers_and_no_label_gradients(inputs):
    _, a, n, t = inputs
    with pytest.raises(ValueError, match="integer"):
        build_prefix_flow(a, n, t, torch.zeros(2))
    f = build_prefix_flow(a.requires_grad_(), n.requires_grad_(), t, torch.tensor([3, 8]))
    assert not f.noisy_actions.requires_grad and not f.target_velocity.requires_grad


def test_episode_window_does_not_cross_terminal():
    a = torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7)
    out, valid = episode_action_window(a, 6)
    assert torch.equal(out[:2], a[6:]) and not out[2:].count_nonzero()
    assert valid.sum() == 2
    with pytest.raises(ValueError):
        episode_action_window(a, 8)


def test_one_tiny_synthetic_optimizer_step_is_finite(inputs):
    m, a, n, t = inputs
    before = {k: p.detach().clone() for k, p in m.named_parameters()}
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    run(m, a, n, t, torch.tensor([3, 8]))["loss"].backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert any(not torch.equal(p, before[k]) for k, p in m.named_parameters())
    assert not torch.cuda.is_initialized()
