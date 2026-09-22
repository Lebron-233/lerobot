"""Small CPU tests for exact zero start, branch selection, gradients and restore."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from smolvla_expert_lora import (  # noqa: E402
    LowRankLinear,
    action_self_attention_targets,
    attach_expert_adapters,
)


def stub():
    layers = nn.ModuleList([nn.Module() for _ in range(4)])
    for layer in layers:
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8).requires_grad_(False)
        layer.self_attn.v_proj = nn.Linear(8, 4).requires_grad_(False)
    wrapper = SimpleNamespace(
        lm_expert=SimpleNamespace(layers=layers),
        attention_mode="cross_attn",
        self_attn_every_n_layers=2,
        get_vlm_model=lambda: SimpleNamespace(text_model=None),
        get_model_layers=lambda models: [[], list(layers)],
    )
    return SimpleNamespace(vlm_with_expert=wrapper)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_start_is_exact_and_base_does_not_update(dtype):
    base = nn.Linear(8, 6, dtype=dtype).requires_grad_(False)
    original = {k: v.clone() for k, v in base.state_dict().items()}
    adapter = LowRankLinear(base, rank=2, alpha=2, seed=41)
    x = torch.randn(2, 3, 8, dtype=dtype)
    assert torch.equal(base(x), adapter(x))
    assert adapter.weight is base.weight
    optimizer = torch.optim.AdamW([adapter.lora_A, adapter.lora_B], lr=1e-3, weight_decay=0)
    adapter(x).float().square().mean().backward()
    assert not adapter.lora_A.grad.count_nonzero()
    assert adapter.lora_B.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    adapter(x).float().square().mean().backward()
    assert adapter.lora_A.grad.abs().sum() > 0
    assert all(torch.equal(v, original[k]) for k, v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())


def test_only_actual_self_branches_selected_and_restored():
    model = stub()
    targets = action_self_attention_targets(model)
    assert [v[-1] for v in targets] == [0, 0, 2, 2]
    ids = [id(v[3]) for v in targets]
    handle = attach_expert_adapters(model, rank=2, alpha=2, seed=42)
    assert len(handle.parameters_by_name()) == 8
    assert isinstance(model.vlm_with_expert.lm_expert.layers[1].self_attn.q_proj, nn.Linear)
    saved = handle.snapshot()
    with torch.no_grad():
        for p in handle.parameters_by_name().values():
            p.add_(1)
    handle.restore(saved)
    assert all(torch.equal(p, saved[k]) for k, p in handle.parameters_by_name().items())
    with pytest.raises(ValueError):
        handle.restore({})
    handle.detach()
    handle.detach()
    assert [id(v[3]) for v in action_self_attention_targets(model)] == ids


def test_noninteger_rank_and_nonfinite_alpha_rejected():
    base = nn.Linear(8, 6).requires_grad_(False)
    for rank, alpha in [(0, 2), (8, 2), (True, 2), (2, float("nan"))]:
        with pytest.raises(ValueError):
            LowRankLinear(base, rank=rank, alpha=alpha, seed=0)


def test_deterministic_initialization_does_not_consume_global_rng():
    base = nn.Linear(8, 6).requires_grad_(False)
    state = torch.random.get_rng_state().clone()
    a = LowRankLinear(base, rank=2, alpha=2, seed=42)
    b = LowRankLinear(base, rank=2, alpha=2, seed=42)
    assert torch.equal(a.lora_A, b.lora_A)
    assert torch.equal(state, torch.random.get_rng_state())


def test_ambiguous_branch_mapping_rejected():
    model = stub()
    layer = model.vlm_with_expert.lm_expert.layers[0]
    model.vlm_with_expert.get_model_layers = lambda models: [[], [layer, layer]]
    with pytest.raises(ValueError, match="both"):
        action_self_attention_targets(model)
