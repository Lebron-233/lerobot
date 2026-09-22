"""Opt-in rank-8 adaptation of ACTUAL expert action-self-attention Q/V only.

No PEFT dependency or production configuration change. Original Linear objects
are restored exactly on detach. Float32 low-rank branch, frozen base dtype, no
dropout, no merge. Zero-B initialization is an exact functional identity.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

ADAPTER_VERSION = "smolvla-self-attn-qv-lora-v1"
RANK = 8
ALPHA = 8.0
SEED = 2026092201


class LowRankLinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, seed: int):
        super().__init__()
        if (
            not isinstance(base, nn.Linear)
            or isinstance(rank, bool)
            or not 1 <= rank <= min(base.weight.shape)
        ):
            raise ValueError("Expected Linear and a feasible positive rank")
        if not math.isfinite(alpha) or alpha <= 0 or any(p.requires_grad for p in base.parameters()):
            raise ValueError("Base must be frozen and alpha positive")
        self.base = base
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / self.rank
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        a.uniform_(-1 / math.sqrt(base.in_features), 1 / math.sqrt(base.in_features), generator=generator)
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    @property
    def weight(self):
        # Existing expert dispatch queries this dtype before its Linear call.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base = self.base(x)
        update = functional.linear(functional.linear(x.float(), self.lora_A), self.lora_B)
        return base + (update * self.scaling).to(base.dtype)


def action_self_attention_targets(model):
    """Resolve execution branches, not just modules named self_attn.

    The same self_attn member supplies Q/V in CROSS layers too. Select by the
    wrapper's actual global-layer dispatch and expert mapping; ambiguous reuse
    across both branch kinds is rejected rather than silently adapting cross.
    """
    wrapper = model.vlm_with_expert
    models = [wrapper.get_vlm_model().text_model, wrapper.lm_expert]
    mapped = wrapper.get_model_layers(models)[1]
    names = {id(layer): i for i, layer in enumerate(wrapper.lm_expert.layers)}
    roles = {}
    for global_index, layer in enumerate(mapped):
        if layer is None:
            continue
        is_self = "cross" not in wrapper.attention_mode or (
            wrapper.self_attn_every_n_layers > 0 and global_index % wrapper.self_attn_every_n_layers == 0
        )
        roles.setdefault(id(layer), set()).add(is_self)
    result = []
    for global_index, layer in enumerate(mapped):
        if layer is None or roles[id(layer)] == {False}:
            continue
        if roles[id(layer)] != {True}:
            raise ValueError("Expert layer reused in both self and cross attention")
        local = names[id(layer)]
        for name in ("q_proj", "v_proj"):
            module = getattr(layer.self_attn, name)
            if not isinstance(module, nn.Linear):
                raise ValueError("Target is already adapted or is not a Linear")
            path = f"vlm_with_expert.lm_expert.layers.{local}.self_attn.{name}"
            if path not in {v[0] for v in result}:
                result.append((path, layer.self_attn, name, module, global_index))
    if not result:
        raise ValueError("No action self-attention target found")
    return result


@dataclass
class ExpertAdapters:
    slots: list
    active: bool = True

    def parameters_by_name(self):
        if not self.active:
            raise ValueError("Adapter already detached")
        return {
            f"{path}.{key}": getattr(adapter, key)
            for path, _parent, _name, _base, adapter, _index in self.slots
            for key in ("lora_A", "lora_B")
        }

    def metadata(self):
        return [
            {
                "path": path,
                "global_layer": index,
                "in_features": base.in_features,
                "out_features": base.out_features,
                "base_dtype": str(base.weight.dtype),
                "rank": adapter.rank,
                "alpha": adapter.alpha,
                "adapter_dtype": "torch.float32",
            }
            for path, _parent, _name, base, adapter, index in self.slots
        ]

    def snapshot(self):
        return {k: p.detach().cpu().clone() for k, p in self.parameters_by_name().items()}

    def restore(self, state):
        named = self.parameters_by_name()
        if set(state) != set(named):
            raise ValueError("Adapter keys mismatch")
        for k, p in named.items():
            v = state[k]
            if p.shape != v.shape or p.dtype != v.dtype or not torch.isfinite(v).all():
                raise ValueError("Adapter shape/dtype/finite mismatch")
        with torch.no_grad():
            for k, p in named.items():
                p.copy_(state[k].to(p.device))
                p.grad = None

    def detach(self):
        if not self.active:
            return
        for _path, parent, name, base, adapter, _index in self.slots:
            if getattr(parent, name) is not adapter:
                raise ValueError("Adapter ownership changed")
            setattr(parent, name, base)
        self.active = False


def attach_expert_adapters(model, *, rank=RANK, alpha=ALPHA, seed=SEED):
    targets = action_self_attention_targets(model)
    if any(p.requires_grad for _, _, _, base, _ in targets for p in base.parameters()):
        raise ValueError("Target base parameters must remain frozen")
    slots = []
    try:
        for i, (path, parent, name, base, index) in enumerate(targets):
            adapter = LowRankLinear(base, rank=rank, alpha=alpha, seed=seed + i)
            setattr(parent, name, adapter)
            slots.append((path, parent, name, base, adapter, index))
    except BaseException:
        ExpertAdapters(slots).detach()
        raise
    return ExpertAdapters(slots)
