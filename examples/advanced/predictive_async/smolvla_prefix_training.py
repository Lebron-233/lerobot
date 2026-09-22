"""Opt-in prefix-conditioned flow training; no mutation of the frozen policy.

This is a training interface, not a switch for an untrained online policy.
Caller supplies checkpoint-normalized actions and whole-episode row windows.
The existing scalar-time forward is the C=0, full-valid reference. Prefix loss
and terminal/padded-coordinate loss are excluded by a shared valid denominator.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding, make_att_2d_masks

INTERFACE_VERSION = "smolvla-prefix-flow-v1"


@dataclass(frozen=True)
class PrefixFlowBatch:
    noisy_actions: Tensor
    target_velocity: Tensor
    flow_times: Tensor
    valid_steps: Tensor
    clean_prefix: Tensor
    loss_mask: Tensor
    counts: Tensor


def build_prefix_flow(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    counts: Tensor,
    valid_steps: Tensor | None = None,
    *,
    action_dim: int = 7,
) -> PrefixFlowBatch:
    """Create clean t=0 prefix, noised suffix, and terminal-aware loss mask.

    `counts[b]` is committed AFTER the observation at the window origin. It is
    not a history-before-observation prefix and must not exceed valid rows.
    Prefix and target tensors are detached labels; model gradients may still
    flow through prefix embeddings because suffix predictions use that context.
    """
    if actions.ndim != 3 or actions.shape != noise.shape or not actions.is_floating_point():
        raise ValueError("Expected equal floating BxHxD action/noise tensors")
    b, h, d = actions.shape
    if not b or not h or not 1 <= action_dim <= d:
        raise ValueError("Invalid action dimensions")
    if noise.dtype != actions.dtype or noise.device != actions.device:
        raise ValueError("Noise dtype/device mismatch")
    if time.shape != (b,) or time.device != actions.device or not time.is_floating_point():
        raise ValueError("Expected scalar flow time [B] on action device")
    if (
        counts.shape != (b,)
        or counts.device != actions.device
        or counts.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected integer prefix counts [B] on action device")
    if not (torch.isfinite(actions).all() and torch.isfinite(noise).all() and torch.isfinite(time).all()):
        raise ValueError("Nonfinite flow inputs")
    if not ((time >= 0) & (time <= 1)).all():
        raise ValueError("Flow time outside [0,1]")
    if valid_steps is None:
        valid_steps = torch.ones((b, h), dtype=torch.bool, device=actions.device)
    if valid_steps.shape != (b, h) or valid_steps.dtype != torch.bool or valid_steps.device != actions.device:
        raise ValueError("Expected boolean valid_steps [B,H] on action device")
    n = valid_steps.sum(1)
    positions = torch.arange(h, device=actions.device)[None, :]
    if not torch.equal(valid_steps, positions < n[:, None]):
        raise ValueError("Terminal padding must be contiguous on the right")
    if not ((counts >= 0) & (counts < n)).all():
        raise ValueError("Each sample needs at least one valid suffix row")
    actions, noise, time = actions.detach(), noise.detach(), time.detach()
    prefix = positions < counts[:, None]
    mixed = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    x = torch.where(prefix[:, :, None], actions, mixed)
    times = torch.where(prefix, torch.zeros_like(time[:, None]), time[:, None].expand(b, h))
    valid_dims = torch.arange(d, device=actions.device)[None, None, :] < action_dim
    mask = (valid_steps & ~prefix)[:, :, None] & valid_dims
    return PrefixFlowBatch(x, noise - actions, times, valid_steps.clone(), prefix, mask, counts.clone())


def suffix_loss(velocity: Tensor, flow: PrefixFlowBatch) -> Tensor:
    if velocity.shape != flow.target_velocity.shape:
        raise ValueError("Velocity shape mismatch")
    elementwise = functional.mse_loss(flow.target_velocity, velocity, reduction="none")
    if bool((flow.counts == 0).all()) and bool(flow.valid_steps.all()):
        # Preserve the native strided reduction and its gradient exactly for C=0.
        # Flattening a selected copy can change floating-point summation order.
        width = int(flow.loss_mask[0, 0].sum())
        return elementwise[..., :width].mean()
    # Selection prevents excluded padded coordinates contaminating the reduction.
    selected = elementwise.masked_select(flow.loss_mask)
    if not selected.numel() or not torch.isfinite(selected).all():
        raise ValueError("Invalid selected suffix loss")
    return selected.mean()


def embed_suffix_per_action(model, noisy_actions: Tensor, timestep: Tensor):
    """Reuse all original projections; only permit scalar or per-action time.

    Scalar time delegates to the frozen method verbatim. This does not monkey
    patch the model, replace its sampler, or enable its guided RTC processor.
    """
    b, h, _ = noisy_actions.shape
    if timestep.ndim == 1:
        if timestep.shape != (b,):
            raise ValueError("Scalar timestep shape mismatch")
        return model.embed_suffix(noisy_actions, timestep)
    if timestep.shape != (b, h) or h != model.config.chunk_size:
        raise ValueError("Per-action time must have shape [B,chunk_size]")
    action_emb = model.action_in_proj(noisy_actions)
    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        model.vlm_with_expert.expert_hidden_size,
        model.config.min_period,
        model.config.max_period,
        device=action_emb.device,
    ).to(dtype=action_emb.dtype)
    emb = model.action_time_mlp_out(
        functional.silu(model.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2)))
    )
    pad = torch.ones((b, h), dtype=torch.bool, device=emb.device)
    att = torch.ones((b, h), dtype=emb.dtype, device=emb.device)
    return emb, pad, att


def prefix_training_forward(
    model,
    images,
    image_masks,
    language_tokens,
    language_masks,
    state,
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    counts: Tensor,
    valid_steps: Tensor | None = None,
    *,
    action_dim: int = 7,
):
    """Full vision/language/action Transformer forward with suffix-only objective.

    C=0/full-valid matches the original model's elementwise training loss, not
    a promise that a different padding reduction matches the public policy's
    fixed-H mean. Both future training arms MUST use this same reduction.
    """
    flow = build_prefix_flow(actions, noise, time, counts, valid_steps, action_dim=action_dim)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images,
        image_masks,
        language_tokens,
        language_masks,
        state=state,
    )
    suffix_time = time if bool((counts == 0).all()) else flow.flow_times
    suffix_embs, suffix_pad, suffix_att = embed_suffix_per_action(model, flow.noisy_actions, suffix_time)
    suffix_pad = suffix_pad & flow.valid_steps
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    att = torch.cat([prefix_att, suffix_att], dim=1)
    mask = make_att_2d_masks(pad, att)
    positions = torch.cumsum(pad, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=mask,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )
    velocity = model.action_out_proj(suffix_out[:, -model.config.chunk_size :].to(torch.float32))
    if not torch.isfinite(velocity).all():
        raise ValueError("Nonfinite full velocity output")
    return {
        "loss": suffix_loss(velocity, flow),
        "velocity": velocity,
        "elementwise": functional.mse_loss(flow.target_velocity, velocity, reduction="none"),
        "flow": flow,
    }


def episode_action_window(actions: Tensor, start: int, horizon: int = 50):
    """Keep one episode's native row order; never resample by FPS metadata."""
    if actions.ndim != 2 or not 0 <= start < len(actions) or horizon < 1:
        raise ValueError("Invalid episode window")
    n = min(horizon, len(actions) - start)
    output = actions.new_zeros((horizon, actions.shape[1]))
    output[:n] = actions[start : start + n]
    valid = torch.arange(horizon, device=actions.device) < n
    return output, valid
