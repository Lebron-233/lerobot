"""Opt-in diagnostic sampler matching prefix-flow training, not production RTC.

No source/method replacement and no optimizer. Prefix is a caller-owned BxCx7
normalized snapshot; no target suffix, terminal labels or future observations
are accepted. C=0 uses the original scalar-time denoiser and Euler arithmetic.
"""

import hashlib
from pathlib import Path

import torch
from smolvla_prefix_training import embed_suffix_per_action

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks

SAMPLER_VERSION = "smolvla-prefix-sampling-v1"
CHECKPOINT_VERSION = "smolvla-prefix-projection-pilot-v1"
MODULES = ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj")


def projection_parameters(model):
    return {
        f"{module}.{name}": p for module in MODULES for name, p in getattr(model, module).named_parameters()
    }


def restore_projections(model, values):
    named = projection_parameters(model)
    if set(named) != set(values) or len(named) != 8:
        raise ValueError("Projection key set mismatch")
    for name, p in named.items():
        v = values[name]
        if p.shape != v.shape or p.dtype != v.dtype or not torch.isfinite(v).all():
            raise ValueError("Projection shape/dtype/nonfinite mismatch")
    with torch.no_grad():
        for name, p in named.items():
            p.copy_(values[name].to(p.device))


def load_projection_checkpoint(model, path, expected_sha, *, arm, base_policy, manifest_sha):
    """Load saved absolute projections, never treat them as additive deltas."""
    path = Path(path)
    with path.open("rb") as f:
        observed = hashlib.file_digest(f, "sha256").hexdigest()
    if observed != expected_sha:
        raise ValueError("Checkpoint digest mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        arm not in ("ordinary", "prefix")
        or saved["arm"] != arm
        or saved["interface_version"] != CHECKPOINT_VERSION
        or saved["updates"] != 128
        or saved["base_policy"] != str(base_policy)
        or saved["data_manifest_sha256"] != manifest_sha
        or saved["deployment_ready"]
        or saved["conditioned_sampler_required"] != (arm == "prefix")
    ):
        raise ValueError("Checkpoint provenance/configuration mismatch")
    restore_projections(model, saved["projections"])
    return {"arm": arm, "sha256": observed, "updates": 128, "absolute_projections": True}


def prefix_buffers(noise, prefix, counts):
    """Clamp all internal coordinates of clean rows, with absent 25 dims zero."""
    if noise.ndim != 3 or noise.shape[1:] != (50, 32) or noise.dtype != torch.float32:
        raise ValueError("Expected float32 Bx50x32 noise")
    b = len(noise)
    if not b or not torch.isfinite(noise).all():
        raise ValueError("Invalid noise")
    if counts is None:
        counts = torch.zeros(b, dtype=torch.long, device=noise.device)
    if (
        counts.shape != (b,)
        or counts.dtype not in (torch.int32, torch.int64)
        or counts.device != noise.device
        or not ((counts >= 0) & (counts <= 8)).all()
    ):
        raise ValueError("Prefix counts must be integers in the trained range [0,8]")
    length = int(counts.max())
    if length == 0:
        if prefix is not None:
            raise ValueError("C0 must not contain a prefix payload")
        return None, None
    if (
        prefix is None
        or prefix.shape != (b, length, 7)
        or prefix.dtype != noise.dtype
        or prefix.device != noise.device
        or not torch.isfinite(prefix).all()
    ):
        raise ValueError("Expected finite normalized Bxmax(C)x7 prefix on noise device")
    clean = torch.zeros_like(noise)
    clean[:, :length, :7] = prefix.detach()
    mask = (torch.arange(50, device=noise.device)[None, :] < counts[:, None])[..., None]
    return clean, mask.expand_as(clean)


def conditional_denoise_step(model, prefix_pad, cache, x, timestep):
    if timestep.ndim == 1:
        return model.denoise_step(prefix_pad, cache, x, timestep)
    suffix, suffix_pad, suffix_att = embed_suffix_per_action(model, x, timestep)
    b, h = suffix_pad.shape
    prefix_len = prefix_pad.shape[1]
    prefix_2d = prefix_pad[:, None, :].expand(b, h, prefix_len)
    suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
    attention = torch.cat([prefix_2d, suffix_2d], dim=2)
    positions = prefix_pad.sum(-1)[:, None] + suffix_pad.cumsum(1) - 1
    try:
        embeddings, _ = model.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=cache,
            inputs_embeds=[None, suffix],
            use_cache=model.config.use_cache,
        )
    finally:
        if cache is not None:
            cache.crop(prefix_len)
    return model.action_out_proj(embeddings[1][:, -model.config.chunk_size :].to(torch.float32))


@torch.no_grad()
def sample_prefix_actions(
    model,
    images,
    image_masks,
    language,
    language_masks,
    state,
    noise,
    *,
    prefix=None,
    counts=None,
    trace=None,
):
    """Original ten-step sampling with clean per-action t=0 prefix conditioning.

    Optional trace contains intermediate states for offline arithmetic audit,
    not deployment telemetry. The native controller/Graph are untouched.
    """
    if (
        model.config.chunk_size != 50
        or model.config.num_steps != 10
        or model._rtc_enabled()
        or not model.config.use_cache
    ):
        raise ValueError("Requires frozen 50x10 cached model with guided RTC disabled")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Diagnostic sampler requires frozen parameters and no gradients")
    clean, mask = prefix_buffers(noise, prefix, counts)
    if state.shape != (len(noise), 32) or state.device != noise.device:
        raise ValueError("State/noise batch or device mismatch")
    image_tokens, token_masks = model.encode_image_tokens(images, image_masks)
    emb, pad, att = model.embed_prefix_from_tokens(image_tokens, token_masks, language, language_masks, state)
    attention = make_att_2d_masks(pad, att)
    positions = torch.cumsum(pad, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=attention,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[emb, None],
        use_cache=model.config.use_cache,
    )

    def denoise(x, timestep):
        velocity = conditional_denoise_step(model, pad, cache, x, timestep)
        if trace is not None:
            trace.append(
                {
                    "x": x.detach().cpu().clone(),
                    "time": timestep.detach().cpu().clone(),
                    "velocity": velocity.detach().cpu().clone(),
                }
            )
        return velocity

    value = euler_integrate(denoise, noise, 10, hard_prefix=clean, hard_prefix_mask=mask)
    if value.shape != noise.shape or not torch.isfinite(value).all():
        raise ValueError("Nonfinite or malformed sampled actions")
    if mask is not None and not torch.equal(value[mask], clean[mask]):
        raise ValueError("Clean prefix was not preserved")
    return value
