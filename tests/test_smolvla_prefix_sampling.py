"""Tiny cached-attention sampler and provenance tests; no pretrained model/Env."""

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_prefix_sampling as audit  # noqa: E402
from smolvla_prefix_sampling import (  # noqa: E402
    CHECKPOINT_VERSION,
    load_projection_checkpoint,
    prefix_buffers,
    projection_parameters,
    sample_prefix_actions,
)

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching  # noqa: E402
from tests.test_smolvla_prefix_training import TinyModel  # noqa: E402


class Cache:
    def __init__(self, prefix):
        self.prefix, self.length, self.crops = prefix, prefix.shape[1], 0

    def crop(self, length):
        assert length == self.prefix.shape[1]
        self.length = length
        self.crops += 1


class CachedExpert(nn.Module):
    expert_hidden_size = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.caches = []

    def forward(self, *, attention_mask, inputs_embeds, past_key_values, **kwargs):
        prefix, suffix = inputs_embeds
        if suffix is None:
            cache = Cache(prefix)
            self.caches.append(cache)
            return (prefix, None), cache
        cache = past_key_values
        assert cache.length == cache.prefix.shape[1], "Cache was not cropped after preceding step"
        keys = torch.cat([cache.prefix, suffix], dim=1)
        value = functional.scaled_dot_product_attention(
            suffix[:, None], keys[:, None], keys[:, None], attn_mask=attention_mask[:, None], dropout_p=0.0
        )[:, 0]
        cache.length += suffix.shape[1]
        return (None, self.proj(suffix + value)), cache


class SamplingModel(TinyModel):
    sample_actions = VLAFlowMatching.sample_actions
    denoise_step = VLAFlowMatching.denoise_step

    def __init__(self):
        super().__init__()
        self.config.num_steps, self.config.use_cache = 10, True
        self.rtc_processor = None
        self.vlm_with_expert = CachedExpert()
        self.eval().requires_grad_(False)

    def _rtc_enabled(self):
        return False

    def encode_image_tokens(self, images, masks):
        return tuple(images), tuple(masks)

    def embed_prefix_from_tokens(self, images, masks, tokens, lang_masks, state):
        return self.embed_prefix(images, masks, tokens, lang_masks, state=state)


@pytest.fixture
def sample():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77)
        model = SamplingModel()
        noise = torch.randn(2, 50, 32)
        prefix = torch.randn(2, 8, 7)
    inputs = (
        [torch.zeros(2, 2, 8)],
        [torch.ones(2, 2, dtype=torch.bool)],
        torch.ones(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 32),
    )
    return model, inputs, noise, prefix


def test_c0_native_all50x32_exact_and_cache_crop(sample):
    model, inputs, noise, _ = sample
    native = model.sample_actions(*inputs, noise=noise)
    trace = []
    out = sample_prefix_actions(model, *inputs, noise, trace=trace)
    assert torch.equal(native, out) and len(trace) == 10
    assert all(c.crops == 10 and c.length == 2 for c in model.vlm_with_expert.caches)
    assert torch.equal(trace[0]["x"], noise)
    assert all(t["time"].shape == (2,) for t in trace)


@pytest.mark.parametrize("counts", [(1, 1), (3, 3), (8, 8), (3, 8), (0, 8)])
def test_clean_prefix_per_action_time_and_ten_step_equation(sample, counts):
    model, inputs, noise, prefix = sample
    cs = torch.tensor(counts)
    prefix = prefix[:, : max(counts)]
    clean, mask = prefix_buffers(noise, prefix, cs)
    trace = []
    result = sample_prefix_actions(model, *inputs, noise, prefix=prefix, counts=cs, trace=trace)
    assert len(trace) == 10 and torch.equal(result[mask], clean[mask])
    wanted = noise.clone()
    for step, row in enumerate(trace):
        wanted = torch.where(mask, clean, wanted)
        assert torch.equal(row["x"], wanted)
        assert not row["time"][mask[..., 0]].count_nonzero()
        assert row["time"].shape == (2, 50)
        expected_time = torch.full((2, 50), 1.0 + step * (-0.1))
        expected_time[mask[..., 0]] = 0
        assert torch.equal(row["time"], expected_time)
        wanted = row["x"] + (-0.1) * row["velocity"]
        wanted = torch.where(mask, clean, wanted)
    assert torch.equal(result, wanted)
    assert not result[..., 7:][mask[..., 7:]].count_nonzero()
    assert model.vlm_with_expert.caches[-1].crops == 10


def test_prefix_changes_suffix_not_only_copied_coordinates(sample):
    model, inputs, noise, prefix = sample
    c = torch.tensor([3, 3])
    a = sample_prefix_actions(model, *inputs, noise, prefix=prefix[:, :3], counts=c)
    b = sample_prefix_actions(model, *inputs, noise, prefix=prefix[:, :3] + 2, counts=c)
    assert not torch.equal(a[:, 3:, :7], b[:, 3:, :7])
    with pytest.raises(TypeError):
        sample_prefix_actions(model, *inputs, noise, valid_steps=torch.ones(2, 50))


@pytest.mark.parametrize("counts", [(-1, 0), (9, 0), (50, 0)])
def test_invalid_count_rejected(sample, counts):
    _, _, noise, prefix = sample
    with pytest.raises(ValueError, match="range"):
        prefix_buffers(noise, prefix, torch.tensor(counts))


def test_nonfinite_wrong_padding_and_wrong_configuration(sample):
    model, inputs, noise, prefix = sample
    with pytest.raises(ValueError, match="integers"):
        prefix_buffers(noise, prefix, torch.zeros(2))
    with pytest.raises(ValueError, match="C0"):
        prefix_buffers(noise, prefix[:, :0], torch.zeros(2, dtype=torch.long))
    prefix[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prefix_buffers(noise, prefix, torch.tensor([8, 8]))
    model.config.num_steps = 5
    with pytest.raises(ValueError, match="50x10"):
        sample_prefix_actions(model, *inputs, noise)
    model.config.num_steps = 10
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        sample_prefix_actions(model, *inputs, noise)


def test_absolute_checkpoint_reload_and_wrong_identity_fail_before_mutation(sample, tmp_path):
    model = sample[0]
    before = {k: p.detach().clone() for k, p in projection_parameters(model).items()}
    values = {k: v + 0.02 for k, v in before.items()}
    checkpoint = {
        "interface_version": CHECKPOINT_VERSION,
        "arm": "prefix",
        "base_policy": "/pinned/base",
        "data_manifest_sha256": "manifest",
        "updates": 128,
        "deployment_ready": False,
        "conditioned_sampler_required": True,
        "projections": values,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for arm, expected in [("ordinary", digest), ("prefix", "0" * 64)]:
        with pytest.raises(ValueError):
            load_projection_checkpoint(
                model, path, expected, arm=arm, base_policy="/pinned/base", manifest_sha="manifest"
            )
        assert all(torch.equal(p, before[k]) for k, p in projection_parameters(model).items())
    load_projection_checkpoint(
        model, path, digest, arm="prefix", base_policy="/pinned/base", manifest_sha="manifest"
    )
    assert all(torch.equal(p, values[k]) for k, p in projection_parameters(model).items())


def test_cpu_trace_audit_rejects_time_prefix_and_output_tampering(sample):
    model, inputs, noise, prefix = sample
    inputs = ([v[:1] for v in inputs[0]], [v[:1] for v in inputs[1]], *(v[:1] for v in inputs[2:]))
    trace = []
    value = sample_prefix_actions(
        model, *inputs, noise[:1], prefix=prefix[:1, :3], counts=torch.tensor([3]), trace=trace
    )
    row = {
        "condition_count": 3,
        "condition_prefix": prefix[:1, :3],
        "mode": "correct_prefix",
        "trace": trace,
        "full": value,
    }
    ref = {"noise": noise[:1]}
    assert audit.inspect_trace(row, ref) == 10
    for field in ("time", "x"):
        bad = copy.deepcopy(row)
        bad["trace"][3][field].flatten()[0] += 1
        with pytest.raises((ValueError, AssertionError)):
            audit.inspect_trace(bad, ref)
    bad = copy.deepcopy(row)
    bad["full"][0, 3, 0] += 0.1
    with pytest.raises(AssertionError):
        audit.inspect_trace(bad, ref)


def test_scores_exclude_copied_prefix_and_terminal_padding():
    target = torch.zeros(1, 50, 32)
    pred = target.clone()
    pred[:, :3, :7] = 100
    pred[:, 3:8, :7] = 2
    pred[:, 8:, :7] = 1000
    valid = torch.arange(50)[None] < 8
    m = audit.metrics(pred, target, valid, 3)
    assert m["suffix"] == m["first"] == m["short"] == 4
    assert m["block"] > 1000


def test_first_action_regression_cannot_hide_behind_chunk_improvement():
    scored = []
    for label in audit.r.LABELS:
        for mode in audit.r.MODES:
            for case in range(16):
                v = 0.8 if (label, mode) == ("prefix", "correct_prefix") else 1.0
                m = dict.fromkeys(("block", "suffix", "first", "short"), v)
                scored.append(
                    {
                        "checkpoint": label,
                        "mode": mode,
                        "case": case,
                        "trajectory_id": str(case // 4),
                        "task": case // 4,
                        "anchor": 20,
                        "C": 3,
                        "normalized": m.copy(),
                        "command": m.copy(),
                    }
                )
    assert audit.reduce_rows(scored)["decoded_action_followup_supported"]
    for row in scored:
        if (row["checkpoint"], row["mode"]) == ("prefix", "correct_prefix"):
            row["normalized"]["first"] = 1.1
    assert not audit.reduce_rows(scored)["decoded_action_followup_supported"]
