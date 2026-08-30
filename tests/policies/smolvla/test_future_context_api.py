#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for SmolVLA's future-context token override API.

The tiny VLM below exercises the real ``SmolVLAPolicy`` and
``VLAFlowMatching`` call chain without loading a checkpoint or accessing the
Hub.  Its prefix cache deliberately influences denoising so the tests can
observe token and state overrides in the returned action chunk.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

_HIDDEN_DIM = 8
_EXPERT_DIM = 6
_IMAGE_TOKEN_COUNT = 4


class _TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, _HIDDEN_DIM)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _TinyConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, _HIDDEN_DIM, bias=False)

    def forward(self, image_patches: Tensor) -> Tensor:
        return self.projection(image_patches)


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = _TinyTextModel()
        self.connector = _TinyConnector()
        self.vision_model = nn.Linear(1, 1, bias=False)
        self.image_seq_len = _IMAGE_TOKEN_COUNT


class _TinyVLMShell(nn.Module):
    def __init__(self, backbone: _TinyBackbone) -> None:
        super().__init__()
        self.model = backbone

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class _TinyCache:
    def __init__(self, prefix_context: Tensor) -> None:
        self.prefix_context = prefix_context

    def crop(self, _length: int) -> None:
        """Match the DynamicCache API used between denoising steps."""


class _TinyVLMWithExpert(nn.Module):
    """Small deterministic stand-in for ``SmolVLMWithExpertModel``."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        backbone = _TinyBackbone()
        self.vlm = _TinyVLMShell(backbone)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_HIDDEN_DIM),
            vision_config=SimpleNamespace(image_size=4, patch_size=1),
            scale_factor=2,
        )
        self.processor = SimpleNamespace(
            image_seq_len=_IMAGE_TOKEN_COUNT,
            tokenizer=SimpleNamespace(fake_image_token_id=1, global_image_token_id=2),
        )
        self.expert_hidden_size = _EXPERT_DIM
        self.context_projection = nn.Linear(_HIDDEN_DIM, _EXPERT_DIM, bias=False)
        self.embed_image_calls = 0

    def get_vlm_model(self) -> _TinyBackbone:
        return self.vlm.model

    def set_requires_grad(self) -> None:
        pass

    def _image_features(self, image: Tensor) -> Tensor:
        patches = F.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        return self.get_vlm_model().connector(patches)

    def embed_image(self, image: Tensor) -> Tensor:
        self.embed_image_calls += 1
        return self._image_features(image)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward(
        self,
        *,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: _TinyCache | None,
        inputs_embeds: list[Tensor | None],
        use_cache: bool,
    ) -> tuple[tuple[Tensor | None, Tensor | None], _TinyCache | None]:
        del attention_mask, position_ids, use_cache
        prefix_embs, suffix_embs = inputs_embeds

        if prefix_embs is not None:
            prefix_context = prefix_embs.mean(dim=1)
            cache = _TinyCache(prefix_context)
            if suffix_embs is None:
                return (prefix_embs, None), cache
            context = self.context_projection(prefix_context)[:, None, :]
            return (prefix_embs, suffix_embs + context), cache

        assert suffix_embs is not None
        assert past_key_values is not None
        context = self.context_projection(past_key_values.prefix_context)[:, None, :]
        return (None, suffix_embs + context), past_key_values


@pytest.fixture
def tiny_policy(monkeypatch) -> SmolVLAPolicy:
    from lerobot.policies.smolvla import modeling_smolvla

    monkeypatch.setattr(modeling_smolvla, "SmolVLMWithExpertModel", _TinyVLMWithExpert)
    torch.manual_seed(0)
    config = SmolVLAConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        num_steps=2,
        max_state_dim=4,
        max_action_dim=3,
        resize_imgs_with_padding=(4, 4),
        use_cache=True,
        load_vlm_weights=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
    )
    return modeling_smolvla.SmolVLAPolicy(config).eval()


def _make_batch(batch_size: int = 2, *, include_images: bool = True) -> dict[str, Tensor]:
    state = torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 2) / 10
    batch = {
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: torch.tensor([[3, 4, 5]] * batch_size, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 3, dtype=torch.bool),
    }
    if include_images:
        batch["observation.images.front"] = torch.full((batch_size, 3, 4, 4), 0.25)
        batch["observation.images.wrist"] = torch.full((batch_size, 3, 4, 4), 0.75)
    return batch


def _clone_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def _replace(values: tuple[Tensor, ...], index: int, value: Tensor) -> tuple[Tensor, ...]:
    updated = list(values)
    updated[index] = value
    return tuple(updated)


def _encode_batch_images(policy, batch: dict[str, Tensor]) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    images, image_masks = policy.prepare_images(batch)
    return policy.model.encode_image_tokens(images, image_masks)


def test_encode_image_tokens_applies_scaling_once_and_preserves_camera_order(tiny_policy) -> None:
    batch = _make_batch()
    images, image_masks = tiny_policy.prepare_images(batch)

    image_tokens, token_masks = tiny_policy.model.encode_image_tokens(images, image_masks)

    assert len(image_tokens) == len(token_masks) == 2
    assert tiny_policy.model.vlm_with_expert.embed_image_calls == 2
    scale = torch.tensor(_HIDDEN_DIM**0.5, dtype=image_tokens[0].dtype)
    for image, image_mask, tokens, masks in zip(images, image_masks, image_tokens, token_masks, strict=True):
        expected = tiny_policy.model.vlm_with_expert._image_features(image) * scale
        torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
        torch.testing.assert_close(masks, image_mask[:, None].expand_as(masks))
        assert tokens.shape == (2, _IMAGE_TOKEN_COUNT, _HIDDEN_DIM)
        assert masks.shape == (2, _IMAGE_TOKEN_COUNT)
        assert masks.dtype is torch.bool

    assert not torch.equal(image_tokens[0], image_tokens[1])


@pytest.mark.parametrize("add_image_special_tokens", [False, True])
def test_legacy_and_token_prefix_paths_are_identical(tiny_policy, add_image_special_tokens: bool) -> None:
    batch = _make_batch()
    images, image_masks = tiny_policy.prepare_images(batch)
    state = tiny_policy.prepare_state(batch)
    language_tokens = batch[OBS_LANGUAGE_TOKENS]
    language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    tiny_policy.model.add_image_special_tokens = add_image_special_tokens
    tiny_policy.model.prefix_length = 20

    legacy_prefix = tiny_policy.model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state
    )
    image_tokens, token_masks = tiny_policy.model.encode_image_tokens(images, image_masks)
    token_prefix = tiny_policy.model.embed_prefix_from_tokens(
        image_tokens, token_masks, language_tokens, language_masks, state
    )

    for legacy, overridden in zip(legacy_prefix, token_prefix, strict=True):
        torch.testing.assert_close(legacy, overridden, rtol=0, atol=0)


def test_override_path_never_calls_embed_image_and_does_not_require_rgb(tiny_policy) -> None:
    image_tokens, token_masks = _encode_batch_images(tiny_policy, _make_batch())
    tiny_policy.model.vlm_with_expert.embed_image_calls = 0
    batch_without_images = _make_batch(include_images=False)
    noise = torch.randn(2, tiny_policy.config.chunk_size, tiny_policy.config.max_action_dim)

    actions = tiny_policy.predict_action_chunk(
        batch_without_images,
        noise=noise,
        future_image_tokens=image_tokens,
        future_image_token_masks=token_masks,
    )

    assert actions.shape == (2, tiny_policy.config.chunk_size, 3)
    assert tiny_policy.model.vlm_with_expert.embed_image_calls == 0


def test_image_and_token_paths_produce_same_actions_with_same_noise(tiny_policy) -> None:
    batch = _make_batch()
    image_tokens, token_masks = _encode_batch_images(tiny_policy, batch)
    noise = torch.randn(2, tiny_policy.config.chunk_size, tiny_policy.config.max_action_dim)

    image_actions = tiny_policy.predict_action_chunk(_clone_batch(batch), noise=noise.clone())
    token_actions = tiny_policy.predict_action_chunk(
        _make_batch(include_images=False),
        noise=noise.clone(),
        future_image_tokens=image_tokens,
        future_image_token_masks=token_masks,
    )

    torch.testing.assert_close(image_actions, token_actions, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_masks",
        "missing_tokens",
        "camera_count",
        "token_rank",
        "mask_rank",
        "batch_size",
        "token_count",
        "hidden_dim",
        "token_dtype",
        "mask_dtype",
        "device",
    ],
)
def test_invalid_future_image_overrides_raise_clear_error(tiny_policy, invalid_case: str) -> None:
    image_tokens, token_masks = _encode_batch_images(tiny_policy, _make_batch())
    kwargs: dict[str, object] = {
        "future_image_tokens": image_tokens,
        "future_image_token_masks": token_masks,
    }

    if invalid_case == "missing_masks":
        kwargs.pop("future_image_token_masks")
    elif invalid_case == "missing_tokens":
        kwargs.pop("future_image_tokens")
    elif invalid_case == "camera_count":
        kwargs["future_image_tokens"] = image_tokens[:-1]
        kwargs["future_image_token_masks"] = token_masks[:-1]
    elif invalid_case == "token_rank":
        kwargs["future_image_tokens"] = _replace(image_tokens, 0, image_tokens[0][:, 0])
    elif invalid_case == "mask_rank":
        kwargs["future_image_token_masks"] = _replace(token_masks, 0, token_masks[0][:, 0])
    elif invalid_case == "batch_size":
        kwargs["future_image_tokens"] = _replace(image_tokens, 0, image_tokens[0][:1])
        kwargs["future_image_token_masks"] = _replace(token_masks, 0, token_masks[0][:1])
    elif invalid_case == "token_count":
        kwargs["future_image_tokens"] = _replace(image_tokens, 0, image_tokens[0][:, :-1])
        kwargs["future_image_token_masks"] = _replace(token_masks, 0, token_masks[0][:, :-1])
    elif invalid_case == "hidden_dim":
        kwargs["future_image_tokens"] = _replace(image_tokens, 0, image_tokens[0][..., :-1])
    elif invalid_case == "token_dtype":
        kwargs["future_image_tokens"] = _replace(image_tokens, 0, image_tokens[0].to(torch.float64))
    elif invalid_case == "mask_dtype":
        kwargs["future_image_token_masks"] = _replace(token_masks, 0, token_masks[0].to(torch.float32))
    elif invalid_case == "device":
        kwargs["future_image_tokens"] = _replace(image_tokens, 0, image_tokens[0].to("meta"))
        kwargs["future_image_token_masks"] = _replace(token_masks, 0, token_masks[0].to("meta"))

    with pytest.raises((TypeError, ValueError), match="future_image"):
        tiny_policy.predict_action_chunk(_make_batch(include_images=False), **kwargs)


def test_model_ready_future_state_changes_actions_without_mutating_inputs(tiny_policy) -> None:
    source_batch = _make_batch()
    image_tokens, token_masks = _encode_batch_images(tiny_policy, source_batch)
    batch = _make_batch(include_images=False)
    batch_before = _clone_batch(batch)
    future_state = tiny_policy.prepare_state(batch).clone()
    future_state[:, 0] += 5
    future_state_before = future_state.clone()
    noise = torch.randn(2, tiny_policy.config.chunk_size, tiny_policy.config.max_action_dim)

    current_actions = tiny_policy.predict_action_chunk(
        _clone_batch(batch),
        noise=noise.clone(),
        future_image_tokens=image_tokens,
        future_image_token_masks=token_masks,
    )
    future_actions = tiny_policy.predict_action_chunk(
        batch,
        noise=noise.clone(),
        future_image_tokens=image_tokens,
        future_image_token_masks=token_masks,
        future_state=future_state,
    )

    assert not torch.allclose(current_actions, future_actions)
    for key in batch:
        torch.testing.assert_close(batch[key], batch_before[key], rtol=0, atol=0)
    torch.testing.assert_close(future_state, future_state_before, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_case", ["shape", "batch_size", "dtype", "device"])
def test_invalid_model_ready_future_state_raises_clear_error(tiny_policy, invalid_case: str) -> None:
    image_tokens, token_masks = _encode_batch_images(tiny_policy, _make_batch())
    state = tiny_policy.prepare_state(_make_batch(include_images=False))
    if invalid_case == "shape":
        state = state[:, :-1]
    elif invalid_case == "batch_size":
        state = state[:1]
    elif invalid_case == "dtype":
        state = state.to(torch.float64)
    elif invalid_case == "device":
        state = state.to("meta")

    with pytest.raises((TypeError, ValueError), match="future_state"):
        tiny_policy.predict_action_chunk(
            _make_batch(include_images=False),
            future_image_tokens=image_tokens,
            future_image_token_masks=token_masks,
            future_state=state,
        )
