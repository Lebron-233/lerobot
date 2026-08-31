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

"""Offline oracle image tokens for predictive-async tests and dataset replay."""

from __future__ import annotations

from typing import Any, TypedDict

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy


class OracleContextUnavailableError(ValueError):
    """Raised when a requested future frame is outside its source episode."""


class OracleFutureTokenKwargs(TypedDict):
    """SmolVLA token overrides accepted by ``predict_action_chunk``."""

    future_image_tokens: tuple[Tensor, ...]
    future_image_token_masks: tuple[Tensor, ...]


def _select_future_image(
    image: Tensor,
    is_pad: Tensor,
    *,
    image_key: str,
    delay_steps: int,
) -> Tensor:
    """Select one horizon from a LeRobotDataset sample or collated batch."""
    if image.ndim == 4 and is_pad.ndim == 1:
        # LeRobotDataset.__getitem__: [T, C, H, W] and [T].
        horizon = image.shape[0]
        if is_pad.shape[0] != horizon:
            raise ValueError(f"{image_key!r} and {image_key + '_is_pad'!r} disagree on horizon length")
        if delay_steps >= horizon:
            raise OracleContextUnavailableError(
                f"{image_key!r} has no oracle frame at delay_steps={delay_steps}; horizon={horizon}"
            )
        if bool(is_pad[delay_steps].item()):
            raise OracleContextUnavailableError(
                f"{image_key!r} oracle frame at delay_steps={delay_steps} is episode padding"
            )
        return image[delay_steps].unsqueeze(0)

    if image.ndim == 5 and is_pad.ndim == 2:
        # DataLoader batch: [B, T, C, H, W] and [B, T].
        batch_size, horizon = image.shape[:2]
        if tuple(is_pad.shape) != (batch_size, horizon):
            raise ValueError(f"{image_key!r} and {image_key + '_is_pad'!r} disagree on batch/horizon shape")
        if delay_steps >= horizon:
            raise OracleContextUnavailableError(
                f"{image_key!r} has no oracle frame at delay_steps={delay_steps}; horizon={horizon}"
            )
        if bool(is_pad[:, delay_steps].any().item()):
            raise OracleContextUnavailableError(
                f"{image_key!r} oracle frame at delay_steps={delay_steps} is episode padding"
            )
        return image[:, delay_steps]

    raise ValueError(
        f"{image_key!r} must be a temporal LeRobotDataset sample [T,C,H,W] or batch "
        f"[B,T,C,H,W] with matching _is_pad metadata; got image {tuple(image.shape)} and "
        f"padding {tuple(is_pad.shape)}"
    )


@torch.inference_mode()
def encode_oracle_future_tokens(
    policy: PreTrainedPolicy,
    temporal_batch: dict[str, Any],
    *,
    delay_steps: int,
) -> OracleFutureTokenKwargs:
    """Encode the true ``t + delay_steps`` dataset frame as SmolVLA overrides.

    Configure ``LeRobotDataset.delta_timestamps`` for every policy camera with
    horizons ``[0 / fps, ..., max_delay / fps]`` before obtaining the sample,
    then run the temporal sample through the policy preprocessor so tensors are
    on the policy device. Image tensors are shaped ``[T,C,H,W]`` with
    ``*_is_pad`` tensors shaped ``[T]``; a collated batch adds a leading batch
    dimension.

    The helper is intentionally offline-only. It selects cameras in
    ``policy.config.image_features`` order, rejects episode-tail padding, and
    invokes ``prepare_images`` and ``encode_image_tokens`` exactly once. The
    returned mapping can be expanded directly into ``predict_action_chunk``.
    State is deliberately not overridden in PR-1.
    """
    if delay_steps < 0:
        raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")

    token_policy: Any = policy
    image_keys = tuple(token_policy.config.image_features)
    if not image_keys:
        raise ValueError("Oracle future context requires at least one policy image feature")

    future_batch = dict(temporal_batch)
    for image_key in image_keys:
        if image_key not in temporal_batch:
            raise KeyError(f"Oracle temporal batch is missing policy image feature {image_key!r}")
        pad_key = f"{image_key}_is_pad"
        if pad_key not in temporal_batch:
            raise KeyError(f"Oracle temporal batch is missing episode-boundary metadata {pad_key!r}")

        image = temporal_batch[image_key]
        is_pad = temporal_batch[pad_key]
        if not isinstance(image, Tensor) or not isinstance(is_pad, Tensor):
            raise TypeError(f"{image_key!r} and {pad_key!r} must both be torch tensors")
        future_batch[image_key] = _select_future_image(
            image,
            is_pad,
            image_key=image_key,
            delay_steps=delay_steps,
        )

    images, image_masks = token_policy.prepare_images(future_batch)
    image_tokens, image_token_masks = token_policy.model.encode_image_tokens(images, image_masks)
    return {
        "future_image_tokens": image_tokens,
        "future_image_token_masks": image_token_masks,
    }
