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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.inference.oracle_context import (
    OracleContextUnavailableError,
    encode_oracle_future_tokens,
)


class _RecordingTokenModel:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.encoded_images: tuple[torch.Tensor, ...] | None = None

    def encode_image_tokens(self, images, image_masks):
        self.encode_calls += 1
        self.encoded_images = tuple(image.clone() for image in images)
        tokens = tuple(image.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1) for image in images)
        masks = tuple(
            mask[:, None].expand(-1, token.shape[1]) for mask, token in zip(image_masks, tokens, strict=True)
        )
        return tokens, masks


class _RecordingPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            image_features={
                "observation.images.front": object(),
                "observation.images.wrist": object(),
            }
        )
        self.model = _RecordingTokenModel()
        self.prepare_calls = 0

    def prepare_images(self, batch):
        self.prepare_calls += 1
        images = [batch[key] for key in self.config.image_features]
        masks = [torch.ones(image.shape[0], dtype=torch.bool) for image in images]
        return images, masks


def test_oracle_selects_true_future_frames_in_policy_camera_order_once() -> None:
    policy = _RecordingPolicy()
    temporal_batch = {
        "observation.images.front": torch.tensor(
            [[[[[10.0]]], [[[11.0]]], [[[12.0]]]], [[[[20.0]]], [[[21.0]]], [[[22.0]]]]]
        ),
        "observation.images.front_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        "observation.images.wrist": torch.tensor(
            [[[[[100.0]]], [[[101.0]]], [[[102.0]]]], [[[[200.0]]], [[[201.0]]], [[[202.0]]]]]
        ),
        "observation.images.wrist_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }

    kwargs = encode_oracle_future_tokens(policy, temporal_batch, delay_steps=2)

    assert policy.prepare_calls == 1
    assert policy.model.encode_calls == 1
    assert policy.model.encoded_images is not None
    assert [image.flatten().tolist() for image in policy.model.encoded_images] == [
        [12.0, 22.0],
        [102.0, 202.0],
    ]
    assert set(kwargs) == {"future_image_tokens", "future_image_token_masks"}
    assert [token.flatten().tolist() for token in kwargs["future_image_tokens"]] == [
        [12.0, 22.0],
        [102.0, 202.0],
    ]
    assert all(mask.dtype is torch.bool and mask.all() for mask in kwargs["future_image_token_masks"])


def test_oracle_rejects_padded_episode_tail_before_encoding() -> None:
    policy = _RecordingPolicy()
    temporal_sample = {
        "observation.images.front": torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1),
        "observation.images.front_is_pad": torch.tensor([False, False, True]),
        "observation.images.wrist": torch.arange(10, 13, dtype=torch.float32).reshape(3, 1, 1, 1),
        "observation.images.wrist_is_pad": torch.tensor([False, False, True]),
    }

    with pytest.raises(OracleContextUnavailableError, match="episode padding"):
        encode_oracle_future_tokens(policy, temporal_sample, delay_steps=2)

    assert policy.prepare_calls == 0
    assert policy.model.encode_calls == 0
