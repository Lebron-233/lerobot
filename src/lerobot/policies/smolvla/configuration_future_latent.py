# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FutureLatentConfig:
    """Configuration for the lightweight future-visual-latent predictor."""

    token_dim: int
    action_dim: int
    state_dim: int
    enabled: bool = False
    rank: int = 64
    action_hidden_dim: int = 128
    state_hidden_dim: int = 64
    delay_embedding_dim: int = 32
    fusion_hidden_dim: int = 128
    max_prediction_delay: int = 8
    max_cameras: int = 4
    token_mixer: str = "depthwise1d"
    risk_head: bool = True
    max_parameter_count: int = 1_000_000

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "token_dim",
            "action_dim",
            "state_dim",
            "rank",
            "action_hidden_dim",
            "state_hidden_dim",
            "delay_embedding_dim",
            "fusion_hidden_dim",
            "max_prediction_delay",
            "max_cameras",
            "max_parameter_count",
        )
        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{field_name}` must be a positive integer, got {value!r}.")

        if not isinstance(self.enabled, bool):
            raise ValueError(f"`enabled` must be a bool, got {self.enabled!r}.")
        if not isinstance(self.risk_head, bool):
            raise ValueError(f"`risk_head` must be a bool, got {self.risk_head!r}.")
        if self.rank > self.token_dim:
            raise ValueError(
                f"`rank` must not exceed `token_dim` for the low-rank projection, got "
                f"rank={self.rank} and token_dim={self.token_dim}."
            )
        if self.action_hidden_dim > 128:
            raise ValueError(f"`action_hidden_dim` must be at most 128, got {self.action_hidden_dim}.")
        if self.state_hidden_dim > 64:
            raise ValueError(f"`state_hidden_dim` must be at most 64, got {self.state_hidden_dim}.")
        if self.fusion_hidden_dim > 128:
            raise ValueError(f"`fusion_hidden_dim` must be at most 128, got {self.fusion_hidden_dim}.")
        if self.max_prediction_delay > 8:
            raise ValueError(
                f"`max_prediction_delay` must be in the Phase A range 1..8, got {self.max_prediction_delay}."
            )
        if self.token_mixer not in {"depthwise1d", "none"}:
            raise ValueError(f"`token_mixer` must be 'depthwise1d' or 'none', got {self.token_mixer!r}.")
        if self.max_parameter_count > 1_000_000:
            raise ValueError(
                f"`max_parameter_count` must preserve the Phase A ceiling of 1,000,000, got "
                f"{self.max_parameter_count}."
            )
