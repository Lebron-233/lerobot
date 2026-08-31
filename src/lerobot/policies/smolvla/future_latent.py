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

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_future_latent import FutureLatentConfig

_COMMITTED_ACTION_HORIZON = 8


def _raise_for_invalid_conditions(conditions: list[tuple[Tensor, str]]) -> None:
    """Evaluate device-side value checks once and report the first failed contract."""
    results = torch.stack([condition.reshape(()) for condition, _ in conditions]).detach().cpu()
    for is_valid, (_, message) in zip(results.tolist(), conditions, strict=True):
        if not is_valid:
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class FutureLatentPrediction:
    delta_tokens: tuple[Tensor, ...]
    predicted_error: Tensor


class LightweightFutureLatentPredictor(nn.Module):
    """Predict a residual over SmolVLA image tokens from short-horizon context."""

    def __init__(self, config: FutureLatentConfig) -> None:
        super().__init__()
        if not isinstance(config, FutureLatentConfig):
            raise TypeError(f"config must be a FutureLatentConfig, got {type(config).__name__}")

        self.config = config
        self.token_norm = nn.LayerNorm(config.token_dim)
        self.down_projection = nn.Linear(config.token_dim, config.rank)
        if config.token_mixer == "depthwise1d":
            self.token_mixer: nn.Conv1d | None = nn.Conv1d(
                config.rank,
                config.rank,
                kernel_size=3,
                padding=1,
                groups=config.rank,
                bias=False,
            )
        else:
            self.token_mixer = None
        self.camera_embedding = nn.Embedding(config.max_cameras, config.rank)

        self.action_projection = nn.Linear(config.action_dim, config.action_hidden_dim)
        self.action_gru = nn.GRU(
            config.action_hidden_dim,
            config.action_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.SiLU(),
        )
        self.delay_embedding = nn.Embedding(config.max_prediction_delay + 1, config.delay_embedding_dim)

        context_dim = config.action_hidden_dim + config.state_hidden_dim + config.delay_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(context_dim, config.fusion_hidden_dim),
            nn.SiLU(),
        )
        self.film = nn.Linear(config.fusion_hidden_dim, 2 * config.rank)
        self.up_projection = nn.Linear(config.rank, config.token_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

        if config.risk_head:
            self.risk_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(config.rank + config.fusion_hidden_dim, config.fusion_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.fusion_hidden_dim, 1),
            )
        else:
            self.risk_head = None

        parameter_count = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        if parameter_count >= config.max_parameter_count:
            raise ValueError(
                "LightweightFutureLatentPredictor has "
                f"{parameter_count:,} trainable parameters, which must be less than the configured "
                f"limit of {config.max_parameter_count:,}."
            )

    def _validate_inputs(
        self,
        image_tokens: tuple[Tensor, ...],
        image_token_masks: tuple[Tensor, ...],
        committed_actions: Tensor,
        committed_mask: Tensor,
        state: Tensor,
        delay_steps: Tensor,
    ) -> tuple[int, torch.device, torch.dtype]:
        if not isinstance(image_tokens, tuple) or not isinstance(image_token_masks, tuple):
            raise TypeError("image_tokens and image_token_masks must be tuples")
        if not image_tokens:
            raise ValueError("image_tokens must contain at least one camera")
        if len(image_tokens) != len(image_token_masks):
            raise ValueError("image_tokens and image_token_masks must contain the same number of cameras")
        if len(image_tokens) > self.config.max_cameras:
            raise ValueError(
                f"received {len(image_tokens)} cameras, but max_cameras is {self.config.max_cameras}"
            )

        first_tokens = image_tokens[0]
        if not isinstance(first_tokens, Tensor):
            raise TypeError("image_tokens[0] must be a Tensor")
        if first_tokens.ndim != 3:
            raise ValueError(f"image_tokens[0] must have shape [B, N, D], got {tuple(first_tokens.shape)}")
        batch_size, first_token_count, token_dim = first_tokens.shape
        if batch_size <= 0 or first_token_count <= 0:
            raise ValueError("image token batches and token sequences must be non-empty")
        if token_dim != self.config.token_dim:
            raise ValueError(f"image token dimension must be {self.config.token_dim}, got {token_dim}")
        if not first_tokens.is_floating_point():
            raise TypeError(f"image tokens must have a floating dtype, got {first_tokens.dtype}")

        device = first_tokens.device
        dtype = first_tokens.dtype
        value_conditions: list[tuple[Tensor, str]] = []
        if device != self.token_norm.weight.device:
            raise ValueError(
                f"input tensors are on {device}, but predictor parameters are on {self.token_norm.weight.device}"
            )

        for camera_index, (tokens, mask) in enumerate(zip(image_tokens, image_token_masks, strict=True)):
            if not isinstance(tokens, Tensor) or not isinstance(mask, Tensor):
                raise TypeError(f"camera {camera_index} tokens and mask must be Tensors")
            if tokens.ndim != 3:
                raise ValueError(
                    f"image_tokens[{camera_index}] must have shape [B, N, D], got {tuple(tokens.shape)}"
                )
            if tokens.shape[0] != batch_size or tokens.shape[2] != self.config.token_dim:
                raise ValueError(
                    f"image_tokens[{camera_index}] must have B={batch_size} and D={self.config.token_dim}, "
                    f"got {tuple(tokens.shape)}"
                )
            if tokens.shape[1] <= 0:
                raise ValueError(f"image_tokens[{camera_index}] must contain at least one token")
            if tokens.device != device or tokens.dtype != dtype:
                raise ValueError("all camera token tensors must share device and dtype")
            if mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"image_token_masks[{camera_index}] must have shape {tuple(tokens.shape[:2])}, "
                    f"got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise TypeError(f"image_token_masks[{camera_index}] must have dtype bool, got {mask.dtype}")
            if mask.device != device:
                raise ValueError("image token masks and image tokens must share a device")
            value_conditions.append(
                (
                    (torch.isfinite(tokens) | ~mask.unsqueeze(-1)).all(),
                    f"image_tokens[{camera_index}] contains non-finite values at valid tokens",
                )
            )

        if not isinstance(committed_actions, Tensor):
            raise TypeError("committed_actions must be a Tensor")
        expected_action_shape = (batch_size, _COMMITTED_ACTION_HORIZON, self.config.action_dim)
        if committed_actions.shape != expected_action_shape:
            raise ValueError(
                f"committed_actions must have shape {expected_action_shape}, got {tuple(committed_actions.shape)}"
            )
        if not committed_actions.is_floating_point():
            raise TypeError(f"committed_actions must have a floating dtype, got {committed_actions.dtype}")
        if committed_actions.device != device:
            raise ValueError("committed_actions and image tokens must share a device")

        if not isinstance(committed_mask, Tensor):
            raise TypeError("committed_mask must be a Tensor")
        expected_mask_shape = (batch_size, _COMMITTED_ACTION_HORIZON)
        if committed_mask.shape != expected_mask_shape:
            raise ValueError(
                f"committed_mask must have shape {expected_mask_shape}, got {tuple(committed_mask.shape)}"
            )
        if committed_mask.dtype != torch.bool:
            raise TypeError(f"committed_mask must have dtype bool, got {committed_mask.dtype}")
        if committed_mask.device != device:
            raise ValueError("committed_mask and image tokens must share a device")

        if not isinstance(state, Tensor):
            raise TypeError("state must be a Tensor")
        expected_state_shape = (batch_size, self.config.state_dim)
        if state.shape != expected_state_shape:
            raise ValueError(f"state must have shape {expected_state_shape}, got {tuple(state.shape)}")
        if not state.is_floating_point():
            raise TypeError(f"state must have a floating dtype, got {state.dtype}")
        if state.device != device:
            raise ValueError("state and image tokens must share a device")
        value_conditions.append((torch.isfinite(state).all(), "state contains non-finite values"))

        if not isinstance(delay_steps, Tensor):
            raise TypeError("delay_steps must be a Tensor")
        if delay_steps.shape != (batch_size,):
            raise ValueError(f"delay_steps must have shape ({batch_size},), got {tuple(delay_steps.shape)}")
        if delay_steps.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"delay_steps must have an integer dtype, got {delay_steps.dtype}")
        if delay_steps.device != device:
            raise ValueError("delay_steps and image tokens must share a device")
        value_conditions.append(
            (
                ((delay_steps >= 1) & (delay_steps <= self.config.max_prediction_delay)).all(),
                f"delay_steps must be in [1, {self.config.max_prediction_delay}]",
            )
        )

        mask_positions = torch.arange(_COMMITTED_ACTION_HORIZON, device=device).unsqueeze(0)
        expected_committed_mask = mask_positions < delay_steps.unsqueeze(1)
        value_conditions.append(
            (
                (committed_mask == expected_committed_mask).all(),
                "committed_mask must be a contiguous prefix whose sum equals delay_steps",
            )
        )
        value_conditions.append(
            (
                (torch.isfinite(committed_actions) | ~committed_mask.unsqueeze(-1)).all(),
                "committed_actions contains non-finite values in the valid prefix",
            )
        )
        _raise_for_invalid_conditions(value_conditions)

        return batch_size, device, dtype

    def _encode_image_tokens(
        self,
        image_tokens: tuple[Tensor, ...],
        image_token_masks: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        low_rank_tokens = []
        for camera_index, (tokens, mask) in enumerate(zip(image_tokens, image_token_masks, strict=True)):
            mask_3d = mask.unsqueeze(-1)
            safe_tokens = tokens.masked_fill(~mask_3d, 0.0).to(dtype=self.token_norm.weight.dtype)
            low_rank = self.down_projection(self.token_norm(safe_tokens))
            camera_embedding = self.camera_embedding.weight[camera_index].view(1, 1, -1)
            low_rank = (low_rank + camera_embedding).masked_fill(~mask_3d, 0.0)
            if self.token_mixer is not None:
                mixed = self.token_mixer(low_rank.transpose(1, 2)).transpose(1, 2)
                low_rank = (low_rank + mixed).masked_fill(~mask_3d, 0.0)
            low_rank_tokens.append(low_rank)
        return tuple(low_rank_tokens)

    def forward(
        self,
        image_tokens: tuple[Tensor, ...],
        image_token_masks: tuple[Tensor, ...],
        committed_actions: Tensor,
        committed_mask: Tensor,
        state: Tensor,
        delay_steps: Tensor,
    ) -> FutureLatentPrediction:
        batch_size, _, _ = self._validate_inputs(
            image_tokens,
            image_token_masks,
            committed_actions,
            committed_mask,
            state,
            delay_steps,
        )

        low_rank_tokens = self._encode_image_tokens(image_tokens, image_token_masks)

        compute_dtype = self.token_norm.weight.dtype
        safe_actions = committed_actions.masked_fill(~committed_mask.unsqueeze(-1), 0.0)
        action_features = F.silu(self.action_projection(safe_actions.to(dtype=compute_dtype)))
        action_sequence, _ = self.action_gru(action_features)
        last_valid_action = (delay_steps - 1).to(dtype=torch.long).view(batch_size, 1, 1)
        last_valid_action = last_valid_action.expand(-1, 1, self.config.action_hidden_dim)
        action_context = action_sequence.gather(dim=1, index=last_valid_action).squeeze(1)

        state_context = self.state_encoder(state.to(dtype=compute_dtype))
        delay_context = self.delay_embedding(delay_steps)
        fused_context = self.fusion(torch.cat((action_context, state_context, delay_context), dim=-1))
        gamma, beta = self.film(fused_context).chunk(2, dim=-1)

        delta_tokens = []
        output_conditions: list[tuple[Tensor, str]] = []
        visual_sum = low_rank_tokens[0].new_zeros((batch_size, self.config.rank))
        valid_token_count = low_rank_tokens[0].new_zeros((batch_size, 1))
        for low_rank, mask, original_tokens in zip(
            low_rank_tokens, image_token_masks, image_tokens, strict=True
        ):
            modulated = low_rank * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            delta = self.up_projection(modulated).to(dtype=original_tokens.dtype)
            delta = delta.masked_fill(~mask.unsqueeze(-1), 0.0)
            output_conditions.append(
                (torch.isfinite(delta).all(), "predicted delta_tokens contains non-finite values")
            )
            delta_tokens.append(delta)

            visual_sum = visual_sum + low_rank.sum(dim=1)
            valid_token_count = valid_token_count + mask.sum(dim=1, keepdim=True).to(low_rank.dtype)

        pooled_visual = visual_sum / valid_token_count.clamp_min(1.0)
        if self.risk_head is None:
            predicted_error = pooled_visual.new_zeros((batch_size,))
        else:
            risk_logits = self.risk_head(torch.cat((pooled_visual, fused_context), dim=-1)).squeeze(-1)
            predicted_error = F.softplus(risk_logits)
        output_conditions.append(
            (
                (torch.isfinite(predicted_error) & (predicted_error >= 0)).all(),
                "predicted_error must be finite and non-negative",
            )
        )
        _raise_for_invalid_conditions(output_conditions)

        return FutureLatentPrediction(delta_tokens=tuple(delta_tokens), predicted_error=predicted_error)
