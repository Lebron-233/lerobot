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

"""Training data and objective helpers for the offline future-latent predictor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset

from lerobot.policies.smolvla.future_latent import (
    FutureLatentPrediction,
    LightweightFutureLatentPredictor,
)
from lerobot.utils.random_utils import get_rng_state, set_rng_state

if __package__:
    from .future_latent_cache import (
        MAX_PREDICTION_DELAY,
        FutureLatentPair,
        build_future_latent_pair,
        load_cache_manifest,
        load_episode_cache,
        validate_episode_cache,
    )
else:
    from future_latent_cache import (
        MAX_PREDICTION_DELAY,
        FutureLatentPair,
        build_future_latent_pair,
        load_cache_manifest,
        load_episode_cache,
        validate_episode_cache,
    )

CACHE_PRODUCER_SHA = "eff8be608c899d0841ad5967d80d5d726cbe4394"

_DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
_CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
_VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
_CAMERA_ORDER = ("observation.images.camera1", "observation.images.camera2")
_CAMERA_MAPPING = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
_STATE_SEMANTICS = "model_ready_normalized_and_padded"
_ACTION_SEMANTICS = "normalized_policy_output_original_action_dim"


@dataclass(frozen=True, slots=True)
class _SplitExpectation:
    episode_ids: tuple[int, ...]
    frame_count: int
    pair_count_by_delay: tuple[int, ...]


_SPLIT_EXPECTATIONS = {
    "train": _SplitExpectation(
        episode_ids=(
            0,
            1,
            2,
            3,
            4,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            32,
            34,
            35,
            36,
            37,
            38,
            40,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ),
        frame_count=15_576,
        pair_count_by_delay=(15_536, 15_496, 15_456, 15_416, 15_376, 15_336, 15_296, 15_256),
    ),
    "val": _SplitExpectation(
        episode_ids=(5, 7, 14, 39, 49),
        frame_count=1_822,
        pair_count_by_delay=(1_817, 1_812, 1_807, 1_802, 1_797, 1_792, 1_787, 1_782),
    ),
}


@dataclass(frozen=True, slots=True)
class FutureLatentBatch:
    current_image_tokens: tuple[Tensor, ...]
    current_image_token_masks: tuple[Tensor, ...]
    committed_actions: Tensor
    committed_mask: Tensor
    current_state: Tensor
    delay_steps: Tensor
    target_image_tokens: tuple[Tensor, ...]
    target_image_token_masks: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class FutureLatentObjective:
    total: Tensor
    latent_smoothl1: Tensor
    cosine: Tensor
    risk_smoothl1: Tensor
    per_sample_smoothl1: Tensor
    per_sample_mse: Tensor
    per_sample_cosine: Tensor
    risk_target: Tensor


@dataclass(frozen=True, slots=True)
class FutureLatentPerSampleMetrics:
    smoothl1: Tensor
    mse: Tensor
    cosine: Tensor


@dataclass(frozen=True, slots=True)
class FutureLatentPairSpec:
    episode_position: int
    frame_offset: int
    delay_steps: int


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    total: float
    latent_smoothl1: float
    cosine: float
    risk_smoothl1: float
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    parameter_delta: float
    sample_count: int
    micro_batch_count: int
    optimizer_step_count: int = 1


@dataclass(frozen=True, slots=True)
class BestMetric:
    val_macro_smoothl1: float
    val_macro_mse: float
    epoch: int


@dataclass(frozen=True, slots=True)
class SelectionState:
    best: BestMetric | None = None
    early_stop_reference_smoothl1: float | None = None
    epochs_without_improvement: int = 0


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _validate_cache_identity(manifest: dict[str, Any], expected_split: str) -> None:
    if expected_split not in _SPLIT_EXPECTATIONS:
        raise ValueError("expected_split must be 'train' or 'val'; test cache access is not permitted")
    expectation = _SPLIT_EXPECTATIONS[expected_split]

    _require_equal(manifest.get("split"), expected_split, name="cache split")
    if manifest.get("complete_split") is not True:
        raise ValueError("training requires a complete_split=true cache")
    _require_equal(
        manifest.get("authoritative_episode_ids"), list(expectation.episode_ids), name="episode split"
    )
    _require_equal(manifest.get("cached_episode_ids"), list(expectation.episode_ids), name="cached episodes")
    _require_equal(manifest.get("episode_count"), len(expectation.episode_ids), name="episode count")
    _require_equal(manifest.get("frame_count"), expectation.frame_count, name="frame count")
    expected_pair_counts = {
        str(delay): expectation.pair_count_by_delay[delay - 1] for delay in range(1, MAX_PREDICTION_DELAY + 1)
    }
    _require_equal(manifest.get("valid_pair_count_by_delay"), expected_pair_counts, name="valid pair counts")

    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("cache manifest must contain producer provenance")
    _require_equal(producer.get("git_sha"), CACHE_PRODUCER_SHA, name="cache producer SHA")

    expected_inputs = {
        "dataset": ("lerobot/svla_so100_pickplace", _DATASET_REVISION),
        "checkpoint": ("lerobot/smolvla_base", _CHECKPOINT_REVISION),
        "vlm": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", _VLM_REVISION),
    }
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("cache manifest must contain pinned input provenance")
    for input_name, (repo_id, revision) in expected_inputs.items():
        entry = inputs.get(input_name)
        if not isinstance(entry, dict):
            raise ValueError(f"cache manifest must contain {input_name} provenance")
        _require_equal(entry.get("repo_id"), repo_id, name=f"{input_name} repo_id")
        _require_equal(entry.get("requested_revision"), revision, name=f"{input_name} requested revision")
        _require_equal(entry.get("resolved_revision"), revision, name=f"{input_name} resolved revision")

    _require_equal(manifest.get("camera_mapping"), _CAMERA_MAPPING, name="camera mapping")
    _require_equal(manifest.get("policy_camera_order"), list(_CAMERA_ORDER), name="camera order")
    _require_equal(
        manifest.get("token_scaling_convention"),
        "native_post_sqrt_hidden_dim",
        name="token scaling convention",
    )
    semantics = manifest.get("semantics")
    if not isinstance(semantics, dict):
        raise ValueError("cache manifest must contain state/action semantics")
    _require_equal(semantics.get("state"), _STATE_SEMANTICS, name="state semantics")
    _require_equal(semantics.get("action"), _ACTION_SEMANTICS, name="action semantics")
    _require_equal(
        semantics.get("processor_config_source"),
        f"lerobot/smolvla_base@{_CHECKPOINT_REVISION}",
        name="processor config source",
    )

    for episode_entry in manifest["episodes"]:
        frame_count = episode_entry["frame_count"]
        metadata = episode_entry["tensor_metadata"]
        expected_specs = {
            "dataset_indices": ([frame_count], "int64"),
            "frame_indices": ([frame_count], "int64"),
            "states": ([frame_count, 32], "float32"),
            "actions": ([frame_count, 6], "float32"),
            "language_tokens": ([frame_count, 48], "int64"),
            "language_attention_mask": ([frame_count, 48], "bool"),
            "image_tokens_0": ([frame_count, 64, 960], "bfloat16"),
            "image_token_masks_0": ([frame_count, 64], "bool"),
            "image_tokens_1": ([frame_count, 64, 960], "bfloat16"),
            "image_token_masks_1": ([frame_count, 64], "bool"),
        }
        for key, (shape, dtype) in expected_specs.items():
            _require_equal(metadata[key]["shape"], shape, name=f"episode tensor {key} shape")
            _require_equal(metadata[key]["dtype"], dtype, name=f"episode tensor {key} dtype")


class FutureLatentCacheDataset(Dataset[FutureLatentPair]):
    """Canonical pair view over one complete, pinned B1 frame cache."""

    def __init__(self, cache_dir: Path, *, expected_split: str):
        self.cache_dir = Path(cache_dir)
        self.expected_split = expected_split
        self.manifest = load_cache_manifest(self.cache_dir)
        _validate_cache_identity(self.manifest, expected_split)

        pair_specs: list[FutureLatentPairSpec] = []
        indices_by_delay: dict[int, list[int]] = {delay: [] for delay in range(1, MAX_PREDICTION_DELAY + 1)}
        for episode_position, episode_entry in enumerate(self.manifest["episodes"]):
            frame_count = episode_entry["frame_count"]
            for frame_offset in range(frame_count):
                for delay_steps in range(1, MAX_PREDICTION_DELAY + 1):
                    if frame_offset + delay_steps >= frame_count:
                        continue
                    pair_index = len(pair_specs)
                    pair_specs.append(
                        FutureLatentPairSpec(
                            episode_position=episode_position,
                            frame_offset=frame_offset,
                            delay_steps=delay_steps,
                        )
                    )
                    indices_by_delay[delay_steps].append(pair_index)

        expectation = _SPLIT_EXPECTATIONS[expected_split]
        expected_length = sum(expectation.pair_count_by_delay)
        if len(pair_specs) != expected_length:
            raise ValueError(
                f"{expected_split} cache has {len(pair_specs)} pairs, expected {expected_length}"
            )
        self.pair_specs = tuple(pair_specs)
        self._indices_by_delay = {delay: tuple(indices) for delay, indices in indices_by_delay.items()}
        self._episode_tensors: dict[int, dict[str, Tensor]] = {}

    def __len__(self) -> int:
        return len(self.pair_specs)

    def indices_for_delay(self, delay_steps: int) -> tuple[int, ...]:
        if delay_steps not in self._indices_by_delay:
            raise ValueError(f"delay_steps must be in [1, {MAX_PREDICTION_DELAY}]")
        return self._indices_by_delay[delay_steps]

    def _load_episode(self, episode_position: int) -> tuple[dict[str, Any], dict[str, Tensor]]:
        episode_entry = self.manifest["episodes"][episode_position]
        episode_index = episode_entry["episode_index"]
        tensors = self._episode_tensors.get(episode_index)
        if tensors is None:
            tensors = load_episode_cache(self.cache_dir, episode_index)
            validate_episode_cache(self.manifest, episode_entry, tensors)
            self._episode_tensors[episode_index] = tensors
        return episode_entry, tensors

    def __getitem__(self, index: int) -> FutureLatentPair:
        spec = self.pair_specs[index]
        episode_entry, tensors = self._load_episode(spec.episode_position)
        return build_future_latent_pair(
            self.manifest,
            episode_entry,
            tensors,
            frame_offset=spec.frame_offset,
            delay_steps=spec.delay_steps,
        )


def collate_future_latent_pairs(pairs: list[FutureLatentPair]) -> FutureLatentBatch:
    if not pairs:
        raise ValueError("cannot collate an empty future-latent batch")
    camera_count = len(pairs[0].current_image_tokens)
    if camera_count == 0 or any(
        len(pair.current_image_tokens) != camera_count
        or len(pair.current_image_token_masks) != camera_count
        or len(pair.target_image_tokens) != camera_count
        or len(pair.target_image_token_masks) != camera_count
        for pair in pairs
    ):
        raise ValueError("all pairs must contain the same non-zero camera count")

    return FutureLatentBatch(
        current_image_tokens=tuple(
            torch.stack([pair.current_image_tokens[camera] for pair in pairs])
            for camera in range(camera_count)
        ),
        current_image_token_masks=tuple(
            torch.stack([pair.current_image_token_masks[camera] for pair in pairs])
            for camera in range(camera_count)
        ),
        committed_actions=torch.stack([pair.committed_actions for pair in pairs]),
        committed_mask=torch.stack([pair.committed_mask for pair in pairs]),
        current_state=torch.stack([pair.current_state for pair in pairs]),
        delay_steps=torch.stack([pair.delay_steps for pair in pairs]),
        target_image_tokens=tuple(
            torch.stack([pair.target_image_tokens[camera] for pair in pairs])
            for camera in range(camera_count)
        ),
        target_image_token_masks=tuple(
            torch.stack([pair.target_image_token_masks[camera] for pair in pairs])
            for camera in range(camera_count)
        ),
    )


def move_future_latent_batch(
    batch: FutureLatentBatch, device: torch.device | str, *, non_blocking: bool = False
) -> FutureLatentBatch:
    return FutureLatentBatch(
        current_image_tokens=tuple(
            tensor.to(device=device, non_blocking=non_blocking) for tensor in batch.current_image_tokens
        ),
        current_image_token_masks=tuple(
            tensor.to(device=device, non_blocking=non_blocking) for tensor in batch.current_image_token_masks
        ),
        committed_actions=batch.committed_actions.to(device=device, non_blocking=non_blocking),
        committed_mask=batch.committed_mask.to(device=device, non_blocking=non_blocking),
        current_state=batch.current_state.to(device=device, non_blocking=non_blocking),
        delay_steps=batch.delay_steps.to(device=device, non_blocking=non_blocking),
        target_image_tokens=tuple(
            tensor.to(device=device, non_blocking=non_blocking) for tensor in batch.target_image_tokens
        ),
        target_image_token_masks=tuple(
            tensor.to(device=device, non_blocking=non_blocking) for tensor in batch.target_image_token_masks
        ),
    )


def forward_predictor(
    predictor: LightweightFutureLatentPredictor, batch: FutureLatentBatch
) -> FutureLatentPrediction:
    return predictor(
        batch.current_image_tokens,
        batch.current_image_token_masks,
        batch.committed_actions,
        batch.committed_mask,
        batch.current_state,
        batch.delay_steps,
    )


@dataclass(frozen=True, slots=True)
class _MetricReduction:
    metrics: FutureLatentPerSampleMetrics
    smoothl1: Tensor
    cosine: Tensor
    valid_element_count: int
    valid_token_count: int


def _reduce_prediction_metrics(
    delta_tokens: tuple[Tensor, ...], batch: FutureLatentBatch
) -> _MetricReduction:
    camera_count = len(batch.current_image_tokens)
    if not (
        len(delta_tokens)
        == len(batch.current_image_token_masks)
        == len(batch.target_image_tokens)
        == len(batch.target_image_token_masks)
        == camera_count
    ):
        raise ValueError("prediction and batch camera counts must match")
    batch_size = batch.delay_steps.shape[0]
    device = batch.current_image_tokens[0].device
    smooth_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
    squared_sums = torch.zeros_like(smooth_sums)
    cosine_sums = torch.zeros_like(smooth_sums)
    element_counts = torch.zeros_like(smooth_sums)
    token_counts = torch.zeros_like(smooth_sums)

    for current, current_mask, delta, target, target_mask in zip(
        batch.current_image_tokens,
        batch.current_image_token_masks,
        delta_tokens,
        batch.target_image_tokens,
        batch.target_image_token_masks,
        strict=True,
    ):
        if delta.shape != current.shape or target.shape != current.shape:
            raise ValueError("current, delta, and target token shapes must match for every camera")
        if current_mask.shape != current.shape[:2] or target_mask.shape != current.shape[:2]:
            raise ValueError("current and target masks must match their camera token shapes")
        valid = current_mask & target_mask
        element_mask = valid.unsqueeze(-1)
        predicted = (current.float() + delta.float()).masked_fill(~element_mask, 0.0)
        detached_target = target.detach().float().masked_fill(~element_mask, 0.0)
        squared = (predicted - detached_target).square()
        smooth = F.smooth_l1_loss(predicted, detached_target, beta=1.0, reduction="none")
        cosine = 1.0 - F.cosine_similarity(predicted, detached_target, dim=-1)

        smooth_sums = smooth_sums + (smooth * element_mask).sum(dim=(1, 2))
        squared_sums = squared_sums + (squared * element_mask).sum(dim=(1, 2))
        cosine_sums = cosine_sums + (cosine * valid).sum(dim=1)
        camera_token_counts = valid.sum(dim=1).to(torch.float32)
        token_counts = token_counts + camera_token_counts
        element_counts = element_counts + camera_token_counts * current.shape[-1]

    if bool((token_counts == 0).any().item()):
        raise ValueError("every sample must contain at least one token valid in current and target masks")

    per_sample_smooth = smooth_sums / element_counts
    per_sample_mse = squared_sums / element_counts
    per_sample_cosine = cosine_sums / token_counts
    return _MetricReduction(
        metrics=FutureLatentPerSampleMetrics(
            smoothl1=per_sample_smooth,
            mse=per_sample_mse,
            cosine=per_sample_cosine,
        ),
        smoothl1=smooth_sums.sum() / element_counts.sum(),
        cosine=cosine_sums.sum() / token_counts.sum(),
        valid_element_count=int(element_counts.sum().item()),
        valid_token_count=int(token_counts.sum().item()),
    )


def compute_future_latent_objective(
    prediction: FutureLatentPrediction,
    batch: FutureLatentBatch,
    *,
    lambda_cos: float,
    lambda_risk: float,
) -> FutureLatentObjective:
    if not math.isfinite(lambda_cos) or lambda_cos < 0:
        raise ValueError("lambda_cos must be finite and non-negative")
    if not math.isfinite(lambda_risk) or lambda_risk < 0:
        raise ValueError("lambda_risk must be finite and non-negative")
    reduction = _reduce_prediction_metrics(prediction.delta_tokens, batch)
    batch_size = batch.delay_steps.shape[0]
    if prediction.predicted_error.shape != (batch_size,):
        raise ValueError(f"predicted_error must have shape ({batch_size},)")
    risk_target = reduction.metrics.mse.detach()
    risk_smoothl1 = F.smooth_l1_loss(
        prediction.predicted_error.float(), risk_target, beta=1.0, reduction="mean"
    )
    total = reduction.smoothl1 + lambda_cos * reduction.cosine + lambda_risk * risk_smoothl1
    return FutureLatentObjective(
        total=total,
        latent_smoothl1=reduction.smoothl1,
        cosine=reduction.cosine,
        risk_smoothl1=risk_smoothl1,
        per_sample_smoothl1=reduction.metrics.smoothl1,
        per_sample_mse=reduction.metrics.mse,
        per_sample_cosine=reduction.metrics.cosine,
        risk_target=risk_target,
    )


def compute_identity_baseline_metrics(batch: FutureLatentBatch) -> FutureLatentPerSampleMetrics:
    identity_delta = tuple(torch.zeros_like(tokens) for tokens in batch.current_image_tokens)
    return _reduce_prediction_metrics(identity_delta, batch).metrics


def deterministic_train_indices(length: int, *, seed: int, epoch: int = 0) -> tuple[int, ...]:
    if length < 0 or epoch < 0:
        raise ValueError("length and epoch must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return tuple(torch.randperm(length, generator=generator).tolist())


def deterministic_val_indices_by_delay(
    dataset: FutureLatentCacheDataset, *, max_pairs_per_delay: int | None = None
) -> tuple[int, ...]:
    if max_pairs_per_delay is None:
        return tuple(range(len(dataset)))
    if max_pairs_per_delay <= 0:
        raise ValueError("max_pairs_per_delay must be positive")
    return tuple(
        index
        for delay in range(1, MAX_PREDICTION_DELAY + 1)
        for index in dataset.indices_for_delay(delay)[:max_pairs_per_delay]
    )


def accumulation_windows(
    num_samples: int, *, micro_batch_size: int, effective_batch_size: int = 64
) -> tuple[tuple[int, int], ...]:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if micro_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if effective_batch_size % micro_batch_size != 0:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    return tuple(
        (start, min(start + effective_batch_size, num_samples))
        for start in range(0, num_samples, effective_batch_size)
    )


def make_predictor_optimizer(
    predictor: LightweightFutureLatentPredictor,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> AdamW:
    parameters = [parameter for parameter in predictor.parameters() if parameter.requires_grad]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("future-latent predictor trainable parameters must remain float32")
    return AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def _gradient_norm(parameters: Sequence[Tensor]) -> Tensor:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return torch.tensor(0.0)
    return torch.stack(squared).sum().sqrt()


def _valid_counts(batch: FutureLatentBatch) -> tuple[int, int]:
    valid_tokens = sum(
        int((current_mask & target_mask).sum().item())
        for current_mask, target_mask in zip(
            batch.current_image_token_masks, batch.target_image_token_masks, strict=True
        )
    )
    if valid_tokens == 0:
        raise ValueError("every optimizer micro batch must contain at least one valid token")
    token_dim = batch.current_image_tokens[0].shape[-1]
    return valid_tokens * token_dim, valid_tokens


def optimizer_step(
    predictor: LightweightFutureLatentPredictor,
    optimizer: Optimizer,
    micro_batches: Sequence[FutureLatentBatch],
    *,
    lambda_cos: float = 0.1,
    lambda_risk: float = 0.1,
    grad_clip_norm: float = 1.0,
) -> OptimizerStepResult:
    if not micro_batches:
        raise ValueError("optimizer_step requires at least one micro batch")
    parameters = [parameter for parameter in predictor.parameters() if parameter.requires_grad]
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in parameters}:
        raise ValueError("optimizer parameter groups must contain exactly the predictor trainable parameters")

    batch_weights: list[tuple[int, int, int]] = []
    for batch in micro_batches:
        valid_elements, valid_tokens = _valid_counts(batch)
        batch_weights.append((valid_elements, valid_tokens, batch.delay_steps.shape[0]))
    total_elements = sum(elements for elements, _, _ in batch_weights)
    total_tokens = sum(tokens for _, tokens, _ in batch_weights)
    total_samples = sum(samples for _, _, samples in batch_weights)

    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.zero_grad(set_to_none=True)
    latent_value = 0.0
    cosine_value = 0.0
    risk_value = 0.0
    for batch, (elements, tokens, samples) in zip(micro_batches, batch_weights, strict=True):
        objective = compute_future_latent_objective(
            forward_predictor(predictor, batch),
            batch,
            lambda_cos=lambda_cos,
            lambda_risk=lambda_risk,
        )
        scaled = (
            objective.latent_smoothl1 * (elements / total_elements)
            + lambda_cos * objective.cosine * (tokens / total_tokens)
            + lambda_risk * objective.risk_smoothl1 * (samples / total_samples)
        )
        scaled.backward()
        latent_value += objective.latent_smoothl1.detach().item() * elements / total_elements
        cosine_value += objective.cosine.detach().item() * tokens / total_tokens
        risk_value += objective.risk_smoothl1.detach().item() * samples / total_samples

    pre_clip = _gradient_norm(parameters)
    if not bool(torch.isfinite(pre_clip).item()):
        raise ValueError("predictor gradient norm is non-finite")
    torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm, error_if_nonfinite=True)
    post_clip = _gradient_norm(parameters)
    optimizer.step()
    parameter_delta = (
        torch.stack(
            [
                (parameter.detach() - original.to(parameter.device)).float().square().sum()
                for parameter, original in zip(parameters, before, strict=True)
            ]
        )
        .sum()
        .sqrt()
    )

    total_value = latent_value + lambda_cos * cosine_value + lambda_risk * risk_value
    return OptimizerStepResult(
        total=total_value,
        latent_smoothl1=latent_value,
        cosine=cosine_value,
        risk_smoothl1=risk_value,
        pre_clip_grad_norm=pre_clip.item(),
        post_clip_grad_norm=post_clip.item(),
        parameter_delta=parameter_delta.item(),
        sample_count=total_samples,
        micro_batch_count=len(micro_batches),
    )


def update_selection(
    state: SelectionState,
    *,
    epoch: int,
    val_macro_smoothl1: float,
    val_macro_mse: float,
    min_relative_improvement: float = 0.001,
    patience: int = 5,
) -> tuple[SelectionState, bool, bool]:
    candidate = BestMetric(val_macro_smoothl1, val_macro_mse, epoch)
    is_best = state.best is None or (
        candidate.val_macro_smoothl1,
        candidate.val_macro_mse,
        candidate.epoch,
    ) < (
        state.best.val_macro_smoothl1,
        state.best.val_macro_mse,
        state.best.epoch,
    )
    best = candidate if is_best else state.best

    reference = state.early_stop_reference_smoothl1
    if reference is None:
        reference = val_macro_smoothl1
        epochs_without_improvement = 0
    else:
        relative_improvement = (reference - val_macro_smoothl1) / reference if reference > 0 else 0.0
        if relative_improvement >= min_relative_improvement:
            reference = val_macro_smoothl1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = state.epochs_without_improvement + 1

    new_state = SelectionState(
        best=best,
        early_stop_reference_smoothl1=reference,
        epochs_without_improvement=epochs_without_improvement,
    )
    return new_state, is_best, epochs_without_improvement >= patience


def capture_rng_state() -> dict[str, Any]:
    return get_rng_state()


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    set_rng_state(rng_state)


def save_predictor_checkpoint(
    path: Path,
    *,
    predictor: LightweightFutureLatentPredictor,
    optimizer: Optimizer | None,
    train_config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_metrics: dict[str, Any],
    cache_provenance: dict[str, Any],
    trainer_git_sha: str,
    kind: Literal["best", "last"],
    selection_state: SelectionState | None = None,
) -> None:
    if kind == "best" and optimizer is not None:
        raise ValueError("best checkpoint must not contain optimizer state")
    if kind == "last" and optimizer is None:
        raise ValueError("last checkpoint must contain optimizer state")
    payload: dict[str, Any] = {
        "checkpoint_kind": kind,
        "predictor_state_dict": predictor.state_dict(),
        "predictor_config": asdict(predictor.config),
        "train_config": train_config,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_metrics": best_val_metrics,
        "cache_provenance": cache_provenance,
        "trainer_git_sha": trainer_git_sha,
        "cache_producer_sha": CACHE_PRODUCER_SHA,
        "selection_state": selection_state,
    }
    if kind == "last":
        assert optimizer is not None
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["rng_state"] = capture_rng_state()
    torch.save(payload, path)


def load_last_checkpoint(
    path: Path,
    *,
    predictor: LightweightFutureLatentPredictor,
    optimizer: Optimizer,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_kind") != "last":
        raise ValueError("resume_from must point to a last checkpoint")
    _require_equal(payload.get("cache_producer_sha"), CACHE_PRODUCER_SHA, name="cache producer SHA")
    _require_equal(payload.get("predictor_config"), asdict(predictor.config), name="predictor config")
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def bounded_run_markers(bounded: bool) -> dict[str, str | bool]:
    if bounded:
        return {
            "run_kind": "bounded_smoke",
            "protocol_complete": False,
            "eligible_for_checkpoint_selection": False,
            "eligible_for_test": False,
        }
    return {
        "run_kind": "train_val",
        "protocol_complete": True,
        "eligible_for_checkpoint_selection": True,
        "eligible_for_test": False,
    }
