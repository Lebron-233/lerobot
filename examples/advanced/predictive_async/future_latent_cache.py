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

"""Schema-v1 data access for frame-level SmolVLA future-latent caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

CACHE_SCHEMA_VERSION = 1
MAX_PREDICTION_DELAY = 8

_CLASSIFICATION = "offline_future_latent_cache_not_task_capability"
_TOKEN_SCALING_CONVENTION = "native_post_sqrt_hidden_dim"
_STATE_SEMANTICS = "model_ready_normalized_and_padded"
_ACTION_SEMANTICS = "normalized_policy_output_original_action_dim"
_BASE_TENSOR_KEYS = (
    "dataset_indices",
    "frame_indices",
    "states",
    "actions",
    "language_tokens",
    "language_attention_mask",
)
_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}


@dataclass(frozen=True, slots=True)
class FutureLatentPair:
    current_image_tokens: tuple[Tensor, ...]
    current_image_token_masks: tuple[Tensor, ...]
    committed_actions: Tensor
    committed_mask: Tensor
    current_state: Tensor
    delay_steps: Tensor
    target_image_tokens: tuple[Tensor, ...]
    target_image_token_masks: tuple[Tensor, ...]
    future_state: Tensor
    current_language_tokens: Tensor
    current_language_attention_mask: Tensor
    future_language_tokens: Tensor
    future_language_attention_mask: Tensor
    episode_index: int
    frame_index: int
    future_frame_index: int


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _expected_tensor_keys(camera_count: int) -> tuple[str, ...]:
    camera_keys = tuple(
        key
        for camera_index in range(camera_count)
        for key in (f"image_tokens_{camera_index}", f"image_token_masks_{camera_index}")
    )
    return _BASE_TENSOR_KEYS + camera_keys


def _validate_tensor_metadata(
    episode_entry: dict[str, Any],
    *,
    camera_count: int,
    storage_device: str,
) -> None:
    episode_index = _require_int(episode_entry.get("episode_index"), name="episode_entry.episode_index")
    frame_count = _require_int(
        episode_entry.get("frame_count"), name=f"episode {episode_index} frame_count", minimum=1
    )
    expected_shard = f"episodes/episode_{episode_index:06d}.safetensors"
    if episode_entry.get("shard") != expected_shard:
        raise ValueError(f"episode {episode_index} shard must be {expected_shard!r}")

    metadata = episode_entry.get("tensor_metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"episode {episode_index} tensor_metadata must be an object")
    expected_keys = set(_expected_tensor_keys(camera_count))
    if set(metadata) != expected_keys:
        raise ValueError(
            f"episode {episode_index} tensor_metadata keys must be exactly {sorted(expected_keys)}"
        )

    for key, entry in metadata.items():
        if not isinstance(entry, dict):
            raise TypeError(f"episode {episode_index} metadata for {key} must be an object")
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError(f"episode {episode_index} metadata for {key} has an invalid shape")
        if shape[0] != frame_count:
            raise ValueError(
                f"episode {episode_index} metadata for {key} must start with frame_count={frame_count}"
            )
        if not isinstance(entry.get("dtype"), str):
            raise TypeError(f"episode {episode_index} metadata for {key} must record dtype")
        if entry.get("storage_device") != storage_device:
            raise ValueError(
                f"episode {episode_index} metadata for {key} must record storage_device={storage_device!r}"
            )

    expected_specs: dict[str, tuple[int, str | set[str]]] = {
        "dataset_indices": (1, "int64"),
        "frame_indices": (1, "int64"),
        "states": (2, "float32"),
        "actions": (2, "float32"),
        "language_tokens": (2, "int64"),
        "language_attention_mask": (2, "bool"),
    }
    for camera_index in range(camera_count):
        expected_specs[f"image_tokens_{camera_index}"] = (3, _FLOAT_DTYPES)
        expected_specs[f"image_token_masks_{camera_index}"] = (2, "bool")
    for key, (ndim, dtype) in expected_specs.items():
        shape = metadata[key]["shape"]
        actual_dtype = metadata[key]["dtype"]
        if len(shape) != ndim:
            raise ValueError(f"episode {episode_index} metadata for {key} must have rank {ndim}")
        if isinstance(dtype, set):
            if actual_dtype not in dtype:
                raise TypeError(
                    f"episode {episode_index} metadata for {key} has invalid dtype {actual_dtype!r}"
                )
        elif actual_dtype != dtype:
            raise TypeError(
                f"episode {episode_index} metadata for {key} must have dtype {dtype}, got {actual_dtype!r}"
            )

    if metadata["language_tokens"]["shape"] != metadata["language_attention_mask"]["shape"]:
        raise ValueError(f"episode {episode_index} language token and mask shapes must match")
    token_dtype = metadata["image_tokens_0"]["dtype"]
    token_dim = metadata["image_tokens_0"]["shape"][2]
    for camera_index in range(camera_count):
        token_key = f"image_tokens_{camera_index}"
        mask_key = f"image_token_masks_{camera_index}"
        token_shape = metadata[token_key]["shape"]
        if token_shape[:2] != metadata[mask_key]["shape"]:
            raise ValueError(f"episode {episode_index} {token_key} and {mask_key} shapes do not match")
        if metadata[token_key]["dtype"] != token_dtype or token_shape[2] != token_dim:
            raise ValueError(
                f"episode {episode_index} camera image tokens must share dtype and hidden dimension"
            )


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise TypeError("cache manifest must contain a JSON object")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(f"cache schema_version must be {CACHE_SCHEMA_VERSION}, got {schema_version!r}")
    if manifest.get("classification") != _CLASSIFICATION:
        raise ValueError(f"cache classification must be {_CLASSIFICATION!r}")
    if manifest.get("split") not in {"train", "val", "test"}:
        raise ValueError("cache split must be 'train', 'val', or 'test'")
    if type(manifest.get("complete_split")) is not bool:
        raise TypeError("cache complete_split must be a boolean")
    if manifest.get("storage_device") != "cpu":
        raise ValueError("cache storage_device must be 'cpu'")
    if manifest.get("token_scaling_convention") != _TOKEN_SCALING_CONVENTION:
        raise ValueError(f"cache token_scaling_convention must be {_TOKEN_SCALING_CONVENTION!r}")

    semantics = manifest.get("semantics")
    if not isinstance(semantics, dict):
        raise TypeError("cache semantics must be an object")
    if semantics.get("state") != _STATE_SEMANTICS:
        raise ValueError(f"cache state semantics must be {_STATE_SEMANTICS!r}")
    if semantics.get("action") != _ACTION_SEMANTICS:
        raise ValueError(f"cache action semantics must be {_ACTION_SEMANTICS!r}")
    if (
        not isinstance(semantics.get("processor_config_source"), str)
        or not semantics["processor_config_source"]
    ):
        raise ValueError("cache semantics must record a non-empty processor_config_source")

    camera_order = manifest.get("policy_camera_order")
    if (
        not isinstance(camera_order, list)
        or not camera_order
        or any(not isinstance(camera, str) or not camera for camera in camera_order)
        or len(set(camera_order)) != len(camera_order)
    ):
        raise ValueError("cache policy_camera_order must be a non-empty list of unique names")

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("cache episodes must be a non-empty list")
    for entry in episodes:
        if not isinstance(entry, dict):
            raise TypeError("each cache episode entry must be an object")
        _validate_tensor_metadata(entry, camera_count=len(camera_order), storage_device="cpu")

    episode_indices = [entry["episode_index"] for entry in episodes]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("cache episode indices must be unique")
    authoritative_episode_ids = manifest.get("authoritative_episode_ids")
    if (
        not isinstance(authoritative_episode_ids, list)
        or any(
            type(episode_index) is not int or episode_index < 0 for episode_index in authoritative_episode_ids
        )
        or len(set(authoritative_episode_ids)) != len(authoritative_episode_ids)
    ):
        raise ValueError("cache authoritative_episode_ids must be a list of unique non-negative integers")
    cached_episode_ids = manifest.get("cached_episode_ids")
    if cached_episode_ids != episode_indices:
        raise ValueError("cache cached_episode_ids must match episode entries in order")
    authoritative_positions = {
        episode_index: position for position, episode_index in enumerate(authoritative_episode_ids)
    }
    if any(episode_index not in authoritative_positions for episode_index in cached_episode_ids) or any(
        authoritative_positions[left] >= authoritative_positions[right]
        for left, right in zip(cached_episode_ids, cached_episode_ids[1:], strict=False)
    ):
        raise ValueError("cache cached_episode_ids must be an ordered subset of authoritative_episode_ids")
    if manifest["complete_split"] and cached_episode_ids != authoritative_episode_ids:
        raise ValueError("a complete_split cache must contain every authoritative episode")
    if manifest.get("episode_count") != len(episodes):
        raise ValueError("cache episode_count does not match episodes")
    total_frames = sum(entry["frame_count"] for entry in episodes)
    if manifest.get("frame_count") != total_frames:
        raise ValueError("cache frame_count does not match episode frame counts")

    pair_counts = manifest.get("valid_pair_count_by_delay")
    expected_pair_counts = {
        str(delay): sum(max(entry["frame_count"] - delay, 0) for entry in episodes)
        for delay in range(1, MAX_PREDICTION_DELAY + 1)
    }
    if pair_counts != expected_pair_counts:
        raise ValueError(
            "cache valid_pair_count_by_delay must equal the legal within-episode pair counts "
            f"{expected_pair_counts}"
        )
    return manifest


def load_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    """Load and validate the schema-v1 manifest in ``cache_dir``."""
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _validate_manifest(manifest)


def load_episode_cache(cache_dir: Path, episode_index: int) -> dict[str, Tensor]:
    """Load one deterministic per-episode safetensors shard onto CPU."""
    episode_index = _require_int(episode_index, name="episode_index")
    shard_path = cache_dir / "episodes" / f"episode_{episode_index:06d}.safetensors"
    return load_file(shard_path, device="cpu")


def validate_episode_cache(
    manifest: dict[str, Any],
    episode_entry: dict[str, Any],
    tensors: dict[str, Tensor],
) -> None:
    """Validate one loaded shard against its manifest entry and frame semantics."""
    manifest = _validate_manifest(manifest)
    if not isinstance(episode_entry, dict):
        raise TypeError("episode_entry must be an object")
    episode_index = _require_int(episode_entry.get("episode_index"), name="episode_index")
    matching_entries = [entry for entry in manifest["episodes"] if entry["episode_index"] == episode_index]
    if len(matching_entries) != 1 or matching_entries[0] != episode_entry:
        raise ValueError(f"episode_entry for episode {episode_index} does not match the manifest")
    if not isinstance(tensors, dict):
        raise TypeError("tensors must be a dictionary")

    metadata = episode_entry["tensor_metadata"]
    if set(tensors) != set(metadata):
        raise ValueError(f"episode {episode_index} tensor keys do not match tensor_metadata")
    for key, expected in metadata.items():
        tensor = tensors[key]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"episode {episode_index} {key} must be a Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"episode {episode_index} {key} must be stored on CPU")
        if list(tensor.shape) != expected["shape"]:
            raise ValueError(
                f"episode {episode_index} {key} shape {list(tensor.shape)} does not match "
                f"metadata {expected['shape']}"
            )
        actual_dtype = _dtype_name(tensor.dtype)
        if actual_dtype != expected["dtype"]:
            raise TypeError(
                f"episode {episode_index} {key} dtype {actual_dtype!r} does not match "
                f"metadata {expected['dtype']!r}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"episode {episode_index} {key} must be contiguous")

    frame_count = episode_entry["frame_count"]
    expected_frames = torch.arange(frame_count, dtype=torch.int64)
    if not torch.equal(tensors["frame_indices"], expected_frames):
        raise ValueError(f"episode {episode_index} frame_indices must equal 0..{frame_count - 1}")
    dataset_indices = tensors["dataset_indices"]
    if frame_count > 1 and not torch.all(dataset_indices[1:] == dataset_indices[:-1] + 1):
        raise ValueError(f"episode {episode_index} dataset_indices must be contiguous and ascending")
    for key in ("states", "actions"):
        if not torch.isfinite(tensors[key]).all():
            raise ValueError(f"episode {episode_index} {key} must contain only finite values")
    for camera_index in range(len(manifest["policy_camera_order"])):
        token_key = f"image_tokens_{camera_index}"
        mask_key = f"image_token_masks_{camera_index}"
        tokens = tensors[token_key]
        mask = tensors[mask_key]
        if not (torch.isfinite(tokens) | ~mask.unsqueeze(-1)).all():
            raise ValueError(
                f"episode {episode_index} {token_key} must be finite wherever {mask_key} is true"
            )


def build_future_latent_pair(
    manifest: dict[str, Any],
    episode_entry: dict[str, Any],
    tensors: dict[str, Tensor],
    *,
    frame_offset: int,
    delay_steps: int,
) -> FutureLatentPair:
    """Build the legal pair ``t -> t+d`` from a validated frame-level shard."""
    frame_offset = _require_int(frame_offset, name="frame_offset")
    delay_steps = _require_int(delay_steps, name="delay_steps", minimum=1)
    if delay_steps > MAX_PREDICTION_DELAY:
        raise ValueError(f"delay_steps must be in [1, {MAX_PREDICTION_DELAY}]")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("policy_camera_order"), list):
        raise TypeError("manifest must provide policy_camera_order")
    if not isinstance(episode_entry, dict):
        raise TypeError("episode_entry must be an object")
    frame_count = _require_int(episode_entry.get("frame_count"), name="episode frame_count", minimum=1)
    future_offset = frame_offset + delay_steps
    if frame_offset >= frame_count or future_offset >= frame_count:
        raise ValueError(
            f"pair crosses episode boundary: frame_offset={frame_offset}, delay_steps={delay_steps}, "
            f"frame_count={frame_count}"
        )

    camera_count = len(manifest["policy_camera_order"])
    current_image_tokens = tuple(
        tensors[f"image_tokens_{index}"][frame_offset] for index in range(camera_count)
    )
    current_image_token_masks = tuple(
        tensors[f"image_token_masks_{index}"][frame_offset] for index in range(camera_count)
    )
    target_image_tokens = tuple(
        tensors[f"image_tokens_{index}"][future_offset] for index in range(camera_count)
    )
    target_image_token_masks = tuple(
        tensors[f"image_token_masks_{index}"][future_offset] for index in range(camera_count)
    )

    actions = tensors["actions"]
    committed_actions = actions.new_zeros((MAX_PREDICTION_DELAY, actions.shape[1]))
    committed_actions[:delay_steps].copy_(actions[frame_offset:future_offset])
    committed_mask = torch.arange(MAX_PREDICTION_DELAY, device=actions.device) < delay_steps

    return FutureLatentPair(
        current_image_tokens=current_image_tokens,
        current_image_token_masks=current_image_token_masks,
        committed_actions=committed_actions,
        committed_mask=committed_mask,
        current_state=tensors["states"][frame_offset],
        delay_steps=torch.tensor(delay_steps, dtype=torch.int64, device=actions.device),
        target_image_tokens=target_image_tokens,
        target_image_token_masks=target_image_token_masks,
        future_state=tensors["states"][future_offset],
        current_language_tokens=tensors["language_tokens"][frame_offset],
        current_language_attention_mask=tensors["language_attention_mask"][frame_offset],
        future_language_tokens=tensors["language_tokens"][future_offset],
        future_language_attention_mask=tensors["language_attention_mask"][future_offset],
        episode_index=_require_int(episode_entry.get("episode_index"), name="episode_index"),
        frame_index=int(tensors["frame_indices"][frame_offset].item()),
        future_frame_index=int(tensors["frame_indices"][future_offset].item()),
    )
