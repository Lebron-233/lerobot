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

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from examples.advanced.predictive_async.future_latent_cache import (
    CACHE_SCHEMA_VERSION,
    MAX_PREDICTION_DELAY,
    FutureLatentPair,
    build_future_latent_pair,
    load_cache_manifest,
    load_episode_cache,
    validate_episode_cache,
)

_CLASSIFICATION = "offline_future_latent_cache_not_task_capability"
_CAMERAS = ("observation.images.camera1", "observation.images.camera2")


def _episode_tensors(frame_count: int = 4) -> dict[str, torch.Tensor]:
    token_dim = 4
    camera_0_tokens = torch.arange(frame_count * 2 * token_dim, dtype=torch.float32).reshape(
        frame_count, 2, token_dim
    )
    camera_1_tokens = 1000 + torch.arange(frame_count * 3 * token_dim, dtype=torch.float32).reshape(
        frame_count, 3, token_dim
    )
    return {
        "dataset_indices": torch.arange(100, 100 + frame_count, dtype=torch.int64),
        "frame_indices": torch.arange(frame_count, dtype=torch.int64),
        "states": torch.arange(frame_count * 5, dtype=torch.float32).reshape(frame_count, 5),
        "actions": torch.arange(frame_count * 2, dtype=torch.float32).reshape(frame_count, 2),
        "language_tokens": torch.arange(frame_count * 3, dtype=torch.int64).reshape(frame_count, 3),
        "language_attention_mask": torch.ones(frame_count, 3, dtype=torch.bool),
        "image_tokens_0": camera_0_tokens.to(torch.bfloat16),
        "image_token_masks_0": torch.ones(frame_count, 2, dtype=torch.bool),
        "image_tokens_1": camera_1_tokens.to(torch.bfloat16),
        "image_token_masks_1": torch.ones(frame_count, 3, dtype=torch.bool),
    }


def _tensor_metadata(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "storage_device": "cpu",
        }
        for key, value in tensors.items()
    }


def _manifest(
    tensors: dict[str, torch.Tensor],
    *,
    complete_split: bool = False,
) -> dict[str, object]:
    frame_count = tensors["frame_indices"].shape[0]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "classification": _CLASSIFICATION,
        "producer": {
            "git_sha": "a" * 40,
            "command": ["cache_smolvla_latents.py", "--split", "val"],
            "created_at_utc": "2026-09-05T00:00:00+00:00",
        },
        "inputs": {
            "dataset": {
                "repo_id": "lerobot/svla_so100_pickplace",
                "requested_revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
                "resolved_revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
            },
            "checkpoint": {
                "repo_id": "lerobot/smolvla_base",
                "requested_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
                "resolved_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
            },
            "vlm": {
                "repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "requested_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                "resolved_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            },
        },
        "split": "val",
        "authoritative_episode_ids": [7],
        "cached_episode_ids": [7],
        "complete_split": complete_split,
        "fps": 30,
        "episode_count": 1,
        "frame_count": frame_count,
        "episodes": [
            {
                "episode_index": 7,
                "frame_count": frame_count,
                "shard": "episodes/episode_000007.safetensors",
                "tensor_metadata": _tensor_metadata(tensors),
            }
        ],
        "valid_pair_count_by_delay": {
            str(delay): max(frame_count - delay, 0) for delay in range(1, MAX_PREDICTION_DELAY + 1)
        },
        "camera_mapping": {
            "observation.images.top": _CAMERAS[0],
            "observation.images.wrist": _CAMERAS[1],
        },
        "policy_camera_order": list(_CAMERAS),
        "token_scaling_convention": "native_post_sqrt_hidden_dim",
        "storage_device": "cpu",
        "semantics": {
            "state": "model_ready_normalized_and_padded",
            "action": "normalized_policy_output_original_action_dim",
            "processor_config_source": ("lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205"),
        },
        "extraction_device": {"type": "cuda", "index": 0, "name": "test-device"},
        "software_versions": {
            "python": "test",
            "torch": torch.__version__,
            "transformers": "test",
            "datasets": "test",
            "lerobot": "test",
            "safetensors": "test",
        },
    }


def _episode_entry(manifest: dict[str, object]) -> dict[str, object]:
    return manifest["episodes"][0]


def test_manifest_and_episode_shard_round_trip_without_casting(tmp_path: Path) -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors, complete_split=False)
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    save_file(tensors, episode_dir / "episode_000007.safetensors")

    loaded_manifest = load_cache_manifest(tmp_path)
    loaded_tensors = load_episode_cache(tmp_path, episode_index=7)
    validate_episode_cache(loaded_manifest, _episode_entry(loaded_manifest), loaded_tensors)

    assert loaded_manifest["schema_version"] == 1
    assert loaded_manifest["complete_split"] is False
    assert tuple(loaded_manifest["policy_camera_order"]) == _CAMERAS
    assert loaded_tensors.keys() == tensors.keys()
    for key, expected in tensors.items():
        assert torch.equal(loaded_tensors[key], expected)
        assert loaded_tensors[key].dtype == expected.dtype
        assert loaded_tensors[key].device.type == "cpu"
        assert loaded_tensors[key].is_contiguous()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest.__setitem__("schema_version", 2), "schema_version"),
        (lambda manifest: manifest.__setitem__("classification", "task_capability"), "classification"),
        (lambda manifest: manifest.__setitem__("complete_split", "false"), "complete_split"),
        (
            lambda manifest: manifest.__setitem__(
                "valid_pair_count_by_delay", {str(delay): 0 for delay in range(1, 9)}
            ),
            "valid_pair_count_by_delay",
        ),
    ],
)
def test_manifest_rejects_a_wrong_schema_or_pair_count(tmp_path: Path, mutation, match: str) -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors)
    mutation(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises((TypeError, ValueError), match=match):
        load_cache_manifest(tmp_path)


def test_partial_manifest_allows_an_ordered_cached_episode_subset(tmp_path: Path) -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors, complete_split=False)
    manifest["authoritative_episode_ids"] = [5, 7]
    manifest["cached_episode_ids"] = [5]
    manifest["episodes"][0]["episode_index"] = 5
    manifest["episodes"][0]["shard"] = "episodes/episode_000005.safetensors"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_cache_manifest(tmp_path)
    assert loaded["authoritative_episode_ids"] == [5, 7]
    assert loaded["cached_episode_ids"] == [5]
    assert loaded["complete_split"] is False

    manifest["complete_split"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete_split"):
        load_cache_manifest(tmp_path)


def test_validate_episode_cache_checks_tensor_contract_and_continuity() -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors)
    validate_episode_cache(manifest, _episode_entry(manifest), tensors)

    noncontiguous = deepcopy(tensors)
    noncontiguous["image_tokens_0"] = (
        noncontiguous["image_tokens_0"].transpose(1, 2).contiguous().transpose(1, 2)
    )
    assert not noncontiguous["image_tokens_0"].is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        validate_episode_cache(manifest, _episode_entry(manifest), noncontiguous)

    discontinuous_frames = deepcopy(tensors)
    discontinuous_frames["frame_indices"] = torch.tensor([0, 1, 3, 4])
    with pytest.raises(ValueError, match="frame_indices"):
        validate_episode_cache(manifest, _episode_entry(manifest), discontinuous_frames)

    discontinuous_dataset = deepcopy(tensors)
    discontinuous_dataset["dataset_indices"] = torch.tensor([100, 101, 103, 104])
    with pytest.raises(ValueError, match="dataset_indices"):
        validate_episode_cache(manifest, _episode_entry(manifest), discontinuous_dataset)


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("states", lambda value: value.to(torch.float64), "states.*dtype"),
        ("actions", lambda value: value[:, :-1], "actions.*shape"),
        (
            "image_token_masks_0",
            lambda value: value.to(torch.int64),
            "image_token_masks_0.*dtype",
        ),
        ("language_tokens", lambda value: value.to(torch.int32), "language_tokens.*dtype"),
        ("states", lambda value: value.to("meta"), "states.*CPU"),
    ],
)
def test_validate_episode_cache_rejects_wrong_shape_dtype_or_device(
    key: str,
    replacement,
    match: str,
) -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors)
    tensors[key] = replacement(tensors[key])

    with pytest.raises((TypeError, ValueError), match=match):
        validate_episode_cache(manifest, _episode_entry(manifest), tensors)


def test_only_valid_tokens_and_all_state_action_values_must_be_finite() -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors)

    tensors["image_token_masks_0"][0, 0] = False
    tensors["image_tokens_0"][0, 0, 0] = torch.nan
    validate_episode_cache(manifest, _episode_entry(manifest), tensors)

    tensors["image_token_masks_0"][0, 0] = True
    with pytest.raises(ValueError, match="image_tokens_0.*finite"):
        validate_episode_cache(manifest, _episode_entry(manifest), tensors)

    tensors = _episode_tensors()
    tensors["actions"][1, 0] = torch.inf
    with pytest.raises(ValueError, match="actions.*finite"):
        validate_episode_cache(manifest, _episode_entry(manifest), tensors)


def test_build_pair_uses_camera_order_action_prefix_and_t_plus_delay_targets() -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors)

    pair = build_future_latent_pair(
        manifest,
        _episode_entry(manifest),
        tensors,
        frame_offset=1,
        delay_steps=2,
    )

    assert isinstance(pair, FutureLatentPair)
    assert pair.episode_index == 7
    assert pair.frame_index == 1
    assert pair.future_frame_index == 3
    assert pair.delay_steps.shape == ()
    assert pair.delay_steps.dtype == torch.int64
    assert pair.delay_steps.item() == 2

    assert len(pair.current_image_tokens) == len(pair.target_image_tokens) == 2
    torch.testing.assert_close(pair.current_image_tokens[0], tensors["image_tokens_0"][1])
    torch.testing.assert_close(pair.current_image_tokens[1], tensors["image_tokens_1"][1])
    torch.testing.assert_close(pair.target_image_tokens[0], tensors["image_tokens_0"][3])
    torch.testing.assert_close(pair.target_image_tokens[1], tensors["image_tokens_1"][3])
    torch.testing.assert_close(pair.current_image_token_masks[0], tensors["image_token_masks_0"][1])
    torch.testing.assert_close(pair.target_image_token_masks[1], tensors["image_token_masks_1"][3])

    assert pair.committed_actions.shape == (MAX_PREDICTION_DELAY, tensors["actions"].shape[1])
    torch.testing.assert_close(pair.committed_actions[:2], tensors["actions"][1:3])
    assert torch.count_nonzero(pair.committed_actions[2:]) == 0
    torch.testing.assert_close(
        pair.committed_mask,
        torch.tensor([True, True, False, False, False, False, False, False]),
    )
    torch.testing.assert_close(pair.current_state, tensors["states"][1])
    torch.testing.assert_close(pair.future_state, tensors["states"][3])
    torch.testing.assert_close(pair.current_language_tokens, tensors["language_tokens"][1])
    torch.testing.assert_close(pair.future_language_tokens, tensors["language_tokens"][3])
    torch.testing.assert_close(pair.current_language_attention_mask, tensors["language_attention_mask"][1])
    torch.testing.assert_close(pair.future_language_attention_mask, tensors["language_attention_mask"][3])


@pytest.mark.parametrize(
    ("frame_offset", "delay_steps", "match"),
    [
        (0, 0, "delay_steps"),
        (0, MAX_PREDICTION_DELAY + 1, "delay_steps"),
        (3, 1, "episode"),
    ],
)
def test_build_pair_rejects_delay_or_episode_boundary_crossing(
    frame_offset: int,
    delay_steps: int,
    match: str,
) -> None:
    tensors = _episode_tensors()
    manifest = _manifest(tensors)

    with pytest.raises(ValueError, match=match):
        build_future_latent_pair(
            manifest,
            _episode_entry(manifest),
            tensors,
            frame_offset=frame_offset,
            delay_steps=delay_steps,
        )
