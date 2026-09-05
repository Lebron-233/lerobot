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

import importlib
import json
import math
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from examples.advanced.predictive_async import future_latent_training as training
from examples.advanced.predictive_async.future_latent_cache import FutureLatentPair
from examples.advanced.predictive_async.future_latent_training import (
    CACHE_PRODUCER_SHA,
    FutureLatentBatch,
    FutureLatentCacheDataset,
    SelectionState,
    accumulation_windows,
    bounded_run_markers,
    capture_rng_state,
    collate_future_latent_pairs,
    compute_future_latent_objective,
    deterministic_train_indices,
    deterministic_val_indices_by_delay,
    forward_predictor,
    load_last_checkpoint,
    make_predictor_optimizer,
    optimizer_step,
    restore_rng_state,
    save_predictor_checkpoint,
    update_selection,
)
from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import FutureLatentPrediction, LightweightFutureLatentPredictor

_CLASSIFICATION = "offline_future_latent_cache_not_task_capability"
_DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
_CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
_VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
_CAMERAS = ("observation.images.camera1", "observation.images.camera2")
_TRAIN_EPISODES = (
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
)
_VAL_EPISODES = (5, 7, 14, 39, 49)
_SPLIT_FRAME_COUNTS = {"train": 15_576, "val": 1_822}


def _episode_lengths(split: str) -> list[int]:
    episode_ids = _TRAIN_EPISODES if split == "train" else _VAL_EPISODES
    frame_count = _SPLIT_FRAME_COUNTS[split]
    quotient, remainder = divmod(frame_count, len(episode_ids))
    return [quotient + (position < remainder) for position in range(len(episode_ids))]


def _tensor_metadata(frame_count: int) -> dict[str, dict[str, object]]:
    shapes_and_dtypes = {
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
    return {
        key: {"shape": shape, "dtype": dtype, "storage_device": "cpu"}
        for key, (shape, dtype) in shapes_and_dtypes.items()
    }


def _complete_manifest(split: str) -> dict[str, object]:
    episode_ids = _TRAIN_EPISODES if split == "train" else _VAL_EPISODES
    lengths = _episode_lengths(split)
    episodes = [
        {
            "episode_index": episode_index,
            "frame_count": frame_count,
            "shard": f"episodes/episode_{episode_index:06d}.safetensors",
            "tensor_metadata": _tensor_metadata(frame_count),
        }
        for episode_index, frame_count in zip(episode_ids, lengths, strict=True)
    ]
    return {
        "schema_version": 1,
        "classification": _CLASSIFICATION,
        "producer": {
            "git_sha": CACHE_PRODUCER_SHA,
            "command": ["cache_smolvla_latents.py", "--split", split],
            "created_at_utc": "2026-09-05T00:00:00+00:00",
        },
        "inputs": {
            "dataset": {
                "repo_id": "lerobot/svla_so100_pickplace",
                "requested_revision": _DATASET_REVISION,
                "resolved_revision": _DATASET_REVISION,
            },
            "checkpoint": {
                "repo_id": "lerobot/smolvla_base",
                "requested_revision": _CHECKPOINT_REVISION,
                "resolved_revision": _CHECKPOINT_REVISION,
            },
            "vlm": {
                "repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "requested_revision": _VLM_REVISION,
                "resolved_revision": _VLM_REVISION,
            },
        },
        "split": split,
        "authoritative_episode_ids": list(episode_ids),
        "cached_episode_ids": list(episode_ids),
        "complete_split": True,
        "fps": 30,
        "episode_count": len(episodes),
        "frame_count": sum(lengths),
        "episodes": episodes,
        "valid_pair_count_by_delay": {
            str(delay): sum(frame_count - delay for frame_count in lengths) for delay in range(1, 9)
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
            "processor_config_source": f"lerobot/smolvla_base@{_CHECKPOINT_REVISION}",
        },
        "extraction_device": {"type": "cuda", "index": 0, "name": "synthetic"},
        "software_versions": {
            "python": "synthetic",
            "torch": torch.__version__,
            "transformers": "synthetic",
            "datasets": "synthetic",
            "lerobot": "synthetic",
            "safetensors": "synthetic",
        },
    }


def _write_manifest(cache_dir: Path, manifest: dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))


def _pair(sample: int, *, delay_steps: int, token_dim: int = 4) -> FutureLatentPair:
    camera_0 = torch.arange(2 * token_dim, dtype=torch.float32).reshape(2, token_dim) + sample * 100
    camera_1 = torch.arange(3 * token_dim, dtype=torch.float32).reshape(3, token_dim) + sample * 1000
    committed_mask = torch.arange(8) < delay_steps
    committed_actions = torch.zeros(8, 6)
    committed_actions[:delay_steps] = sample + torch.arange(delay_steps * 6).reshape(delay_steps, 6) / 10
    return FutureLatentPair(
        current_image_tokens=(camera_0.to(torch.bfloat16), camera_1.to(torch.bfloat16)),
        current_image_token_masks=(torch.tensor([True, sample % 2 == 0]), torch.ones(3, dtype=torch.bool)),
        committed_actions=committed_actions,
        committed_mask=committed_mask,
        current_state=torch.arange(32, dtype=torch.float32) + sample,
        delay_steps=torch.tensor(delay_steps, dtype=torch.int64),
        target_image_tokens=((camera_0 + 1).to(torch.bfloat16), (camera_1 + 2).to(torch.bfloat16)),
        target_image_token_masks=(torch.ones(2, dtype=torch.bool), torch.tensor([True, True, False])),
        future_state=torch.arange(32, dtype=torch.float32) + sample + 1,
        current_language_tokens=torch.tensor([1, 2, 3]),
        current_language_attention_mask=torch.tensor([True, True, False]),
        future_language_tokens=torch.tensor([2, 3, 4]),
        future_language_attention_mask=torch.tensor([True, True, True]),
        episode_index=sample,
        frame_index=sample * 10,
        future_frame_index=sample * 10 + delay_steps,
    )


def _small_episode_tensors(frame_count: int = 3) -> dict[str, torch.Tensor]:
    return {
        "dataset_indices": torch.arange(100, 100 + frame_count, dtype=torch.int64),
        "frame_indices": torch.arange(frame_count, dtype=torch.int64),
        "states": torch.arange(frame_count * 32, dtype=torch.float32).reshape(frame_count, 32),
        "actions": torch.arange(frame_count * 6, dtype=torch.float32).reshape(frame_count, 6),
        "language_tokens": torch.ones(frame_count, 48, dtype=torch.int64),
        "language_attention_mask": torch.ones(frame_count, 48, dtype=torch.bool),
        "image_tokens_0": torch.arange(frame_count * 2 * 4, dtype=torch.float32)
        .reshape(frame_count, 2, 4)
        .to(torch.bfloat16),
        "image_token_masks_0": torch.ones(frame_count, 2, dtype=torch.bool),
        "image_tokens_1": torch.arange(frame_count * 3 * 4, dtype=torch.float32)
        .reshape(frame_count, 3, 4)
        .to(torch.bfloat16),
        "image_token_masks_1": torch.ones(frame_count, 3, dtype=torch.bool),
    }


def _tiny_config() -> FutureLatentConfig:
    return FutureLatentConfig(
        token_dim=4,
        action_dim=6,
        state_dim=32,
        enabled=True,
        rank=2,
        action_hidden_dim=4,
        state_hidden_dim=4,
        delay_embedding_dim=2,
        fusion_hidden_dim=4,
        max_prediction_delay=8,
        max_cameras=2,
        token_mixer="none",
        risk_head=True,
    )


def _tiny_batch(batch_size: int = 4) -> FutureLatentBatch:
    return collate_future_latent_pairs(
        [_pair(sample, delay_steps=sample % 3 + 1) for sample in range(batch_size)]
    )


def _slice_batch(batch: FutureLatentBatch, start: int, stop: int) -> FutureLatentBatch:
    return FutureLatentBatch(
        current_image_tokens=tuple(tokens[start:stop] for tokens in batch.current_image_tokens),
        current_image_token_masks=tuple(mask[start:stop] for mask in batch.current_image_token_masks),
        committed_actions=batch.committed_actions[start:stop],
        committed_mask=batch.committed_mask[start:stop],
        current_state=batch.current_state[start:stop],
        delay_steps=batch.delay_steps[start:stop],
        target_image_tokens=tuple(tokens[start:stop] for tokens in batch.target_image_tokens),
        target_image_token_masks=tuple(mask[start:stop] for mask in batch.target_image_token_masks),
    )


def test_complete_frozen_train_and_val_manifests_have_the_canonical_pair_counts(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_manifest(train_dir, _complete_manifest("train"))
    _write_manifest(val_dir, _complete_manifest("val"))

    train = FutureLatentCacheDataset(train_dir, expected_split="train")
    val = FutureLatentCacheDataset(val_dir, expected_split="val")

    assert len(train) == 123_168
    assert len(val) == 14_396
    assert [(spec.episode_position, spec.frame_offset, spec.delay_steps) for spec in val.pair_specs[:10]] == [
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 0, 4),
        (0, 0, 5),
        (0, 0, 6),
        (0, 0, 7),
        (0, 0, 8),
        (0, 1, 1),
        (0, 1, 2),
    ]
    first_episode_frames = _episode_lengths("val")[0]
    first_episode_pair_count = sum(first_episode_frames - delay for delay in range(1, 9))
    before_boundary = val.pair_specs[first_episode_pair_count - 1]
    after_boundary = val.pair_specs[first_episode_pair_count]
    assert (
        before_boundary.episode_position,
        before_boundary.frame_offset,
        before_boundary.delay_steps,
    ) == (0, first_episode_frames - 2, 1)
    assert (
        after_boundary.episode_position,
        after_boundary.frame_offset,
        after_boundary.delay_steps,
    ) == (1, 0, 1)


@pytest.mark.parametrize(
    ("mutation", "expected_split", "match"),
    [
        (lambda manifest: manifest.__setitem__("complete_split", False), "val", "complete_split"),
        (lambda manifest: manifest.__setitem__("split", "test"), "val", "split"),
        (lambda manifest: manifest["producer"].__setitem__("git_sha", "0" * 40), "val", "producer"),
        (
            lambda manifest: manifest["inputs"]["dataset"].__setitem__("resolved_revision", "0" * 40),
            "val",
            "dataset",
        ),
        (
            lambda manifest: manifest["inputs"]["checkpoint"].__setitem__("requested_revision", "0" * 40),
            "val",
            "checkpoint",
        ),
        (
            lambda manifest: manifest["inputs"]["vlm"].__setitem__("resolved_revision", "0" * 40),
            "val",
            "vlm",
        ),
        (lambda manifest: None, "train", "split"),
    ],
)
def test_dataset_rejects_partial_test_wrong_split_producer_or_pins(
    tmp_path: Path, mutation, expected_split: str, match: str
) -> None:
    manifest = _complete_manifest("val")
    mutation(manifest)
    _write_manifest(tmp_path, manifest)

    with pytest.raises((TypeError, ValueError), match=match):
        FutureLatentCacheDataset(tmp_path, expected_split=expected_split)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest["policy_camera_order"].reverse(), "camera"),
        (
            lambda manifest: manifest["episodes"][0]["tensor_metadata"]["image_tokens_0"][
                "shape"
            ].__setitem__(2, 959),
            "token",
        ),
        (
            lambda manifest: manifest["episodes"][0]["tensor_metadata"]["actions"]["shape"].__setitem__(1, 7),
            "action",
        ),
        (lambda manifest: manifest["semantics"].__setitem__("state", "raw"), "state"),
    ],
)
def test_dataset_rejects_camera_tensor_or_semantic_drift(tmp_path: Path, mutation, match: str) -> None:
    manifest = _complete_manifest("val")
    mutation(manifest)
    _write_manifest(tmp_path, manifest)

    with pytest.raises((TypeError, ValueError), match=match):
        FutureLatentCacheDataset(tmp_path, expected_split="val")


def test_train_shuffle_and_bounded_val_subset_are_deterministic(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_manifest(train_dir, _complete_manifest("train"))
    _write_manifest(val_dir, _complete_manifest("val"))
    train = FutureLatentCacheDataset(train_dir, expected_split="train")
    val = FutureLatentCacheDataset(val_dir, expected_split="val")

    epoch_0 = deterministic_train_indices(len(train), seed=0, epoch=0)
    assert epoch_0 == deterministic_train_indices(len(train), seed=0, epoch=0)
    assert epoch_0 != deterministic_train_indices(len(train), seed=0, epoch=1)
    assert sorted(epoch_0) == list(range(len(train)))

    bounded = deterministic_val_indices_by_delay(val, max_pairs_per_delay=2)
    expected = tuple(index for delay in range(1, 9) for index in val.indices_for_delay(delay)[:2])
    assert bounded == expected
    bounded_delays = [val.pair_specs[index].delay_steps for index in bounded]
    assert bounded_delays.count(1) == 2
    assert all(bounded_delays.count(delay) == 2 for delay in range(1, 9))
    assert deterministic_val_indices_by_delay(val, max_pairs_per_delay=None) == tuple(range(len(val)))


def test_dataset_uses_the_b1_pair_builder_and_caches_each_loaded_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, _complete_manifest("val"))
    tensors = _small_episode_tensors()
    calls = {"load": 0, "validate": 0, "build": 0}
    b1_build_pair = training.build_future_latent_pair

    def load_episode(cache_dir: Path, episode_index: int) -> dict[str, torch.Tensor]:
        assert cache_dir == tmp_path
        assert episode_index == _VAL_EPISODES[0]
        calls["load"] += 1
        return tensors

    def validate_episode(manifest, episode_entry, loaded_tensors) -> None:
        assert loaded_tensors is tensors
        calls["validate"] += 1

    def build_pair(manifest, episode_entry, loaded_tensors, *, frame_offset: int, delay_steps: int):
        calls["build"] += 1
        return b1_build_pair(
            manifest,
            episode_entry,
            loaded_tensors,
            frame_offset=frame_offset,
            delay_steps=delay_steps,
        )

    monkeypatch.setattr(training, "load_episode_cache", load_episode)
    monkeypatch.setattr(training, "validate_episode_cache", validate_episode)
    monkeypatch.setattr(training, "build_future_latent_pair", build_pair)
    dataset = FutureLatentCacheDataset(tmp_path, expected_split="val")

    pair_at_d1 = dataset[0]
    pair_at_d2 = dataset[1]

    assert pair_at_d1.future_frame_index == 1
    assert pair_at_d2.future_frame_index == 2
    assert calls == {"load": 1, "validate": 1, "build": 2}


def test_collate_preserves_camera_order_and_predictor_input_contract() -> None:
    pairs = [_pair(0, delay_steps=1), _pair(1, delay_steps=3)]

    batch = collate_future_latent_pairs(pairs)

    assert [tokens.shape for tokens in batch.current_image_tokens] == [(2, 2, 4), (2, 3, 4)]
    assert [tokens.shape for tokens in batch.target_image_tokens] == [(2, 2, 4), (2, 3, 4)]
    assert all(tokens.dtype == torch.bfloat16 for tokens in batch.current_image_tokens)
    assert all(mask.dtype == torch.bool for mask in batch.current_image_token_masks)
    assert batch.committed_actions.shape == (2, 8, 6)
    assert batch.committed_actions.dtype == torch.float32
    assert batch.committed_mask.shape == (2, 8)
    assert batch.committed_mask.dtype == torch.bool
    assert batch.current_state.shape == (2, 32)
    assert batch.current_state.dtype == torch.float32
    assert batch.delay_steps.dtype == torch.int64
    torch.testing.assert_close(batch.current_image_tokens[0][1], pairs[1].current_image_tokens[0])
    torch.testing.assert_close(batch.current_image_tokens[1][0], pairs[0].current_image_tokens[1])


def _hand_computed_objective_inputs() -> tuple[FutureLatentPrediction, FutureLatentBatch]:
    current_0 = torch.tensor([[[1.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [0.0, 2.0]]], dtype=torch.bfloat16)
    current_1 = torch.tensor([[[1.0, 1.0]], [[1.0, -1.0]]], dtype=torch.bfloat16)
    target_0 = torch.tensor(
        [[[2.0, 0.0], [10_000.0, 10_000.0]], [[0.0, 3.0], [10_000.0, 10_000.0]]],
        dtype=torch.bfloat16,
    )
    target_1 = torch.tensor([[[1.0, 3.0]], [[10_000.0, 10_000.0]]], dtype=torch.bfloat16)
    current_masks = (torch.tensor([[True, True], [True, False]]), torch.tensor([[True], [True]]))
    target_masks = (torch.tensor([[True, False], [True, False]]), torch.tensor([[True], [False]]))
    batch = FutureLatentBatch(
        current_image_tokens=(current_0, current_1),
        current_image_token_masks=current_masks,
        committed_actions=torch.zeros(2, 8, 6),
        committed_mask=torch.tensor([[True, False, False, False, False, False, False, False]] * 2),
        current_state=torch.zeros(2, 32),
        delay_steps=torch.ones(2, dtype=torch.int64),
        target_image_tokens=(target_0, target_1),
        target_image_token_masks=target_masks,
    )
    prediction = FutureLatentPrediction(
        delta_tokens=(
            torch.tensor(
                [[[1.0, 0.0], [99.0, 99.0]], [[0.0, 1.0], [99.0, 99.0]]],
                requires_grad=True,
            ),
            torch.tensor([[[0.0, 1.0]], [[99.0, 99.0]]], requires_grad=True),
        ),
        predicted_error=torch.tensor([0.0, 1.5], requires_grad=True),
    )
    return prediction, batch


def test_objective_matches_hand_computed_float32_masked_losses() -> None:
    prediction, batch = _hand_computed_objective_inputs()

    objective = compute_future_latent_objective(prediction, batch, lambda_cos=0.1, lambda_risk=0.1)

    cosine_error = 1.0 - 7.0 / math.sqrt(50.0)
    expected_cosine = cosine_error / 3.0
    expected_risk = (0.5 * 0.25**2 + 0.5) / 2.0
    assert objective.latent_smoothl1.dtype == torch.float32
    assert objective.cosine.dtype == torch.float32
    assert objective.risk_smoothl1.dtype == torch.float32
    torch.testing.assert_close(objective.latent_smoothl1, torch.tensor(1.0 / 6.0))
    torch.testing.assert_close(objective.cosine, torch.tensor(expected_cosine))
    torch.testing.assert_close(objective.risk_smoothl1, torch.tensor(expected_risk))
    torch.testing.assert_close(objective.per_sample_smoothl1, torch.tensor([1.0 / 8.0, 1.0 / 4.0]))
    torch.testing.assert_close(objective.per_sample_mse, torch.tensor([1.0 / 4.0, 1.0 / 2.0]))
    torch.testing.assert_close(objective.per_sample_cosine, torch.tensor([cosine_error / 2.0, 0.0]))
    torch.testing.assert_close(objective.risk_target, torch.tensor([1.0 / 4.0, 1.0 / 2.0]))
    torch.testing.assert_close(
        objective.total,
        torch.tensor(1.0 / 6.0 + 0.1 * expected_cosine + 0.1 * expected_risk),
    )


def test_objective_rejects_a_sample_with_no_jointly_valid_tokens() -> None:
    prediction, batch = _hand_computed_objective_inputs()
    empty_for_second_sample = tuple(mask.clone() for mask in batch.target_image_token_masks)
    for mask in empty_for_second_sample:
        mask[1] = False
    batch = replace(batch, target_image_token_masks=empty_for_second_sample)

    with pytest.raises(ValueError, match="token valid"):
        compute_future_latent_objective(prediction, batch, lambda_cos=0.1, lambda_risk=0.1)


def test_objective_ignores_non_finite_values_outside_the_joint_mask() -> None:
    clean_prediction, clean_batch = _hand_computed_objective_inputs()
    expected = compute_future_latent_objective(clean_prediction, clean_batch, lambda_cos=0.1, lambda_risk=0.1)
    prediction, batch = _hand_computed_objective_inputs()
    with torch.no_grad():
        for current, current_mask, delta, target, target_mask in zip(
            batch.current_image_tokens,
            batch.current_image_token_masks,
            prediction.delta_tokens,
            batch.target_image_tokens,
            batch.target_image_token_masks,
            strict=True,
        ):
            invalid = ~(current_mask & target_mask)
            current[invalid] = torch.nan
            delta[invalid] = torch.nan
            target[invalid] = torch.nan

    actual = compute_future_latent_objective(prediction, batch, lambda_cos=0.1, lambda_risk=0.1)

    for field in (
        "total",
        "latent_smoothl1",
        "cosine",
        "risk_smoothl1",
        "per_sample_smoothl1",
        "per_sample_mse",
        "per_sample_cosine",
        "risk_target",
    ):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))


def test_risk_target_is_detached_from_residual_prediction_path() -> None:
    torch.manual_seed(0)
    predictor = LightweightFutureLatentPredictor(_tiny_config())
    batch = _tiny_batch(2)
    prediction = forward_predictor(predictor, batch)
    prediction.delta_tokens[0].retain_grad()
    objective = compute_future_latent_objective(prediction, batch, lambda_cos=0.1, lambda_risk=0.1)

    assert objective.risk_target.requires_grad is False
    objective.risk_smoothl1.backward()

    assert prediction.delta_tokens[0].grad is None
    assert predictor.up_projection.weight.grad is None
    assert predictor.up_projection.bias.grad is None
    assert predictor.risk_head is not None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in predictor.risk_head.parameters()
    )


def test_cpu_optimizer_performs_two_clipped_predictor_only_updates() -> None:
    torch.manual_seed(0)
    predictor = LightweightFutureLatentPredictor(_tiny_config())
    optimizer = make_predictor_optimizer(predictor)
    trainable = [parameter for parameter in predictor.parameters() if parameter.requires_grad]
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    before = [parameter.detach().clone() for parameter in trainable]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert {id(parameter) for parameter in optimizer_parameters} == {id(parameter) for parameter in trainable}
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in trainable)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    batch = _tiny_batch(4)
    results = [
        optimizer_step(
            predictor,
            optimizer,
            [_slice_batch(batch, 0, 2), _slice_batch(batch, 2, 4)],
            grad_clip_norm=1e-4,
        )
        for _ in range(2)
    ]

    metric_names = (
        "total",
        "latent_smoothl1",
        "cosine",
        "risk_smoothl1",
        "pre_clip_grad_norm",
        "post_clip_grad_norm",
        "parameter_delta",
    )
    assert all(math.isfinite(getattr(result, name)) for result in results for name in metric_names)
    assert all(result.parameter_delta > 0 for result in results)
    assert all(result.pre_clip_grad_norm > result.post_clip_grad_norm for result in results)
    assert all(result.post_clip_grad_norm <= 1.01e-4 for result in results)
    assert all(
        (result.sample_count, result.micro_batch_count, result.optimizer_step_count) == (4, 2, 1)
        for result in results
    )
    assert any(
        not torch.equal(previous, parameter) for previous, parameter in zip(before, trainable, strict=True)
    )
    optimizer_state_steps = [int(state["step"].item()) for state in optimizer.state.values()]
    assert optimizer_state_steps and set(optimizer_state_steps) == {2}


def test_gradient_accumulation_matches_one_effective_batch_and_keeps_the_remainder() -> None:
    assert accumulation_windows(10, micro_batch_size=4, effective_batch_size=8) == ((0, 8), (8, 10))
    assert (
        sum(
            stop - start
            for start, stop in accumulation_windows(10, micro_batch_size=4, effective_batch_size=8)
        )
        == 10
    )

    torch.manual_seed(0)
    single_batch_predictor = LightweightFutureLatentPredictor(_tiny_config())
    micro_batch_predictor = deepcopy(single_batch_predictor)
    single_batch_optimizer = make_predictor_optimizer(
        single_batch_predictor, learning_rate=1e-3, weight_decay=0
    )
    micro_batch_optimizer = make_predictor_optimizer(
        micro_batch_predictor, learning_rate=1e-3, weight_decay=0
    )
    batch = _tiny_batch(5)

    single_result = optimizer_step(
        single_batch_predictor,
        single_batch_optimizer,
        [batch],
        grad_clip_norm=1_000.0,
    )
    micro_result = optimizer_step(
        micro_batch_predictor,
        micro_batch_optimizer,
        [_slice_batch(batch, 0, 2), _slice_batch(batch, 2, 5)],
        grad_clip_norm=1_000.0,
    )

    assert single_result.sample_count == micro_result.sample_count == 5
    assert single_result.micro_batch_count == 1
    assert micro_result.micro_batch_count == 2
    assert single_result.optimizer_step_count == micro_result.optimizer_step_count == 1
    torch.testing.assert_close(torch.tensor(single_result.total), torch.tensor(micro_result.total))
    for single_parameter, micro_parameter in zip(
        single_batch_predictor.parameters(), micro_batch_predictor.parameters(), strict=True
    ):
        torch.testing.assert_close(single_parameter, micro_parameter, rtol=1e-5, atol=1e-6)


def test_bounded_run_markers_forbid_protocol_selection_and_test_claims() -> None:
    assert bounded_run_markers(True) == {
        "run_kind": "bounded_smoke",
        "protocol_complete": False,
        "eligible_for_checkpoint_selection": False,
        "eligible_for_test": False,
    }


def test_resume_rejects_cross_protocol_checkpoint_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_dir = Path(__file__).resolve().parents[3] / "examples" / "advanced" / "predictive_async"
    monkeypatch.syspath_prepend(str(script_dir))
    cli = importlib.import_module("examples.advanced.predictive_async.train_future_latent")

    payload = {"train_config": bounded_run_markers(True)}
    run_config = bounded_run_markers(False)

    with pytest.raises(ValueError, match="run_kind"):
        cli._validate_resume_payload(payload, run_config=run_config)


def test_checkpoint_selection_ties_and_early_stopping_follow_the_frozen_rule() -> None:
    state = SelectionState()
    state, is_best, should_stop = update_selection(state, epoch=0, val_macro_smoothl1=1.0, val_macro_mse=0.5)
    assert is_best is True
    assert should_stop is False

    state, is_best, should_stop = update_selection(state, epoch=1, val_macro_smoothl1=1.0, val_macro_mse=0.4)
    assert is_best is True
    assert should_stop is False
    state, is_best, should_stop = update_selection(state, epoch=2, val_macro_smoothl1=1.0, val_macro_mse=0.4)
    assert is_best is False
    assert should_stop is False

    state = SelectionState()
    state, _, _ = update_selection(state, epoch=0, val_macro_smoothl1=1.0, val_macro_mse=0.5)
    for epoch in range(1, 6):
        state, _, should_stop = update_selection(
            state, epoch=epoch, val_macro_smoothl1=1.0, val_macro_mse=0.5
        )
        assert should_stop is (epoch == 5)

    state = SelectionState()
    state, _, _ = update_selection(state, epoch=0, val_macro_smoothl1=1.0, val_macro_mse=0.5)
    state, is_best, should_stop = update_selection(
        state, epoch=1, val_macro_smoothl1=0.9995, val_macro_mse=0.5
    )
    assert is_best is True
    assert should_stop is False
    for epoch in range(2, 6):
        state, _, should_stop = update_selection(
            state, epoch=epoch, val_macro_smoothl1=0.9995, val_macro_mse=0.5
        )
    assert should_stop is True


def test_best_and_last_checkpoints_reload_predictor_optimizer_step_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(0)
    predictor = LightweightFutureLatentPredictor(_tiny_config())
    optimizer = make_predictor_optimizer(predictor)
    optimizer_step(predictor, optimizer, [_tiny_batch(2)])
    expected_parameters = {key: value.detach().clone() for key, value in predictor.state_dict().items()}
    train_config = {"learning_rate": 3e-4, "weight_decay": 1e-4, "seed": 0}
    best_metrics = {"smoothl1": 0.5, "mse": 0.75, "epoch": 0}
    cache_provenance = {"train": {"split": "train"}, "val": {"split": "val"}}
    state, _, _ = update_selection(SelectionState(), epoch=0, val_macro_smoothl1=0.5, val_macro_mse=0.75)
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"
    save_predictor_checkpoint(
        best_path,
        predictor=predictor,
        optimizer=None,
        train_config=train_config,
        epoch=0,
        global_step=1,
        best_val_metrics=best_metrics,
        cache_provenance=cache_provenance,
        trainer_git_sha="b" * 40,
        kind="best",
        selection_state=state,
    )

    random.seed(123)
    torch.manual_seed(123)
    save_predictor_checkpoint(
        last_path,
        predictor=predictor,
        optimizer=optimizer,
        train_config=train_config,
        epoch=0,
        global_step=1,
        best_val_metrics=best_metrics,
        cache_provenance=cache_provenance,
        trainer_git_sha="b" * 40,
        kind="last",
        selection_state=state,
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(3)

    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
    assert best_payload["checkpoint_kind"] == "best"
    assert "optimizer_state_dict" not in best_payload
    assert "rng_state" not in best_payload
    assert last_payload["checkpoint_kind"] == "last"
    assert "optimizer_state_dict" in last_payload
    assert "rng_state" in last_payload
    assert set(best_payload["predictor_state_dict"]) == set(predictor.state_dict())
    forbidden_checkpoint_keys = {"backbone_state_dict", "smolvla_state_dict", "vlm_state_dict"}
    assert forbidden_checkpoint_keys.isdisjoint(best_payload)
    assert forbidden_checkpoint_keys.isdisjoint(last_payload)

    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.add_(10)
    optimizer.param_groups[0]["lr"] = 0.5
    random.seed(999)
    torch.manual_seed(999)
    resumed = load_last_checkpoint(last_path, predictor=predictor, optimizer=optimizer)

    assert resumed["global_step"] == 1
    assert optimizer.param_groups[0]["lr"] == 3e-4
    for key, expected in expected_parameters.items():
        torch.testing.assert_close(predictor.state_dict()[key], expected, rtol=0, atol=0)
    assert {int(entry["step"].item()) for entry in optimizer.state.values()} == {1}
    assert random.random() == expected_python_random
    torch.testing.assert_close(torch.rand(3), expected_torch_random, rtol=0, atol=0)


def test_rng_capture_and_restore_round_trip() -> None:
    random.seed(321)
    torch.manual_seed(321)
    state = capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(2)
    random.seed(0)
    torch.manual_seed(0)

    restore_rng_state(state)

    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(2), expected_torch, rtol=0, atol=0)
