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

import argparse
import importlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.advanced.predictive_async import (
    eval_future_latent_test as heldout_cli,
    future_latent_cache as cache_schema,
    future_latent_evaluation as evaluation,
    future_latent_training as training,
)
from examples.advanced.predictive_async.future_latent_cache import FutureLatentPair
from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import FutureLatentPrediction
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

_PREDICTIVE_ASYNC_DIR = Path(__file__).parents[3] / "examples/advanced/predictive_async"
sys.path.insert(0, str(_PREDICTIVE_ASYNC_DIR))
try:
    cache_cli = importlib.import_module("examples.advanced.predictive_async.cache_smolvla_latents")
finally:
    sys.path.pop(0)

_TEST_EPISODES = (15, 29, 31, 33, 41)
_SYNTHETIC_EPISODE_LENGTHS = (447, 447, 447, 446, 446)
_TEST_PAIR_COUNTS = (2_228, 2_223, 2_218, 2_213, 2_208, 2_203, 2_198, 2_193)
_TEST_PAIR_COUNT = 17_684
_SYNTHETIC_IMPLEMENTATION_SHA = "b" * 40

_INPUTS = {
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
}
_CAMERAS = ("observation.images.camera1", "observation.images.camera2")
_CAMERA_MAPPING = {
    "observation.images.top": _CAMERAS[0],
    "observation.images.wrist": _CAMERAS[1],
}
_SEMANTICS = {
    "state": "model_ready_normalized_and_padded",
    "action": "normalized_policy_output_original_action_dim",
    "processor_config_source": ("lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205"),
}


def _tensor_metadata(frame_count: int) -> dict[str, dict[str, object]]:
    specs = {
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
        for key, (shape, dtype) in specs.items()
    }


def _test_manifest(*, producer_sha: str = _SYNTHETIC_IMPLEMENTATION_SHA) -> dict[str, object]:
    episodes = [
        {
            "episode_index": episode_index,
            "frame_count": frame_count,
            "shard": f"episodes/episode_{episode_index:06d}.safetensors",
            "tensor_metadata": _tensor_metadata(frame_count),
        }
        for episode_index, frame_count in zip(_TEST_EPISODES, _SYNTHETIC_EPISODE_LENGTHS, strict=True)
    ]
    return {
        "schema_version": 1,
        "classification": "offline_future_latent_cache_not_task_capability",
        "producer": {
            "git_sha": producer_sha,
            "command": ["synthetic-cache", "--split", "test"],
            "created_at_utc": "2026-09-05T00:00:00+00:00",
        },
        "inputs": deepcopy(_INPUTS),
        "split": "test",
        "authoritative_episode_ids": list(_TEST_EPISODES),
        "cached_episode_ids": list(_TEST_EPISODES),
        "complete_split": True,
        "fps": 30,
        "episode_count": len(_TEST_EPISODES),
        "frame_count": sum(_SYNTHETIC_EPISODE_LENGTHS),
        "episodes": episodes,
        "valid_pair_count_by_delay": {
            str(delay): count for delay, count in enumerate(_TEST_PAIR_COUNTS, start=1)
        },
        "camera_mapping": dict(_CAMERA_MAPPING),
        "policy_camera_order": list(_CAMERAS),
        "token_scaling_convention": "native_post_sqrt_hidden_dim",
        "storage_device": "cpu",
        "semantics": dict(_SEMANTICS),
        "extraction_device": {"type": "cuda", "index": 0, "name": "synthetic"},
        "software_versions": {"torch": torch.__version__},
    }


def _write_manifest(cache_dir: Path, manifest: dict[str, object]) -> None:
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_generic_schema_accepts_complete_test_but_training_dataset_still_rejects_it(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "synthetic-test-cache"
    _write_manifest(cache_dir, _test_manifest())

    manifest = cache_schema.load_cache_manifest(cache_dir)
    assert manifest["split"] == "test"
    assert manifest["complete_split"] is True
    assert set(training._SPLIT_EXPECTATIONS) == {"train", "val"}
    with pytest.raises(ValueError, match="test cache access is not permitted"):
        training.FutureLatentCacheDataset(cache_dir, expected_split="test")


def test_cache_cli_allows_only_complete_test_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cache_smolvla_latents.py", "--output-dir", str(output_dir), "--split", "test"],
    )
    args = cache_cli.parse_args()
    assert args.split == "test"
    assert args.max_episodes is None
    assert args.max_frames_per_episode is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_smolvla_latents.py",
            "--output-dir",
            str(output_dir),
            "--split",
            "val",
            "--max-episodes",
            "1",
        ],
    )
    assert cache_cli.parse_args().max_episodes == 1


@pytest.mark.parametrize("partial_flag", ["--max-episodes", "--max-frames-per-episode"])
def test_cache_cli_rejects_each_partial_test_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_flag: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_smolvla_latents.py",
            "--output-dir",
            str(tmp_path / "cache"),
            "--split",
            "test",
            partial_flag,
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cache_cli.parse_args()
    assert error.value.code == 2


def _make_test_dataset(tmp_path: Path, manifest: dict[str, object] | None = None):
    cache_dir = tmp_path / "synthetic-test-cache"
    _write_manifest(cache_dir, manifest or _test_manifest())
    return evaluation.FutureLatentTestDataset(
        cache_dir,
        expected_test_cache_producer_sha=_SYNTHETIC_IMPLEMENTATION_SHA,
    )


def test_test_dataset_freezes_ids_counts_and_canonical_within_episode_pairs(tmp_path: Path) -> None:
    dataset = _make_test_dataset(tmp_path)

    assert evaluation.FROZEN_TEST_EPISODES == _TEST_EPISODES
    assert evaluation.FROZEN_TEST_FRAME_COUNT == 2_233
    assert evaluation.FROZEN_TEST_PAIR_COUNTS == _TEST_PAIR_COUNTS
    assert evaluation.FROZEN_TEST_PAIR_COUNT == _TEST_PAIR_COUNT
    assert len(dataset) == _TEST_PAIR_COUNT
    assert tuple(len(dataset.indices_for_delay(delay)) for delay in range(1, 9)) == _TEST_PAIR_COUNTS

    specs = iter(dataset.pair_specs)
    observed_count = 0
    for episode_position, frame_count in enumerate(_SYNTHETIC_EPISODE_LENGTHS):
        for frame_offset in range(frame_count):
            for delay_steps in range(1, 9):
                if frame_offset + delay_steps >= frame_count:
                    continue
                spec = next(specs)
                assert (spec.episode_position, spec.frame_offset, spec.delay_steps) == (
                    episode_position,
                    frame_offset,
                    delay_steps,
                )
                assert spec.frame_offset + spec.delay_steps < frame_count
                observed_count += 1
    assert next(specs, None) is None
    assert observed_count == _TEST_PAIR_COUNT


def _wrong_episode_ids(manifest: dict[str, object]) -> None:
    manifest["authoritative_episode_ids"][0] = 14
    manifest["cached_episode_ids"][0] = 14
    manifest["episodes"][0]["episode_index"] = 14
    manifest["episodes"][0]["shard"] = "episodes/episode_000014.safetensors"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest.__setitem__("split", "val"), "test cache split"),
        (_wrong_episode_ids, "test episode split"),
        (lambda manifest: manifest.__setitem__("complete_split", False), "test cache completeness"),
        (lambda manifest: manifest.__setitem__("frame_count", 2_234), "frame_count|frame count"),
        (
            lambda manifest: manifest["valid_pair_count_by_delay"].__setitem__("8", 2_194),
            "valid_pair_count_by_delay|valid pair counts",
        ),
        (
            lambda manifest: manifest["inputs"]["dataset"].__setitem__("requested_revision", "0" * 40),
            "dataset requested revision",
        ),
        (
            lambda manifest: manifest.__setitem__("policy_camera_order", list(reversed(_CAMERAS))),
            "test camera order",
        ),
        (
            lambda manifest: manifest["camera_mapping"].__setitem__("observation.images.top", _CAMERAS[1]),
            "test camera mapping",
        ),
        (
            lambda manifest: manifest["episodes"][0]["tensor_metadata"]["states"]["shape"].__setitem__(1, 31),
            "states.*shape",
        ),
        (
            lambda manifest: manifest["episodes"][0]["tensor_metadata"]["actions"]["shape"].__setitem__(1, 5),
            "actions.*shape",
        ),
        (
            lambda manifest: manifest["semantics"].__setitem__("state", "raw"),
            "state semantics",
        ),
        (
            lambda manifest: manifest["semantics"].__setitem__("action", "raw"),
            "action semantics",
        ),
        (
            lambda manifest: manifest["producer"].__setitem__("git_sha", "0" * 40),
            "producer SHA",
        ),
    ],
)
def test_test_dataset_rejects_identity_or_semantic_drift(tmp_path: Path, mutation, match: str) -> None:
    manifest = _test_manifest()
    mutation(manifest)
    cache_dir = tmp_path / "synthetic-test-cache"
    _write_manifest(cache_dir, manifest)

    with pytest.raises((TypeError, ValueError), match=match):
        evaluation.FutureLatentTestDataset(
            cache_dir,
            expected_test_cache_producer_sha=_SYNTHETIC_IMPLEMENTATION_SHA,
        )


_TRAIN_COUNTS = {
    "episode_count": 40,
    "frame_count": 15_576,
    "valid_pair_count_by_delay": {
        "1": 15_536,
        "2": 15_496,
        "3": 15_456,
        "4": 15_416,
        "5": 15_376,
        "6": 15_336,
        "7": 15_296,
        "8": 15_256,
    },
}
_VAL_COUNTS = {
    "episode_count": 5,
    "frame_count": 1_822,
    "valid_pair_count_by_delay": {
        "1": 1_817,
        "2": 1_812,
        "3": 1_807,
        "4": 1_802,
        "5": 1_797,
        "6": 1_792,
        "7": 1_787,
        "8": 1_782,
    },
}


def _training_cache_provenance(path: Path, split: str) -> dict[str, object]:
    counts = _TRAIN_COUNTS if split == "train" else _VAL_COUNTS
    return {
        "path": str(path.resolve()),
        "split": split,
        "complete_split": True,
        "producer_git_sha": evaluation.EXPECTED_CACHE_PRODUCER_SHA,
        "inputs": deepcopy(_INPUTS),
        "policy_camera_order": list(_CAMERAS),
        "episode_count": counts["episode_count"],
        "frame_count": counts["frame_count"],
        "valid_pair_count_by_delay": deepcopy(counts["valid_pair_count_by_delay"]),
        "semantics": dict(_SEMANTICS),
        "token_scaling_convention": "native_post_sqrt_hidden_dim",
    }


def _frozen_checkpoint_payload(tmp_path: Path) -> dict[str, object]:
    predictor_config = asdict(FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, enabled=True))
    train_cache = _training_cache_provenance(tmp_path / "train-cache", "train")
    val_cache = _training_cache_provenance(tmp_path / "val-cache", "val")
    train_config = {
        "schema_version": 1,
        "classification": "offline_future_latent_predictor_training_not_task_capability",
        "run_kind": "train_val",
        "protocol_complete": True,
        "eligible_for_checkpoint_selection": True,
        "eligible_for_test": False,
        "trainer_git_sha": evaluation.EXPECTED_TRAINER_SHA,
        "cache_producer_sha": evaluation.EXPECTED_CACHE_PRODUCER_SHA,
        "train_cache": deepcopy(train_cache),
        "val_cache": deepcopy(val_cache),
        "future_latent_config": predictor_config,
        "device": "cuda",
        "micro_batch_size": 16,
        "effective_batch_size": 64,
        "gradient_accumulation_steps": 4,
        "num_workers": 0,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 1e-4,
        },
        "scheduler": None,
        "grad_clip_norm": 1.0,
        "lambda_cos": 0.1,
        "lambda_risk": 0.05,
        "max_epochs": 30,
        "seed": 0,
        "max_optimizer_steps": None,
        "max_val_pairs_per_delay": None,
        "resume_from": None,
    }
    return {
        "checkpoint_kind": "best",
        "predictor_state_dict": {"synthetic": torch.tensor(1.0)},
        "predictor_config": predictor_config,
        "train_config": train_config,
        "epoch": 29,
        "global_step": 57_750,
        "best_val_metrics": {"epoch": 29, "global_step": 57_750},
        "cache_provenance": {"train": train_cache, "val": val_cache},
        "trainer_git_sha": evaluation.EXPECTED_TRAINER_SHA,
        "cache_producer_sha": evaluation.EXPECTED_CACHE_PRODUCER_SHA,
        "selection_state": SimpleNamespace(best=SimpleNamespace(epoch=29)),
    }


class _SyntheticLoadedPredictor:
    def __init__(self, config: FutureLatentConfig):
        self.config = config
        self.loaded_strictly = False
        self.requested_device: torch.device | None = None
        self.grad_enabled = True
        self.training = True

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool) -> None:
        assert set(state_dict) == {"synthetic"}
        assert state_dict["synthetic"].item() == pytest.approx(1.0)
        self.loaded_strictly = strict

    def to(self, *, device: torch.device, dtype: torch.dtype):
        assert dtype == torch.float32
        self.requested_device = device
        return self

    def requires_grad_(self, requires_grad: bool):
        self.grad_enabled = requires_grad
        return self

    def eval(self):
        self.training = False
        return self


def test_frozen_test_loader_accepts_only_the_frozen_best_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _make_test_dataset(tmp_path)
    checkpoint = tmp_path / "best.pt"
    torch.save(_frozen_checkpoint_payload(tmp_path), checkpoint)
    monkeypatch.setattr(evaluation, "LightweightFutureLatentPredictor", _SyntheticLoadedPredictor)

    frozen = evaluation.load_frozen_test_predictor(
        checkpoint,
        test_cache=dataset,
        expected_test_cache_producer_sha=_SYNTHETIC_IMPLEMENTATION_SHA,
        device="cpu",
    )

    assert frozen.test_dataset is dataset
    assert frozen.checkpoint["checkpoint_kind"] == "best"
    assert frozen.checkpoint["train_config"]["lambda_risk"] == pytest.approx(0.05)
    assert frozen.predictor.loaded_strictly is True
    assert frozen.predictor.requested_device == torch.device("cpu")
    assert frozen.predictor.grad_enabled is False
    assert frozen.predictor.training is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.__setitem__("checkpoint_kind", "last"), "checkpoint kind"),
        (
            lambda payload: payload["train_config"].__setitem__("lambda_risk", 0.1),
            "lambda_risk",
        ),
    ],
)
def test_frozen_test_loader_rejects_nonbest_or_wrong_candidate(tmp_path: Path, mutation, match: str) -> None:
    dataset = _make_test_dataset(tmp_path)
    payload = _frozen_checkpoint_payload(tmp_path)
    mutation(payload)
    checkpoint = tmp_path / "candidate.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=match):
        evaluation.load_frozen_test_predictor(
            checkpoint,
            test_cache=dataset,
            expected_test_cache_producer_sha=_SYNTHETIC_IMPLEMENTATION_SHA,
            device="cpu",
        )


def _pair(
    *,
    episode_index: int,
    frame_index: int,
    delay_steps: int,
    current_value: float = 1.0,
    target_value: float = 4.0,
) -> FutureLatentPair:
    current_tokens = (
        torch.full((2, 2), current_value),
        torch.full((1, 2), current_value),
    )
    target_tokens = (
        torch.full((2, 2), target_value),
        torch.full((1, 2), target_value),
    )
    language_tokens = torch.tensor([11, 12, 0], dtype=torch.int64)
    language_mask = torch.tensor([True, True, False])
    return FutureLatentPair(
        current_image_tokens=current_tokens,
        current_image_token_masks=(torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool)),
        committed_actions=torch.arange(48, dtype=torch.float32).reshape(8, 6),
        committed_mask=torch.arange(8) < delay_steps,
        current_state=torch.full((32,), 10.0),
        delay_steps=torch.tensor(delay_steps, dtype=torch.int64),
        target_image_tokens=target_tokens,
        target_image_token_masks=(torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool)),
        future_state=torch.full((32,), 20.0),
        current_language_tokens=language_tokens,
        current_language_attention_mask=language_mask,
        future_language_tokens=language_tokens.clone(),
        future_language_attention_mask=language_mask.clone(),
        episode_index=episode_index,
        frame_index=frame_index,
        future_frame_index=frame_index + delay_steps,
    )


class _SyntheticAnchorDataset:
    expected_split = "test"

    def __init__(self) -> None:
        self.manifest = {
            "episodes": [
                {"episode_index": episode_index, "frame_count": 40} for episode_index in _TEST_EPISODES
            ]
        }
        self.pair_specs = tuple(
            SimpleNamespace(
                episode_position=episode_position,
                frame_offset=frame_offset,
                delay_steps=delay_steps,
            )
            for episode_position in range(len(_TEST_EPISODES))
            for frame_offset in range(40)
            for delay_steps in range(1, 9)
            if frame_offset + delay_steps < 40
        )
        self._episode_tensors = tuple(
            {
                "dataset_indices": torch.arange(
                    episode_position * 1_000,
                    episode_position * 1_000 + 40,
                    dtype=torch.int64,
                ),
                "frame_indices": torch.arange(40, dtype=torch.int64),
            }
            for episode_position in range(len(_TEST_EPISODES))
        )

    def _load_episode(self, episode_position: int):
        return self.manifest["episodes"][episode_position], self._episode_tensors[episode_position]

    def __getitem__(self, index: int) -> FutureLatentPair:
        spec = self.pair_specs[index]
        return _pair(
            episode_index=self.manifest["episodes"][spec.episode_position]["episode_index"],
            frame_index=spec.frame_offset,
            delay_steps=spec.delay_steps,
        )


def test_seed_zero_test_anchor_selection_returns_128_common_anchors_times_eight() -> None:
    dataset = _SyntheticAnchorDataset()

    anchors = evaluation.select_test_anchor_pairs(dataset, count=128, seed=0)
    repeated = evaluation.select_test_anchor_pairs(dataset, count=128, seed=0)

    def identity(anchor):
        return anchor.anchor_id, anchor.episode_index, anchor.frame_index

    assert [identity(anchor) for anchor in anchors] == [identity(anchor) for anchor in repeated]
    assert len(anchors) == len({anchor.anchor_id for anchor in anchors}) == 128
    episode_counts = {
        episode_index: sum(anchor.episode_index == episode_index for anchor in anchors)
        for episode_index in _TEST_EPISODES
    }
    assert sorted(episode_counts.values()) == [25, 25, 26, 26, 26]
    for anchor in anchors:
        assert tuple(int(pair.delay_steps.item()) for pair in anchor.pairs) == tuple(range(1, 9))
        assert all(pair.episode_index == anchor.episode_index for pair in anchor.pairs)
        assert all(pair.frame_index == anchor.frame_index for pair in anchor.pairs)
        assert all(pair.future_frame_index < 40 for pair in anchor.pairs)


class _FixedDeltaPredictor(torch.nn.Module):
    def forward(
        self,
        image_tokens: tuple[torch.Tensor, ...],
        image_token_masks: tuple[torch.Tensor, ...],
        committed_actions: torch.Tensor,
        committed_mask: torch.Tensor,
        state: torch.Tensor,
        delay_steps: torch.Tensor,
    ) -> FutureLatentPrediction:
        del image_token_masks, committed_actions, committed_mask, state
        return FutureLatentPrediction(
            delta_tokens=tuple(torch.full_like(tokens, 2.0) for tokens in image_tokens),
            predicted_error=delay_steps.float() / 10,
        )


class _TinyTestDataset:
    expected_split = "test"

    def __init__(self) -> None:
        self.pairs = [
            _pair(episode_index=15, frame_index=0, delay_steps=1),
            _pair(episode_index=29, frame_index=2, delay_steps=2),
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> FutureLatentPair:
        return self.pairs[index]


def test_test_latent_evaluator_consumes_every_synthetic_pair_and_keeps_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation, "FROZEN_TEST_PAIR_COUNT", 2)
    monkeypatch.setattr(evaluation, "FROZEN_TEST_PAIR_COUNTS", (1, 1))
    monkeypatch.setattr(evaluation, "FROZEN_DELAYS", (1, 2))

    result = evaluation.evaluate_test_latent_risk(
        _FixedDeltaPredictor(),
        _TinyTestDataset(),
        device="cpu",
        batch_size=2,
    )

    assert result.summary.total_record_count == 2
    assert [entry.sample_count for entry in result.summary.per_delay] == [1, 1]
    assert [
        (record.episode_index, record.frame_index, record.future_frame_index, record.delay_steps)
        for record in result.records
    ] == [(15, 0, 1, 1), (29, 2, 4, 2)]


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.received_noises: list[torch.Tensor] = []

    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        *,
        noise: torch.Tensor,
        future_image_tokens: tuple[torch.Tensor, ...],
        future_image_token_masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        self.received_noises.append(noise)
        self.calls.append(
            {
                "batch": {key: value.clone() for key, value in batch.items()},
                "tokens": tuple(value.clone() for value in future_image_tokens),
                "masks": tuple(value.clone() for value in future_image_token_masks),
                "noise": noise.clone(),
                "noise_data_ptr": noise.data_ptr(),
            }
        )
        first_token = float(future_image_tokens[0][0, 0, 0].item())
        state_value = float(batch[OBS_STATE][0, 0].item())
        noise.add_(100)
        base = first_token + state_value / 100
        return torch.tensor([[[base, base * 2]]], dtype=torch.float32)


def test_four_paths_reuse_one_noise_across_delays_with_independent_clones() -> None:
    policy = _RecordingPolicy()
    predictor = _FixedDeltaPredictor()
    noise = torch.zeros(1, 2, 2)

    records = tuple(
        evaluation.evaluate_four_path_pair(
            policy,
            predictor,
            _pair(episode_index=15, frame_index=10, delay_steps=delay),
            noise,
            lambda action: action * 2,
            anchor_id=91,
        )
        for delay in range(1, 9)
    )

    assert len(records) == 8
    assert torch.equal(noise, torch.zeros_like(noise))
    assert len(policy.calls) == 32
    assert len({call["noise_data_ptr"] for call in policy.calls}) == 32
    assert all(torch.equal(call["noise"], torch.zeros_like(noise)) for call in policy.calls)
    for offset in range(0, len(policy.calls), 4):
        current, oracle, predicted, teacher = policy.calls[offset : offset + 4]
        assert [float(call["tokens"][0][0, 0, 0]) for call in (current, oracle, predicted, teacher)] == [
            1.0,
            4.0,
            3.0,
            4.0,
        ]
        assert [float(call["batch"][OBS_STATE][0, 0]) for call in (current, oracle, predicted, teacher)] == [
            10.0,
            10.0,
            10.0,
            20.0,
        ]
        assert all(
            set(call["batch"]) == {OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK}
            for call in (current, oracle, predicted, teacher)
        )
    assert records[0].post_policy.predicted_visual_vs_teacher.l1 == pytest.approx(
        records[0].policy_output.predicted_visual_vs_teacher.l1 * 2
    )


def _action_errors(
    *,
    current_l1: float,
    oracle_l1: float,
    predicted_l1: float,
    current_rmse: float,
    oracle_rmse: float,
    predicted_rmse: float,
) -> evaluation.ActionSpaceErrors:
    return evaluation.ActionSpaceErrors(
        current_vs_teacher=evaluation.ActionError(current_l1, current_rmse),
        oracle_visual_vs_teacher=evaluation.ActionError(oracle_l1, oracle_rmse),
        predicted_visual_vs_teacher=evaluation.ActionError(predicted_l1, predicted_rmse),
    )


def _action_record(
    anchor_id: int,
    delay_steps: int,
    *,
    policy_output: evaluation.ActionSpaceErrors,
    post_policy: evaluation.ActionSpaceErrors,
) -> evaluation.FourPathActionRecord:
    return evaluation.FourPathActionRecord(
        anchor_id=anchor_id,
        episode_index=_TEST_EPISODES[anchor_id % len(_TEST_EPISODES)],
        frame_index=anchor_id,
        future_frame_index=anchor_id + delay_steps,
        delay_steps=delay_steps,
        policy_output=policy_output,
        post_policy=post_policy,
    )


def test_action_aggregation_keeps_negative_capture_and_unchanged_primary_gate() -> None:
    records = []
    for anchor_id in range(128):
        for delay in range(1, 9):
            predicted_l1 = 5.0 if delay <= 6 else (10.5 if delay == 7 else 10.0)
            records.append(
                _action_record(
                    anchor_id,
                    delay,
                    policy_output=_action_errors(
                        current_l1=10.0,
                        oracle_l1=2.0,
                        predicted_l1=predicted_l1,
                        current_rmse=10.0,
                        oracle_rmse=11.0,
                        predicted_rmse=5.0,
                    ),
                    post_policy=_action_errors(
                        current_l1=3.0,
                        oracle_l1=4.0,
                        predicted_l1=2.0,
                        current_rmse=4.0,
                        oracle_rmse=2.0,
                        predicted_rmse=5.0,
                    ),
                )
            )

    summary = evaluation.aggregate_action_records(records)

    assert summary.total_record_count == 1_024
    assert summary.positive_policy_l1_capture_delay_count == 6
    assert summary.policy_l1_nonpositive_oracle_gap_delays == ()
    assert summary.per_delay[6].policy_output.l1.capture == pytest.approx(-0.0625)
    assert summary.per_delay[6].policy_output.rmse.capture is None
    assert summary.per_delay[6].post_policy.l1.capture is None
    assert summary.per_delay[6].post_policy.rmse.capture == pytest.approx(-0.5)
    assert summary.max_policy_l1_regression == pytest.approx(0.05)
    assert summary.test_protocol_prerequisites_met is True


def test_nonpositive_gap_hard_stop_is_scoped_only_to_primary_policy_l1() -> None:
    record = _action_record(
        1,
        1,
        policy_output=_action_errors(
            current_l1=1.0,
            oracle_l1=1.5,
            predicted_l1=0.5,
            current_rmse=2.0,
            oracle_rmse=1.0,
            predicted_rmse=1.5,
        ),
        post_policy=_action_errors(
            current_l1=3.0,
            oracle_l1=4.0,
            predicted_l1=2.0,
            current_rmse=4.0,
            oracle_rmse=2.0,
            predicted_rmse=5.0,
        ),
    )

    summary = evaluation.aggregate_action_records([record])
    delay = summary.per_delay[0]

    assert delay.policy_output.l1.capture_defined is False
    assert delay.policy_output.l1.capture is None
    assert delay.policy_output.rmse.capture == pytest.approx(0.5)
    assert delay.post_policy.l1.capture is None
    assert delay.post_policy.rmse.capture == pytest.approx(-0.5)
    assert summary.policy_l1_nonpositive_oracle_gap_delays == (1,)
    assert summary.test_protocol_prerequisites_met is False


def _passing_latent_summary() -> dict[str, object]:
    return {
        "total_record_count": _TEST_PAIR_COUNT,
        "per_delay": [
            {
                "delay_steps": delay,
                "sample_count": _TEST_PAIR_COUNTS[delay - 1],
                "predicted": {"smoothl1": 1.0 if delay <= 7 else 3.0, "mse": 2.0, "cosine": 0.1},
                "identity": {"smoothl1": 2.0, "mse": 3.0, "cosine": 0.2},
                "risk_smoothl1": 0.5,
            }
            for delay in range(1, 9)
        ],
        "macro_predicted": {"smoothl1": 1.25, "mse": 2.0, "cosine": 0.1},
        "macro_identity": {"smoothl1": 2.0, "mse": 3.0, "cosine": 0.2},
        "macro_risk_smoothl1": 0.5,
    }


def test_heldout_cli_has_only_five_options_and_writes_frozen_schema(
    tmp_path: Path,
) -> None:
    parser = heldout_cli.build_parser()
    long_options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert long_options == {
        "--test-cache",
        "--predictor-checkpoint",
        "--output-dir",
        "--device",
        "--local-files-only",
    }
    args = parser.parse_args(
        [
            "--test-cache",
            str(tmp_path / "cache"),
            "--predictor-checkpoint",
            str(tmp_path / "best.pt"),
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    assert args.device == "cuda"
    assert args.local_files_only is False

    markers = heldout_cli.evaluation_protocol_markers()
    assert markers == {
        "test_protocol_complete": True,
        "test_data_read": True,
        "used_for_checkpoint_or_hyperparameter_selection": False,
        "risk_thresholds": None,
        "online_or_m5_authorized": False,
    }

    latent_summary = _passing_latent_summary()
    action_summary = {"test_protocol_prerequisites_met": True}
    risk_summary = {"risk_gating_eligible": True}
    manifest = _test_manifest()
    summary = heldout_cli.build_summary(
        implementation_sha=_SYNTHETIC_IMPLEMENTATION_SHA,
        started_at=datetime(2026, 9, 5, tzinfo=UTC),
        completed_at=datetime(2026, 9, 5, 0, 1, tzinfo=UTC),
        command=["eval_future_latent_test.py"],
        predictor_checkpoint=tmp_path / "best.pt",
        checkpoint={
            "checkpoint_kind": "best",
            "epoch": 29,
            "global_step": 57_750,
            "predictor_config": {"enabled": True},
        },
        test_cache=tmp_path / "cache",
        test_manifest=manifest,
        latent_summary=latent_summary,
        latent_record_count=_TEST_PAIR_COUNT,
        risk_summary=risk_summary,
        action_summary=action_summary,
        action_record_count=1_024,
    )

    assert summary["schema_version"] == 1
    assert summary["classification"] == "offline_future_latent_heldout_test_not_task_capability"
    assert {key: summary[key] for key in markers} == markers
    assert summary["test_cache"]["episode_ids"] == list(_TEST_EPISODES)
    assert summary["test_cache"]["frame_count"] == 2_233
    assert summary["latent_risk_record_count"] == _TEST_PAIR_COUNT
    assert summary["action_record_count"] == 1_024
    assert summary["fixed_protocol"]["seed"] == 0
    assert summary["fixed_protocol"]["delays"] == list(range(1, 9))
    assert summary["fixed_protocol"]["anchor_count"] == 128
    assert summary["fixed_protocol"]["fallback"] is None
    assert summary["heldout_gates"]["m3_heldout_passed"] is True
    assert summary["heldout_gates"]["risk_diagnostic"]["alters_latent_or_action_gate"] is False

    summary_path = tmp_path / "summary.json"
    heldout_cli._write_json(summary_path, summary)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["classification"] == summary["classification"]
    assert written["risk_thresholds"] is None


def test_heldout_gate_is_not_changed_by_risk_and_output_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latent_summary = _passing_latent_summary()
    action_summary = {"test_protocol_prerequisites_met": True}
    gates = heldout_cli.summarize_heldout_gates(
        latent_summary,
        action_summary,
        {"risk_gating_eligible": False},
    )
    assert gates["latent"]["improved_delay_count"] == 7
    assert gates["latent"]["passed"] is True
    assert gates["action"]["passed"] is True
    assert gates["risk_diagnostic"]["generalization_eligible"] is False
    assert gates["m3_heldout_passed"] is True

    monkeypatch.setattr(
        heldout_cli,
        "parse_args",
        lambda: argparse.Namespace(output_dir=tmp_path),
    )
    with pytest.raises(FileExistsError, match="output directory already exists"):
        heldout_cli.main()
