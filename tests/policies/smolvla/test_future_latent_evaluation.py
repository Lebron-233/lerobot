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

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.advanced.predictive_async import (
    benchmark_predictor as benchmark,
    eval_future_latent as eval_cli,
    future_latent_evaluation as evaluation,
)
from examples.advanced.predictive_async.future_latent_cache import FutureLatentPair
from examples.advanced.predictive_async.future_latent_training import FutureLatentPairSpec
from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import FutureLatentPrediction
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

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
_SPLIT_COUNTS = {
    "train": {
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
    },
    "val": {
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
    },
}


def _manifest(split: str) -> dict[str, object]:
    counts = _SPLIT_COUNTS[split]
    return {
        "split": split,
        "complete_split": True,
        "producer": {"git_sha": evaluation.EXPECTED_CACHE_PRODUCER_SHA},
        "inputs": deepcopy(_INPUTS),
        "policy_camera_order": [
            "observation.images.camera1",
            "observation.images.camera2",
        ],
        "episode_count": counts["episode_count"],
        "frame_count": counts["frame_count"],
        "valid_pair_count_by_delay": deepcopy(counts["valid_pair_count_by_delay"]),
        "semantics": {
            "state": "model_ready_normalized_and_padded",
            "action": "normalized_policy_output_original_action_dim",
            "processor_config_source": ("lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205"),
        },
        "token_scaling_convention": "native_post_sqrt_hidden_dim",
    }


def _provenance(path: Path, split: str) -> dict[str, object]:
    manifest = _manifest(split)
    return {
        "path": str(path.resolve()),
        "split": manifest["split"],
        "complete_split": manifest["complete_split"],
        "producer_git_sha": manifest["producer"]["git_sha"],
        "inputs": manifest["inputs"],
        "policy_camera_order": manifest["policy_camera_order"],
        "episode_count": manifest["episode_count"],
        "frame_count": manifest["frame_count"],
        "valid_pair_count_by_delay": manifest["valid_pair_count_by_delay"],
        "semantics": manifest["semantics"],
        "token_scaling_convention": manifest["token_scaling_convention"],
    }


def _frozen_checkpoint_payload(tmp_path: Path) -> dict[str, object]:
    predictor_config = asdict(FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, enabled=True))
    train_cache = _provenance(tmp_path / "recorded-train-cache", "train")
    val_cache = _provenance(tmp_path / "recorded-val-cache", "val")
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


class _SyntheticFrozenValDataset:
    def __init__(self, cache_dir: Path, *, expected_split: str):
        self.cache_dir = Path(cache_dir)
        self.expected_split = expected_split
        self.manifest = _manifest("val")

    def __len__(self) -> int:
        return 14_396


class _SyntheticLoadedPredictor:
    def __init__(self, config: FutureLatentConfig):
        self.config = config
        self.loaded_strictly = False
        self.requested_device: torch.device | None = None
        self.grad_enabled = True
        self.training = True

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool) -> None:
        assert state_dict == {"synthetic": torch.tensor(1.0)}
        self.loaded_strictly = strict

    def to(self, *, device: torch.device, dtype: torch.dtype) -> _SyntheticLoadedPredictor:
        assert dtype == torch.float32
        self.requested_device = device
        return self

    def requires_grad_(self, requires_grad: bool) -> _SyntheticLoadedPredictor:
        self.grad_enabled = requires_grad
        return self

    def eval(self) -> _SyntheticLoadedPredictor:
        self.training = False
        return self


def _install_synthetic_loader_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "FutureLatentCacheDataset", _SyntheticFrozenValDataset)
    monkeypatch.setattr(evaluation, "LightweightFutureLatentPredictor", _SyntheticLoadedPredictor)


def test_frozen_best_loader_accepts_only_pathless_matching_identity_and_freezes_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_loader_dependencies(monkeypatch)
    checkpoint = tmp_path / "best.pt"
    torch.save(_frozen_checkpoint_payload(tmp_path), checkpoint)

    frozen = evaluation.load_frozen_best_predictor(
        checkpoint,
        val_cache=tmp_path / "different-but-identical-val-cache",
        device="cpu",
    )

    assert frozen.checkpoint["checkpoint_kind"] == "best"
    assert frozen.checkpoint["epoch"] == 29
    assert frozen.checkpoint["global_step"] == 57_750
    assert frozen.checkpoint["train_config"]["lambda_risk"] == pytest.approx(0.05)
    assert frozen.predictor.loaded_strictly is True
    assert frozen.predictor.requested_device == torch.device("cpu")
    assert frozen.predictor.grad_enabled is False
    assert frozen.predictor.training is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.__setitem__("checkpoint_kind", "last"), "checkpoint kind"),
        (lambda payload: payload.__setitem__("trainer_git_sha", "0" * 40), "trainer SHA"),
        (lambda payload: payload.__setitem__("cache_producer_sha", "0" * 40), "producer SHA"),
        (lambda payload: payload.__setitem__("epoch", 28), "best epoch"),
        (lambda payload: payload.__setitem__("global_step", 57_749), "best global step"),
        (
            lambda payload: payload["predictor_config"].__setitem__("rank", 63),
            "predictor config",
        ),
        (
            lambda payload: payload["train_config"].__setitem__("run_kind", "bounded_smoke"),
            "run_kind",
        ),
        (
            lambda payload: payload["train_config"].__setitem__("max_optimizer_steps", 1),
            "max_optimizer_steps",
        ),
        (
            lambda payload: payload["cache_provenance"]["train"].__setitem__("producer_git_sha", "0" * 40),
            "train cache producer SHA",
        ),
        (lambda payload: payload.__setitem__("optimizer_state_dict", {}), "optimizer or RNG"),
    ],
)
def test_frozen_best_loader_rejects_wrong_or_bounded_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    _install_synthetic_loader_dependencies(monkeypatch)
    payload = _frozen_checkpoint_payload(tmp_path)
    mutation(payload)
    checkpoint = tmp_path / "candidate.pt"
    torch.save(payload, checkpoint)

    with pytest.raises((TypeError, ValueError), match=match):
        evaluation.load_frozen_best_predictor(checkpoint, val_cache=tmp_path / "val", device="cpu")


def test_frozen_best_loader_rejects_requested_val_cache_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongValDataset(_SyntheticFrozenValDataset):
        def __init__(self, cache_dir: Path, *, expected_split: str):
            super().__init__(cache_dir, expected_split=expected_split)
            self.manifest["producer"]["git_sha"] = "0" * 40

    monkeypatch.setattr(evaluation, "FutureLatentCacheDataset", WrongValDataset)
    monkeypatch.setattr(evaluation, "LightweightFutureLatentPredictor", _SyntheticLoadedPredictor)
    checkpoint = tmp_path / "best.pt"
    torch.save(_frozen_checkpoint_payload(tmp_path), checkpoint)

    with pytest.raises(ValueError, match="requested val cache identity"):
        evaluation.load_frozen_best_predictor(checkpoint, val_cache=tmp_path / "val", device="cpu")


def test_frozen_best_loader_rejects_original_lambda_risk_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_synthetic_loader_dependencies(monkeypatch)
    payload = _frozen_checkpoint_payload(tmp_path)
    payload["train_config"]["lambda_risk"] = 0.1
    checkpoint = tmp_path / "original-lambda-risk.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="train config lambda_risk"):
        evaluation.load_frozen_best_predictor(checkpoint, val_cache=tmp_path / "val", device="cpu")


def _pair(
    *,
    delay_steps: int,
    frame_index: int,
    episode_index: int = 5,
    current_tokens: tuple[torch.Tensor, ...] | None = None,
    current_masks: tuple[torch.Tensor, ...] | None = None,
    target_tokens: tuple[torch.Tensor, ...] | None = None,
    target_masks: tuple[torch.Tensor, ...] | None = None,
    current_state: torch.Tensor | None = None,
    future_state: torch.Tensor | None = None,
    current_language_tokens: torch.Tensor | None = None,
    future_language_tokens: torch.Tensor | None = None,
) -> FutureLatentPair:
    current_tokens = current_tokens or (torch.zeros(2, 2), torch.zeros(1, 2))
    current_masks = current_masks or (
        torch.ones(2, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    )
    target_tokens = target_tokens or tuple(tokens + 1 for tokens in current_tokens)
    target_masks = target_masks or tuple(mask.clone() for mask in current_masks)
    current_state = current_state if current_state is not None else torch.arange(32, dtype=torch.float32)
    future_state = future_state if future_state is not None else current_state + 10
    current_language_tokens = (
        current_language_tokens
        if current_language_tokens is not None
        else torch.tensor([11, 12, 0], dtype=torch.int64)
    )
    future_language_tokens = (
        future_language_tokens if future_language_tokens is not None else current_language_tokens.clone()
    )
    committed_mask = torch.arange(8) < delay_steps
    return FutureLatentPair(
        current_image_tokens=current_tokens,
        current_image_token_masks=current_masks,
        committed_actions=torch.arange(48, dtype=torch.float32).reshape(8, 6),
        committed_mask=committed_mask,
        current_state=current_state,
        delay_steps=torch.tensor(delay_steps, dtype=torch.int64),
        target_image_tokens=target_tokens,
        target_image_token_masks=target_masks,
        future_state=future_state,
        current_language_tokens=current_language_tokens,
        current_language_attention_mask=torch.tensor([True, True, False]),
        future_language_tokens=future_language_tokens,
        future_language_attention_mask=torch.tensor([True, True, False]),
        episode_index=episode_index,
        frame_index=frame_index,
        future_frame_index=frame_index + delay_steps,
    )


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
        scale = torch.where(delay_steps == 1, -1.0, 1.0)
        delta_tokens = tuple(torch.ones_like(tokens) * scale[:, None, None] for tokens in image_tokens)
        return FutureLatentPrediction(
            delta_tokens=delta_tokens,
            predicted_error=delay_steps.float() / 10,
        )


class _TinyValDataset:
    expected_split = "val"

    def __init__(self, pairs: list[FutureLatentPair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> FutureLatentPair:
        return self.pairs[index]


def test_latent_evaluation_uses_joint_masks_retains_d1_and_aggregates_delay_macros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d1 = _pair(
        delay_steps=1,
        frame_index=10,
        target_tokens=(
            torch.tensor([[1.0, 3.0], [1_000.0, 1_000.0]]),
            torch.tensor([[1_000.0, 1_000.0]]),
        ),
        current_masks=(torch.tensor([True, False]), torch.tensor([True])),
        target_masks=(torch.tensor([True, True]), torch.tensor([False])),
    )
    d2 = _pair(delay_steps=2, frame_index=20)
    dataset = _TinyValDataset([d1, d2])
    monkeypatch.setattr(evaluation, "FROZEN_VAL_PAIR_COUNT", 2)
    monkeypatch.setattr(evaluation, "FROZEN_DELAYS", (1, 2))
    monkeypatch.setitem(
        evaluation._EXPECTED_SPLIT_COUNTS,
        "val",
        {"valid_pair_count_by_delay": {"1": 1, "2": 1}},
    )

    result = evaluation.evaluate_latent_risk(
        _FixedDeltaPredictor(),
        dataset,
        device="cpu",
        batch_size=2,
    )

    assert [record.delay_steps for record in result.records] == [1, 2]
    d1_record = result.records[0]
    assert d1_record.predicted_risk == pytest.approx(0.1)
    assert d1_record.actual_mse == pytest.approx(10.0)
    assert d1_record.predicted.mse == pytest.approx(10.0)
    assert d1_record.predicted.smoothl1 == pytest.approx(2.5)
    assert d1_record.predicted.cosine == pytest.approx(1.8944272)
    assert d1_record.identity.mse == pytest.approx(5.0)
    assert d1_record.predicted.mse > d1_record.identity.mse
    assert result.summary.total_record_count == 2
    assert [entry.sample_count for entry in result.summary.per_delay] == [1, 1]
    assert [entry.risk_smoothl1 for entry in result.summary.per_delay] == pytest.approx([9.4, 0.02])
    assert result.summary.macro_predicted.mse == pytest.approx(5.0)
    assert result.summary.macro_identity.mse == pytest.approx(3.0)
    assert result.summary.macro_risk_smoothl1 == pytest.approx(4.71)


def _risk_record(index: int, *, predicted_risk: float, actual_mse: float, delay: int = 1):
    predicted = evaluation.LatentMetrics(
        smoothl1=actual_mse / 2,
        mse=actual_mse,
        cosine=actual_mse / 10,
    )
    identity = evaluation.LatentMetrics(
        smoothl1=actual_mse,
        mse=actual_mse * 2,
        cosine=actual_mse / 5,
    )
    return evaluation.LatentRiskRecord(
        episode_index=index // 10_000,
        frame_index=index,
        future_frame_index=index + delay,
        delay_steps=delay,
        predicted_risk=predicted_risk,
        actual_mse=actual_mse,
        predicted=predicted,
        identity=identity,
    )


def test_tie_aware_spearman_uses_average_ranks_and_constant_side_is_undefined() -> None:
    assert evaluation.tie_aware_spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    assert evaluation.tie_aware_spearman([7, 7, 7], [1, 2, 3]) is None
    assert evaluation.tie_aware_spearman([1, 2, 3], [4, 4, 4]) is None


def test_full_val_sized_risk_bins_and_quintiles_are_equal_count_and_recomputable() -> None:
    records = tuple(
        _risk_record(
            index,
            predicted_risk=float(index // 2),
            actual_mse=float(index + 1),
            delay=index % 8 + 1,
        )
        for index in range(14_396)
    )

    summary = evaluation.compute_risk_calibration(records)

    assert summary.record_count == 14_396
    assert [entry.count for entry in summary.bins] == [1_440] * 6 + [1_439] * 4
    assert summary.bottom_quintile_count == 2_879
    assert summary.top_quintile_count == 2_879
    assert summary.overall_spearman == pytest.approx(1.0)
    assert summary.top_bottom_actual_mse_ratio == pytest.approx(
        summary.top_quintile_actual_mse_mean / summary.bottom_quintile_actual_mse_mean
    )
    assert summary.risk_gating_eligible is True
    assert summary.risk_thresholds is None


def test_constant_risk_has_explicit_null_spearman_numeric_tie_order_bins_and_no_gate() -> None:
    records = tuple(
        _risk_record(index, predicted_risk=7.0, actual_mse=float(index + 1)) for index in range(20)
    )

    summary = evaluation.compute_risk_calibration(records)

    assert summary.overall_spearman is None
    assert summary.per_delay_spearman == {"1": None}
    assert [entry.count for entry in summary.bins] == [2] * 10
    assert summary.bottom_quintile_actual_mse_mean == pytest.approx(2.5)
    assert summary.top_quintile_actual_mse_mean == pytest.approx(18.5)
    assert summary.top_bottom_actual_mse_ratio == pytest.approx(7.4)
    assert summary.top_bottom_ratio_defined is True
    assert summary.risk_gating_eligible is False
    assert summary.to_dict()["risk_thresholds"] is None


class _SyntheticAnchorDataset:
    expected_split = "val"

    def __init__(self) -> None:
        episode_ids = (5, 7, 14, 39, 49)
        self.manifest = {
            "episodes": [{"episode_index": episode_index, "frame_count": 40} for episode_index in episode_ids]
        }
        self.pair_specs = tuple(
            FutureLatentPairSpec(
                episode_position=episode_position,
                frame_offset=frame_offset,
                delay_steps=delay_steps,
            )
            for episode_position in range(len(episode_ids))
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
            for episode_position in range(len(episode_ids))
        )

    def _load_episode(self, episode_position: int):
        return self.manifest["episodes"][episode_position], self._episode_tensors[episode_position]

    def __getitem__(self, index: int) -> FutureLatentPair:
        spec = self.pair_specs[index]
        episode_index = self.manifest["episodes"][spec.episode_position]["episode_index"]
        return _pair(
            episode_index=episode_index,
            frame_index=spec.frame_offset,
            delay_steps=spec.delay_steps,
        )


def test_val_anchor_selection_is_seeded_episode_stratified_and_has_all_eight_delays() -> None:
    dataset = _SyntheticAnchorDataset()

    anchors = evaluation.select_val_anchor_pairs(dataset, count=128, seed=0)
    repeated = evaluation.select_val_anchor_pairs(dataset, count=128, seed=0)

    assert [(anchor.anchor_id, anchor.episode_index, anchor.frame_index) for anchor in anchors] == [
        (anchor.anchor_id, anchor.episode_index, anchor.frame_index) for anchor in repeated
    ]
    assert len(anchors) == len({anchor.anchor_id for anchor in anchors}) == 128
    assert {anchor.episode_index for anchor in anchors} == {5, 7, 14, 39, 49}
    for anchor in anchors:
        assert len(anchor.pairs) == 8
        assert tuple(int(pair.delay_steps.item()) for pair in anchor.pairs) == tuple(range(1, 9))
        assert all(pair.episode_index == anchor.episode_index for pair in anchor.pairs)
        assert all(pair.frame_index == anchor.frame_index for pair in anchor.pairs)
        assert all(
            pair.future_frame_index == anchor.frame_index + delay
            for delay, pair in enumerate(anchor.pairs, 1)
        )


class _DeltaTwoPredictor(torch.nn.Module):
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
            delta_tokens=tuple(torch.full_like(tokens, 2) for tokens in image_tokens),
            predicted_error=torch.zeros_like(delay_steps, dtype=torch.float32),
        )


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
        first_token = float(future_image_tokens[0][0, 0, 0].item())
        state_value = float(batch[OBS_STATE][0, 0].item())
        self.calls.append(
            {
                "batch": {key: value.clone() for key, value in batch.items()},
                "tokens": tuple(value.clone() for value in future_image_tokens),
                "masks": tuple(value.clone() for value in future_image_token_masks),
                "noise_before_mutation": noise.clone(),
                "noise_data_ptr": noise.data_ptr(),
            }
        )
        noise.add_(100)
        base = first_token + state_value / 100
        return torch.tensor([[[base, base * 2]]], dtype=torch.float32)


def _four_path_pair(delay_steps: int) -> FutureLatentPair:
    return _pair(
        delay_steps=delay_steps,
        frame_index=10,
        current_tokens=(torch.ones(2, 2), torch.ones(1, 2)),
        target_tokens=(torch.full((2, 2), 4.0), torch.full((1, 2), 4.0)),
        current_state=torch.full((32,), 10.0),
        future_state=torch.full((32,), 20.0),
    )


def test_four_path_uses_one_noise_across_delays_independent_clones_and_exact_token_state_paths() -> None:
    policy = _RecordingPolicy()
    predictor = _DeltaTwoPredictor()
    noise = torch.zeros(1, 2, 2)

    records = tuple(
        evaluation.evaluate_four_path_pair(
            policy,
            predictor,
            _four_path_pair(delay),
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
    assert all(torch.equal(call["noise_before_mutation"], torch.zeros_like(noise)) for call in policy.calls)
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
        assert all(
            torch.equal(call["batch"][OBS_LANGUAGE_TOKENS], current["batch"][OBS_LANGUAGE_TOKENS])
            for call in (oracle, predicted, teacher)
        )

    first = records[0]
    assert first.policy_output.current_vs_teacher.l1 == pytest.approx(4.65)
    assert first.policy_output.current_vs_teacher.rmse == pytest.approx(24.025**0.5)
    assert first.policy_output.oracle_visual_vs_teacher.l1 == pytest.approx(0.15)
    assert first.policy_output.predicted_visual_vs_teacher.l1 == pytest.approx(1.65)
    assert first.post_policy.current_vs_teacher.l1 == pytest.approx(9.3)
    assert first.post_policy.predicted_visual_vs_teacher.rmse == pytest.approx(
        first.policy_output.predicted_visual_vs_teacher.rmse * 2
    )


def test_four_path_stops_before_policy_call_when_future_language_differs() -> None:
    policy = _RecordingPolicy()
    pair = replace(_four_path_pair(1), future_language_tokens=torch.tensor([99, 12, 0]))

    with pytest.raises(ValueError, match="language rollout is not authorized"):
        evaluation.evaluate_four_path_pair(
            policy,
            _DeltaTwoPredictor(),
            pair,
            torch.zeros(1, 2, 2),
            lambda action: action,
            anchor_id=1,
        )

    assert policy.calls == []


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
        episode_index=anchor_id % 5,
        frame_index=anchor_id,
        future_frame_index=anchor_id + delay_steps,
        delay_steps=delay_steps,
        policy_output=policy_output,
        post_policy=post_policy,
    )


def test_action_aggregate_applies_primary_gate_and_keeps_negative_and_secondary_undefined_capture() -> None:
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
    assert len(summary.anchor_ids) == 128
    assert summary.positive_policy_l1_capture_delay_count == 6
    assert summary.policy_l1_nonpositive_oracle_gap_delays == ()
    assert summary.per_delay[6].policy_output.l1.capture == pytest.approx(-0.0625)
    assert summary.per_delay[6].policy_output.rmse.capture is None
    assert summary.per_delay[6].post_policy.l1.capture is None
    assert summary.per_delay[6].post_policy.rmse.capture == pytest.approx(-0.5)
    assert summary.max_policy_l1_regression == pytest.approx(0.05)
    assert summary.test_protocol_prerequisites_met is True
    assert summary.eligible_for_test is False


def test_nonpositive_primary_l1_gap_is_the_only_capture_hard_stop() -> None:
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

    assert delay.policy_output.l1.oracle_gap == pytest.approx(-0.5)
    assert delay.policy_output.l1.capture_defined is False
    assert delay.policy_output.l1.capture is None
    assert delay.policy_output.rmse.capture == pytest.approx(0.5)
    assert delay.post_policy.l1.capture is None
    assert delay.post_policy.rmse.capture == pytest.approx(-0.5)
    assert summary.policy_l1_nonpositive_oracle_gap_delays == (1,)
    assert summary.test_protocol_prerequisites_met is False


def test_action_aggregate_rejects_nonfinite_record_instead_of_dropping_it() -> None:
    errors = _action_errors(
        current_l1=float("nan"),
        oracle_l1=1.0,
        predicted_l1=1.0,
        current_rmse=1.0,
        oracle_rmse=1.0,
        predicted_rmse=1.0,
    )
    record = _action_record(1, 1, policy_output=errors, post_policy=errors)

    with pytest.raises(ValueError, match="invalid error metric"):
        evaluation.aggregate_action_records([record])


class _PublicPredictorSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.arguments: tuple[object, ...] | None = None

    def _validate_inputs(self, *args) -> None:
        raise AssertionError("benchmark helper must not call a private validation shortcut")

    def forward(self, *args):
        self.arguments = args
        return "public-forward-result"


def _benchmark_inputs() -> benchmark.PredictorBenchmarkInputs:
    return benchmark.PredictorBenchmarkInputs(
        current_image_tokens=(torch.zeros(1, 2, 4), torch.zeros(1, 3, 4)),
        current_image_token_masks=(
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
        ),
        committed_actions=torch.zeros(1, 8, 6),
        committed_mask=torch.tensor([[True, False, False, False, False, False, False, False]]),
        current_state=torch.zeros(1, 32),
        delay_steps=torch.ones(1, dtype=torch.int64),
    )


def test_benchmark_calls_normal_public_predictor_and_policy_chunk_paths() -> None:
    predictor = _PublicPredictorSpy()
    inputs = _benchmark_inputs()

    assert benchmark.call_public_predictor(predictor, inputs) == "public-forward-result"
    assert predictor.arguments == (
        inputs.current_image_tokens,
        inputs.current_image_token_masks,
        inputs.committed_actions,
        inputs.committed_mask,
        inputs.current_state,
        inputs.delay_steps,
    )

    class Policy:
        def predict_action_chunk(self, batch, **kwargs):
            self.call = (batch, kwargs)
            return torch.ones(1, 1, 1)

    policy = Policy()
    batch = {OBS_STATE: torch.zeros(1, 32)}
    noise = torch.zeros(1, 1, 1)
    benchmark.call_public_policy_chunk(policy, batch, noise)
    assert policy.call == (batch, {"noise": noise})


def test_benchmark_measurement_uses_fixed_warmup_samples_sync_and_peak_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    synchronizations = []

    def measured_call():
        nonlocal calls
        calls += 1
        return object()

    timestamps = iter((0.0, 0.001, 0.010, 0.012, 0.020, 0.023))
    monkeypatch.setattr(benchmark, "_synchronize", lambda device: synchronizations.append(device))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(benchmark.torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(benchmark.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda device: 700)

    result = benchmark.measure_cuda_latency(
        measured_call,
        device=torch.device("cuda"),
        warmup_count=2,
        measurement_count=3,
        measure_peak_memory=True,
    )

    assert calls == 5
    assert len(synchronizations) == 11
    assert result.samples_ms == pytest.approx((1.0, 2.0, 3.0))
    assert result.p50_ms == pytest.approx(2.0)
    assert result.p90_ms == pytest.approx(2.8)
    assert result.peak_extra_allocated_bytes == 600


def test_snapshot_resolution_never_requests_unfiltered_dataset_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator_calls: list[dict[str, object]] = []

    def evaluator_snapshot_download(**kwargs):
        evaluator_calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(eval_cli, "snapshot_download", evaluator_snapshot_download)
    eval_cli._resolve_evaluator_snapshots(local_files_only=False)
    evaluator_dataset_call = next(call for call in evaluator_calls if call.get("repo_type") == "dataset")
    assert evaluator_dataset_call["allow_patterns"] == "meta/**"

    benchmark_calls: list[dict[str, object]] = []

    def benchmark_snapshot_download(**kwargs):
        benchmark_calls.append(kwargs)
        return str(tmp_path)

    class FakeMetadata:
        video_keys = ("observation.images.top", "observation.images.wrist")

        def __init__(self, *args, **kwargs):
            pass

        def get_data_file_path(self, episode_index: int) -> Path:
            return Path(f"data/episode_{episode_index}.parquet")

        def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
            return Path(f"videos/{video_key}/episode_{episode_index}.mp4")

    monkeypatch.setattr(benchmark, "snapshot_download", benchmark_snapshot_download)
    monkeypatch.setattr(benchmark, "LeRobotDatasetMetadata", FakeMetadata)
    benchmark._resolve_baseline_snapshots(local_files_only=False)
    benchmark_dataset_calls = [call for call in benchmark_calls if call.get("repo_type") == "dataset"]
    assert benchmark_dataset_calls[0]["allow_patterns"] == "meta/**"
    assert set(benchmark_dataset_calls[1]["allow_patterns"]) == {
        "meta/**",
        *(f"data/episode_{episode}.parquet" for episode in benchmark.VAL_EPISODE_IDS),
        *(
            f"videos/{video_key}/episode_{episode}.mp4"
            for video_key in FakeMetadata.video_keys
            for episode in benchmark.VAL_EPISODE_IDS
        ),
    }


@pytest.mark.parametrize(
    "forbidden",
    [
        ("--split", "test"),
        ("--test-cache", "/tmp/test"),
        ("--risk-threshold", "0.5"),
        ("--fallback", "identity"),
        ("--anchor-count", "1"),
    ],
)
def test_evaluator_cli_has_only_the_five_frozen_options(forbidden: tuple[str, str]) -> None:
    parser = eval_cli.build_parser()
    base = [
        "--val-cache",
        "/tmp/val",
        "--predictor-checkpoint",
        "/tmp/best.pt",
        "--output-dir",
        "/tmp/output",
    ]
    args = parser.parse_args(base)
    assert set(vars(args)) == {
        "val_cache",
        "predictor_checkpoint",
        "output_dir",
        "device",
        "local_files_only",
    }
    with pytest.raises(SystemExit):
        parser.parse_args([*base, *forbidden])


def test_benchmark_cli_has_no_validation_off_and_summaries_keep_val_only_markers() -> None:
    base = [
        "--val-cache",
        "/tmp/val",
        "--predictor-checkpoint",
        "/tmp/best.pt",
        "--output-json",
        "/tmp/benchmark.json",
    ]
    args = benchmark.parse_args(base)
    assert set(vars(args)) == {
        "val_cache",
        "predictor_checkpoint",
        "output_json",
        "device",
        "local_files_only",
    }
    with pytest.raises(SystemExit):
        benchmark.parse_args([*base, "--validation-off"])

    assert eval_cli.evaluation_protocol_markers() == {
        "protocol_complete": True,
        "test_data_read": False,
        "eligible_for_test": False,
        "risk_thresholds": None,
    }
    summary = benchmark.build_benchmark_summary(
        anchor_record={"anchor_id": 1, "episode_index": 5, "frame_index": 0, "delay_steps": 1},
        predictor_record={"p90_ms": 1.0},
        baseline_record={"p90_ms": 100.0},
        device_record={"type": "cuda"},
        provenance={},
    )
    assert summary["protocol"]["warmup_count"] == 50
    assert summary["protocol"]["measurement_count"] == 200
    assert summary["protocol"]["predictor_public_forward"] is True
    assert summary["protocol"]["predictor_input_host_check_included"] is True
    assert summary["protocol"]["predictor_output_host_check_included"] is True
    assert summary["m5_efficiency_eligible"] is True
    assert summary["test_data_read"] is False
    assert summary["eligible_for_test"] is False
    assert summary["risk_thresholds"] is None
