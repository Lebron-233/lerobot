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

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import TransitionKey, UnnormalizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.rollout.inference.oracle_evaluation import (
    ActionTriplet,
    OracleAnchorCandidate,
    OracleEvaluationError,
    OracleEvaluationRecord,
    PairedActionErrors,
    aggregate_by_delay,
    make_evaluation_record,
    postprocess_action_triplet,
    remap_checkpoint_action_stats,
    run_with_shared_noise,
    select_common_anchor_ids,
    slice_temporal_sample,
)
from lerobot.utils.constants import ACTION

_FRONT = "observation.images.front"
_WRIST = "observation.images.wrist"
_STATE = "observation.state"
_TEMPORAL_KEYS = (_FRONT, _WRIST, _STATE)


def _temporal_sample() -> dict[str, object]:
    return {
        _FRONT: torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1),
        f"{_FRONT}_is_pad": torch.zeros(4, dtype=torch.bool),
        _WRIST: torch.arange(10, 14, dtype=torch.float32).reshape(4, 1, 1, 1),
        f"{_WRIST}_is_pad": torch.zeros(4, dtype=torch.bool),
        _STATE: torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
        f"{_STATE}_is_pad": torch.zeros(4, dtype=torch.bool),
        "task": "pick the cube",
        "episode_index": torch.tensor(3),
        "frame_index": torch.tensor(7),
    }


def test_slice_temporal_sample_builds_independent_full_future_observation() -> None:
    sample = _temporal_sample()

    future = slice_temporal_sample(sample, temporal_keys=_TEMPORAL_KEYS, step=2)

    torch.testing.assert_close(future[_FRONT], torch.tensor([[[2.0]]]))
    torch.testing.assert_close(future[_WRIST], torch.tensor([[[12.0]]]))
    torch.testing.assert_close(future[_STATE], torch.tensor([4.0, 5.0]))
    assert future["task"] == "pick the cube"
    assert all(f"{key}_is_pad" not in future for key in _TEMPORAL_KEYS)

    future[_STATE][0] = -100
    assert sample[_STATE][2, 0].item() == 4.0


@pytest.mark.parametrize("padded_key", [_FRONT, _WRIST, _STATE])
def test_slice_temporal_sample_rejects_camera_or_state_episode_padding(padded_key: str) -> None:
    sample = _temporal_sample()
    sample[f"{padded_key}_is_pad"][2] = True

    with pytest.raises(OracleEvaluationError, match=f"{padded_key!r}.*episode padding"):
        slice_temporal_sample(sample, temporal_keys=_TEMPORAL_KEYS, step=2)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda sample: sample.__setitem__(_STATE, sample[_STATE][:-1]), "disagree on horizon"),
        (
            lambda sample: sample.__setitem__(f"{_STATE}_is_pad", torch.zeros(3, dtype=torch.bool)),
            "disagree on horizon",
        ),
    ],
)
def test_slice_temporal_sample_reports_inconsistent_horizons(mutate, match: str) -> None:
    sample = _temporal_sample()
    mutate(sample)

    with pytest.raises(OracleEvaluationError, match=match):
        slice_temporal_sample(sample, temporal_keys=_TEMPORAL_KEYS, step=2)


def _candidates(episode_count: int, frames_per_episode: int) -> list[OracleAnchorCandidate]:
    return [
        OracleAnchorCandidate(
            anchor_id=episode * 100 + frame,
            episode_index=episode,
            frame_index=frame,
            episode_length=frames_per_episode,
        )
        for episode in range(episode_count)
        for frame in range(frames_per_episode)
    ]


def test_common_anchor_selection_is_deterministic_episode_aware_and_valid_for_all_delays() -> None:
    candidates = _candidates(episode_count=3, frames_per_episode=30)

    selected = select_common_anchor_ids(candidates, delays=(1, 8, 20), count=12, seed=7)
    repeated = select_common_anchor_ids(tuple(reversed(candidates)), delays=(1, 8, 20), count=12, seed=7)

    assert selected == repeated
    selected_candidates = {candidate.anchor_id: candidate for candidate in candidates}
    assert all(
        selected_candidates[anchor_id].frame_index + 20 < selected_candidates[anchor_id].episode_length
        for anchor_id in selected
    )
    assert {selected_candidates[anchor_id].episode_index for anchor_id in selected} == {0, 1, 2}
    assert len(selected) == len(set(selected)) == 12


def test_common_anchor_selection_fails_instead_of_shrinking_below_128() -> None:
    candidates = _candidates(episode_count=1, frames_per_episode=147)
    # With max delay 20, only frame indices 0..126 are valid: 127 common anchors.
    with pytest.raises(OracleEvaluationError, match=r"has 127 candidates; required 128"):
        select_common_anchor_ids(candidates, delays=(1, 8, 20), seed=0)


def test_run_with_shared_noise_gives_each_path_an_independent_equal_clone() -> None:
    noise = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    received: list[torch.Tensor] = []

    def predictor(value: torch.Tensor) -> torch.Tensor:
        received.append(value)
        return value + len(received)

    outputs = run_with_shared_noise(
        noise,
        current=predictor,
        oracle_visual=predictor,
        full_future_teacher=predictor,
    )

    assert len(received) == 3
    assert all(torch.equal(value, noise) for value in received)
    assert len({value.data_ptr() for value in received}) == 3
    torch.testing.assert_close(outputs.current, noise + 1)
    torch.testing.assert_close(outputs.oracle_visual, noise + 2)
    torch.testing.assert_close(outputs.full_future_teacher, noise + 3)


def test_remap_checkpoint_action_stats_selects_only_the_approved_embodiment() -> None:
    checkpoint_stats = {
        "so100.buffer.action.mean": torch.tensor([10.0, -5.0]),
        "so100.buffer.action.std": torch.tensor([2.0, 4.0]),
        "so100-blue.buffer.action.mean": torch.tensor([100.0, 100.0]),
        "so100-blue.buffer.action.std": torch.tensor([10.0, 10.0]),
    }

    mapped = remap_checkpoint_action_stats(
        checkpoint_stats,
        source_key="so100.buffer.action",
        action_dim=2,
    )

    torch.testing.assert_close(mapped[ACTION]["mean"], torch.tensor([10.0, -5.0]))
    torch.testing.assert_close(mapped[ACTION]["std"], torch.tensor([2.0, 4.0]))
    mapped[ACTION]["mean"][0] = -100
    assert checkpoint_stats["so100.buffer.action.mean"][0].item() == 10.0


def test_remap_checkpoint_action_stats_fails_when_approved_stats_are_missing() -> None:
    checkpoint_stats = {
        "so100-blue.buffer.action.mean": torch.zeros(2),
        "so100-blue.buffer.action.std": torch.ones(2),
    }

    with pytest.raises(OracleEvaluationError, match=r"missing.*so100\.buffer\.action\.mean"):
        remap_checkpoint_action_stats(
            checkpoint_stats,
            source_key="so100.buffer.action",
            action_dim=2,
        )


def test_remap_checkpoint_action_stats_fails_on_wrong_action_dimension() -> None:
    checkpoint_stats = {
        "so100.buffer.action.mean": torch.zeros(3),
        "so100.buffer.action.std": torch.ones(3),
    }

    with pytest.raises(OracleEvaluationError, match=r"shape \(2,\).*got \(3,\)"):
        remap_checkpoint_action_stats(
            checkpoint_stats,
            source_key="so100.buffer.action",
            action_dim=2,
        )


def test_checkpoint_postprocessing_is_non_identity_and_keeps_policy_output_unchanged() -> None:
    mapped_stats = remap_checkpoint_action_stats(
        {
            "so100.buffer.action.mean": torch.tensor([10.0, -5.0]),
            "so100.buffer.action.std": torch.tensor([2.0, 4.0]),
        },
        source_key="so100.buffer.action",
        action_dim=2,
    )
    unnormalizer = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats=mapped_stats,
    )

    def postprocessor(action: torch.Tensor) -> torch.Tensor:
        return unnormalizer(create_transition(action=action))[TransitionKey.ACTION]

    policy_output = ActionTriplet(
        current=torch.tensor([[[1.0, 2.0]]]),
        oracle_visual=torch.tensor([[[0.5, -0.5]]]),
        full_future_teacher=torch.tensor([[[0.0, 1.0]]]),
    )
    original_chunks = tuple(chunk.clone() for chunk in policy_output.__dict__.values())

    post_policy = postprocess_action_triplet(postprocessor, policy_output)

    torch.testing.assert_close(post_policy.current, torch.tensor([[[12.0, 3.0]]]))
    assert not torch.equal(post_policy.current, policy_output.current)
    for chunk, original in zip(policy_output.__dict__.values(), original_chunks, strict=True):
        torch.testing.assert_close(chunk, original)


def _errors(current_l1: float, current_l2: float, oracle_l1: float, oracle_l2: float):
    return PairedActionErrors(
        current_vs_teacher_l1=current_l1,
        current_vs_teacher_l2_rmse=current_l2,
        oracle_visual_vs_teacher_l1=oracle_l1,
        oracle_visual_vs_teacher_l2_rmse=oracle_l2,
    )


def test_make_record_computes_policy_and_post_policy_l1_and_rmse() -> None:
    teacher = torch.zeros(1, 2, 2)
    policy = ActionTriplet(
        current=torch.tensor([[[1.0, -1.0], [1.0, -1.0]]]),
        oracle_visual=torch.tensor([[[0.5, -0.5], [0.5, -0.5]]]),
        full_future_teacher=teacher,
    )
    post_policy = ActionTriplet(
        current=policy.current * 2,
        oracle_visual=policy.oracle_visual * 2,
        full_future_teacher=teacher,
    )

    record = make_evaluation_record(
        anchor_id=9,
        episode_index=2,
        frame_index=4,
        delay_steps=3,
        policy_output=policy,
        post_policy=post_policy,
    )

    assert record.policy_output == _errors(1.0, 1.0, 0.5, 0.5)
    assert record.post_policy == _errors(2.0, 2.0, 1.0, 1.0)


def _record(
    anchor_id: int,
    delay: int,
    *,
    policy: PairedActionErrors,
    post_policy: PairedActionErrors,
) -> OracleEvaluationRecord:
    return OracleEvaluationRecord(
        anchor_id=anchor_id,
        episode_index=anchor_id // 10,
        frame_index=anchor_id,
        delay_steps=delay,
        policy_output=policy,
        post_policy=post_policy,
    )


def test_aggregate_by_delay_uses_ratio_of_means_in_both_action_spaces() -> None:
    records = [
        _record(1, 1, policy=_errors(1, 2, 0.5, 1), post_policy=_errors(2, 4, 1.5, 3)),
        _record(2, 1, policy=_errors(3, 4, 1.5, 2), post_policy=_errors(6, 8, 2.5, 5)),
        _record(1, 2, policy=_errors(2, 3, 1, 1), post_policy=_errors(4, 6, 2, 3)),
        _record(2, 2, policy=_errors(4, 5, 3, 4), post_policy=_errors(8, 10, 6, 8)),
    ]

    summary = aggregate_by_delay(records)

    assert summary.anchor_ids == (1, 2)
    assert summary.total_record_count == 4
    delay_1 = summary.delays[0]
    assert delay_1.sample_count == 2
    assert delay_1.policy_output.current_vs_teacher_l1_mean == 2
    assert delay_1.policy_output.oracle_visual_vs_teacher_l1_mean == 1
    assert delay_1.policy_output.relative_l1_error_reduction == 0.5
    assert delay_1.policy_output.relative_l2_error_reduction == 0.5
    # Ratio of means: (4 - 2) / 4, rather than averaging per-sample ratios.
    assert delay_1.post_policy.relative_l1_error_reduction == 0.5
    assert delay_1.post_policy.relative_l2_error_reduction == pytest.approx(1 / 3)


def test_aggregate_returns_none_when_relative_reduction_denominator_is_zero() -> None:
    zero = _errors(0, 0, 0, 0)
    summary = aggregate_by_delay([_record(1, 1, policy=zero, post_policy=zero)])

    assert summary.delays[0].policy_output.relative_l1_error_reduction is None
    assert summary.delays[0].policy_output.relative_l2_error_reduction is None
    assert summary.delays[0].post_policy.relative_l1_error_reduction is None
    assert summary.delays[0].post_policy.relative_l2_error_reduction is None


def test_aggregate_rejects_per_delay_cohort_drift() -> None:
    errors = _errors(1, 1, 0.5, 0.5)
    records = [
        _record(1, 1, policy=errors, post_policy=errors),
        _record(2, 1, policy=errors, post_policy=errors),
        _record(1, 2, policy=errors, post_policy=errors),
        _record(3, 2, policy=errors, post_policy=errors),
    ]

    with pytest.raises(OracleEvaluationError, match="does not use the common anchor cohort"):
        aggregate_by_delay(records)
