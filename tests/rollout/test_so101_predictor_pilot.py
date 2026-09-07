"""Causal prefix and masked scoring tests for the separately pre-split L6 pilot."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from train_so101_predictor_pilot import ForecastPairs, per_sample_errors, valid_pairs


def test_prefix_never_uses_actions_from_a_future_observation_chunk():
    pairs = valid_pairs(101, torch.arange(100) // 50)
    assert (49, 1) in pairs and (49, 2) not in pairs
    assert (42, 8) in pairs and (43, 8) not in pairs
    assert (50, 8) in pairs and (99, 1) in pairs
    # Exclude terminal auto-reset observation: last action may have no valid successor.
    terminal_pairs = valid_pairs(100, torch.arange(100) // 50)
    assert (99, 1) not in terminal_pairs
    assert all(t + d < 100 for t, d in terminal_pairs)


def test_masked_forecast_metrics_ignore_invalid_tokens_and_identity_is_exact():
    target = torch.ones(2, 2, 3, 4)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)
    mask[:, :, -1] = False
    prediction = target.clone()
    prediction[:, :, -1] = 100
    smooth, cosine = per_sample_errors(prediction, target, mask)
    torch.testing.assert_close(smooth, torch.zeros(2))
    torch.testing.assert_close(cosine, torch.zeros(2))
    smooth, _ = per_sample_errors(target + 1, target, mask)
    torch.testing.assert_close(smooth, torch.full((2,), 0.5))


def test_saved_pre_failure_prefix_remains_usable_without_cross_split_reads(tmp_path):
    episode = {
        "tokens": torch.zeros(4, 2, 3, 4),
        "masks": torch.ones(4, 2, 3, dtype=torch.bool),
        "states": torch.zeros(4, 32),
        "actions": torch.ones(3, 6),
        "chunk_ids": torch.zeros(3, dtype=torch.long),
    }
    file = tmp_path / "prior_episode.pt"
    torch.save(episode, file)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"episode_files": {"0": str(file), "5": "must_not_be_opened"}})
    )
    data = ForecastPairs(tmp_path, [0])
    assert len(data) == 6
    sample = data[0]
    assert int(sample[-1]) == 1
    torch.testing.assert_close(sample[-2][0], torch.ones(6))
    torch.testing.assert_close(sample[-2][1:], torch.zeros(7, 6))
