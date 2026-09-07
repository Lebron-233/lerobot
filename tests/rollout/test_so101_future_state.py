"""L12 future-state causality and coherent-context summary boundaries."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from so101_future_state import FutureStateResidual, state_pairs  # noqa: E402
from validate_so101_joint_context import summarize_contexts  # noqa: E402


def test_zero_initialization_and_padding_are_exact():
    model = FutureStateResidual()
    state = torch.randn(2, 32)
    actions = torch.randn(2, 8, 6)
    delay = torch.tensor([1, 8])
    mask = torch.arange(8)[None] < delay[:, None]
    actions[0, 1:] = float("nan")
    torch.testing.assert_close(model(state, actions, mask, delay), state, rtol=0, atol=0)
    model.network[-1].bias.data.fill_(1)
    predicted = model(state, actions, mask, delay)
    torch.testing.assert_close(predicted[:, 6:], state[:, 6:], rtol=0, atol=0)
    assert torch.isfinite(predicted).all()


def test_reject_uncommitted_holes_and_out_of_range_delay():
    model = FutureStateResidual()
    state, actions = torch.zeros(1, 32), torch.zeros(1, 8, 6)
    with pytest.raises(ValueError):
        model(state, actions, torch.zeros(1, 8, dtype=torch.bool), torch.tensor([2]))
    with pytest.raises(ValueError):
        model(state, actions, torch.ones(1, 8, dtype=torch.bool), torch.tensor([9]))


def test_pairs_do_not_read_new_chunk_or_post_reset_target():
    episode = {
        "states": torch.arange(5).float()[:, None].expand(5, 32),
        "actions": torch.arange(5).float()[:, None].expand(5, 6),
        "chunk_ids": torch.tensor([0, 0, 1, 1, 1]),
    }
    states, actions, masks, delays, targets = state_pairs(episode)
    assert len(states) == 6
    for s, a, m, d, target in zip(states, actions, masks, delays, targets, strict=True):
        t = int(s[0])
        assert torch.all(episode["chunk_ids"][t : t + int(d)] == episode["chunk_ids"][t])
        assert int(target[0]) < 5 and int(target[0]) == t + int(d)
        assert torch.equal(a[~m], torch.zeros_like(a[~m]))


def test_joint_benefit_does_not_imply_visual_contribution():
    rows = [
        {
            "episode": e,
            "identity_l1_25": 1.0,
            "visual_only_l1_25": 1.1,
            "state_only_l1_25": 0.5,
            "joint_l1_25": 0.6,
            "oracle_visual_only_l1_25": 0.8,
            "oracle_state_only_l1_25": 0.4,
        }
        for e in range(6)
        for _ in range(48)
    ]
    report = summarize_contexts(rows, 288)
    assert report["contrasts"]["identity"]["stable_gate"]
    assert report["contrasts"]["visual_only"]["stable_gate"]
    assert not report["contrasts"]["state_only"]["stable_gate"]
    assert report["contrasts"]["state_only"]["reduction_percent"] < 0
    assert not summarize_contexts(rows[:-1], 288)["contrasts"]["identity"]["stable_gate"]
