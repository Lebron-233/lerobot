"""Causal input selection and gradient isolation of the action-aware pilot."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from train_so101_action_distillation import causal_pairs, query_from_data, training_loss  # noqa: E402

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


def test_action_distillation_never_uses_actions_from_a_future_replan():
    data = {
        "tokens": torch.zeros(101, 2, 3, 4),
        "actions": torch.zeros(100, 6),
        "chunk_ids": torch.arange(100) // 50,
    }
    pairs = causal_pairs(data, stride=1, delays=(1, 8))
    assert (42, 8) in pairs and (43, 8) not in pairs
    assert (92, 8) in pairs and (99, 8) not in pairs


def test_query_uses_current_state_and_exact_already_committed_prefix():
    data = {
        "tokens": torch.arange(101.0)[:, None, None, None].expand(-1, 2, 3, 4),
        "masks": torch.ones(101, 2, 3, dtype=torch.bool),
        "states": torch.arange(101.0)[:, None].expand(-1, 32),
        "actions": torch.arange(100.0)[:, None].expand(-1, 6),
        "chunk_ids": torch.arange(100) // 50,
        "language": {},
    }
    query = query_from_data(data, 25, 8, noise_seed=1, device="cpu")
    assert (query["state"] == 25).all() and (query["future"] == 33).all()
    torch.testing.assert_close(query["actions"][0], data["actions"][25:33])


def test_action_loss_backpropagates_only_to_predictor_with_current_state():
    states_seen = []

    class FrozenDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 6).requires_grad_(False)

        def sample_actions(self, images, image_masks, language, language_mask, state, **kwargs):
            states_seen.append(state)
            x = torch.stack(kwargs["future_image_tokens"], dim=1).mean((1, 2))
            return self.projection(x)[:, None, :].expand(-1, 50, -1)

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.delta = torch.nn.Parameter(torch.zeros(4))

        def forward(self, tokens, *args):
            return SimpleNamespace(delta_tokens=tuple(torch.zeros_like(z) + self.delta for z in tokens))

    predictor, decoder = Predictor(), FrozenDecoder()
    query = {
        "z": torch.ones(1, 2, 3, 4),
        "future": torch.ones(1, 2, 3, 4) * 1.1,
        "masks": torch.ones(1, 2, 3, dtype=torch.bool),
        "state": torch.ones(1, 32),
        "actions": torch.zeros(1, 8, 6),
        "valid": torch.ones(1, 8, dtype=torch.bool),
        "delay": torch.tensor([8]),
        "noise": torch.zeros(1, 50, 32),
        "language": {
            OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
        },
    }
    loss, _ = training_loss(SimpleNamespace(model=decoder), lambda x: x, predictor, query)
    loss.backward()
    assert torch.isfinite(predictor.delta.grad).all() and predictor.delta.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in decoder.parameters())
    assert len(states_seen) == 2 and all(state is query["state"] for state in states_seen)
