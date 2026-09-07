"""L12 learned model-ready proprioception; never reads a future observation."""

from __future__ import annotations

import torch
from torch import nn


class FutureStateResidual(nn.Module):
    """Predict six joint-state residuals from the already committed action prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 6)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state, actions, mask, delay):
        batch = state.shape[0]
        if state.shape != (batch, 32) or actions.shape != (batch, 8, 6):
            raise ValueError("Expected model-ready state[B,32] and committed actions[B,8,6]")
        if mask.shape != (batch, 8) or mask.dtype != torch.bool or delay.shape != (batch,):
            raise ValueError("Expected ordered boolean prefix mask and batch delay")
        expected = torch.arange(8, device=state.device)[None] < delay[:, None]
        valid = torch.stack(
            [
                ((delay >= 1) & (delay <= 8)).all(),
                (mask == expected).all(),
                torch.isfinite(state).all(),
                (torch.isfinite(actions) | ~mask[..., None]).all(),
            ]
        ).all()
        if not bool(valid):
            raise ValueError("Delay/prefix/finiteness contract changed")
        safe = actions.masked_fill(~mask[..., None], 0)
        features = torch.cat((state[:, :6], safe.flatten(1), mask.to(state.dtype), delay[:, None] / 8), 1)
        residual = self.network(features.to(self.network[0].weight.dtype)).to(state.dtype)
        result = torch.cat((state[:, :6] + residual, state[:, 6:]), 1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite predicted state")
        return result


def state_pairs(episode: dict) -> tuple[torch.Tensor, ...]:
    """Only same-commitment prefixes and real, pre-terminal successor states."""
    states, actions, chunks = episode["states"], episode["actions"], episode["chunk_ids"]
    pairs = [
        (t, d)
        for t in range(len(actions))
        for d in range(1, 9)
        if t + d < len(states) and t + d <= len(actions) and bool((chunks[t : t + d] == chunks[t]).all())
    ]
    if not pairs:
        raise ValueError("No causal state prediction pairs")
    starts = torch.tensor([t for t, _ in pairs])
    delays = torch.tensor([d for _, d in pairs])
    mask = torch.arange(8)[None] < delays[:, None]
    positions = (starts[:, None] + torch.arange(8)[None]).clamp(max=len(actions) - 1)
    prefix = actions[positions].masked_fill(~mask[..., None], 0)
    return states[starts].float(), prefix.float(), mask, delays, states[starts + delays].float()
