"""Experimental absolute-action-index noise; not RTC or a continuity guarantee."""

from __future__ import annotations

import torch


class ActionIndexedNoise:
    def __init__(self, policy_seed: int, *, device: str, limit: int = 3600) -> None:
        self.policy_seed, self.device, self.limit = policy_seed, device, limit
        # Separate fixed-shape draws preserve the old 50x32 bootstrap exactly.
        fields = []
        for start in range(0, limit + 50, 50):
            generator = torch.Generator(device=device).manual_seed(policy_seed * 100000 + start)
            fields.append(torch.randn((1, 50, 32), generator=generator, device=device))
        self.field = torch.cat(fields, dim=1)

    def window(self, action_start: int) -> torch.Tensor:
        if not isinstance(action_start, int) or not 0 <= action_start <= self.limit:
            raise ValueError("Absolute noise window is outside the registered action horizon")
        return self.field[:, action_start : action_start + 50].clone()

    def __call__(self, request_step: int, takeover_index: int) -> torch.Tensor:
        if takeover_index != request_step + 7:
            raise ValueError("Noise is indexed by the intended seven-step takeover, not the request")
        return self.window(takeover_index)
