"""Explicit L8 execution contract: projection before policy-space commitment."""

from __future__ import annotations

from pathlib import Path

import torch
from leisaac_so101_predicted import SO101PredictiveAsyncInferenceEngine

CONTRACT = "so101_feasible_actions_v1"


class FeasibleActionProjector:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (6,) or std.shape != (6,) or not bool((std > 0).all()):
            raise ValueError("Projection requires the checkpoint's six positive action standard deviations")
        self.mean, self.std = mean, std
        self.low = mean.new_tensor([-100, -100, -100, -100, -100, 0])
        self.high = mean.new_tensor([100] * 6)
        self.records: list[dict] = []

    @classmethod
    def from_snapshot(cls, snapshot: Path, device: str):
        from safetensors.torch import load_file

        stats = load_file(str(snapshot / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        return cls(stats["action.mean"].to(device), stats["action.std"].to(device))

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != 6 or not bool(torch.isfinite(actions).all()):
            raise ValueError("Cannot project malformed or nonfinite normalized actions")
        native = actions * self.std + self.mean
        feasible = native.clamp(min=self.low, max=self.high)
        changed = native != feasible
        normalized = torch.where(changed, (feasible - self.mean) / self.std, actions)
        self.records.append(
            {
                "components": native.numel(),
                "projected_components": int(changed.sum().item()),
                "max_native_adjustment": float((native - feasible).abs().max().item()),
            }
        )
        return normalized


class SO101FeasibleAsyncEngine(SO101PredictiveAsyncInferenceEngine):
    """The existing engine/queue with mutually consistent feasible commitments."""

    def __init__(self, *, action_projector: FeasibleActionProjector, **kwargs) -> None:
        self.action_projector = action_projector
        super().__init__(**kwargs)

    def _prepare_queue_actions(self, actions: torch.Tensor, metrics: dict):
        projected = self.action_projector(actions)
        metrics["action_projection"] = dict(self.action_projector.records[-1])
        return projected.squeeze(0).clone(), self._postprocessor(projected).squeeze(0)
