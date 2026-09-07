"""Separately bound L13 context predictor: learned future state, never oracle state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from leisaac_so101_predicted import validate_so101_prediction_candidate
from so101_feasible_actions import SO101FeasibleAsyncEngine
from so101_future_state import FutureStateResidual
from torch import nn

STATE_SOURCE = "bf4e025dbf0cacf4777289b90498a8b70d4974fa"
STATE_EPOCH = 29


@dataclass(frozen=True)
class JointContextPrediction:
    delta_tokens: tuple[torch.Tensor, ...]
    predicted_error: torch.Tensor
    future_state: torch.Tensor
    variant: str


def load_future_state(path: Path, device: str) -> FutureStateResidual:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("kind") != "so101_future_state_l12_v1"
        or saved.get("source_commit") != STATE_SOURCE
        or saved.get("epoch") != STATE_EPOCH
        or saved.get("parameters") != 25478
    ):
        raise ValueError("The joint deployment requires frozen L12 epoch29 state weights")
    model = FutureStateResidual()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    model._l12_binding = {"source": STATE_SOURCE, "epoch": STATE_EPOCH, "parameters": 25478}
    return model


class JointContextPredictor(nn.Module):
    def __init__(self, visual, state_model: FutureStateResidual, variant: str) -> None:
        super().__init__()
        if variant not in ("joint", "state_only"):
            raise ValueError("L13 requires an explicit joint/state-only variant")
        if getattr(state_model, "_l12_binding", {}).get("source") != STATE_SOURCE:
            raise ValueError("Unbound state predictor")
        self.visual, self.state_model, self.variant = visual, state_model, variant
        self.config = visual.config
        self._so101_joint_binding = {
            "id": "so101_l13_" + variant,
            "variant": variant,
            "visual_parent": visual._so101_l6_binding,
            "state_predictor": dict(state_model._l12_binding),
            "future_state_source": "current_state_and_committed_normalized_prefix",
            "risk_thresholds": None,
        }

    def forward(self, tokens, masks, actions, prefix, state, delay):
        future_state = self.state_model(state, actions, prefix, delay)
        if self.variant == "joint":
            visual = self.visual(tokens, masks, actions, prefix, state, delay)
            delta, error = visual.delta_tokens, visual.predicted_error
        else:
            delta = tuple(torch.zeros_like(z) for z in tokens)
            error = state.new_zeros(state.shape[0])
        return JointContextPrediction(delta, error, future_state, self.variant)


class SO101JointAsyncEngine(SO101FeasibleAsyncEngine):
    def _validate_prediction_candidate(self, policy, preprocessor, postprocessor, predictor):
        if not isinstance(predictor, JointContextPredictor):
            raise ValueError("Joint runtime cannot silently accept a visual-only predictor")
        return validate_so101_prediction_candidate(policy, preprocessor, postprocessor, predictor.visual)

    def _future_state_override(self, prediction, metrics):
        if not isinstance(prediction, JointContextPrediction):
            raise ValueError("Missing causal future-state prediction")
        if metrics is not None:
            metrics["context_variant"] = prediction.variant
            metrics["state_predictor_calls"] = 1
            metrics["visual_predictor_calls"] = int(prediction.variant == "joint")
            metrics["future_state_source"] = "learned_current_state_and_committed_prefix"
        return prediction.future_state
