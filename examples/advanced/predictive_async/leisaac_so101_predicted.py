"""L7 binding of the frozen L6 predictor to its independent SO101 policy.

Only candidate validation differs from the production engine. No alternate queue,
worker, predictor math, late handling or SO100 identity is introduced here.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from leisaac_so101_matched import (
    CAMERA_KEYS,
    VLM_REVISION,
    WSAGI_REVISION,
    MotorStatePreprocessor,
    PhysicalActionPostprocessor,
)

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor
from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine

TRAINING_SOURCE = "40e00e1654779d5e6bfec8f2f06ab57a6c697ff1"
SELECTED_EPOCH = 3
PREDICTOR_PARAMETERS = 273824
PREDICTOR_ID = "so101_wsagi_l6_epoch3"


def validate_l6_checkpoint(checkpoint: dict) -> FutureLatentConfig:
    """Reject another model/epoch/domain rather than silently applying old weights."""
    if (
        checkpoint.get("source_commit") != TRAINING_SOURCE
        or checkpoint.get("epoch") != SELECTED_EPOCH
        or checkpoint.get("parameters") != PREDICTOR_PARAMETERS
    ):
        raise ValueError("L7 requires the frozen L6 epoch-3 checkpoint and training source")
    collection = checkpoint["collection_manifest"]
    if (
        collection["candidate"]["policy_revision"] != WSAGI_REVISION
        or collection["candidate"]["vlm_config_tokenizer_revision"] != VLM_REVISION
        or collection["control_fps"] != 30
        or collection["camera_backend"] != "standard"
        or collection["physics"] != "cpu_physx_rtx_v1"
    ):
        raise ValueError("L6 predictor collection does not match the bound SO101 deployment")
    config = FutureLatentConfig(**checkpoint["config"])
    expected = FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, max_cameras=2, risk_head=False)
    if config != expected:
        raise ValueError("L6 predictor architecture differs from the selected model")
    return config


def load_so101_l6_predictor(path: Path, *, device: str) -> LightweightFutureLatentPredictor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = validate_l6_checkpoint(checkpoint)
    predictor = LightweightFutureLatentPredictor(config)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(p.numel() for p in predictor.parameters()) != PREDICTOR_PARAMETERS:
        raise ValueError("L6 predictor parameter count differs")
    predictor.to(device).eval().requires_grad_(False)
    predictor._so101_l6_binding = {
        "id": PREDICTOR_ID,
        "policy_revision": WSAGI_REVISION,
        "training_source": TRAINING_SOURCE,
        "epoch": SELECTED_EPOCH,
        "parameters": PREDICTOR_PARAMETERS,
        "config": asdict(config),
    }
    return predictor


def validate_so101_prediction_candidate(
    policy: Any, preprocessor: Any, postprocessor: Any, predictor: Any
) -> tuple[str, ...]:
    association = getattr(policy, "_so101_matched_candidate", {})
    if (
        association.get("policy_revision") != WSAGI_REVISION
        or association.get("vlm_config_tokenizer_revision") != VLM_REVISION
        or association.get("preprocessor") is not preprocessor
        or association.get("postprocessor") is not postprocessor
        or association.get("execution_steps") != 50
    ):
        raise ValueError("Predicted SO101 requires the exact WSAGI policy and its processor instances")
    if not isinstance(preprocessor, MotorStatePreprocessor) or not isinstance(
        postprocessor, PhysicalActionPostprocessor
    ):
        raise ValueError("Predicted SO101 requires the native motor/state processor adapters")
    config = policy.config
    expected_fields = {
        "type": "smolvla",
        "max_state_dim": 32,
        "max_action_dim": 32,
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_steps": 10,
        "use_amp": False,
        "use_peft": False,
        "empty_cameras": 0,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
    }
    if any(getattr(config, field, None) != expected for field, expected in expected_fields.items()):
        raise ValueError("Predicted SO101 policy configuration changed")
    if tuple(config.image_features) != CAMERA_KEYS or any(
        tuple(feature.shape) != (3, 480, 640) for feature in config.image_features.values()
    ):
        raise ValueError("Predicted SO101 requires native ordered front/wrist camera features")
    if config.rtc_config is not None and config.rtc_config.enabled:
        raise ValueError("L7 does not enable full RTC guidance")
    if any(
        tuple(getattr(config, field).shape) != (6,) for field in ("robot_state_feature", "action_feature")
    ):
        raise ValueError("Predicted SO101 requires six state/action scalars")
    binding = getattr(predictor, "_so101_l6_binding", {})
    if (
        binding.get("id") != PREDICTOR_ID
        or binding.get("policy_revision") != WSAGI_REVISION
        or binding.get("training_source") != TRAINING_SOURCE
        or binding.get("epoch") != SELECTED_EPOCH
    ):
        raise ValueError("Predicted SO101 requires the independently loaded L6 predictor")
    return CAMERA_KEYS


class SO101PredictiveAsyncInferenceEngine(PredictiveAsyncInferenceEngine):
    """Same production execution, with the explicitly qualified SO101 binding."""

    def _validate_prediction_candidate(self, policy, preprocessor, postprocessor, predictor):
        return validate_so101_prediction_candidate(policy, preprocessor, postprocessor, predictor)
