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

"""Load the frozen future-latent candidate without opening its offline caches."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .configuration_future_latent import FutureLatentConfig
from .future_latent import LightweightFutureLatentPredictor

POLICY_REPO_ID = "lerobot/smolvla_base"
POLICY_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
DATASET_REPO_ID = "lerobot/svla_so100_pickplace"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
RAW_CAMERA_KEYS = ("observation.images.top", "observation.images.wrist")
POLICY_CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
CAMERA_RENAME_MAP = dict(zip(RAW_CAMERA_KEYS, POLICY_CAMERA_KEYS, strict=True))
RUNTIME_SCALAR_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
DATASET_FEATURE_NAMES = (
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper",
)
RAW_IMAGE_SHAPE = (480, 640, 3)
TRAINER_SHA = "9e618076f617751c297d92626ad422dbbf30c03b"
CACHE_PRODUCER_SHA = "eff8be608c899d0841ad5967d80d5d726cbe4394"
BEST_EPOCH = 29
BEST_GLOBAL_STEP = 57_750


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Frozen future-latent {name}: expected {expected!r}, got {actual!r}")


def _frozen_predictor_config() -> FutureLatentConfig:
    return FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, enabled=True)


def _validate_recorded_provenance(provenance: Any, *, split: str) -> None:
    """Consume the checkpoint's recorded identity, never the recorded cache paths."""
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Frozen future-latent checkpoint is missing {split} cache provenance")
    expected_fields = {
        "split": split,
        "complete_split": True,
        "producer_git_sha": CACHE_PRODUCER_SHA,
        "policy_camera_order": list(POLICY_CAMERA_KEYS),
        "token_scaling_convention": "native_post_sqrt_hidden_dim",
    }
    for field, expected in expected_fields.items():
        _require_equal(provenance.get(field), expected, name=f"{split} {field}")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"Frozen future-latent {split} provenance is missing pinned inputs")
    for source, (repo_id, revision) in {
        "dataset": (DATASET_REPO_ID, DATASET_REVISION),
        "checkpoint": (POLICY_REPO_ID, POLICY_REVISION),
        "vlm": (VLM_REPO_ID, VLM_REVISION),
    }.items():
        entry = inputs.get(source)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Frozen future-latent {split} provenance is missing {source}")
        for field, expected in {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": revision,
        }.items():
            _require_equal(entry.get(field), expected, name=f"{split} {source} {field}")
    semantics = provenance.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError(f"Frozen future-latent {split} provenance is missing processor semantics")
    for field, expected in {
        # This is the existing artifact's label. Its six raw state values stay
        # unchanged because the pinned processor has no observation.state stats.
        "state": "model_ready_normalized_and_padded",
        "action": "normalized_policy_output_original_action_dim",
        "processor_config_source": f"{POLICY_REPO_ID}@{POLICY_REVISION}",
    }.items():
        _require_equal(semantics.get(field), expected, name=f"{split} {field} semantics")


def load_frozen_future_latent_predictor(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> LightweightFutureLatentPredictor:
    """Load the approved B3.3a best checkpoint as a frozen float32 predictor."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Frozen future-latent checkpoint must contain a dictionary")
    for field, expected in {
        "checkpoint_kind": "best",
        "trainer_git_sha": TRAINER_SHA,
        "cache_producer_sha": CACHE_PRODUCER_SHA,
        "epoch": BEST_EPOCH,
        "global_step": BEST_GLOBAL_STEP,
    }.items():
        _require_equal(payload.get(field), expected, name=field)
    if "optimizer_state_dict" in payload or "rng_state" in payload:
        raise ValueError("Frozen best checkpoint must not contain optimizer or RNG state")

    config = _frozen_predictor_config()
    expected_config = asdict(config)
    _require_equal(payload.get("predictor_config"), expected_config, name="predictor config")
    train_config = payload.get("train_config")
    if not isinstance(train_config, Mapping):
        raise ValueError("Frozen best checkpoint is missing train config")
    for field, expected in {
        "run_kind": "train_val",
        "protocol_complete": True,
        "trainer_git_sha": TRAINER_SHA,
        "cache_producer_sha": CACHE_PRODUCER_SHA,
        "future_latent_config": expected_config,
        "lambda_cos": 0.1,
        "lambda_risk": 0.05,
        "seed": 0,
    }.items():
        _require_equal(train_config.get(field), expected, name=f"train config {field}")

    recorded_caches = payload.get("cache_provenance")
    if not isinstance(recorded_caches, Mapping):
        raise ValueError("Frozen best checkpoint is missing cache provenance")
    for split in ("train", "val"):
        provenance = recorded_caches.get(split)
        _validate_recorded_provenance(provenance, split=split)
        configured = train_config.get(f"{split}_cache")
        if not isinstance(configured, Mapping):
            raise ValueError(f"Frozen train config is missing {split} cache provenance")
        _require_equal(
            {key: value for key, value in configured.items() if key != "path"},
            {key: value for key, value in provenance.items() if key != "path"},
            name=f"{split} cache provenance in checkpoint and train config",
        )

    predictor = LightweightFutureLatentPredictor(config)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    predictor.to(device=torch.device(device), dtype=torch.float32)
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor


def _bind_frozen_candidate(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    *,
    policy_revision: str,
    vlm_revision: str,
) -> None:
    """Associate objects after context has constructed them from exact snapshots."""
    policy._future_latent_candidate = {
        "policy_revision": policy_revision,
        "vlm_revision": vlm_revision,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
    }


def _validate_frozen_candidate(policy: Any, preprocessor: Any, postprocessor: Any) -> None:
    """Reject unbound objects or a changed candidate at the engine boundary."""
    association = getattr(policy, "_future_latent_candidate", None)
    if not isinstance(association, dict):
        raise ValueError("Predicted mode requires the frozen candidate constructed by build_rollout_context")
    _require_equal(association.get("policy_revision"), POLICY_REVISION, name="active policy revision")
    _require_equal(association.get("vlm_revision"), VLM_REVISION, name="active VLM revision")
    if (
        association.get("preprocessor") is not preprocessor
        or association.get("postprocessor") is not postprocessor
    ):
        raise ValueError("Predicted mode requires the same frozen policy processor instances")

    config = policy.config
    for field, expected in {
        "type": "smolvla",
        "max_state_dim": 32,
        "max_action_dim": 32,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
        "empty_cameras": 0,
        "use_peft": False,
    }.items():
        _require_equal(getattr(config, field, None), expected, name=f"active policy {field}")
    _require_equal(tuple(config.image_features), POLICY_CAMERA_KEYS, name="active policy camera order")
    for key, feature in config.image_features.items():
        _require_equal(tuple(feature.shape), (3, 480, 640), name=f"active policy {key} shape")
    for field in ("robot_state_feature", "action_feature"):
        feature = getattr(config, field, None)
        shape = None if feature is None else tuple(feature.shape)
        _require_equal(shape, (6,), name=f"active policy {field} shape")
    if config.rtc_config is not None and config.rtc_config.enabled:
        raise ValueError("Frozen future-latent candidate does not support full RTC guidance")
