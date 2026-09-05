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

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla import future_latent_checkpoint as checkpoint
from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor


def _recorded_cache(split: str) -> dict:
    return {
        "path": f"/never-open-recorded-{split}-cache",
        "split": split,
        "complete_split": True,
        "producer_git_sha": "eff8be608c899d0841ad5967d80d5d726cbe4394",
        "policy_camera_order": ["observation.images.camera1", "observation.images.camera2"],
        "token_scaling_convention": "native_post_sqrt_hidden_dim",
        "inputs": {
            source: {"repo_id": repo_id, "requested_revision": revision, "resolved_revision": revision}
            for source, repo_id, revision in (
                ("dataset", "lerobot/svla_so100_pickplace", "728583b5eaf9e739a7f119e2def466fa1d552402"),
                ("checkpoint", "lerobot/smolvla_base", "c83c3163b8ca9b7e67c509fffd9121e66cb96205"),
                (
                    "vlm",
                    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                ),
            )
        },
        "semantics": {
            "state": "model_ready_normalized_and_padded",
            "action": "normalized_policy_output_original_action_dim",
            "processor_config_source": "lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        },
    }


@pytest.fixture
def payload() -> dict:
    config = FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, enabled=True)
    # Non-float32 source tensors make the loader's dtype conversion observable.
    model = LightweightFutureLatentPredictor(config).double()
    caches = {split: _recorded_cache(split) for split in ("train", "val")}
    return {
        "checkpoint_kind": "best",
        "trainer_git_sha": "9e618076f617751c297d92626ad422dbbf30c03b",
        "cache_producer_sha": "eff8be608c899d0841ad5967d80d5d726cbe4394",
        "epoch": 29,
        "global_step": 57_750,
        "predictor_config": asdict(config),
        "predictor_state_dict": model.state_dict(),
        "train_config": {
            "run_kind": "train_val",
            "protocol_complete": True,
            "trainer_git_sha": "9e618076f617751c297d92626ad422dbbf30c03b",
            "cache_producer_sha": "eff8be608c899d0841ad5967d80d5d726cbe4394",
            "future_latent_config": asdict(config),
            "lambda_cos": 0.1,
            "lambda_risk": 0.05,
            "seed": 0,
            "train_cache": deepcopy(caches["train"]),
            "val_cache": deepcopy(caches["val"]),
        },
        "cache_provenance": caches,
    }


def _save(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "synthetic-best.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_loads_exact_synthetic_weights_once_without_opening_recorded_caches(
    tmp_path, monkeypatch, payload, device
):
    path = _save(tmp_path, payload)
    real_load = torch.load
    calls = []

    def record_load(actual_path, **kwargs):
        calls.append((actual_path, kwargs))
        assert actual_path == path
        return real_load(actual_path, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", record_load)
    predictor = checkpoint.load_frozen_future_latent_predictor(path, device=device)

    assert calls == [(path, {"map_location": "cpu", "weights_only": False})]
    assert not predictor.training
    assert all(not parameter.requires_grad for parameter in predictor.parameters())
    assert all(parameter.device == torch.device(device) for parameter in predictor.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in predictor.parameters())
    for name, tensor in predictor.state_dict().items():
        assert torch.equal(tensor, payload["predictor_state_dict"][name].float())


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_kind", "last"),
        ("trainer_git_sha", "other-trainer"),
        ("cache_producer_sha", "other-cache"),
        ("epoch", 28),
        ("global_step", 57_749),
    ],
)
def test_rejects_other_candidate_before_constructing_predictor(tmp_path, monkeypatch, payload, key, value):
    payload[key] = value

    def unexpected_construction(config):
        pytest.fail("An incompatible checkpoint must be rejected before predictor construction")

    monkeypatch.setattr(checkpoint, "LightweightFutureLatentPredictor", unexpected_construction)
    with pytest.raises(ValueError, match=key):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("key", ["optimizer_state_dict", "rng_state"])
def test_rejects_training_state(tmp_path, payload, key):
    payload[key] = {}
    with pytest.raises(ValueError, match="optimizer or RNG"):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize(
    "key,value",
    [
        ("lambda_cos", 0.0),
        ("lambda_risk", 0.1),
        ("run_kind", "bounded_smoke"),
        ("protocol_complete", False),
        ("trainer_git_sha", "other-trainer"),
        ("cache_producer_sha", "other-cache"),
        ("seed", 1),
    ],
)
def test_rejects_changed_training_identity(tmp_path, payload, key, value):
    payload["train_config"][key] = value
    with pytest.raises(ValueError, match=f"train config {key}"):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("location", ["predictor_config", "future_latent_config"])
def test_rejects_changed_architecture(tmp_path, payload, location):
    target = payload if location == "predictor_config" else payload["train_config"]
    target[location]["risk_head"] = False
    with pytest.raises(ValueError, match="config"):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("source", ["dataset", "checkpoint", "vlm"])
def test_rejects_same_shape_wrong_recorded_revision(tmp_path, payload, split, source):
    payload["cache_provenance"][split]["inputs"][source]["resolved_revision"] = "different-revision"
    with pytest.raises(ValueError, match=f"{split} {source} resolved_revision"):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "dataset_standardized"),
        ("action", "post_policy"),
        ("processor_config_source", "same-shape-other-processor"),
    ],
)
def test_rejects_wrong_processor_semantics(tmp_path, payload, field, value):
    payload["cache_provenance"]["train"]["semantics"][field] = value
    with pytest.raises(ValueError, match=f"{field} semantics"):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("producer_git_sha", "other-producer"),
        ("policy_camera_order", ["observation.images.camera2", "observation.images.camera1"]),
        ("token_scaling_convention", "unscaled"),
        ("complete_split", False),
    ],
)
def test_rejects_incompatible_recorded_cache_identity(tmp_path, payload, field, value):
    payload["cache_provenance"]["val"][field] = value
    with pytest.raises(ValueError, match=field):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


def test_rejects_inconsistent_duplicate_provenance(tmp_path, payload):
    payload["train_config"]["val_cache"]["semantics"]["processor_config_source"] = "other-processor"
    with pytest.raises(ValueError, match="provenance in checkpoint and train config"):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape"])
def test_weights_must_load_strictly(tmp_path, payload, change):
    state = payload["predictor_state_dict"]
    if change == "missing":
        del state["token_norm.weight"]
    elif change == "unexpected":
        state["unexpected_weight"] = torch.zeros(1)
    else:
        state["token_norm.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        checkpoint.load_frozen_future_latent_predictor(_save(tmp_path, payload), device="cpu")


def _candidate():
    config = SimpleNamespace(
        type="smolvla",
        max_state_dim=32,
        max_action_dim=32,
        adapt_to_pi_aloha=False,
        use_delta_joint_actions_aloha=False,
        empty_cameras=0,
        use_peft=False,
        rtc_config=None,
        image_features={key: SimpleNamespace(shape=(3, 480, 640)) for key in checkpoint.POLICY_CAMERA_KEYS},
        robot_state_feature=SimpleNamespace(shape=(6,)),
        action_feature=SimpleNamespace(shape=(6,)),
    )
    policy, preprocessor, postprocessor = SimpleNamespace(config=config), object(), object()
    checkpoint._bind_frozen_candidate(
        policy,
        preprocessor,
        postprocessor,
        policy_revision=checkpoint.POLICY_REVISION,
        vlm_revision=checkpoint.VLM_REVISION,
    )
    return policy, preprocessor, postprocessor


def test_candidate_association_accepts_same_loaded_objects():
    checkpoint._validate_frozen_candidate(*_candidate())


@pytest.mark.parametrize(
    "change", ["missing", "policy_revision", "vlm_revision", "preprocessor", "postprocessor"]
)
def test_candidate_association_rejects_missing_or_different_source(change):
    policy, preprocessor, postprocessor = _candidate()
    if change == "missing":
        del policy._future_latent_candidate
    else:
        policy._future_latent_candidate[change] = "another-source"
    with pytest.raises(ValueError, match="frozen|Frozen"):
        checkpoint._validate_frozen_candidate(policy, preprocessor, postprocessor)


@pytest.mark.parametrize(
    "camera_order",
    [
        ("observation.images.camera1",),
        ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"),
        (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
            "observation.images.camera4",
        ),
        ("observation.images.camera2", "observation.images.camera1"),
    ],
)
def test_candidate_rejects_untrained_camera_count_or_order(camera_order):
    policy, preprocessor, postprocessor = _candidate()
    policy.config.image_features = {key: SimpleNamespace(shape=(3, 480, 640)) for key in camera_order}
    with pytest.raises(ValueError, match="camera order"):
        checkpoint._validate_frozen_candidate(policy, preprocessor, postprocessor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_state_dim", 31),
        ("max_action_dim", 31),
        ("adapt_to_pi_aloha", True),
        ("use_delta_joint_actions_aloha", True),
        ("empty_cameras", 1),
        ("use_peft", True),
    ],
)
def test_candidate_rejects_changed_state_or_action_adaptation(field, value):
    policy, preprocessor, postprocessor = _candidate()
    setattr(policy.config, field, value)
    with pytest.raises(ValueError, match=field):
        checkpoint._validate_frozen_candidate(policy, preprocessor, postprocessor)


def test_candidate_rejects_full_rtc_guidance():
    policy, preprocessor, postprocessor = _candidate()
    policy.config.rtc_config = SimpleNamespace(enabled=True)
    with pytest.raises(ValueError, match="RTC"):
        checkpoint._validate_frozen_candidate(policy, preprocessor, postprocessor)
