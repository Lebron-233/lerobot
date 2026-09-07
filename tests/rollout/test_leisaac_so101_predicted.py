"""L7 candidate isolation and actual production worker integration; no task scores."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from eval_leisaac_so101 import MemoryMetrics, SnapshotRobot, prepare_engine, prime_episode_engine, stop_engine
from leisaac_so101_contract import SCALAR_KEYS, hardware_features
from leisaac_so101_matched import (
    CAMERA_KEYS,
    VLM_REVISION,
    WSAGI_REVISION,
    MotorStatePreprocessor,
    PhysicalActionPostprocessor,
    physical_to_motor,
)
from leisaac_so101_predicted import (
    PREDICTOR_ID,
    PREDICTOR_PARAMETERS,
    SELECTED_EPOCH,
    TRAINING_SOURCE,
    SO101PredictiveAsyncInferenceEngine,
    validate_l6_checkpoint,
    validate_so101_prediction_candidate,
)

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import FutureLatentPrediction
from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot


def checkpoint_metadata():
    return {
        "source_commit": TRAINING_SOURCE,
        "epoch": SELECTED_EPOCH,
        "parameters": PREDICTOR_PARAMETERS,
        "config": asdict(
            FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, max_cameras=2, risk_head=False)
        ),
        "collection_manifest": {
            "candidate": {"policy_revision": WSAGI_REVISION, "vlm_config_tokenizer_revision": VLM_REVISION},
            "control_fps": 30,
            "camera_backend": "standard",
            "physics": "cpu_physx_rtx_v1",
        },
    }


@pytest.mark.parametrize("field,value", [("epoch", 4), ("source_commit", "other"), ("parameters", 1)])
def test_another_selected_checkpoint_is_rejected(field, value):
    metadata = checkpoint_metadata()
    validate_l6_checkpoint(metadata)
    metadata[field] = value
    with pytest.raises(ValueError, match="frozen L6"):
        validate_l6_checkpoint(metadata)


class NativePipeline:
    steps = ()

    def __init__(self, post=False):
        self.post = post

    def reset(self):
        pass

    def __call__(self, value):
        if self.post:
            return value * 10
        batch = dict(value)
        batch[CAMERA_KEYS[0]] = batch.pop("observation.images.top")
        batch["observation.state"] = batch["observation.state"] / 10
        return batch


class Policy:
    def __init__(self):
        self.config = SimpleNamespace(
            type="smolvla",
            max_state_dim=32,
            max_action_dim=32,
            chunk_size=50,
            n_action_steps=50,
            num_steps=10,
            use_amp=False,
            use_peft=False,
            empty_cameras=0,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
            rtc_config=None,
            image_features={key: SimpleNamespace(shape=(3, 480, 640)) for key in CAMERA_KEYS},
            robot_state_feature=SimpleNamespace(shape=(6,)),
            action_feature=SimpleNamespace(shape=(6,)),
        )
        self.model = self
        self.last_override = None

    def reset(self):
        pass

    def prepare_images(self, batch):
        assert {key for key in batch if key.startswith("observation.images.")} == set(CAMERA_KEYS)
        return [batch[key] for key in CAMERA_KEYS], [torch.ones(1, dtype=torch.bool)] * 2

    def prepare_state(self, batch):
        return torch.nn.functional.pad(batch["observation.state"], (0, 26))

    def encode_image_tokens(self, images, masks):
        return tuple(torch.zeros(1, 2, 4) for _ in images), tuple(
            torch.ones(1, 2, dtype=torch.bool) for _ in images
        )

    def predict_action_chunk(self, batch, **kwargs):
        tokens = kwargs.get("future_image_tokens")
        if tokens is None:
            self.prepare_images(batch)
            value = 0.01
        else:
            self.last_override = tokens
            value = 0.01 + float(tokens[0].mean())
        return torch.full((1, 50, 6), value)


class Predictor:
    def __init__(self):
        self._so101_l6_binding = {
            "id": PREDICTOR_ID,
            "policy_revision": WSAGI_REVISION,
            "training_source": TRAINING_SOURCE,
            "epoch": SELECTED_EPOCH,
        }
        self.seen = []

    def __call__(self, tokens, masks, actions, prefix, state, delay):
        self.seen.append((actions.clone(), prefix.clone(), state.clone(), delay.clone()))
        return FutureLatentPrediction(tuple(torch.full_like(z, 0.1) for z in tokens), torch.zeros(1))


def arguments():
    policy = Policy()
    pre, post = MotorStatePreprocessor(NativePipeline()), PhysicalActionPostprocessor(NativePipeline(True))
    policy._so101_matched_candidate = {
        "policy_revision": WSAGI_REVISION,
        "vlm_config_tokenizer_revision": VLM_REVISION,
        "preprocessor": pre,
        "postprocessor": post,
        "execution_steps": 50,
    }
    raw = dict.fromkeys(SCALAR_KEYS, 0.0)
    raw.update({name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("top", "wrist")})
    return raw, {
        "policy": policy,
        "preprocessor": pre,
        "postprocessor": post,
        "future_latent_predictor": Predictor(),
        "robot_wrapper": ThreadSafeRobot(SnapshotRobot(raw)),
        "hw_features": hardware_features(),
        "task": "fixture",
        "fps": 30,
        "device": "cpu",
        "queue_threshold": 49,
        "context_mode": "predicted",
        "min_prediction_delay": 1,
        "max_prediction_delay": 8,
        "metrics_sink": MemoryMetrics(),
    }


def test_so101_is_not_a_forged_frozen_so100_candidate():
    _, args = arguments()
    with pytest.raises(ValueError, match="frozen candidate"):
        PredictiveAsyncInferenceEngine(**args)
    assert (
        validate_so101_prediction_candidate(
            args["policy"], args["preprocessor"], args["postprocessor"], args["future_latent_predictor"]
        )
        == CAMERA_KEYS
    )
    args["policy"]._so101_matched_candidate["postprocessor"] = object()
    with pytest.raises(ValueError, match="processor instances"):
        SO101PredictiveAsyncInferenceEngine(**args)


def test_new_predictor_uses_native_state_normalized_prefix_and_real_takeover():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    raw, args = arguments()
    engine = SO101PredictiveAsyncInferenceEngine(**args)
    predictor, policy = args["future_latent_predictor"], args["policy"]
    try:
        prepare_engine(engine, raw, timeout=10)
        prime_episode_engine(engine, raw, timeout=10)
        torch.testing.assert_close(physical_to_motor(engine.get_action(None)), torch.full((6,), 0.1))
        engine.notify_observation(raw)
        deadline = time.monotonic() + 5
        while not engine.queue.has_staged_chunk():
            assert not engine.failed and time.monotonic() < deadline
            time.sleep(0.005)
        plan = engine.queue.plan_snapshot()
        actions, mask, state, delay = predictor.seen[-1]
        assert plan is not None and delay.item() == plan.planned_delay_steps
        torch.testing.assert_close(actions[mask], torch.full_like(actions[mask], 0.01))
        assert state.shape == (1, 32)
        assert state[0, 2].item() == pytest.approx((100 / 19 - 0) / 10, rel=1e-5)
        for _ in range(plan.planned_delay_steps):
            engine.get_action(None)
        takeover = physical_to_motor(engine.get_action(None))
        torch.testing.assert_close(takeover, torch.full((6,), 1.1), atol=1e-5, rtol=1e-5)
        assert policy.last_override[0].mean().item() == pytest.approx(0.1)
    finally:
        stop_engine(engine)
        torch.set_num_threads(old_threads)
    assert args["metrics_sink"].closed
    assert any(event.get("predictor_calls", 0) == 1 for event in args["metrics_sink"].events)
