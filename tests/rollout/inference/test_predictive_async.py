# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from copy import deepcopy
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc.scheduled_action_queue import InstallOutcome
from lerobot.policies.smolvla.future_latent import FutureLatentPrediction
from lerobot.policies.smolvla.future_latent_checkpoint import (
    CAMERA_RENAME_MAP,
    POLICY_CAMERA_KEYS,
    POLICY_REVISION,
    RAW_CAMERA_KEYS,
    RAW_IMAGE_SHAPE,
    RUNTIME_SCALAR_KEYS,
    VLM_REVISION,
    _bind_frozen_candidate,
)
from lerobot.processor import RelativeActionsProcessorStep
from lerobot.rollout.inference import predictive_async
from lerobot.rollout.inference.metrics import JsonlMetricsSink
from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine


class _IdentityPipeline:
    steps: tuple = ()

    def __call__(self, batch):
        return batch

    def reset(self) -> None:
        pass


class _TokenModel:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_image_tokens(self, images, image_masks):
        self.encode_calls += 1
        batch_size = images[0].shape[0]
        tokens = (torch.full((batch_size, 2, 4), float(self.encode_calls)),)
        masks = (torch.ones(batch_size, 2, dtype=torch.bool),)
        return tokens, masks


class _ChunkPolicy:
    def __init__(self, *, chunk_size: int = 6, block_call: int | None = None) -> None:
        self.config = SimpleNamespace(type="smolvla")
        self.model = _TokenModel()
        self.chunk_size = chunk_size
        self.block_call = block_call
        self.calls = 0
        self.kwargs: list[dict] = []
        self.entered = Event()
        self.release = Event()
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def prepare_images(self, batch):
        batch_size = batch["observation.state"].shape[0]
        return [torch.zeros(batch_size, 3, 4, 4)], [torch.ones(batch_size, dtype=torch.bool)]

    def predict_action_chunk(self, batch, **kwargs):
        self.calls += 1
        self.kwargs.append(dict(kwargs))
        if "future_image_tokens" not in kwargs:
            images, masks = self.prepare_images(batch)
            self.model.encode_image_tokens(images, masks)
        if self.calls == self.block_call:
            self.entered.set()
            assert self.release.wait(timeout=3.0)
        base = self.calls * 100
        rows = torch.arange(self.chunk_size, dtype=torch.float32).view(1, self.chunk_size, 1)
        return (rows + base).expand(1, self.chunk_size, 2).clone()


def _make_engine(
    policy: _ChunkPolicy,
    *,
    fps: float = 1.0,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    context_mode: str = "identity",
    fallback_mode: str = "identity",
    delay_safety_margin_steps: int = 0,
    metrics_sink=None,
) -> PredictiveAsyncInferenceEngine:
    return PredictiveAsyncInferenceEngine(
        policy=policy,
        preprocessor=_IdentityPipeline(),
        postprocessor=_IdentityPipeline(),
        robot_wrapper=SimpleNamespace(robot_type="mock"),
        hw_features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a.pos", "joint_b.pos"],
            }
        },
        task="task A",
        fps=fps,
        device="cpu",
        queue_threshold=10,
        latency_quantile=0.9,
        latency_window=10,
        delay_safety_margin_steps=delay_safety_margin_steps,
        min_prediction_delay=1,
        max_prediction_delay=8,
        committed_guard_steps=2,
        max_late_steps=2,
        context_mode=context_mode,
        fallback_mode=fallback_mode,
        use_torch_compile=use_torch_compile,
        compile_warmup_inferences=compile_warmup_inferences,
        metrics_sink=metrics_sink,
    )


def _obs() -> dict[str, float]:
    return {"joint_a.pos": 1.0, "joint_b.pos": 2.0}


def _wait_for(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _bootstrap(engine: PredictiveAsyncInferenceEngine) -> None:
    engine.notify_observation(_obs())
    assert _wait_for(lambda: engine.queue.qsize() == 6)


def test_identity_request_stages_without_early_switch_or_second_vision_pass() -> None:
    policy = _ChunkPolicy()
    engine = _make_engine(policy)
    engine.start()
    engine.resume()
    try:
        _bootstrap(engine)
        assert engine.get_action(None).tolist() == [100.0, 100.0]

        # notify occurs before this tick's get: the plan snapshots next index 1 and d=1.
        engine.notify_observation(_obs())
        assert _wait_for(engine.queue.has_staged_chunk)
        plan = engine.queue.plan_snapshot()
        assert plan is not None
        assert plan.next_action_index == 1
        assert plan.takeover_index == 2

        assert engine.get_action(None).tolist() == [101.0, 101.0]
        takeover = engine.queue.get_with_task()
        assert takeover.post_policy_action.tolist() == [200.0, 200.0]
        assert takeover.action_index == 2

        # One native vision pass per request: internal bootstrap, external override.
        assert policy.model.encode_calls == 2
        assert "future_image_tokens" in policy.kwargs[1]
        assert "inference_delay" not in policy.kwargs[1]
        assert "prev_chunk_left_over" not in policy.kwargs[1]
    finally:
        engine.stop()


def test_chunk_completion_barrier_precedes_queue_publication(monkeypatch) -> None:
    policy = _ChunkPolicy()
    engine = _make_engine(policy)
    events: list[str] = []
    install = engine.queue.install_active_chunk

    monkeypatch.setattr(
        predictive_async,
        "_synchronize_policy_device",
        lambda _device: events.append("synchronized"),
    )

    def _recording_install(*args, **kwargs):
        events.append("published")
        return install(*args, **kwargs)

    monkeypatch.setattr(engine.queue, "install_active_chunk", _recording_install)
    engine.start()
    engine.resume()
    try:
        _bootstrap(engine)
        assert events == ["synchronized", "published"]
    finally:
        engine.stop()


def test_late_predictive_chunk_is_discarded_without_skipping_its_prefix() -> None:
    policy = _ChunkPolicy(block_call=2)
    engine = _make_engine(policy)
    engine.start()
    engine.resume()
    try:
        _bootstrap(engine)
        assert engine.get_action(None).tolist() == [100.0, 100.0]
        engine.notify_observation(_obs())
        assert policy.entered.wait(timeout=2.0)

        assert engine.get_action(None).tolist() == [101.0, 101.0]
        # No staged result at takeover: the old guard action is what actually executes.
        assert engine.get_action(None).tolist() == [102.0, 102.0]
        policy.release.set()
        assert _wait_for(lambda: engine.stats.deadline_misses == 1)

        assert engine.queue.plan_snapshot() is None
        assert engine.get_action(None).tolist() == [103.0, 103.0]
        assert all(value < 200 for value in engine.get_action(None).tolist())
    finally:
        policy.release.set()
        engine.stop()


def test_task_change_invalidates_inflight_result_but_preserves_active_provenance() -> None:
    policy = _ChunkPolicy(block_call=2)
    engine = _make_engine(policy)
    engine.start()
    engine.resume()
    try:
        _bootstrap(engine)
        assert engine.get_action(None) is not None
        engine.notify_observation(_obs())
        assert policy.entered.wait(timeout=2.0)

        assert engine.set_task("task B")
        assert engine.task_snapshot == ("task B", 1)
        policy.release.set()
        assert _wait_for(lambda: engine.stats.stale_results == 1)

        assert engine.get_action(None) is not None
        assert engine.dispatched_task == "task A"
        assert engine.queue.plan_snapshot() is None
    finally:
        policy.release.set()
        engine.stop()


def test_reset_discards_inflight_result_without_reusing_action_index() -> None:
    policy = _ChunkPolicy(block_call=2)
    engine = _make_engine(policy)
    engine.start()
    engine.resume()
    try:
        _bootstrap(engine)
        assert engine.get_action(None) is not None
        next_index = engine.queue.next_action_index
        engine.notify_observation(_obs())
        assert policy.entered.wait(timeout=2.0)

        engine.reset()
        policy.release.set()
        assert _wait_for(lambda: engine.stats.stale_results == 1)
        assert engine.queue.qsize() == 0
        assert engine.queue.next_action_index == next_index
    finally:
        policy.release.set()
        engine.stop()


def test_prediction_cap_is_explicit_and_does_not_create_a_covered_plan() -> None:
    policy = _ChunkPolicy()
    engine = _make_engine(policy, fps=30.0, delay_safety_margin_steps=1)
    installed = engine.queue.install_active_chunk(
        torch.zeros(10, 2),
        torch.zeros(10, 2),
        task="task A",
        reset_epoch=0,
        task_epoch=0,
    )
    assert installed.outcome is InstallOutcome.INSTALLED
    engine._latency_tracker.add(0.5)
    engine.resume()

    engine.notify_observation(_obs())

    plan = engine.last_delay_plan
    assert plan is not None
    assert plan.raw_required_delay_steps == 16
    assert plan.planned_delay_steps == 8
    assert plan.prediction_cap_exceeded
    assert engine.stats.prediction_cap_exceeded == 1
    assert engine.queue.plan_snapshot() is None
    assert policy.calls == 0


@pytest.mark.parametrize("mode", ["oracle", "predicted"])
def test_live_engine_rejects_oracle_and_predicted_without_predictor(mode: str) -> None:
    policy = _ChunkPolicy()
    with pytest.raises(ValueError):
        _make_engine(policy, context_mode=mode)


def test_engine_constructor_rejects_relative_action_pipeline() -> None:
    policy = _ChunkPolicy()
    preprocessor = _IdentityPipeline()
    preprocessor.steps = (RelativeActionsProcessorStep(enabled=True),)

    with pytest.raises(NotImplementedError, match="relative-action anchor/rebase"):
        PredictiveAsyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=_IdentityPipeline(),
            robot_wrapper=SimpleNamespace(robot_type="mock"),
            hw_features={},
            task="task A",
            fps=30.0,
            device="cpu",
        )


def test_compile_warmup_covers_rgb_and_token_override_then_marks_backend_ready() -> None:
    policy = _ChunkPolicy()
    engine = _make_engine(policy, use_torch_compile=True, compile_warmup_inferences=1)
    engine.start()
    engine.resume()
    try:
        assert not engine.ready
        engine.notify_observation(_obs())
        assert _wait_for(lambda: policy.calls == 1 and not engine._request_in_flight)
        engine.notify_observation(_obs())
        assert _wait_for(lambda: policy.calls == 2 and engine.ready)
        assert engine.queue.qsize() == 0
        assert "future_image_tokens" not in policy.kwargs[0]
        assert "future_image_tokens" in policy.kwargs[1]
        assert policy.model.encode_calls == 2
    finally:
        engine.stop()


class _CandidatePreprocessor(_IdentityPipeline):
    def __init__(self) -> None:
        self.drop_camera: str | None = None
        self.add_cameras: tuple[str, ...] = ()

    def __call__(self, batch):
        result = {CAMERA_RENAME_MAP.get(key, key): value for key, value in batch.items()}
        if self.drop_camera is not None:
            result.pop(self.drop_camera)
        for key in self.add_cameras:
            result[key] = result[POLICY_CAMERA_KEYS[0]]
        return result


class _CandidatePostprocessor(_IdentityPipeline):
    def __call__(self, actions):
        return actions + 1000


class _CandidateTokenModel:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.tokens = (
            torch.full((1, 64, 960), 2048.0, dtype=torch.float16),
            torch.full((1, 64, 960), -2.0, dtype=torch.float16),
        )
        self.masks = (
            torch.ones(1, 64, dtype=torch.bool),
            torch.arange(64).unsqueeze(0) < 61,
        )

    def encode_image_tokens(self, images, image_masks):
        self.encode_calls += 1
        assert len(images) == len(image_masks) == 2
        return self.tokens, self.masks


class _CandidatePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            type="smolvla",
            image_features={
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for key in POLICY_CAMERA_KEYS
            },
            robot_state_feature=PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            max_state_dim=32,
            max_action_dim=32,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
            empty_cameras=0,
            use_peft=False,
            rtc_config=None,
        )
        self.model = _CandidateTokenModel()
        self.prepare_image_calls = 0
        self.prepare_state_calls = 0
        self.kwargs: list[dict] = []
        self.batches: list[dict] = []
        self.output = torch.arange(72, dtype=torch.float32).reshape(1, 12, 6) / 100

    def reset(self) -> None:
        pass

    def prepare_images(self, batch):
        self.prepare_image_calls += 1
        return [batch[key] for key in POLICY_CAMERA_KEYS], [torch.ones(1, dtype=torch.bool)] * 2

    def prepare_state(self, batch):
        self.prepare_state_calls += 1
        return torch.nn.functional.pad(batch["observation.state"], (0, 26))

    def predict_action_chunk(self, batch, **kwargs):
        self.batches.append(batch)
        self.kwargs.append(kwargs)
        if "future_image_tokens" not in kwargs:
            images, masks = self.prepare_images(batch)
            self.model.encode_image_tokens(images, masks)
        return self.output.clone()


class _RecordingPredictor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.prediction: FutureLatentPrediction | None = None

    def __call__(self, tokens, masks, actions, action_mask, state, delay):
        self.calls.append((tokens, masks, actions, action_mask, state, delay))
        # A double residual at a half-precision tie distinguishes the required
        # float32 addition from either a float64 sum or pre-casting the residual.
        self.prediction = FutureLatentPrediction(
            delta_tokens=tuple(
                torch.full_like(token, residual, dtype=torch.float64)
                for token, residual in zip(tokens, (1.00000005, 1.0005), strict=True)
            ),
            predicted_error=torch.tensor([1e9]),
        )
        return self.prediction


def _candidate_arguments() -> dict:
    policy = _CandidatePolicy()
    preprocessor = _CandidatePreprocessor()
    postprocessor = _CandidatePostprocessor()
    _bind_frozen_candidate(
        policy,
        preprocessor,
        postprocessor,
        policy_revision=POLICY_REVISION,
        vlm_revision=VLM_REVISION,
    )
    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "future_latent_predictor": _RecordingPredictor(),
        "robot_wrapper": SimpleNamespace(robot_type="so_follower"),
        "hw_features": {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": list(RUNTIME_SCALAR_KEYS),
            },
            **{key: {"dtype": "video", "shape": RAW_IMAGE_SHAPE} for key in RAW_CAMERA_KEYS},
        },
        "task": "task A",
        "fps": 30.0,
        "device": "cpu",
        "queue_threshold": 20,
        "delay_safety_margin_steps": 0,
        "min_prediction_delay": 1,
        "max_prediction_delay": 8,
        "context_mode": "predicted",
    }


def _candidate_observation() -> dict:
    return {
        **dict(zip(RUNTIME_SCALAR_KEYS, [15.0, -25.0, 35.0, -45.0, 55.0, 65.0], strict=True)),
        "top": np.zeros(RAW_IMAGE_SHAPE, dtype=np.uint8),
        "wrist": np.full(RAW_IMAGE_SHAPE, 255, dtype=np.uint8),
    }


def _take_pending(engine):
    request = engine._pending_request
    assert request is not None
    engine._pending_request = None
    engine._request_ready.clear()
    return request


def _mark_candidate_steady(engine) -> None:
    """Put manually installed steady-state fixtures beyond the real startup tests."""
    if engine._startup_phase is not None:
        engine._startup_phase = "complete"
        engine._ready_event.set()


def _candidate_planned_request(engine, delay: int):
    _mark_candidate_steady(engine)
    count = 2 if delay == 0 else 12
    old_actions = torch.arange(count * 6, dtype=torch.float32).reshape(count, 6) / 10 + 10
    installed = engine.queue.install_active_chunk(
        old_actions,
        old_actions + 2000,
        task="task A",
        reset_epoch=0,
        task_epoch=0,
    )
    assert installed.outcome is InstallOutcome.INSTALLED
    engine._latency_tracker.add(max(0.001, (delay - 0.5) / 30))
    engine.resume()
    engine.notify_observation(_candidate_observation())
    request = _take_pending(engine)
    assert request.kind == "planned"
    assert request.plan.planned_delay_steps == delay
    return request, old_actions


@pytest.mark.parametrize("delay,cap", [(1, 8), (8, 8), (3, 6)])
def test_predicted_tensor_flow_and_local_padding_preserve_scheduled_takeover(delay: int, cap: int) -> None:
    arguments = _candidate_arguments()
    arguments["max_prediction_delay"] = cap
    engine = PredictiveAsyncInferenceEngine(**arguments)
    policy = arguments["policy"]
    predictor = arguments["future_latent_predictor"]
    request, old_actions = _candidate_planned_request(engine, delay)
    frozen_actions = request.plan.committed_policy_actions.clone()
    frozen_mask = request.plan.committed_mask.clone()

    engine._run_request(request)

    assert policy.prepare_image_calls == policy.model.encode_calls == 1
    assert policy.prepare_state_calls == 1
    assert len(predictor.calls) == 1
    tokens, masks, actions, mask, state, actual_delay = predictor.calls[0]
    assert tokens is policy.model.tokens
    assert masks is policy.model.masks
    assert actions.shape == (1, 8, 6)
    assert mask.shape == (1, 8)
    torch.testing.assert_close(actions[0, :delay], old_actions[:delay])
    assert torch.count_nonzero(actions[0, delay:]) == 0
    assert torch.equal(mask, torch.arange(8).unsqueeze(0) < delay)
    assert actual_delay.tolist() == [delay]
    assert state.shape == (1, 32)
    assert state.dtype == torch.float32
    torch.testing.assert_close(state[0, :6], torch.tensor([15.0, -25.0, 35.0, -45.0, 55.0, 65.0]))
    assert torch.count_nonzero(state[0, 6:]) == 0
    assert request.plan.committed_policy_actions.shape == (cap, 6)
    assert request.plan.committed_mask.shape == (cap,)
    assert torch.equal(request.plan.committed_policy_actions, frozen_actions)
    assert torch.equal(request.plan.committed_mask, frozen_mask)
    kwargs = policy.kwargs[0]
    assert set(kwargs) == {"future_image_tokens", "future_image_token_masks"}
    assert kwargs["future_image_token_masks"] is masks
    for original, delta, predicted in zip(
        tokens, predictor.prediction.delta_tokens, kwargs["future_image_tokens"], strict=True
    ):
        assert predicted.dtype == original.dtype
        assert torch.equal(predicted, (original.float() + delta.float()).to(original.dtype))
    assert kwargs["future_image_tokens"][0][0, 0, 0].item() == 2048.0
    assert policy.batches[0][POLICY_CAMERA_KEYS[0]].mean().item() == 0
    assert policy.batches[0][POLICY_CAMERA_KEYS[1]].mean().item() == 1
    assert engine.queue.has_staged_chunk()
    assert engine.queue.next_action_index == request.plan.next_action_index
    for index in range(delay):
        torch.testing.assert_close(engine.get_action(None), old_actions[index] + 2000)
    takeover = engine.queue.get_with_task()
    assert takeover.action_index == request.plan.takeover_index
    torch.testing.assert_close(takeover.post_policy_action, policy.output[0, 0] + 1000)


@pytest.mark.parametrize("timing", ["early", "on_time", "late"])
@pytest.mark.parametrize("delay", [0, 1])
def test_predicted_d0_identity_and_d1_keep_planned_deadline_semantics(delay: int, timing: str) -> None:
    arguments = _candidate_arguments()
    engine = PredictiveAsyncInferenceEngine(**arguments)
    request, old_actions = _candidate_planned_request(engine, delay)
    consumed = 0 if timing == "early" else delay + (timing == "late")
    for index in range(consumed):
        torch.testing.assert_close(engine.get_action(None), old_actions[index] + 2000)

    engine._run_request(request)

    policy = arguments["policy"]
    assert engine.stats.planned_requests == 1
    assert engine.stats.bootstrap_requests == 0
    assert len(arguments["future_latent_predictor"].calls) == (delay > 0)
    assert policy.prepare_state_calls == (delay > 0)
    assert policy.model.encode_calls == 1
    if delay == 0:
        assert policy.kwargs[0]["future_image_tokens"] is policy.model.tokens
        assert policy.kwargs[0]["future_image_token_masks"] is policy.model.masks
    if timing == "late":
        assert engine.stats.deadline_misses == 1
        assert engine.queue.plan_snapshot() is None
        torch.testing.assert_close(engine.get_action(None), old_actions[consumed] + 2000)
    else:
        for index in range(consumed, delay):
            torch.testing.assert_close(engine.get_action(None), old_actions[index] + 2000)
        result = engine.queue.get_with_task()
        assert result.action_index == request.plan.takeover_index
        torch.testing.assert_close(result.post_policy_action, policy.output[0, 0] + 1000)


@pytest.mark.parametrize("change", ["reset", "task"])
@pytest.mark.parametrize("delay", [0, 1])
def test_predicted_plans_keep_stale_reset_and_task_rejection(change: str, delay: int) -> None:
    arguments = _candidate_arguments()
    engine = PredictiveAsyncInferenceEngine(**arguments)
    request, old_actions = _candidate_planned_request(engine, delay)
    if change == "reset":
        engine.reset()
    else:
        engine.set_task("task B")

    engine._run_request(request)

    assert engine.stats.stale_results == 1
    assert engine.queue.plan_snapshot() is None
    assert engine.queue.next_action_index == request.plan.next_action_index
    if change == "reset":
        assert engine.queue.qsize() == 0
    else:
        torch.testing.assert_close(engine.get_action(None), old_actions[0] + 2000)
        assert engine.dispatched_task == "task A"


def test_predicted_compile_warmup_and_bootstrap_never_call_predictor() -> None:
    arguments = _candidate_arguments()
    engine = PredictiveAsyncInferenceEngine(**arguments, use_torch_compile=True)
    engine.resume()
    for warmup_index in range(2):
        engine.notify_observation(_candidate_observation())
        request = _take_pending(engine)
        assert request.kind == "warmup"
        engine._run_request(request)
        assert engine.ready == (warmup_index == 1)
        assert engine.queue.qsize() == 0
    assert arguments["policy"].kwargs[0] == {}
    assert arguments["policy"].kwargs[1]["future_image_tokens"] is arguments["policy"].model.tokens
    engine.notify_observation(_candidate_observation())
    bootstrap = _take_pending(engine)
    assert bootstrap.kind == "bootstrap"
    engine._run_request(bootstrap)
    assert arguments["policy"].kwargs[2] == {}
    assert arguments["policy"].model.encode_calls == 3
    assert arguments["policy"].prepare_state_calls == 0
    assert arguments["future_latent_predictor"].calls == []
    assert engine.queue.qsize() == 12


@pytest.mark.parametrize("fallback_mode", ["identity", "discard"])
@pytest.mark.parametrize("available", [0, 12])
def test_predicted_cap_exceeded_keeps_existing_empty_queue_behavior(
    fallback_mode: str, available: int
) -> None:
    arguments = _candidate_arguments()
    engine = PredictiveAsyncInferenceEngine(**arguments, fallback_mode=fallback_mode)
    _mark_candidate_steady(engine)
    if available:
        engine.queue.install_active_chunk(
            torch.zeros(available, 6),
            torch.zeros(available, 6),
            task="task A",
            reset_epoch=0,
            task_epoch=0,
        )
    engine._latency_tracker.add(0.5)
    engine.resume()
    engine.notify_observation(_candidate_observation())
    assert engine.stats.prediction_cap_exceeded == 1
    assert engine.queue.plan_snapshot() is None
    if available == 0 and fallback_mode == "identity":
        request = _take_pending(engine)
        assert request.kind == "bootstrap"
        engine._run_request(request)
        assert engine.queue.qsize() == 12
    else:
        assert engine._pending_request is None
    assert arguments["future_latent_predictor"].calls == []


@pytest.mark.parametrize(
    "mismatch", ["missing_association", "policy_revision", "vlm_revision", "preprocessor", "postprocessor"]
)
def test_predicted_direct_engine_rejects_missing_or_wrong_source_association(mismatch: str) -> None:
    arguments = _candidate_arguments()
    if mismatch == "missing_association":
        arguments["policy"] = _CandidatePolicy()
    elif mismatch in ("policy_revision", "vlm_revision"):
        _bind_frozen_candidate(
            arguments["policy"],
            arguments["preprocessor"],
            arguments["postprocessor"],
            policy_revision="wrong" if mismatch == "policy_revision" else POLICY_REVISION,
            vlm_revision="wrong" if mismatch == "vlm_revision" else VLM_REVISION,
        )
    else:
        arguments[mismatch] = _IdentityPipeline()
    with pytest.raises(ValueError):
        PredictiveAsyncInferenceEngine(**arguments)
    assert arguments["future_latent_predictor"].calls == []


@pytest.mark.parametrize("minimum,maximum", [(0, 8), (1, 9), (7, 6)])
def test_predicted_direct_engine_rejects_invalid_delay_contract(minimum: int, maximum: int) -> None:
    arguments = _candidate_arguments()
    arguments.update(min_prediction_delay=minimum, max_prediction_delay=maximum)
    with pytest.raises(ValueError, match="1 <= min_prediction_delay"):
        PredictiveAsyncInferenceEngine(**arguments)


def test_identity_engine_rejects_injected_predictor_without_using_it() -> None:
    arguments = _candidate_arguments()
    arguments["context_mode"] = "identity"
    with pytest.raises(ValueError, match="only supported"):
        PredictiveAsyncInferenceEngine(**arguments)
    assert arguments["future_latent_predictor"].calls == []


@pytest.mark.parametrize(
    "mismatch",
    [
        "scalar_order",
        "scalar_names",
        "state_shape",
        "state_dtype",
        "camera_missing",
        "camera_wrong",
        "camera_third",
        "camera_fourth",
        "camera_shape",
        "fps",
    ],
)
def test_predicted_engine_rejects_incompatible_runtime_schema(mismatch: str) -> None:
    arguments = _candidate_arguments()
    features = deepcopy(arguments["hw_features"])
    arguments["hw_features"] = features
    state = features["observation.state"]
    if mismatch == "scalar_order":
        state["names"].reverse()
    elif mismatch == "scalar_names":
        state["names"][0] = "main_shoulder_pan"
    elif mismatch == "state_shape":
        state["shape"] = (32,)
    elif mismatch == "state_dtype":
        state["dtype"] = "float64"
    elif mismatch == "camera_missing":
        features.pop(RAW_CAMERA_KEYS[1])
    elif mismatch == "camera_wrong":
        features["observation.images.side"] = features.pop(RAW_CAMERA_KEYS[1])
    elif mismatch in ("camera_third", "camera_fourth"):
        features["observation.images.side"] = deepcopy(features[RAW_CAMERA_KEYS[1]])
        if mismatch == "camera_fourth":
            features["observation.images.front"] = deepcopy(features[RAW_CAMERA_KEYS[1]])
    elif mismatch == "camera_shape":
        features[RAW_CAMERA_KEYS[0]]["shape"] = (640, 480, 3)
    else:
        arguments["fps"] = 25.0
    with pytest.raises(ValueError, match="frozen future-latent candidate"):
        PredictiveAsyncInferenceEngine(**arguments)
    assert arguments["future_latent_predictor"].calls == []


def test_predicted_engine_accepts_raw_camera_dictionary_order_with_same_mapping() -> None:
    arguments = _candidate_arguments()
    arguments["hw_features"] = dict(reversed(arguments["hw_features"].items()))
    engine = PredictiveAsyncInferenceEngine(**arguments)
    request, _ = _candidate_planned_request(engine, 1)
    engine._run_request(request)
    batch = arguments["policy"].batches[0]
    assert batch[POLICY_CAMERA_KEYS[0]].mean().item() == 0
    assert batch[POLICY_CAMERA_KEYS[1]].mean().item() == 1
    assert len(arguments["future_latent_predictor"].calls) == 1


@pytest.mark.parametrize("camera_count", [1, 3, 4])
def test_predicted_planned_batch_requires_exactly_the_frozen_camera_streams(camera_count: int) -> None:
    arguments = _candidate_arguments()
    preprocessor = arguments["preprocessor"]
    if camera_count == 1:
        preprocessor.drop_camera = POLICY_CAMERA_KEYS[1]
    else:
        preprocessor.add_cameras = tuple(
            f"observation.images.camera{index}" for index in range(3, camera_count + 1)
        )
    engine = PredictiveAsyncInferenceEngine(**arguments)
    request, _ = _candidate_planned_request(engine, 1)
    with pytest.raises(ValueError, match="exactly camera1 and camera2"):
        engine._run_request(request)
    assert arguments["policy"].prepare_image_calls == 0
    assert arguments["policy"].model.encode_calls == 0
    assert arguments["future_latent_predictor"].calls == []


def test_predicted_prepare_images_cannot_silently_drop_a_camera(monkeypatch) -> None:
    arguments = _candidate_arguments()
    policy = arguments["policy"]
    prepare_images = policy.prepare_images

    def drop_stream(batch):
        images, masks = prepare_images(batch)
        return images[:1], masks[:1]

    monkeypatch.setattr(policy, "prepare_images", drop_stream)
    engine = PredictiveAsyncInferenceEngine(**arguments)
    request, _ = _candidate_planned_request(engine, 1)
    with pytest.raises(ValueError, match="preserve both frozen camera streams"):
        engine._run_request(request)
    assert policy.model.encode_calls == 0
    assert arguments["future_latent_predictor"].calls == []


_METRIC_PHASES = {
    "observation_preparation",
    "preprocessor",
    "vision_encode",
    "predictor_input_preparation",
    "predictor_forward",
    "residual_application",
    "policy_total",
    "postprocessor",
}
_PREDICTOR_PHASES = {"predictor_input_preparation", "predictor_forward", "residual_application"}


class _RecordingMetrics:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.close_calls = 0

    def emit(self, event) -> None:
        # Exercise the complete public JSON payload, including enums and optional values.
        self.events.append(json.loads(json.dumps(dict(event))))

    def close(self) -> None:
        self.close_calls += 1


class _StartupClock:
    """A controlled host clock; waiting for the real worker never changes latency."""

    def __init__(self) -> None:
        self.value = 100.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.value += 0.0001
        return self.value


def _startup_observation(marker: int) -> dict:
    observation = _candidate_observation()
    observation[RUNTIME_SCALAR_KEYS[0]] = float(marker)
    observation["top"].fill(marker)
    observation["wrist"].fill(255 - marker)
    return observation


def _startup_harness(monkeypatch, *, sink=None, cap=8, fallback_mode="identity"):
    arguments = _candidate_arguments()
    arguments["max_prediction_delay"] = cap
    shutdown = Event()
    engine = PredictiveAsyncInferenceEngine(
        **arguments, metrics_sink=sink, shutdown_event=shutdown, fallback_mode=fallback_mode
    )
    clock = _StartupClock()
    harness = SimpleNamespace(
        engine=engine,
        arguments=arguments,
        sink=sink,
        shutdown=shutdown,
        clock=clock,
        requests=[],
        finished=[],
        active=None,
        clock_counts={},
        policy_attempts=[],
        outputs={},
        publications=[],
        syncs=[],
        durations={"cold_temporary": 0.6, "probe": 0.005, "fresh_warmed": 0.4},
        fail=None,
        nonfinite=None,
        block_phase=None,
        block_request_id=None,
        entered=Event(),
        release=Event(),
        fail_after_release=False,
    )
    monkeypatch.setattr(predictive_async, "time", SimpleNamespace(perf_counter=clock, sleep=time.sleep))
    run = engine._run_request
    policy = arguments["policy"]
    predict = policy.predict_action_chunk
    predictor = arguments["future_latent_predictor"]
    postprocess = _CandidatePostprocessor.__call__

    def run_request(request):
        harness.active = request
        harness.requests.append(request)
        before = clock.calls
        try:
            return run(request)
        finally:
            harness.clock_counts[request.request_id] = clock.calls - before
            harness.finished.append(request.request_id)
            harness.active = None

    def public_policy(batch, **kwargs):
        request = harness.active
        harness.policy_attempts.append(request.request_id)
        if (
            request.startup_phase == harness.block_phase and harness.block_phase is not None
        ) or request.request_id == harness.block_request_id:
            harness.entered.set()
            if not harness.release.wait(timeout=3):
                raise RuntimeError("Synthetic policy was not released")
            if harness.fail_after_release:
                raise ValueError("Synthetic public policy failed after interruption")
        if harness.fail == (request.startup_phase, "policy_total"):
            raise ValueError("Synthetic policy failure")
        actions = predict(batch, **kwargs) + batch["observation.state"][0, 0]
        clock.value += harness.durations.get(request.startup_phase, 0.001)
        if request.kind == "startup_probe" and harness.nonfinite == "policy":
            actions[0, 0, 0] = float("nan")
        harness.outputs[request.request_id] = actions.clone()
        return actions

    def public_predictor(*args):
        if harness.fail == (harness.active.startup_phase, "predictor_forward"):
            raise ValueError("Synthetic predictor failure")
        return predictor(*args)

    def public_postprocessor(self, actions):
        if harness.fail == (harness.active.startup_phase, "postprocessor"):
            raise ValueError("Synthetic postprocessor failure")
        result = postprocess(self, actions)
        if harness.active.kind == "startup_probe" and harness.nonfinite == "post_policy":
            result[0, 0, 0] = float("inf")
        return result

    def synchronize(device):
        harness.syncs.append(harness.active.request_id)
        if harness.active.kind == "startup_probe" and harness.nonfinite == "total":
            clock.value = float("nan")

    monkeypatch.setattr(engine, "_run_request", run_request)
    monkeypatch.setattr(policy, "predict_action_chunk", public_policy)
    monkeypatch.setattr(engine, "_future_latent_predictor", public_predictor)
    monkeypatch.setattr(_CandidatePostprocessor, "__call__", public_postprocessor)
    monkeypatch.setattr(predictive_async, "_synchronize_policy_device", synchronize)
    for method_name in ("install_active_chunk", "stage_chunk"):
        original = getattr(engine.queue, method_name)

        def publish(*args, _original=original, _name=method_name, **kwargs):
            harness.publications.append(
                (None if harness.active is None else harness.active.request_id, _name)
            )
            return _original(*args, **kwargs)

        monkeypatch.setattr(engine.queue, method_name, publish)
    return harness


def _submit_startup(harness, marker: int):
    count = len(harness.finished) + 1
    harness.engine.notify_observation(_startup_observation(marker))
    _wait_for(lambda: len(harness.finished) == count and not harness.engine._request_in_flight)
    return harness.requests[-1]


def _startup_evidence(tmp_path, harness, **extra) -> None:
    """Keep the synthetic inputs, terminal timing and action provenance reviewable."""
    (tmp_path / "startup_evidence.json").write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "id": request.request_id,
                        "kind": request.kind,
                        "phase": request.startup_phase,
                        "reset_epoch": request.reset_epoch,
                        "task_epoch": request.task_epoch,
                        "task": request.task,
                        "observation_marker": request.observation[RUNTIME_SCALAR_KEYS[0]],
                    }
                    for request in harness.requests
                ],
                "events": [] if harness.sink is None else harness.sink.events,
                "probe_record": harness.engine._startup_probe_record,
                "tracker_samples_s": list(harness.engine._latency_tracker._values),
                "publications": harness.publications,
                "final_barrier_request_ids": harness.syncs,
                "policy_attempt_request_ids": harness.policy_attempts,
                "requests_started": harness.engine.stats.requests_started,
                "queue_reset_epoch": harness.engine.queue.reset_epoch,
                "queue_task_epoch": harness.engine.queue.task_epoch,
                "next_action_index": harness.engine.queue.next_action_index,
                "phase": harness.engine._startup_phase,
                "ready": harness.engine.ready,
                "failed": harness.engine.failed,
                **extra,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("fallback_mode", ["identity", "discard"])
def test_predicted_startup_cold_is_temporary_and_not_admitted(monkeypatch, tmp_path, fallback_mode):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics(), fallback_mode=fallback_mode)
    engine = harness.engine
    assert not engine.ready
    with pytest.raises(RuntimeError, match="cold_temporary"):
        engine.get_action(None)
    engine.start()
    engine.resume()
    try:
        cold = _submit_startup(harness, 10)
        assert (cold.request_id, cold.kind, cold.startup_phase) == (0, "bootstrap", "cold_temporary")
        assert engine._startup_phase == "probe" and not engine.ready
        assert engine.queue.qsize() == 12
        assert len(engine._latency_tracker) == 0
        assert harness.arguments["policy"].kwargs == [{}]
        assert harness.arguments["policy"].model.encode_calls == 1
        assert harness.arguments["future_latent_predictor"].calls == []
        with pytest.raises(RuntimeError, match="probe"):
            engine.get_action(None)
        assert engine.queue.next_action_index == 0
        assert engine.stats.underflows == 0
        assert engine.stats.requests_started == engine.stats.bootstrap_requests == 1
        assert engine.stats.planned_requests == 0
        assert harness.sink.events[0]["total_chunk_s"] > 0.6
        assert not harness.sink.events[0]["latency_tracker_admitted"]
        assert [event["event"] for event in harness.sink.events] == ["chunk_request"]
        assert len(harness.requests) == 1 and engine._pending_request is None
        _startup_evidence(tmp_path, harness, temporary_policy_actions=harness.outputs[0][0].tolist())
    finally:
        engine.stop()


def test_predicted_startup_probe_uses_fresh_observation_and_true_d8_prefix(monkeypatch, tmp_path):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics(), cap=3)
    engine = harness.engine
    engine._queue_threshold = 0
    creations = []
    create = engine.queue.create_takeover_plan

    def create_plan(**kwargs):
        result = create(**kwargs)
        creations.append((kwargs, result.plan))
        return result

    monkeypatch.setattr(engine.queue, "create_takeover_plan", create_plan)
    engine.start()
    engine.resume()
    try:
        _submit_startup(harness, 10)
        assert engine.stats.requests_started == 1
        probe = _submit_startup(harness, 20)
        assert probe.kind == "startup_probe" and probe.startup_phase == "probe"
        assert len(creations) == 1
        kwargs, plan = creations[0]
        assert kwargs["planned_delay_steps"] == kwargs["max_prediction_delay"] == 8
        assert kwargs["committed_guard_steps"] == engine._committed_guard_steps
        assert (plan.request_id, plan.reset_epoch, plan.task_epoch, plan.task) == (1, 0, 0, "task A")
        assert plan.next_action_index == 0 and plan.takeover_index == 8
        assert torch.equal(plan.committed_policy_actions, harness.outputs[0][0, :8])
        assert plan.committed_mask.tolist() == [True] * 8
        predictor = harness.arguments["future_latent_predictor"]
        assert len(predictor.calls) == 1
        tokens, masks, actions, action_mask, state, delay = predictor.calls[0]
        assert torch.equal(actions[0], harness.outputs[0][0, :8])
        assert action_mask.tolist() == [[True] * 8] and delay.tolist() == [8]
        assert state.shape == (1, 32) and state[0, 0].item() == 20
        assert torch.count_nonzero(state[0, 6:]) == 0
        policy = harness.arguments["policy"]
        assert policy.model.encode_calls == policy.prepare_image_calls == 2
        assert policy.prepare_state_calls == 1 and len(policy.kwargs) == 2
        assert policy.batches[1][POLICY_CAMERA_KEYS[0]].mean().item() == pytest.approx(20 / 255)
        assert policy.batches[1][POLICY_CAMERA_KEYS[1]].mean().item() == pytest.approx(235 / 255)
        assert policy.kwargs[1]["future_image_token_masks"] is masks
        for token, residual, future in zip(
            tokens, predictor.prediction.delta_tokens, policy.kwargs[1]["future_image_tokens"], strict=True
        ):
            assert torch.equal(future, (token.float() + residual.float()).to(token.dtype))
        assert harness.publications == [(0, "install_active_chunk")]
        assert harness.syncs == [0, 1]
        assert engine.last_delay_plan is None and engine.stats.planned_requests == 0
        assert not any(event["event"] == "planner_decision" for event in harness.sink.events)
        record = engine._startup_probe_record
        assert record["outcome"] == "probe_discarded"
        assert record["planned_delay_steps"] == 8
        assert (record["plan_next_action_index"], record["takeover_index"]) == (0, 8)
        assert all(
            record[key] is None
            for key in ("result_next_action_index", "late_steps", "consumed_steps_at_stage")
        )
        _startup_evidence(
            tmp_path, harness, committed_policy_actions=actions[0].tolist(), state=state.tolist()
        )
    finally:
        engine.stop()


@pytest.mark.parametrize("cap", [1, 8])
def test_predicted_startup_pass_preserves_only_seed_and_waits_for_fresh_bootstrap(monkeypatch, tmp_path, cap):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics(), cap=cap, fallback_mode="discard")
    harness.durations["probe"] = (cap - 0.5) / 30
    engine = harness.engine
    queue, tracker = engine.queue, engine._latency_tracker
    reset_calls = []
    for name in ("policy", "preprocessor", "postprocessor"):
        monkeypatch.setattr(harness.arguments[name], "reset", lambda _name=name: reset_calls.append(_name))
    engine.start()
    engine.resume()
    try:
        _submit_startup(harness, 10)
        _submit_startup(harness, 20)
        record = engine._startup_probe_record
        seed = record["total_chunk_s"]
        assert record["startup_gate_raw_required_delay_steps"] == cap
        assert record["startup_gate_outcome"] == "passed" and record["latency_tracker_admitted"]
        assert list(tracker._values) == [seed]
        assert engine._startup_phase == "fresh_warmed" and not engine.ready
        assert engine._reset_epoch == queue.reset_epoch == record["startup_queue_reset_epoch_after"] == 1
        assert queue.qsize() == 0 and queue.plan_snapshot() is None and not queue.has_staged_chunk()
        assert queue.next_action_index == 0
        assert reset_calls == [] and engine.queue is queue and engine._latency_tracker is tracker
        assert engine._pending_request is None and len(harness.requests) == 2
        with pytest.raises(RuntimeError, match="fresh_warmed"):
            engine.get_action(None)
        fresh = _submit_startup(harness, 30)
        assert (fresh.request_id, fresh.kind, fresh.reset_epoch) == (2, "bootstrap", 1)
        assert engine._startup_phase == "complete" and engine.ready
        assert list(tracker._values) == [seed]
        assert harness.arguments["policy"].kwargs[2] == {}
        assert engine.stats.requests_started == 3 and engine.stats.bootstrap_requests == 2
        assert engine.stats.planned_requests == 0 and engine.stats.prediction_cap_exceeded == 0
        first_action = engine.get_action(None)
        assert torch.equal(first_action, harness.outputs[2][0, 0] + 1000)
        assert not torch.equal(first_action, harness.outputs[0][0, 0] + 1000)
        assert not torch.equal(first_action, harness.outputs[1][0, 0] + 1000)
        planned = _submit_startup(harness, 40)
        assert (planned.request_id, planned.kind, planned.reset_epoch) == (3, "planned", 1)
        assert planned.plan.planned_delay_steps == cap
        assert planned.plan.next_action_index == 1 and planned.plan.takeover_index == 1 + cap
        assert len(tracker) == 2 and tracker._values[0] == seed
        assert reset_calls == []
        _startup_evidence(tmp_path, harness, actual_cap=cap, first_action=first_action.tolist())
    finally:
        engine.stop()


@pytest.mark.parametrize(
    "failure", ["cap", "policy", "post_policy", "total", "phase_missing", "phase_nonfinite"]
)
def test_predicted_startup_probe_gate_stops_once(monkeypatch, tmp_path, failure):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics(), cap=3)
    engine = harness.engine
    if failure == "cap":
        harness.durations["probe"] = 3.5 / 30
    elif failure in ("policy", "post_policy", "total"):
        harness.nonfinite = failure
    else:
        record_phase = engine._record_metrics_phase

        @contextmanager
        def broken_phase(phase, metrics, cuda_events):
            with record_phase(phase, metrics, cuda_events):
                yield
            if harness.active.kind == "startup_probe" and phase == "vision_encode":
                metrics["phase_host_wall_s"][phase] = None if failure == "phase_missing" else float("nan")

        monkeypatch.setattr(engine, "_record_metrics_phase", broken_phase)
    engine.start()
    engine.resume()
    try:
        _submit_startup(harness, 10)
        _submit_startup(harness, 20)
        _wait_for(lambda: engine.failed)
        assert harness.shutdown.is_set() and not engine.ready and engine._startup_phase == "failed"
        assert engine.failure_traceback
        assert len(engine._latency_tracker) == 0
        assert harness.publications == [(0, "install_active_chunk")]
        assert engine.stats.prediction_cap_exceeded == (failure == "cap")
        assert engine.stats.deadline_misses == engine.stats.stale_results == 0
        errors = [event for event in harness.sink.events if event["request_id"] == 1]
        assert len(errors) == 1 and errors[0]["event"] == "request_error"
        error = errors[0]
        assert error["failed_phase"] == "startup_gate"
        assert not error["latency_tracker_admitted"]
        expected_gate = (
            "cap_exceeded"
            if failure == "cap"
            else "telemetry_missing"
            if failure == "phase_missing"
            else "nonfinite"
        )
        assert error["startup_gate_outcome"] == expected_gate
        if failure == "cap":
            assert error["startup_gate_raw_required_delay_steps"] == 4
        assert engine._startup_probe_record["request_id"] == error["request_id"]
        assert engine._startup_probe_record["startup_gate_outcome"] == expected_gate
        assert not any(event["event"] == "planner_decision" for event in harness.sink.events)
        engine.notify_observation(_startup_observation(30))
        assert engine.stats.requests_started == 2 and len(harness.requests) == 2
        with pytest.raises(RuntimeError, match="failed"):
            engine.get_action(None)
        assert engine.queue.next_action_index == engine.stats.underflows == 0
        _startup_evidence(tmp_path, harness, failure=failure)
    finally:
        engine.stop()


@pytest.mark.parametrize(
    "startup_phase,failed_phase",
    [
        ("cold_temporary", "policy_total"),
        ("probe", "predictor_forward"),
        ("probe", "policy_total"),
        ("probe", "postprocessor"),
        ("fresh_warmed", "policy_total"),
    ],
)
def test_predicted_startup_probe_error_is_fatal_on_first_request(
    monkeypatch, tmp_path, startup_phase, failed_phase
):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics())
    harness.fail = (startup_phase, failed_phase)
    engine = harness.engine
    engine.start()
    engine.resume()
    try:
        phases = ["cold_temporary", "probe", "fresh_warmed"]
        for index in range(phases.index(startup_phase) + 1):
            _submit_startup(harness, 10 * (index + 1))
        _wait_for(lambda: engine.failed)
        assert not engine.ready and harness.shutdown.is_set()
        errors = [event for event in harness.sink.events if event["event"] == "request_error"]
        assert len(errors) == 1
        error = errors[0]
        assert error["failed_phase"] == failed_phase and error["startup_phase"] == startup_phase
        assert error["startup_gate_outcome"] == "error" and error["error_type"] == "ValueError"
        assert not error["latency_tracker_admitted"]
        assert len(harness.requests) == phases.index(startup_phase) + 1
        assert engine.stats.requests_started == len(harness.requests)
        assert len(engine._latency_tracker) == (startup_phase == "fresh_warmed")
        engine.notify_observation(_startup_observation(40))
        assert engine._pending_request is None
        assert sum(event["request_id"] == error["request_id"] for event in harness.sink.events) == 1
        assert engine.failure_traceback and "Synthetic" in engine.failure_traceback
        _startup_evidence(tmp_path, harness, injected_failure=list(harness.fail))
    finally:
        engine.stop()


@pytest.mark.parametrize("sink_mode", ["off", "record", "emit_failure", "close_failure"])
def test_predicted_startup_probe_internal_record_with_optional_sink(monkeypatch, tmp_path, sink_mode):
    class OptionalSink(_RecordingMetrics):
        def emit(self, event):
            super().emit(event)
            if sink_mode == "emit_failure":
                raise OSError("Synthetic metrics emit failure")

        def close(self):
            super().close()
            if sink_mode == "close_failure":
                raise OSError("Synthetic metrics close failure")

    sink = None if sink_mode == "off" else OptionalSink()
    harness = _startup_harness(monkeypatch, sink=sink)
    engine = harness.engine
    engine.start()
    engine.resume()
    try:
        _submit_startup(harness, 10)
        assert engine._startup_probe_record is None
        _submit_startup(harness, 20)
        record = engine._startup_probe_record
        assert record["schema_version"] == 1 and record["backend"] == "predictive_async"
        assert record["context_mode"] == "predicted" and record["request_kind"] == "startup_probe"
        assert record["startup_gate_outcome"] == "passed" and record["latency_tracker_admitted"]
        assert set(record["phase_host_wall_s"]) == _METRIC_PHASES
        assert all(math.isfinite(value) and value >= 0 for value in record["phase_host_wall_s"].values())
        assert record["phase_cuda_stream_elapsed_ms"] is None
        assert record["predictor_calls"] == 1 and not record["policy_includes_vision"]
        _submit_startup(harness, 30)
        assert engine.ready and not engine.failed and not harness.shutdown.is_set()
        assert engine._startup_probe_record is record
        if sink is None:
            assert harness.clock_counts[0] == harness.clock_counts[2] == 1
            assert harness.clock_counts[1] > 1
        else:
            assert [event for event in sink.events if event["request_kind"] == "startup_probe"] == [record]
            assert len(sink.events) == 3
    finally:
        engine.stop()
    assert not engine.failed and not harness.shutdown.is_set()
    if sink is not None:
        assert sink.close_calls == 1
    _startup_evidence(tmp_path, harness, sink_mode=sink_mode, host_clock_calls=harness.clock_counts)


@pytest.mark.parametrize("cuda_timing", ["complete", "missing_phase"])
def test_predicted_startup_probe_timing_uses_original_final_barrier(monkeypatch, tmp_path, cuda_timing):
    harness = _startup_harness(monkeypatch)
    engine = harness.engine
    engine.start()
    engine.resume()
    order = []
    try:
        _submit_startup(harness, 10)
        engine._device = torch.device("cuda:0")
        stream = object()
        prepare = predictive_async.prepare_observation_for_inference
        tensor, zeros, finite, item = torch.tensor, torch.zeros, torch.isfinite, torch.Tensor.item
        synchronize = predictive_async._synchronize_policy_device

        class FakeEvent:
            def __init__(self, *, enable_timing):
                assert enable_timing

            def record(self, actual_stream):
                assert actual_stream is stream and "barrier" not in order
                order.append("record")

            def elapsed_time(self, other):
                assert "barrier" in order
                order.append("elapsed")
                if cuda_timing == "missing_phase" and order.count("elapsed") == 1:
                    raise RuntimeError("Synthetic CUDA event elapsed is unavailable")
                return 0.25

        def cpu_factory(factory):
            def allocate(*args, **kwargs):
                if torch.device(kwargs.get("device", "cpu")).type == "cuda":
                    kwargs["device"] = "cpu"
                return factory(*args, **kwargs)

            return allocate

        def finite_reduction(value):
            assert "barrier" not in order
            order.append("finite_enqueued")
            return finite(value)

        def read_scalar(value, *args):
            if value.ndim == 0 and value.dtype == torch.bool:
                assert "barrier" in order
                order.append("finite_read")
            return item(value, *args)

        def final_barrier(device):
            assert device == torch.device("cuda:0")
            order.append("barrier")
            synchronize(device)

        monkeypatch.setattr(
            predictive_async,
            "prepare_observation_for_inference",
            lambda batch, device, task, robot: prepare(batch, torch.device("cpu"), task, robot),
        )
        monkeypatch.setattr(torch, "tensor", cpu_factory(tensor))
        monkeypatch.setattr(torch, "zeros", cpu_factory(zeros))
        monkeypatch.setattr(torch, "isfinite", finite_reduction)
        monkeypatch.setattr(torch.Tensor, "item", read_scalar)
        monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda *args: pytest.fail("Probe added a second CUDA synchronization")
        )
        monkeypatch.setattr(predictive_async, "_synchronize_policy_device", final_barrier)
        _submit_startup(harness, 20)
        assert order.count("record") == 16 and order.count("elapsed") == 8
        assert order.count("barrier") == order.count("finite_read") == 1
        assert order.index("finite_enqueued") < order.index("barrier") < order.index("finite_read")
        record = engine._startup_probe_record
        assert set(record["phase_cuda_stream_elapsed_ms"]) == _METRIC_PHASES
        if cuda_timing == "complete":
            assert not engine.failed and engine._startup_phase == "fresh_warmed"
            assert set(record["phase_cuda_stream_elapsed_ms"].values()) == {0.25}
        else:
            _wait_for(lambda: engine.failed)
            assert not engine.ready and harness.shutdown.is_set()
            assert (
                record["event"] == "request_error" and record["startup_gate_outcome"] == "telemetry_missing"
            )
            assert record["phase_cuda_stream_elapsed_ms"]["observation_preparation"] is None
            assert len(engine._latency_tracker) == 0
        assert all(math.isfinite(value) and value >= 0 for value in record["phase_host_wall_s"].values())
        assert record["total_chunk_s"] == pytest.approx(
            record["cuda_completed_at_s"] - record["requested_at_s"]
        )
        assert record["device_completion_wait_s"] >= 0
        assert harness.publications == [(0, "install_active_chunk")]
        _startup_evidence(tmp_path, harness, fake_cuda_order=order, cuda_timing=cuda_timing)
    finally:
        engine.stop()


@pytest.mark.parametrize(
    "boundary",
    [
        "initial",
        "noops",
        "complete_reset",
        "complete_task",
        "pending_cold_reset",
        "pending_probe_task",
        "pending_fresh_reset",
        "inflight_cold_task",
        "inflight_probe_reset",
        "inflight_fresh_task",
        "inflight_probe_task_error",
        "gap_probe_reset",
        "gap_fresh_task",
    ],
)
def test_predicted_startup_reset_and_task_boundaries(monkeypatch, tmp_path, boundary):
    class DispatchGate(Event):
        def __init__(self):
            super().__init__()
            self.release = Event()
            self.release.set()
            self.blocked = Event()

        def wait(self, timeout=None):
            signaled = super().wait(timeout)
            if signaled and not self.release.is_set():
                self.blocked.set()
                if not self.release.wait(timeout=3):
                    raise RuntimeError("Synthetic pending dispatch was not released")
            return signaled

    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics())
    engine = harness.engine
    dispatch = DispatchGate()
    engine._request_ready = dispatch
    if boundary == "initial":
        engine.reset()
        engine.reset()
        assert engine.set_task("task B")
        assert not engine.failed and not engine.ready
    engine.start()
    engine.resume()
    try:
        if boundary in ("initial", "noops", "complete_reset", "complete_task"):
            cold = _submit_startup(harness, 10)
            if boundary == "initial":
                assert (cold.reset_epoch, cold.task_epoch, cold.task) == (2, 1, "task B")
            if boundary == "noops":
                assert not engine.set_task("task A")
                engine.pause()
                engine.notify_observation(_startup_observation(15))
                assert engine.stats.requests_started == 1
                engine.resume()
            _submit_startup(harness, 20)
            _submit_startup(harness, 30)
            assert engine.ready and not engine.failed
            seed = list(engine._latency_tracker._values)
            assert len(seed) == 1
            record = engine._startup_probe_record
            if boundary.startswith("complete"):
                if boundary == "complete_reset":
                    engine.reset()
                else:
                    assert engine.set_task("task B")
                assert engine.ready and engine._startup_phase == "complete"
                assert list(engine._latency_tracker._values) == seed
                followup = _submit_startup(harness, 40)
                assert followup.startup_phase is None
                assert followup.kind == ("bootstrap" if boundary == "complete_reset" else "planned")
                assert engine._startup_probe_record is record
                assert sum(request.kind == "startup_probe" for request in harness.requests) == 1
            assert not harness.shutdown.is_set()
        else:
            window, phase_name, change, *suffix = boundary.split("_")
            phase = {"cold": "cold_temporary", "probe": "probe", "fresh": "fresh_warmed"}[phase_name]
            prerequisites = {"cold": 0, "probe": 1, "fresh": 2}[phase_name]
            for index in range(prerequisites):
                _submit_startup(harness, 10 * (index + 1))
            assert engine._startup_phase == phase
            before = len(harness.sink.events)
            cancelled_id = None
            if window == "pending":
                dispatch.release.clear()
                engine.notify_observation(_startup_observation(40))
                assert dispatch.blocked.wait(timeout=3)
                pending = engine._pending_request
                assert pending is not None and pending.startup_phase == phase
                cancelled_id = pending.request_id
            elif window == "inflight":
                harness.block_phase = phase
                harness.fail_after_release = suffix == ["error"]
                engine.notify_observation(_startup_observation(40))
                assert harness.entered.wait(timeout=3)
                cancelled_id = harness.requests[-1].request_id
            if change == "reset":
                engine.reset()
            else:
                assert engine.set_task("task B")
            harness.release.set()
            dispatch.release.set()
            _wait_for(lambda: engine.failed and not engine._request_in_flight)
            _wait_for(lambda: not engine._worker.is_alive())
            assert not engine.ready and engine._startup_phase == "failed"
            assert harness.shutdown.is_set() and engine.failure_traceback
            assert engine._pending_request is None
            assert len(harness.sink.events) == before + (window != "gap")
            if cancelled_id is not None:
                terminals = [event for event in harness.sink.events if event["request_id"] == cancelled_id]
                assert len(terminals) == 1
                terminal = terminals[0]
                assert (
                    terminal["event"] == "request_error" and terminal["failed_phase"] == "startup_interrupted"
                )
                assert (
                    terminal["error_type"] == "RuntimeError" and terminal["startup_gate_outcome"] == "error"
                )
                assert ("reset" if change == "reset" else "task change") in terminal["error_message"]
                assert not terminal["latency_tracker_admitted"]
                if window == "pending":
                    assert all(value is None for value in terminal["phase_host_wall_s"].values())
                    assert all(
                        terminal[key] is None
                        for key in ("started_at_s", "total_chunk_s", "cuda_completed_at_s")
                    )
                    assert cancelled_id not in harness.policy_attempts
                else:
                    assert terminal["phase_host_wall_s"]["preprocessor"] is not None
                    assert harness.policy_attempts.count(cancelled_id) == 1
                    if suffix == ["error"]:
                        assert terminal["phase_host_wall_s"]["policy_total"] is None
                        assert terminal["total_chunk_s"] is None
                    else:
                        assert terminal["phase_host_wall_s"]["policy_total"] is not None
                        assert terminal["total_chunk_s"] is not None
                assert not any(request_id == cancelled_id for request_id, _ in harness.publications)
            terminal_count = len(harness.sink.events)
            accepted_count = engine.stats.requests_started
            engine.reset()
            engine.set_task("task C")
            engine.notify_observation(_startup_observation(50))
            assert len(harness.sink.events) == terminal_count
            assert engine.stats.requests_started == accepted_count
            with pytest.raises(RuntimeError, match="failed"):
                engine.get_action(None)
            assert engine.queue.next_action_index == engine.stats.underflows == 0
        _startup_evidence(tmp_path, harness, boundary=boundary)
    finally:
        harness.release.set()
        dispatch.release.set()
        engine.stop()


def test_predicted_startup_fresh_completion_linearizes_concurrent_task_change(monkeypatch, tmp_path):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics())
    engine = harness.engine
    installed = Event()
    release_completion = Event()
    task_lock_attempted = Event()
    task_finished = Event()
    task_thread = None
    task_results = []
    task_errors = []
    order = []
    snapshots = {}
    request_lock = engine._request_lock
    install = engine.queue.install_active_chunk
    set_ready = engine._ready_event.set

    def snapshot():
        task, task_epoch = engine.task_snapshot
        return {
            "task": task,
            "task_epoch": task_epoch,
            "queue_task_epoch": engine.queue.task_epoch,
            "startup_phase": engine._startup_phase,
            "ready": engine.ready,
        }

    class ObservedRequestLock:
        def __enter__(self):
            is_task_thread = current_thread() is task_thread
            if is_task_thread:
                order.append("task_lock_attempt")
                task_lock_attempted.set()
            request_lock.acquire()
            if is_task_thread:
                snapshots["task_lock_acquired"] = snapshot()
                order.append("task_lock_acquired")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            request_lock.release()

    def install_before_completion(*args, **kwargs):
        result = install(*args, **kwargs)
        if harness.active.startup_phase == "fresh_warmed":
            assert result.outcome is InstallOutcome.INSTALLED
            assert request_lock.locked()
            snapshots["fresh_installed"] = snapshot()
            order.append("fresh_installed")
            installed.set()
            if not release_completion.wait(timeout=3):
                raise RuntimeError("Synthetic fresh completion was not released")
        return result

    def commit_ready():
        assert request_lock.locked()
        set_ready()
        snapshots["ready_committed"] = snapshot()
        order.append("ready_committed")

    def change_task():
        try:
            task_results.append(engine.set_task("task B"))
            snapshots["task_changed"] = snapshot()
            order.append("task_changed")
        except Exception as error:
            task_errors.append(error)
        finally:
            task_finished.set()

    monkeypatch.setattr(engine, "_request_lock", ObservedRequestLock())
    monkeypatch.setattr(engine.queue, "install_active_chunk", install_before_completion)
    monkeypatch.setattr(engine._ready_event, "set", commit_ready)
    engine.start()
    engine.resume()
    try:
        _submit_startup(harness, 10)
        _submit_startup(harness, 20)
        seed = list(engine._latency_tracker._values)
        engine.notify_observation(_startup_observation(30))
        assert installed.wait(timeout=3)
        assert snapshots["fresh_installed"] == {
            "task": "task A",
            "task_epoch": 0,
            "queue_task_epoch": 0,
            "startup_phase": "fresh_warmed",
            "ready": False,
        }
        assert engine.queue.qsize() == 12 and engine.queue.next_action_index == 0
        task_thread = Thread(target=change_task, name="SyntheticConcurrentTaskChange")
        task_thread.start()
        assert task_lock_attempted.wait(timeout=3)
        # The old set_task mutated both identities before this observed lock
        # attempt. The completion lock must now keep them at A until ready.
        snapshots["task_waiting"] = snapshot()
        assert snapshots["task_waiting"] == snapshots["fresh_installed"]
        assert not task_finished.is_set()
        release_completion.set()
        assert task_finished.wait(timeout=3)
        task_thread.join(timeout=3)
        assert not task_thread.is_alive() and task_errors == [] and task_results == [True]
        _wait_for(lambda: len(harness.finished) == 3 and not engine._request_in_flight)
        completed_identity = {
            "task": "task A",
            "task_epoch": 0,
            "queue_task_epoch": 0,
            "startup_phase": "complete",
            "ready": True,
        }
        assert snapshots["ready_committed"] == snapshots["task_lock_acquired"] == completed_identity
        assert snapshots["task_changed"] == {
            **completed_identity,
            "task": "task B",
            "task_epoch": 1,
            "queue_task_epoch": 1,
        }
        assert order == [
            "fresh_installed",
            "task_lock_attempt",
            "ready_committed",
            "task_lock_acquired",
            "task_changed",
        ]
        fresh = harness.requests[-1]
        terminals = [event for event in harness.sink.events if event.get("request_id") == fresh.request_id]
        assert len(terminals) == 1
        terminal = terminals[0]
        assert (
            terminal["event"],
            terminal["request_kind"],
            terminal["startup_phase"],
            terminal["outcome"],
        ) == ("chunk_request", "bootstrap", "fresh_warmed", "installed")
        assert (terminal["task"], terminal["task_epoch"], terminal["reset_epoch"]) == ("task A", 0, 1)
        assert not terminal["latency_tracker_admitted"]
        assert not any(event["event"] == "request_error" for event in harness.sink.events)
        assert engine._startup_interruption_reason is None
        assert engine.ready and not engine.failed and not harness.shutdown.is_set()
        assert list(engine._latency_tracker._values) == seed
        assert engine.stats.requests_started == 3 and engine.stats.bootstrap_requests == 2
        assert engine.stats.planned_requests == 0 and harness.policy_attempts == [0, 1, 2]
        assert engine.queue.next_action_index == 0
        _startup_evidence(tmp_path, harness, task_completion_order=order, boundary_states=snapshots)
    finally:
        release_completion.set()
        if task_thread is not None:
            task_thread.join(timeout=3)
        engine.stop()


@pytest.mark.parametrize(
    "outcome", ["early", "on_time", "late", "reset", "task", "publication_error", "d0", "fallback"]
)
def test_predicted_tracker_admits_only_probe_and_positive_planned(monkeypatch, tmp_path, outcome):
    harness = _startup_harness(monkeypatch, sink=_RecordingMetrics())
    engine = harness.engine
    engine.start()
    engine.resume()
    try:
        for marker in (10, 20, 30):
            _submit_startup(harness, marker)
        assert engine.ready
        seed = list(engine._latency_tracker._values)
        assert len(seed) == 1
        calls_before = len(harness.arguments["future_latent_predictor"].calls)
        if outcome in ("d0", "fallback"):
            remaining = 2 if outcome == "d0" else 0
            while engine.queue.qsize() > remaining:
                engine.get_action(None)
        harness.block_request_id = 3
        engine.notify_observation(_startup_observation(40))
        assert harness.entered.wait(timeout=3)
        request = harness.requests[-1]
        assert request.request_id == 3 and request.startup_phase is None
        if outcome == "fallback":
            assert request.kind == "bootstrap" and request.plan is None
        else:
            assert request.kind == "planned"
            assert request.plan.planned_delay_steps == (0 if outcome == "d0" else 1)
        if outcome in ("on_time", "late"):
            for _ in range(1 + (outcome == "late")):
                engine.get_action(None)
        elif outcome == "reset":
            engine.reset()
        elif outcome == "task":
            engine.set_task("task B")
        elif outcome == "publication_error":

            def fail_publication(*args, **kwargs):
                raise RuntimeError("Synthetic stage publication failure")

            monkeypatch.setattr(engine.queue, "stage_chunk", fail_publication)
        harness.release.set()
        _wait_for(lambda: len(harness.finished) == 4 and not engine._request_in_flight)
        event = [
            event
            for event in harness.sink.events
            if event.get("request_id") == 3 and event["event"] != "planner_decision"
        ]
        assert len(event) == 1
        event = event[0]
        admitted = outcome not in ("d0", "fallback")
        assert event["latency_tracker_admitted"] == admitted
        assert len(engine._latency_tracker) == 1 + admitted
        assert engine._latency_tracker._values[0] == seed[0]
        assert len(harness.arguments["future_latent_predictor"].calls) == calls_before + admitted
        assert not engine.failed and engine._startup_phase == "complete"
        if outcome == "publication_error":
            assert event["event"] == "request_error" and event["failed_phase"] == "queue_publication"
            assert engine._latency_tracker._values[-1] > 0
        else:
            expected = {
                "early": "staged_early",
                "on_time": "staged_on_time",
                "late": "deadline_miss",
                "reset": "stale",
                "task": "stale",
                "d0": "staged_on_time",
                "fallback": "installed",
            }
            assert event["outcome"] == expected[outcome]
            if admitted:
                assert engine._latency_tracker._values[-1] == event["total_chunk_s"]
        assert engine.stats.deadline_misses == (outcome == "late")
        assert engine.stats.stale_results == (outcome in ("reset", "task"))
        assert (
            sum(
                event["latency_tracker_admitted"]
                for event in harness.sink.events
                if event["event"] in ("chunk_request", "request_error")
            )
            == 1 + admitted
        )
        _startup_evidence(tmp_path, harness, planned_outcome=outcome)
    finally:
        harness.release.set()
        engine.stop()


def test_predictive_metrics_off_preserves_calls_and_barrier(monkeypatch) -> None:
    arguments = _candidate_arguments()
    engine = PredictiveAsyncInferenceEngine(**arguments)
    request, _ = _candidate_planned_request(engine, 1)
    calls = []
    stage = engine.queue.stage_chunk

    def clock():
        calls.append("total_clock")
        return request.requested_at + 0.25

    def publish(*args, **kwargs):
        calls.append("stage")
        return stage(*args, **kwargs)

    def unexpected_event(*args, **kwargs):
        pytest.fail("metrics disabled must not create a CUDA event")

    def unexpected_payload(event):
        pytest.fail("metrics disabled must not emit an event payload")

    monkeypatch.setattr(predictive_async, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(
        predictive_async, "_synchronize_policy_device", lambda device: calls.append("barrier")
    )
    monkeypatch.setattr(torch.cuda, "Event", unexpected_event)
    monkeypatch.setattr(engine, "_emit_metrics", unexpected_payload)
    monkeypatch.setattr(engine.queue, "stage_chunk", publish)
    engine._run_request(request)
    assert calls == ["barrier", "total_clock", "stage"]
    assert arguments["policy"].prepare_image_calls == arguments["policy"].model.encode_calls == 1
    assert len(arguments["future_latent_predictor"].calls) == len(arguments["policy"].kwargs) == 1
    assert engine.queue.has_staged_chunk()
    assert len(engine._latency_tracker) == 2


@pytest.mark.parametrize(
    "case",
    [
        "predicted_early",
        "predicted_on_time",
        "predicted_late",
        "predicted_reset",
        "predicted_task",
        "predicted_d0",
        "identity",
        "bootstrap",
        "bootstrap_stale",
        "warmup_rgb",
        "warmup_override",
    ],
)
def test_predictive_metrics_request_phases_and_outcomes(case, tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlMetricsSink(path)
    arguments = _candidate_arguments()
    if case == "identity":
        arguments.update(context_mode="identity", future_latent_predictor=None)
    engine = PredictiveAsyncInferenceEngine(
        **arguments, metrics_sink=sink, use_torch_compile=case.startswith("warmup")
    )
    delay = 0 if case == "predicted_d0" else 1
    consumed = 0
    if case.startswith("predicted") or case == "identity":
        request, _ = _candidate_planned_request(engine, delay)
        if case in ("predicted_on_time", "predicted_late"):
            consumed = 1 + (case == "predicted_late")
            for _ in range(consumed):
                engine.get_action(None)
        elif case == "predicted_reset":
            engine.reset()
        elif case == "predicted_task":
            engine.set_task("task B")
    else:
        if case in ("bootstrap", "bootstrap_stale"):
            _mark_candidate_steady(engine)
        engine.resume()
        if case == "warmup_override":
            engine.notify_observation(_candidate_observation())
            engine._run_request(_take_pending(engine))
        engine.notify_observation(_candidate_observation())
        request = _take_pending(engine)
        if case == "bootstrap_stale":
            engine.reset()
    tracker_before = len(engine._latency_tracker)
    engine._run_request(request)
    engine.stop()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    event = [event for event in events if event["event"] == "chunk_request"][-1]
    assert event["schema_version"] == 1
    assert event["backend"] == "predictive_async"
    assert event["request_id"] == request.request_id
    assert event["request_kind"] == request.kind
    assert event["context_mode"] == arguments["context_mode"]
    assert event["reset_epoch"] == request.reset_epoch
    assert event["task_epoch"] == request.task_epoch
    assert event["task"] == request.task
    host = event["phase_host_wall_s"]
    assert set(host) == _METRIC_PHASES
    assert event["phase_cuda_stream_elapsed_ms"] is None
    predictor_called = case.startswith("predicted") and case != "predicted_d0"
    assert event["predictor_calls"] == int(predictor_called)
    for key in _PREDICTOR_PHASES:
        assert (host[key] is not None) == predictor_called
    native_vision = case in ("bootstrap", "bootstrap_stale", "warmup_rgb")
    assert event["policy_includes_vision"] == native_vision
    assert (host["vision_encode"] is None) == native_vision
    assert all(value is None or value >= 0 for value in host.values())
    assert event["requested_at_s"] <= event["started_at_s"] <= event["cuda_completed_at_s"]
    assert event["total_chunk_s"] == pytest.approx(event["cuda_completed_at_s"] - request.requested_at)
    assert event["dispatch_wait_s"] == pytest.approx(event["started_at_s"] - request.requested_at)
    assert event["device_completion_wait_s"] >= 0
    assert event["d_actual_wall"] == predictive_async.latency_to_steps(event["total_chunk_s"], engine._fps)
    admitted = request.kind != "warmup" and case not in ("predicted_d0", "bootstrap", "bootstrap_stale")
    assert event["latency_tracker_admitted"] == admitted
    assert len(engine._latency_tracker) == tracker_before + admitted
    assert event["underflow_total"] == 0
    if request.plan is None:
        assert event["planned_delay_steps"] is None
        assert event["plan_next_action_index"] is None
        assert event["consumed_steps_at_stage"] is None
        assert event["outcome"] == (
            "warmup_completed"
            if request.kind == "warmup"
            else "stale"
            if case == "bootstrap_stale"
            else "installed"
        )
    else:
        stale = case in ("predicted_reset", "predicted_task")
        expected = (
            "stale"
            if stale
            else "deadline_miss"
            if case == "predicted_late"
            else ("staged_on_time" if consumed == delay else "staged_early")
        )
        assert event["outcome"] == expected
        assert event["planned_delay_steps"] == delay
        assert event["plan_next_action_index"] == request.plan.next_action_index
        assert event["result_next_action_index"] == request.plan.next_action_index + consumed
        assert event["consumed_steps_at_stage"] == (None if stale else consumed)
        assert event["late_steps"] == (1 if case == "predicted_late" else 0)
        assert engine.stats.deadline_misses == (case == "predicted_late")
        assert engine.stats.stale_results == stale


def test_predictive_metrics_cuda_events_resolve_after_completion(monkeypatch) -> None:
    sink = _RecordingMetrics()
    engine = _make_engine(_ChunkPolicy(), metrics_sink=sink)
    engine.queue.install_active_chunk(
        torch.zeros(6, 2), torch.zeros(6, 2), task="task A", reset_epoch=0, task_epoch=0
    )
    engine._latency_tracker.add(0.001)
    engine.resume()
    engine.notify_observation(_obs())
    request = _take_pending(engine)
    order = []
    stream = object()
    stage = engine.queue.stage_chunk

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, actual_stream):
            assert actual_stream is stream
            assert "barrier" not in order
            order.append("record")

        def elapsed_time(self, other):
            assert "stage" in order
            order.append("elapsed")
            return 2.5

    def publish(*args, **kwargs):
        assert order.count("barrier") == 1
        result = stage(*args, **kwargs)
        order.append("stage")
        return result

    def unexpected_sync(*args, **kwargs):
        pytest.fail("phase timing must not add CUDA synchronization")

    engine._device = torch.device("cuda")
    monkeypatch.setattr(predictive_async, "prepare_observation_for_inference", lambda batch, *args: batch)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    monkeypatch.setattr(
        predictive_async, "_synchronize_policy_device", lambda device: order.append("barrier")
    )
    monkeypatch.setattr(engine.queue, "stage_chunk", publish)
    engine._run_request(request)
    event = sink.events[-1]
    assert event["event"] == "chunk_request"
    assert order.count("barrier") == 1
    cuda = event["phase_cuda_stream_elapsed_ms"]
    assert set(cuda) == _METRIC_PHASES
    assert all(cuda[key] is None for key in _PREDICTOR_PHASES)
    assert all(cuda[key] == 2.5 for key in _METRIC_PHASES - _PREDICTOR_PHASES)
    assert order.count("record") == 10
    assert order.count("elapsed") == 5
    assert engine.queue.has_staged_chunk()


@pytest.mark.parametrize("case", ["cap_wait", "cap_discard", "cap_bootstrap", "plan_failed"])
def test_predictive_metrics_cap_decision_without_request(case, tmp_path) -> None:
    sink = _RecordingMetrics()
    arguments = _candidate_arguments()
    engine = PredictiveAsyncInferenceEngine(
        **arguments, metrics_sink=sink, fallback_mode="discard" if case == "cap_discard" else "identity"
    )
    _mark_candidate_steady(engine)
    if case in ("cap_wait", "plan_failed"):
        engine.queue.install_active_chunk(
            torch.zeros(12, 6), torch.zeros(12, 6), task="task A", reset_epoch=0, task_epoch=0
        )
    engine._latency_tracker.add(0.001 if case == "plan_failed" else 0.5)
    if case == "plan_failed":
        engine.queue.create_takeover_plan(
            request_id=99,
            planned_delay_steps=1,
            max_prediction_delay=8,
            committed_guard_steps=2,
            reset_epoch=0,
            task_epoch=0,
            task="task A",
        )
    engine.resume()
    engine.notify_observation(_candidate_observation())
    if case != "cap_bootstrap":
        engine.notify_observation(_candidate_observation())
    events = sink.events
    assert len(events) == (1 if case == "cap_bootstrap" else 2)
    assert all(event["event"] == "planner_decision" for event in events)
    for event in events:
        assert event["request_id"] == (0 if case == "cap_bootstrap" else None)
        assert event["decision"] == ("bootstrap" if case == "cap_bootstrap" else case)
        assert event["prediction_cap_exceeded"] == (case != "plan_failed")
        if case != "plan_failed":
            assert event["raw_required_delay_steps"] == 15
            assert event["planned_delay_steps"] == (8 if case == "cap_wait" else 0)
        else:
            assert event["plan_outcome"] == "plan_in_flight"
    assert engine._request_id == (1 if case == "cap_bootstrap" else 0)
    assert arguments["future_latent_predictor"].calls == []
    assert arguments["policy"].kwargs == []
    (tmp_path / "planner_events.json").write_text(json.dumps(events, indent=2))
    engine.stop()


def test_predictive_metrics_queue_get_preserves_underflow_and_takeover(tmp_path) -> None:
    sink = _RecordingMetrics()
    engine = _make_engine(_ChunkPolicy(), metrics_sink=sink)
    assert engine.get_action(None) is None
    assert engine.get_action(None) is None
    engine.queue.install_active_chunk(
        torch.zeros(6, 2), torch.ones(6, 2), task="task A", reset_epoch=0, task_epoch=0
    )
    assert engine.get_action(None).tolist() == [1, 1]
    engine.set_task("task B")
    assert engine.get_action(None).tolist() == [1, 1]
    assert engine.dispatched_task == "task A"
    engine.queue.create_takeover_plan(
        request_id=0,
        planned_delay_steps=0,
        max_prediction_delay=8,
        committed_guard_steps=2,
        reset_epoch=0,
        task_epoch=1,
        task="task B",
    )
    engine.queue.stage_chunk(
        torch.zeros(6, 2), torch.full((6, 2), 7), request_id=0, reset_epoch=0, task_epoch=1, task="task B"
    )
    assert engine.get_action(None).tolist() == [7, 7]
    assert engine.dispatched_task == "task B"
    events = sink.events
    assert [event["event"] for event in events] == ["queue_get"] * 5
    assert [event["outcome"] for event in events] == [
        "underflow",
        "underflow",
        "action",
        "action",
        "takeover",
    ]
    assert [event["action_index"] for event in events] == [0, 0, 0, 1, 2]
    assert [event["task"] for event in events] == [None, None, "task A", "task A", "task B"]
    assert [event["underflow_total"] for event in events] == [1, 2, 2, 2, 2]
    assert all("reset_epoch" not in event and "task_epoch" not in event for event in events)
    (tmp_path / "queue_events.json").write_text(json.dumps(events, indent=2))
    engine.stop()


@pytest.mark.parametrize(
    "failure", ["preprocessor", "predictor_forward", "device_completion", "worker_budget"]
)
def test_predictive_metrics_request_error_preserves_error_policy(failure, monkeypatch, tmp_path) -> None:
    sink = _RecordingMetrics()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic inference failure")

    if failure == "worker_budget":
        engine = _make_engine(_ChunkPolicy(), metrics_sink=sink)
        monkeypatch.setattr(engine._policy, "predict_action_chunk", fail)
        engine.start()
        engine.resume()
        try:
            for index in range(5):
                engine.notify_observation(_obs())
                assert _wait_for(
                    lambda index=index: (
                        len(sink.events) >= index + 1
                        and not engine._request_in_flight
                        and (index < 4 or engine.failed)
                    )
                )
                assert engine.failed == (index == 4)
            assert engine._failure_traceback is not None
            assert len(engine._latency_tracker) == 0
            assert engine.queue.qsize() == 0
        finally:
            engine.stop()
        assert len(sink.events) == 5
        assert all(event["event"] == "request_error" for event in sink.events)
        assert all(event["failed_phase"] == "policy_total" for event in sink.events)
    else:
        arguments = _candidate_arguments()
        engine = PredictiveAsyncInferenceEngine(**arguments, metrics_sink=sink)
        request, _ = _candidate_planned_request(engine, 1)
        tracker_before = len(engine._latency_tracker)
        if failure == "preprocessor":
            engine._preprocessor = fail
        elif failure == "predictor_forward":
            engine._future_latent_predictor = fail
        else:
            monkeypatch.setattr(predictive_async, "_synchronize_policy_device", fail)
        with pytest.raises(RuntimeError, match="synthetic inference failure"):
            engine._run_request(request)
        event = sink.events[-1]
        assert event["event"] == "request_error"
        assert event["failed_phase"] == failure
        assert event["request_id"] == request.request_id
        assert event["phase_host_wall_s"]["observation_preparation"] is not None
        if failure in _METRIC_PHASES:
            assert event["phase_host_wall_s"][failure] is None
        assert len(engine._latency_tracker) == tracker_before
        assert not engine.queue.has_staged_chunk()
        engine.stop()
    errors = [event for event in sink.events if event["event"] == "request_error"]
    for event in errors:
        assert event["error_type"] == "RuntimeError"
        assert event["error_message"] == "synthetic inference failure"
        assert event["total_chunk_s"] is None
        assert event["d_actual_wall"] is None
        assert event["phase_cuda_stream_elapsed_ms"] is None
    (tmp_path / "error_events.json").write_text(json.dumps(errors, indent=2))


@pytest.mark.parametrize("failure", ["emit", "close"])
def test_predictive_metrics_sink_failures_leave_results_unchanged(failure, caplog) -> None:
    class FailingSink(_RecordingMetrics):
        def emit(self, event):
            if failure == "emit":
                raise OSError("synthetic metrics write failure")
            super().emit(event)

        def close(self):
            super().close()
            if failure == "close":
                raise OSError("synthetic metrics close failure")

    sink = FailingSink()
    policy = _ChunkPolicy()
    engine = _make_engine(policy, metrics_sink=sink)
    engine.start()
    engine.resume()
    try:
        engine.notify_observation(_obs())
        assert _wait_for(
            lambda: policy.calls == 1 and not engine._request_in_flight and engine.queue.qsize() == 6
        )
        engine.get_action(None)
        for count in range(2, 7):
            engine.notify_observation(_obs())
            assert _wait_for(lambda count=count: policy.calls == count and not engine._request_in_flight)
            assert engine.get_action(None) is not None
            assert engine.get_action(None).tolist() == [count * 100] * 2
        assert not engine.failed
        assert engine.stats.requests_started == 6
        assert engine.stats.deadline_misses == 0
        assert len(engine._latency_tracker) == 6
    finally:
        engine.stop()
    assert sink.close_calls == 1
    assert "synthetic metrics" in caplog.text
    assert not any(event["event"] == "request_error" for event in sink.events)


@pytest.mark.parametrize("lifecycle", ["never_started", "normal", "join_timeout"])
def test_predictive_metrics_sink_closes_after_worker_finishes(lifecycle, monkeypatch) -> None:
    sink = _RecordingMetrics()
    policy = _ChunkPolicy(block_call=1 if lifecycle == "join_timeout" else None)
    engine = _make_engine(policy, metrics_sink=sink)
    if lifecycle != "never_started":
        engine.start()
    if lifecycle == "join_timeout":
        engine.resume()
        engine.notify_observation(_obs())
        assert policy.entered.wait(timeout=2)
        monkeypatch.setattr(predictive_async, "_JOIN_TIMEOUT_S", 0.0)
        try:
            engine.stop()
            assert sink.close_calls == 0
            assert engine._worker.is_alive()
        finally:
            policy.release.set()
            assert _wait_for(lambda: sink.close_calls == 1)
            engine._worker.join(timeout=2)
    engine.stop()
    assert sink.close_calls == 1
    assert engine._worker is None
