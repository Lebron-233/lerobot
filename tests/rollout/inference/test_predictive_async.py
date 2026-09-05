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

import time
from copy import deepcopy
from threading import Event
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


def _candidate_planned_request(engine, delay: int):
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
