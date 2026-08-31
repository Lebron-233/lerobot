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
from threading import Event
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.policies.rtc.scheduled_action_queue import InstallOutcome
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
def test_live_engine_rejects_non_identity_pr1_context_modes(mode: str) -> None:
    policy = _ChunkPolicy()
    expected = ValueError if mode == "oracle" else NotImplementedError
    with pytest.raises(expected):
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
