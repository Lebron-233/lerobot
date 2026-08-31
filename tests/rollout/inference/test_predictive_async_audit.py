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

from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine


class _IdentityPipeline:
    steps: tuple = ()

    def __call__(self, batch):
        return batch

    def reset(self) -> None:
        pass


class _TokenModel:
    def encode_image_tokens(self, images, image_masks):
        batch_size = images[0].shape[0]
        return (torch.zeros(batch_size, 2, 4),), (torch.ones(batch_size, 2, dtype=torch.bool),)


class _Policy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(type="smolvla")
        self.model = _TokenModel()
        self.calls = 0
        self.called = Event()

    def reset(self) -> None:
        pass

    def prepare_images(self, batch):
        batch_size = batch["observation.state"].shape[0]
        return [torch.zeros(batch_size, 3, 4, 4)], [torch.ones(batch_size, dtype=torch.bool)]

    def predict_action_chunk(self, batch, **kwargs):
        self.calls += 1
        self.called.set()
        return torch.zeros(1, 6, 2)


def _make_engine(*, fallback_mode: str = "identity") -> PredictiveAsyncInferenceEngine:
    return PredictiveAsyncInferenceEngine(
        policy=_Policy(),
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
        fps=30.0,
        device="cpu",
        queue_threshold=10,
        latency_window=10,
        committed_guard_steps=2,
        max_late_steps=2,
        fallback_mode=fallback_mode,
    )


def _obs() -> dict[str, float]:
    return {"joint_a.pos": 1.0, "joint_b.pos": 2.0}


def _capture(errors: list[Exception], fn) -> None:
    try:
        fn()
    except Exception as error:
        errors.append(error)


def test_notify_drops_stale_task_snapshot_instead_of_raising(monkeypatch) -> None:
    engine = _make_engine()
    engine.resume()
    stale_notify_reached_queue = Event()
    newer_epoch_installed = Event()
    release_stale_notify = Event()
    original_invalidate = engine.queue.invalidate_task

    def interleaved_invalidate(task_epoch: int) -> bool:
        if task_epoch == 0:
            stale_notify_reached_queue.set()
            assert release_stale_notify.wait(timeout=2.0)
        result = original_invalidate(task_epoch)
        if task_epoch == 1:
            newer_epoch_installed.set()
        return result

    monkeypatch.setattr(engine.queue, "invalidate_task", interleaved_invalidate)
    errors: list[Exception] = []
    notify_thread = Thread(target=_capture, args=(errors, lambda: engine.notify_observation(_obs())))
    task_thread = Thread(target=_capture, args=(errors, lambda: engine.set_task("task B")))

    notify_thread.start()
    assert stale_notify_reached_queue.wait(timeout=2.0)
    task_thread.start()
    assert newer_epoch_installed.wait(timeout=2.0)
    release_stale_notify.set()
    notify_thread.join(timeout=2.0)
    task_thread.join(timeout=2.0)

    assert not notify_thread.is_alive()
    assert not task_thread.is_alive()
    assert errors == []
    assert engine.task_snapshot == ("task B", 1)
    assert engine.queue.task_epoch == 1
    assert engine._pending_request is None
    assert engine.queue.plan_snapshot() is None


def test_reset_with_stale_task_snapshot_clears_state_without_rewinding_epoch(monkeypatch) -> None:
    engine = _make_engine()
    engine.resume()
    engine.notify_observation(_obs())
    assert engine._pending_request is not None

    stale_reset_reached_queue = Event()
    newer_epoch_installed = Event()
    release_stale_reset = Event()
    original_reset = engine.queue.reset
    original_invalidate = engine.queue.invalidate_task

    def interleaved_reset(reset_epoch: int, *, task_epoch: int | None = None) -> None:
        stale_reset_reached_queue.set()
        assert release_stale_reset.wait(timeout=2.0)
        original_reset(reset_epoch, task_epoch=task_epoch)

    def observed_invalidate(task_epoch: int) -> bool:
        result = original_invalidate(task_epoch)
        if task_epoch == 1:
            newer_epoch_installed.set()
        return result

    monkeypatch.setattr(engine.queue, "reset", interleaved_reset)
    monkeypatch.setattr(engine.queue, "invalidate_task", observed_invalidate)
    errors: list[Exception] = []
    reset_thread = Thread(target=_capture, args=(errors, engine.reset))
    task_thread = Thread(target=_capture, args=(errors, lambda: engine.set_task("task B")))

    reset_thread.start()
    assert stale_reset_reached_queue.wait(timeout=2.0)
    task_thread.start()
    assert newer_epoch_installed.wait(timeout=2.0)
    release_stale_reset.set()
    reset_thread.join(timeout=2.0)
    task_thread.join(timeout=2.0)

    assert not reset_thread.is_alive()
    assert not task_thread.is_alive()
    assert errors == []
    assert engine.task_snapshot == ("task B", 1)
    assert engine.queue.task_epoch == 1
    assert engine.queue.reset_epoch == 1
    assert engine._pending_request is None
    assert engine.queue.qsize() == 0


def test_older_set_task_completion_does_not_clear_newer_pending_request(monkeypatch) -> None:
    engine = _make_engine()
    engine.resume()
    older_setter_reached_queue = Event()
    release_older_setter = Event()
    original_invalidate = engine.queue.invalidate_task

    def interleaved_invalidate(task_epoch: int) -> bool:
        if task_epoch == 1:
            older_setter_reached_queue.set()
            assert release_older_setter.wait(timeout=2.0)
        return original_invalidate(task_epoch)

    monkeypatch.setattr(engine.queue, "invalidate_task", interleaved_invalidate)
    errors: list[Exception] = []
    older_task_thread = Thread(target=_capture, args=(errors, lambda: engine.set_task("task B")))

    older_task_thread.start()
    assert older_setter_reached_queue.wait(timeout=2.0)
    assert engine.set_task("task C")
    engine.notify_observation(_obs())
    assert engine._pending_request is not None
    assert engine._pending_request.task_epoch == 2

    release_older_setter.set()
    older_task_thread.join(timeout=2.0)

    assert not older_task_thread.is_alive()
    assert errors == []
    assert engine.task_snapshot == ("task C", 2)
    assert engine.queue.task_epoch == 2
    assert engine._pending_request is not None
    assert engine._pending_request.task == "task C"
    assert engine._pending_request.task_epoch == 2


def _prime_planned_request(engine: PredictiveAsyncInferenceEngine) -> None:
    engine.queue.install_active_chunk(
        torch.zeros(6, 2),
        torch.zeros(6, 2),
        task="task A",
        reset_epoch=0,
        task_epoch=0,
    )
    engine._latency_tracker.add(0.01)
    engine.notify_observation(_obs())
    assert engine._pending_request is not None
    assert engine._pending_request.plan is not None
    assert engine.queue.plan_snapshot() is not None


def test_stop_cancels_pending_plan_before_worker_can_claim_it(monkeypatch) -> None:
    engine = _make_engine()
    policy = engine._policy
    worker_waiting = Event()
    release_wait = Event()
    original_wait = engine._request_ready.wait

    def gated_wait(timeout: float | None = None) -> bool:
        worker_waiting.set()
        assert release_wait.wait(timeout=2.0)
        return original_wait(timeout=0.0)

    monkeypatch.setattr(engine._request_ready, "wait", gated_wait)
    engine.start()
    engine.resume()
    assert worker_waiting.wait(timeout=2.0)
    _prime_planned_request(engine)

    stop_thread = Thread(target=engine.stop)
    stop_thread.start()
    assert engine._shutdown_event.wait(timeout=2.0)
    release_wait.set()
    stop_thread.join(timeout=2.0)

    assert not stop_thread.is_alive()
    assert policy.calls == 0
    assert engine._pending_request is None
    assert engine.queue.plan_snapshot() is None
    assert engine._worker is None


def test_pause_does_not_claim_request_that_wakes_worker_after_pause(monkeypatch) -> None:
    engine = _make_engine()
    policy = engine._policy
    worker_waiting = Event()
    release_wait = Event()
    original_wait = engine._request_ready.wait

    def gated_wait(timeout: float | None = None) -> bool:
        worker_waiting.set()
        assert release_wait.wait(timeout=2.0)
        return original_wait(timeout=0.0)

    monkeypatch.setattr(engine._request_ready, "wait", gated_wait)
    engine.start()
    engine.resume()
    assert worker_waiting.wait(timeout=2.0)
    _prime_planned_request(engine)

    engine.pause()
    release_wait.set()
    try:
        assert not policy.called.wait(timeout=0.1)
        assert engine._pending_request is not None
        assert engine.queue.plan_snapshot() is not None
    finally:
        engine.stop()
    assert engine._pending_request is None
    assert engine.queue.plan_snapshot() is None


def test_prediction_cap_identity_fallback_is_baseline_bootstrap_not_predictive_plan() -> None:
    engine = _make_engine()
    engine._latency_tracker.add(0.5)
    engine.resume()

    engine.notify_observation(_obs())

    delay_plan = engine.last_delay_plan
    assert delay_plan is not None
    assert delay_plan.raw_required_delay_steps == 16
    assert delay_plan.planned_delay_steps == 0
    assert delay_plan.prediction_cap_exceeded
    assert engine._pending_request is not None
    assert engine._pending_request.kind == "bootstrap"
    assert engine._pending_request.plan is None
    assert engine._pending_request.delay_plan == delay_plan
    assert engine.queue.plan_snapshot() is None
    assert engine.stats.bootstrap_requests == 1
    assert engine.stats.planned_requests == 0


def test_prediction_cap_discard_fallback_stays_underflow_without_advancing_index() -> None:
    engine = _make_engine(fallback_mode="discard")
    engine._latency_tracker.add(0.5)
    engine.resume()

    engine.notify_observation(_obs())

    assert engine.last_delay_plan is not None
    assert engine.last_delay_plan.prediction_cap_exceeded
    assert engine._pending_request is None
    assert engine.queue.plan_snapshot() is None
    assert engine.get_action(None) is None
    assert engine.queue.next_action_index == 0
    assert engine.stats.underflows == 1
