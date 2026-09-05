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

"""Latency-planned asynchronous inference for SmolVLA.

The control thread creates a request from one observation and one atomic
scheduled-queue snapshot before it pops that tick's action.  A single worker
then produces either a bootstrap chunk or a chunk staged for an explicit
takeover index. Identity context is the default; predicted context uses the
frozen short-horizon future-latent candidate.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Literal

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.rtc.scheduled_action_queue import (
    InstallOutcome,
    ScheduledActionQueue,
    StageOutcome,
    TakeoverPlan,
)
from lerobot.policies.smolvla.future_latent_checkpoint import (
    POLICY_CAMERA_KEYS,
    RAW_CAMERA_KEYS,
    RAW_IMAGE_SHAPE,
    RUNTIME_SCALAR_KEYS,
    _validate_frozen_candidate,
)
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .latency_replay import DelayPlan, compute_delay_plan, latency_to_steps
from .metrics import InferenceMetricsSink

if TYPE_CHECKING:
    from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.005
_JOIN_TIMEOUT_S = 3.0
_MAX_CONSECUTIVE_ERRORS = 5
_METRIC_PHASES = (
    "observation_preparation",
    "preprocessor",
    "vision_encode",
    "predictor_input_preparation",
    "predictor_forward",
    "residual_application",
    "policy_total",
    "postprocessor",
)


def _synchronize_policy_device(device: torch.device) -> None:
    """Make a completed chunk and its latency visible at the same boundary."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class PredictiveAsyncStats:
    """Small control-plane counter snapshot for tests and runtime diagnostics."""

    requests_started: int = 0
    bootstrap_requests: int = 0
    planned_requests: int = 0
    prediction_cap_exceeded: int = 0
    deadline_misses: int = 0
    stale_results: int = 0
    underflows: int = 0


@dataclass(frozen=True)
class _InferenceRequest:
    request_id: int
    kind: Literal["warmup", "bootstrap", "planned", "startup_probe"]
    observation: dict[str, Any]
    requested_at: float
    reset_epoch: int
    task: str
    task_epoch: int
    plan: TakeoverPlan | None = None
    delay_plan: DelayPlan | None = None
    startup_phase: Literal["cold_temporary", "probe", "fresh_warmed"] | None = None


class PredictiveAsyncInferenceEngine(InferenceEngine):
    """Single-worker predictive asynchronous backend for SmolVLA.

    ``predicted`` accepts only the frozen candidate and its processor instances.
    ``oracle`` is deliberately offline-only.
    """

    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        queue_threshold: int = 30,
        latency_quantile: float = 0.9,
        latency_window: int = 50,
        delay_safety_margin_steps: int = 1,
        min_prediction_delay: int = 0,
        max_prediction_delay: int = 8,
        committed_guard_steps: int = 2,
        max_late_steps: int = 2,
        context_mode: str = "identity",
        future_latent_predictor: LightweightFutureLatentPredictor | None = None,
        fallback_mode: str = "identity",
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        shutdown_event: Event | None = None,
        metrics_sink: InferenceMetricsSink | None = None,
    ) -> None:
        super().__init__(task=task)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be finite and > 0, got {fps}")
        if queue_threshold < 0:
            raise ValueError(f"queue_threshold must be >= 0, got {queue_threshold}")
        if committed_guard_steps < max_late_steps:
            raise ValueError(
                "committed_guard_steps must be >= max_late_steps, got "
                f"{committed_guard_steps} < {max_late_steps}"
            )
        if fallback_mode not in ("identity", "discard"):
            raise ValueError(f"Unsupported fallback_mode: {fallback_mode!r}")
        if context_mode == "oracle":
            raise ValueError("context_mode='oracle' is offline-only and cannot be used by live rollout")
        if context_mode == "predicted":
            if future_latent_predictor is None:
                raise ValueError("context_mode='predicted' requires a future_latent_predictor")
            if not 1 <= min_prediction_delay <= max_prediction_delay <= 8:
                raise ValueError(
                    "predicted context requires 1 <= min_prediction_delay <= max_prediction_delay <= 8"
                )
            _validate_frozen_candidate(policy, preprocessor, postprocessor)
            if not math.isclose(fps, 30.0):
                raise ValueError("The frozen future-latent candidate requires fps=30")
            state_feature = hw_features.get("observation.state", {})
            if (
                state_feature.get("dtype") != "float32"
                or tuple(state_feature.get("shape", ())) != (6,)
                or tuple(state_feature.get("names", ())) != RUNTIME_SCALAR_KEYS
            ):
                raise ValueError(
                    "The frozen future-latent candidate requires the ordered six runtime scalar keys"
                )
            visual_features = {
                key: feature
                for key, feature in hw_features.items()
                if key.startswith("observation.images.") or feature.get("dtype") in ("image", "video")
            }
            if set(visual_features) != set(RAW_CAMERA_KEYS) or any(
                feature.get("dtype") != "video" or tuple(feature.get("shape", ())) != RAW_IMAGE_SHAPE
                for feature in visual_features.values()
            ):
                raise ValueError(
                    "The frozen future-latent candidate requires exactly top/wrist RGB cameras at 480x640"
                )
        elif context_mode != "identity":
            raise ValueError(f"Unsupported context_mode: {context_mode!r}")
        elif future_latent_predictor is not None:
            raise ValueError("future_latent_predictor is only supported with context_mode='predicted'")
        if any(
            isinstance(step, RelativeActionsProcessorStep) and step.enabled
            for step in getattr(preprocessor, "steps", ())
        ):
            raise NotImplementedError(
                "PredictiveAsyncInferenceEngine does not yet define relative-action anchor/rebase semantics. "
                "Use RTC or remove relative action processor steps from the policy pipeline."
            )

        model = getattr(policy, "model", None)
        if not callable(getattr(policy, "prepare_images", None)) or not callable(
            getattr(model, "encode_image_tokens", None)
        ):
            raise TypeError("predictive_async currently requires the SmolVLA image-token override API")

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._hw_features = hw_features
        self._fps = fps
        self._device = torch.device(device or "cpu")
        self._queue_threshold = queue_threshold
        self._latency_quantile = latency_quantile
        self._delay_safety_margin_steps = delay_safety_margin_steps
        self._min_prediction_delay = min_prediction_delay
        self._max_prediction_delay = max_prediction_delay
        self._committed_guard_steps = committed_guard_steps
        self._max_late_steps = max_late_steps
        self._context_mode = context_mode
        self._future_latent_predictor = future_latent_predictor
        self._fallback_mode = fallback_mode
        self._use_torch_compile = use_torch_compile
        # The normal RGB and token-override calls specialize different branches of
        # ``sample_actions``.  Warm each at least once so the first scheduled request
        # does not compile inside its takeover deadline or pollute the latency window.
        self._compile_warmup_inferences = max(2, compile_warmup_inferences)
        self._global_shutdown_event = shutdown_event
        self._metrics_sink = metrics_sink

        self._queue = ScheduledActionQueue(reset_epoch=0, task_epoch=0)
        self._latency_tracker = LatencyTracker(maxlen=latency_window)
        self._request_lock = Lock()
        self._pending_request: _InferenceRequest | None = None
        self._request_in_flight = False
        self._request_id = 0
        self._reset_epoch = 0
        self._stats = PredictiveAsyncStats()
        self._last_delay_plan: DelayPlan | None = None

        self._request_ready = Event()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._ready_event = Event()
        self._worker_error = Event()
        self._failure_traceback: str | None = None
        self._worker: Thread | None = None
        self._warmup_completed = 0
        self._startup_phase = (
            "cold_temporary"
            if not use_torch_compile
            and (
                context_mode == "predicted" or (context_mode == "identity" and 1 <= max_prediction_delay <= 8)
            )
            else None
        )
        self._startup_interruption_reason: str | None = None
        self._startup_probe_record: dict[str, Any] | None = None
        if not use_torch_compile and self._startup_phase is None:
            self._ready_event.set()

    @property
    def queue(self) -> ScheduledActionQueue:
        """Expose the scheduled queue for diagnostics and deterministic tests."""
        return self._queue

    @property
    def stats(self) -> PredictiveAsyncStats:
        with self._request_lock:
            return self._stats

    @property
    def last_delay_plan(self) -> DelayPlan | None:
        """Last calibrated plan, including raw requirement and cap status."""
        with self._request_lock:
            return self._last_delay_plan

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set()

    @property
    def failed(self) -> bool:
        return self._worker_error.is_set()

    @property
    def failure_traceback(self) -> str | None:
        return self._failure_traceback

    @property
    def control_thread_owns_policy(self) -> bool:
        """The background worker exclusively owns policy inference."""
        return False

    def start(self) -> None:
        """Start the single policy worker."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._shutdown_event.clear()
        self._worker = Thread(target=self._worker_loop, daemon=True, name="PredictiveAsyncInference")
        self._worker.start()

    def stop(self) -> None:
        """Stop the worker after its current policy call returns."""
        self._shutdown_event.set()
        self._policy_active.clear()
        with self._request_lock:
            pending = self._pending_request
            self._pending_request = None
            self._request_ready.clear()
        if pending is not None and pending.plan is not None:
            self._queue.cancel_plan(
                request_id=pending.request_id,
                reset_epoch=pending.reset_epoch,
                task_epoch=pending.task_epoch,
            )
        # Wake a worker blocked in Event.wait so it can observe shutdown.
        self._request_ready.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=_JOIN_TIMEOUT_S)
            if worker.is_alive():
                logger.warning("Predictive async worker did not join within %.1fs", _JOIN_TIMEOUT_S)
                return
        self._worker = None
        self._request_ready.clear()
        self._close_metrics_sink()

    def _emit_metrics(self, event: dict[str, Any]) -> None:
        sink = self._metrics_sink
        if sink is None:
            return
        try:
            sink.emit(
                {
                    "schema_version": 1,
                    "backend": "predictive_async",
                    "context_mode": self._context_mode,
                    **event,
                }
            )
        except Exception:
            logger.exception("Failed to write predictive async inference metrics")

    def _close_metrics_sink(self) -> None:
        sink = self._metrics_sink
        if sink is None:
            return
        self._metrics_sink = None
        try:
            sink.close()
        except Exception:
            logger.exception("Failed to close predictive async inference metrics")

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
        self._policy_active.set()

    def _latch_startup_failure_locked(self, message: str) -> None:
        self._startup_phase = "failed"
        self._ready_event.clear()
        self._failure_traceback = f"RuntimeError: {message}"

    def _signal_startup_failure(self) -> None:
        self._worker_error.set()
        self._shutdown_event.set()
        self._request_ready.set()
        if self._global_shutdown_event is not None:
            self._global_shutdown_event.set()

    def _interrupt_startup_locked(self, reason: str) -> tuple[bool, _InferenceRequest | None]:
        if self._startup_phase in (None, "complete", "failed") or self._stats.requests_started == 0:
            return False, None
        message = f"Inference startup interrupted by {reason}"
        self._startup_interruption_reason = message
        self._latch_startup_failure_locked(message)
        # A running request owns its terminal; a pending request will never run
        # after removal and must receive its terminal from this caller.
        cancelled = None
        if self._pending_request is not None:
            cancelled = self._pending_request
            self._pending_request = None
            self._request_ready.clear()
        return True, cancelled

    def _finish_startup_interruption(self, interrupted: bool, request: _InferenceRequest | None) -> None:
        if not interrupted:
            return
        if request is not None and (self._metrics_sink is not None or request.kind == "startup_probe"):
            metrics = self._new_request_metrics(request, started_at=None)
            metrics.update(
                event="request_error",
                failed_phase="startup_interrupted",
                error_type="RuntimeError",
                error_message=self._startup_interruption_reason,
                startup_gate_outcome="error",
            )
            self._publish_request_metrics(metrics)
        # Emit a cancelled pending request before waking the worker to close its
        # sink. In-flight requests instead emit from their existing worker call.
        self._signal_startup_failure()

    def reset(self) -> None:
        """Clear episode state while retaining lifetime action and request indices."""
        with self._request_lock:
            interrupted, cancelled = self._interrupt_startup_locked("reset")
        self._finish_startup_interruption(interrupted, cancelled)
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        _, task_epoch = self.task_snapshot
        with self._request_lock:
            self._reset_epoch += 1
            self._queue.reset(self._reset_epoch, task_epoch=task_epoch)
            self._pending_request = None
            self._request_ready.clear()
        self._discard_task_change()

    def set_task(self, task: str) -> bool:
        with self._request_lock:
            changed = super().set_task(task)
            if not changed:
                return False
            _, task_epoch = self.task_snapshot
            self._queue.invalidate_task(task_epoch)
            interrupted, cancelled = self._interrupt_startup_locked("task change")
            pending = self._pending_request
            current_task_epoch = self._queue.task_epoch
            if pending is not None and pending.task_epoch < current_task_epoch:
                self._pending_request = None
                self._request_ready.clear()
        self._finish_startup_interruption(interrupted, cancelled)
        return True

    def notify_observation(self, obs: dict) -> None:
        """Atomically pair this control-tick observation with a takeover plan."""
        if (
            not self._policy_active.is_set()
            or self._shutdown_event.is_set()
            or self._startup_phase == "failed"
        ):
            return
        metrics_event = {} if self._metrics_sink is not None else None
        with self._request_lock:
            if not self._policy_active.is_set() or self._shutdown_event.is_set():
                return
            if self._pending_request is not None or self._request_in_flight:
                return

            task, task_epoch = self.task_snapshot
            self._queue.invalidate_task(task_epoch)
            if self._queue.task_epoch != task_epoch:
                # A newer set_task won the race after this snapshot.  The next
                # control tick will pair its observation with the current epoch.
                return
            request = self._make_request_locked(
                copy(obs), task=task, task_epoch=task_epoch, metrics_event=metrics_event
            )
            if request is not None:
                self._pending_request = request
                self._stats = replace(
                    self._stats,
                    requests_started=self._stats.requests_started + 1,
                    bootstrap_requests=(self._stats.bootstrap_requests + (request.kind == "bootstrap")),
                    planned_requests=self._stats.planned_requests + (request.kind == "planned"),
                )
                self._request_ready.set()
        if metrics_event:
            self._emit_metrics(metrics_event)

    def _make_request_locked(
        self,
        observation: dict[str, Any],
        *,
        task: str,
        task_epoch: int,
        metrics_event: dict[str, Any] | None = None,
    ) -> _InferenceRequest | None:
        request_id = self._request_id
        now = time.perf_counter()
        if self._startup_phase not in (None, "complete"):
            if self._startup_phase == "failed":
                return None
            plan = None
            if self._startup_phase == "probe":
                creation = self._queue.create_takeover_plan(
                    request_id=request_id,
                    planned_delay_steps=8,
                    max_prediction_delay=8,
                    committed_guard_steps=self._committed_guard_steps,
                    reset_epoch=self._reset_epoch,
                    task_epoch=task_epoch,
                    task=task,
                )
                plan = creation.plan
            self._request_id += 1
            return _InferenceRequest(
                request_id=request_id,
                kind="startup_probe" if self._startup_phase == "probe" else "bootstrap",
                observation=observation,
                requested_at=now,
                reset_epoch=self._reset_epoch,
                task=task,
                task_epoch=task_epoch,
                plan=plan,
                startup_phase=self._startup_phase,
            )
        if self._use_torch_compile and not self._ready_event.is_set():
            self._request_id += 1
            return _InferenceRequest(
                request_id=request_id,
                kind="warmup",
                observation=observation,
                requested_at=now,
                reset_epoch=self._reset_epoch,
                task=task,
                task_epoch=task_epoch,
            )

        available = self._queue.available_steps()
        if len(self._latency_tracker) == 0:
            if available != 0:
                return None
            self._request_id += 1
            return _InferenceRequest(
                request_id=request_id,
                kind="bootstrap",
                observation=observation,
                requested_at=now,
                reset_epoch=self._reset_epoch,
                task=task,
                task_epoch=task_epoch,
            )

        if available > self._queue_threshold:
            return None

        delay_plan = compute_delay_plan(
            self._latency_tracker,
            fps=self._fps,
            latency_quantile=self._latency_quantile,
            delay_safety_margin_steps=self._delay_safety_margin_steps,
            min_prediction_delay=self._min_prediction_delay,
            max_prediction_delay=self._max_prediction_delay,
            available_actions=available,
            committed_guard_steps=self._committed_guard_steps,
        )
        if delay_plan is None:
            return None
        self._last_delay_plan = delay_plan
        if metrics_event is not None:
            metrics_event.update(
                event="planner_decision",
                timestamp_s=time.perf_counter(),
                reset_epoch=self._reset_epoch,
                task_epoch=task_epoch,
                task=task,
                request_id=None,
                next_action_index_snapshot=self._queue.next_action_index,
                available_steps=available,
                required_steps=delay_plan.planned_delay_steps + self._committed_guard_steps,
                committed_guard_steps=self._committed_guard_steps,
                estimated_latency_s=delay_plan.estimated_latency_s,
                raw_required_delay_steps=delay_plan.raw_required_delay_steps,
                planned_delay_steps=delay_plan.planned_delay_steps,
                available_after_guard_steps=delay_plan.available_after_guard_steps,
                prediction_cap_exceeded=delay_plan.prediction_cap_exceeded,
                plan_outcome=None,
                decision=None,
            )
        if delay_plan.prediction_cap_exceeded:
            self._stats = replace(
                self._stats,
                prediction_cap_exceeded=self._stats.prediction_cap_exceeded + 1,
            )
            logger.debug(
                "predictive_async request=%d cap_exceeded=true raw_required=%d planned=%d next=%d",
                request_id,
                delay_plan.raw_required_delay_steps,
                delay_plan.planned_delay_steps,
                self._queue.next_action_index,
            )
            if available != 0 or self._fallback_mode == "discard":
                if metrics_event is not None:
                    metrics_event["decision"] = (
                        "cap_discard" if self._fallback_mode == "discard" else "cap_wait"
                    )
                return None
            self._request_id += 1
            if metrics_event is not None:
                metrics_event.update(decision="bootstrap", request_id=request_id)
            return _InferenceRequest(
                request_id=request_id,
                kind="bootstrap",
                observation=observation,
                requested_at=now,
                reset_epoch=self._reset_epoch,
                task=task,
                task_epoch=task_epoch,
                delay_plan=delay_plan,
            )

        if available == 0:
            if self._fallback_mode == "discard":
                if metrics_event is not None:
                    metrics_event["decision"] = "empty_discard"
                return None
            self._request_id += 1
            if metrics_event is not None:
                metrics_event.update(decision="bootstrap", request_id=request_id)
            return _InferenceRequest(
                request_id=request_id,
                kind="bootstrap",
                observation=observation,
                requested_at=now,
                reset_epoch=self._reset_epoch,
                task=task,
                task_epoch=task_epoch,
                delay_plan=delay_plan,
            )
        creation = self._queue.create_takeover_plan(
            request_id=request_id,
            planned_delay_steps=delay_plan.planned_delay_steps,
            max_prediction_delay=self._max_prediction_delay,
            committed_guard_steps=self._committed_guard_steps,
            reset_epoch=self._reset_epoch,
            task_epoch=task_epoch,
            task=task,
        )
        if metrics_event is not None:
            metrics_event.update(
                plan_outcome=creation.outcome.value,
                available_steps=creation.available_steps,
                required_steps=creation.required_steps,
                decision="plan_failed" if creation.plan is None else "planned",
            )
        if creation.plan is None:
            return None
        self._request_id += 1
        if metrics_event is not None:
            metrics_event["request_id"] = request_id
        return _InferenceRequest(
            request_id=request_id,
            kind="planned",
            observation=observation,
            requested_at=now,
            reset_epoch=self._reset_epoch,
            task=task,
            task_epoch=task_epoch,
            plan=creation.plan,
            delay_plan=delay_plan,
        )

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return one post-policy action; ``obs_frame`` is ignored."""
        if self._startup_phase not in (None, "complete"):
            raise RuntimeError(f"Inference startup is not ready: {self._startup_phase}")
        result = self._queue.get_with_task()
        if result.post_policy_action is None:
            with self._request_lock:
                self._stats = replace(self._stats, underflows=self._stats.underflows + 1)
        elif result.task is None:
            raise RuntimeError("ScheduledActionQueue returned an action without task provenance")
        else:
            self._set_dispatched_task(result.task)
        if self._metrics_sink is not None:
            self._emit_metrics(
                {
                    "event": "queue_get",
                    "timestamp_s": time.perf_counter(),
                    "outcome": result.outcome.value,
                    "action_index": result.action_index,
                    "task": result.task,
                    "underflow_total": self._stats.underflows,
                }
            )
        return result.post_policy_action

    def _worker_loop(self) -> None:
        consecutive_errors = 0
        try:
            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_IDLE_WAIT_S)
                    continue
                if not self._request_ready.wait(timeout=_IDLE_WAIT_S):
                    continue
                with self._request_lock:
                    if self._shutdown_event.is_set() or not self._policy_active.is_set():
                        continue
                    request = self._pending_request
                    self._pending_request = None
                    self._request_ready.clear()
                    if request is None:
                        continue
                    self._request_in_flight = True

                try:
                    self._run_request(request)
                    consecutive_errors = 0
                except Exception:
                    if request.plan is not None:
                        self._queue.cancel_plan(
                            request_id=request.request_id,
                            reset_epoch=request.reset_epoch,
                            task_epoch=request.task_epoch,
                        )
                    if request.startup_phase is not None:
                        raise
                    consecutive_errors += 1
                    logger.exception(
                        "Predictive async inference error (%d/%d)",
                        consecutive_errors,
                        _MAX_CONSECUTIVE_ERRORS,
                    )
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise
                finally:
                    with self._request_lock:
                        self._request_in_flight = False
        except Exception as error:
            self._failure_traceback = traceback.format_exc()
            logger.error("Fatal predictive async worker error: %s", error)
            self._worker_error.set()
            if self._startup_phase == "failed":
                self._ready_event.clear()
            else:
                self._ready_event.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
        finally:
            self._close_metrics_sink()

    @contextmanager
    def _record_metrics_phase(
        self,
        phase: str,
        metrics: dict[str, Any] | None,
        cuda_events: dict[str, tuple[Any, Any]] | None,
    ) -> Iterator[None]:
        if metrics is None:
            yield
            return
        metrics["failed_phase"] = phase
        started_at = None
        start_event = end_event = None
        try:
            if cuda_events is not None:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(self._device))
            started_at = time.perf_counter()
        except Exception:
            logger.exception("Failed to start predictive async phase metrics: %s", phase)
        # A public-call error propagates without completing the failed phase or
        # reading CUDA events. Only observational setup/record failures are caught.
        yield
        try:
            if started_at is not None:
                metrics["phase_host_wall_s"][phase] = time.perf_counter() - started_at
            if start_event is not None and end_event is not None:
                end_event.record(torch.cuda.current_stream(self._device))
                cuda_events[phase] = (start_event, end_event)
        except Exception:
            logger.exception("Failed to finish predictive async phase metrics: %s", phase)

    def _new_request_metrics(self, request: _InferenceRequest, *, started_at: float | None) -> dict[str, Any]:
        delay_plan = request.delay_plan
        plan = request.plan
        return {
            "event": "chunk_request",
            "request_id": request.request_id,
            "request_kind": request.kind,
            "reset_epoch": request.reset_epoch,
            "task_epoch": request.task_epoch,
            "task": request.task,
            "requested_at_s": request.requested_at,
            "started_at_s": started_at,
            "cuda_completed_at_s": None,
            "dispatch_wait_s": None if started_at is None else started_at - request.requested_at,
            "device_completion_wait_s": None,
            "total_chunk_s": None,
            "d_actual_wall": None,
            "phase_host_wall_s": dict.fromkeys(_METRIC_PHASES),
            "phase_cuda_stream_elapsed_ms": dict.fromkeys(_METRIC_PHASES)
            if self._device.type == "cuda"
            else None,
            "predictor_calls": 0,
            "policy_includes_vision": None,
            "estimated_latency_s": None if delay_plan is None else delay_plan.estimated_latency_s,
            "raw_required_delay_steps": None if delay_plan is None else delay_plan.raw_required_delay_steps,
            "planned_delay_steps": None if plan is None else plan.planned_delay_steps,
            "available_after_guard_steps": None
            if delay_plan is None
            else delay_plan.available_after_guard_steps,
            "prediction_cap_exceeded": None if delay_plan is None else delay_plan.prediction_cap_exceeded,
            "plan_next_action_index": None if plan is None else plan.next_action_index,
            "takeover_index": None if plan is None else plan.takeover_index,
            "result_next_action_index": None,
            "outcome": None,
            "late_steps": None,
            "consumed_steps_at_stage": None,
            "underflow_total": self._stats.underflows,
            "failed_phase": None,
            "startup_phase": request.startup_phase,
            "latency_tracker_admitted": False,
            "startup_gate_raw_required_delay_steps": None,
            "startup_gate_outcome": None,
            "startup_queue_reset_epoch_after": None,
        }

    def _publish_request_metrics(self, metrics: dict[str, Any]) -> None:
        if metrics["request_kind"] == "startup_probe":
            metrics = {
                "schema_version": 1,
                "backend": "predictive_async",
                "context_mode": self._context_mode,
                **metrics,
            }
            self._startup_probe_record = metrics
        self._emit_metrics(metrics)

    def _complete_request_metrics(
        self,
        metrics: dict[str, Any] | None,
        cuda_events: dict[str, tuple[Any, Any]] | None,
        *,
        latency_s: float,
        completion_started_at: float | None,
        outcome: str,
        result_next_action_index: int | None = None,
        late_steps: int | None = None,
        consumed_steps_at_stage: int | None = None,
    ) -> None:
        if metrics is None:
            return
        # Ordinary requests call this after publication. The startup probe must
        # inspect its complete measurements before admitting the only seed.
        if cuda_events is not None:
            for phase, (start_event, end_event) in cuda_events.items():
                try:
                    metrics["phase_cuda_stream_elapsed_ms"][phase] = start_event.elapsed_time(end_event)
                except Exception:
                    logger.exception("Failed to resolve predictive async phase metrics: %s", phase)
        completed_at = metrics["requested_at_s"] + latency_s
        metrics.update(
            cuda_completed_at_s=completed_at,
            device_completion_wait_s=completed_at - completion_started_at,
            total_chunk_s=latency_s,
            d_actual_wall=latency_to_steps(latency_s, self._fps) if math.isfinite(latency_s) else None,
            outcome=outcome,
            result_next_action_index=result_next_action_index,
            late_steps=late_steps,
            consumed_steps_at_stage=consumed_steps_at_stage,
            underflow_total=self._stats.underflows,
        )

    def _finish_request_metrics(
        self,
        metrics: dict[str, Any] | None,
        cuda_events: dict[str, tuple[Any, Any]] | None,
        *,
        latency_s: float,
        completion_started_at: float | None,
        outcome: str,
        result_next_action_index: int | None = None,
        late_steps: int | None = None,
        consumed_steps_at_stage: int | None = None,
    ) -> None:
        if metrics is None:
            return
        self._complete_request_metrics(
            metrics,
            cuda_events,
            latency_s=latency_s,
            completion_started_at=completion_started_at,
            outcome=outcome,
            result_next_action_index=result_next_action_index,
            late_steps=late_steps,
            consumed_steps_at_stage=consumed_steps_at_stage,
        )
        metrics.pop("failed_phase")
        self._publish_request_metrics(metrics)

    def _check_startup_request_locked(self, request: _InferenceRequest) -> None:
        if self._startup_interruption_reason is not None:
            raise RuntimeError(self._startup_interruption_reason)
        task, task_epoch = self.task_snapshot
        if (
            self._startup_phase != request.startup_phase
            or self._reset_epoch != request.reset_epoch
            or task_epoch != request.task_epoch
            or task != request.task
            or self._queue.reset_epoch != request.reset_epoch
            or self._queue.task_epoch != request.task_epoch
        ):
            raise RuntimeError("Inference startup request identity is no longer current")

    def _validate_startup_probe(
        self, metrics: dict[str, Any], *, latency_s: float, actions_finite: bool
    ) -> None:
        metrics["failed_phase"] = "startup_gate"
        if not actions_finite or not math.isfinite(latency_s) or latency_s < 0:
            metrics["startup_gate_outcome"] = "nonfinite"
            raise RuntimeError("Startup probe actions and latency must be finite and latency non-negative")
        families = ["phase_host_wall_s"]
        if self._device.type == "cuda":
            families.append("phase_cuda_stream_elapsed_ms")
        unused_phases = (
            ("predictor_input_preparation", "predictor_forward", "residual_application")
            if self._context_mode == "identity"
            else ()
        )
        for family in families:
            for phase in _METRIC_PHASES:
                value = metrics[family][phase]
                if phase in unused_phases:
                    if value is not None:
                        metrics["startup_gate_outcome"] = "error"
                        raise RuntimeError(
                            f"Identity startup probe requires unused {family}.{phase} to be null"
                        )
                    continue
                if value is None:
                    metrics["startup_gate_outcome"] = "telemetry_missing"
                    raise RuntimeError(f"Startup probe requires {family}.{phase}")
                if not math.isfinite(value) or value < 0:
                    metrics["startup_gate_outcome"] = "nonfinite"
                    raise RuntimeError(f"Startup probe requires finite non-negative {family}.{phase}")
        raw_required = latency_to_steps(latency_s, self._fps) + self._delay_safety_margin_steps
        metrics["startup_gate_raw_required_delay_steps"] = raw_required
        if raw_required > self._max_prediction_delay:
            metrics["startup_gate_outcome"] = "cap_exceeded"
            with self._request_lock:
                self._stats = replace(
                    self._stats, prediction_cap_exceeded=self._stats.prediction_cap_exceeded + 1
                )
            raise RuntimeError(
                f"Startup probe requires {raw_required} delay steps, exceeding runtime cap {self._max_prediction_delay}"
            )

    def _run_request(self, request: _InferenceRequest) -> None:
        metrics = None
        cuda_events = None
        completion_started_at = None
        latency_s = None
        if self._metrics_sink is not None or request.kind == "startup_probe":
            started_at = time.perf_counter()
            cuda_events = {} if self._device.type == "cuda" else None
            metrics = self._new_request_metrics(request, started_at=started_at)
        try:
            if request.kind == "startup_probe" and request.plan is None:
                metrics["failed_phase"] = "startup_gate"
                raise RuntimeError("Startup probe could not obtain a real eight-row committed prefix")
            with self._record_metrics_phase("observation_preparation", metrics, cuda_events):
                batch = build_dataset_frame(self._hw_features, request.observation, prefix="observation")
                batch = prepare_observation_for_inference(
                    batch, self._device, request.task, self._robot.robot_type
                )
                batch["task"] = [request.task]
            with self._record_metrics_phase("preprocessor", metrics, cuda_events):
                batch = self._preprocessor(batch)

            with torch.inference_mode():
                predict_kwargs: dict[str, Any] = {}
                warmup_token_override = request.kind == "warmup" and self._warmup_completed > 0
                if metrics is not None:
                    metrics["policy_includes_vision"] = not (
                        request.kind in ("planned", "startup_probe") or warmup_token_override
                    )
                if request.kind in ("planned", "startup_probe") or warmup_token_override:
                    token_policy: Any = self._policy
                    predicted_request = (
                        request.kind in ("planned", "startup_probe") and self._context_mode == "predicted"
                    )
                    with self._record_metrics_phase("vision_encode", metrics, cuda_events):
                        if predicted_request and {
                            key for key in batch if key.startswith("observation.images.")
                        } != set(POLICY_CAMERA_KEYS):
                            raise ValueError(
                                "A predicted planned observation requires exactly camera1 and camera2"
                            )
                        images, image_masks = token_policy.prepare_images(batch)
                        if predicted_request and (len(images) != 2 or len(image_masks) != 2):
                            raise ValueError("prepare_images must preserve both frozen camera streams")
                        image_tokens, image_token_masks = token_policy.model.encode_image_tokens(
                            images, image_masks
                        )
                    future_tokens = image_tokens
                    if predicted_request:
                        plan = request.plan
                        assert plan is not None
                        if plan.planned_delay_steps > 0:
                            # Queue snapshots keep the configured runtime cap. Only
                            # the predictor boundary extends them to eight rows.
                            with self._record_metrics_phase(
                                "predictor_input_preparation", metrics, cuda_events
                            ):
                                committed_actions = torch.zeros(
                                    (1, 8, 6),
                                    dtype=plan.committed_policy_actions.dtype,
                                    device=self._device,
                                )
                                committed_mask = torch.zeros((1, 8), dtype=torch.bool, device=self._device)
                                prefix_rows = plan.committed_policy_actions.shape[0]
                                committed_actions[0, :prefix_rows].copy_(plan.committed_policy_actions)
                                committed_mask[0, :prefix_rows].copy_(plan.committed_mask)
                                state = token_policy.prepare_state(batch)
                                delay_steps = torch.tensor(
                                    [plan.planned_delay_steps], dtype=torch.long, device=self._device
                                )
                            with self._record_metrics_phase("predictor_forward", metrics, cuda_events):
                                if metrics is not None:
                                    metrics["predictor_calls"] += 1
                                prediction = self._future_latent_predictor(
                                    image_tokens,
                                    image_token_masks,
                                    committed_actions,
                                    committed_mask,
                                    state,
                                    delay_steps,
                                )
                            with self._record_metrics_phase("residual_application", metrics, cuda_events):
                                future_tokens = tuple(
                                    (tokens.float() + delta.float()).to(tokens.dtype)
                                    for tokens, delta in zip(
                                        image_tokens, prediction.delta_tokens, strict=True
                                    )
                                )
                    predict_kwargs.update(
                        future_image_tokens=future_tokens,
                        future_image_token_masks=image_token_masks,
                    )
                with self._record_metrics_phase("policy_total", metrics, cuda_events):
                    actions = self._policy.predict_action_chunk(batch, **predict_kwargs)
            with self._record_metrics_phase("postprocessor", metrics, cuda_events):
                policy_actions = actions.squeeze(0).clone()
                post_policy_actions = self._postprocessor(actions).squeeze(0)
                if request.kind == "startup_probe":
                    probe_actions_finite = (
                        torch.isfinite(policy_actions).all() & torch.isfinite(post_policy_actions).all()
                    )
            # Keep the original CUDA-complete sample before publication. Metrics
            # add event records, never an extra phase synchronization.
            if metrics is not None:
                metrics["failed_phase"] = "device_completion"
                completion_started_at = time.perf_counter()
            _synchronize_policy_device(self._device)
            latency_s = time.perf_counter() - request.requested_at

            if request.kind == "startup_probe":
                self._complete_request_metrics(
                    metrics,
                    cuda_events,
                    latency_s=latency_s,
                    completion_started_at=completion_started_at,
                    outcome="probe_discarded",
                )
                with self._request_lock:
                    self._check_startup_request_locked(request)
                self._validate_startup_probe(
                    metrics, latency_s=latency_s, actions_finite=bool(probe_actions_finite.item())
                )
                with self._request_lock:
                    self._check_startup_request_locked(request)
                    self._latency_tracker.add(latency_s)
                    metrics["latency_tracker_admitted"] = True
                    self._reset_epoch += 1
                    self._queue.reset(self._reset_epoch, task_epoch=request.task_epoch)
                    metrics["startup_queue_reset_epoch_after"] = self._reset_epoch
                    metrics["startup_gate_outcome"] = "passed"
                    self._startup_phase = "fresh_warmed"
                metrics.pop("failed_phase")
                self._publish_request_metrics(metrics)
                return

            if metrics is not None:
                metrics["failed_phase"] = "queue_publication"
            if request.kind == "warmup":
                with self._request_lock:
                    self._warmup_completed += 1
                    if self._warmup_completed >= self._compile_warmup_inferences:
                        self._ready_event.set()
                self._finish_request_metrics(
                    metrics,
                    cuda_events,
                    latency_s=latency_s,
                    completion_started_at=completion_started_at,
                    outcome="warmup_completed",
                )
                return

            if request.startup_phase is not None:
                with self._request_lock:
                    self._check_startup_request_locked(request)
                    installed = self._queue.install_active_chunk(
                        policy_actions,
                        post_policy_actions,
                        task=request.task,
                        reset_epoch=request.reset_epoch,
                        task_epoch=request.task_epoch,
                    )
                    if metrics is not None:
                        metrics["outcome"] = installed.outcome.value
                        metrics["result_next_action_index"] = installed.next_action_index
                    if installed.outcome is not InstallOutcome.INSTALLED:
                        self._stats = replace(self._stats, stale_results=self._stats.stale_results + 1)
                        raise RuntimeError(
                            f"Startup bootstrap installation failed: {installed.outcome.value}"
                        )
                    self._startup_phase = "probe" if request.startup_phase == "cold_temporary" else "complete"
                    if self._startup_phase == "complete":
                        self._ready_event.set()
                self._finish_request_metrics(
                    metrics,
                    cuda_events,
                    latency_s=latency_s,
                    completion_started_at=completion_started_at,
                    outcome=installed.outcome.value,
                    result_next_action_index=installed.next_action_index,
                )
                return

            admit_latency = self._startup_phase is None or (
                request.kind == "planned"
                and (self._context_mode == "identity" or request.plan.planned_delay_steps > 0)
            )
            if admit_latency:
                self._latency_tracker.add(latency_s)
                if metrics is not None:
                    metrics["latency_tracker_admitted"] = True
            if request.kind == "bootstrap":
                installed = self._queue.install_active_chunk(
                    policy_actions,
                    post_policy_actions,
                    task=request.task,
                    reset_epoch=request.reset_epoch,
                    task_epoch=request.task_epoch,
                )
                if installed.outcome is not InstallOutcome.INSTALLED:
                    with self._request_lock:
                        self._stats = replace(self._stats, stale_results=self._stats.stale_results + 1)
                self._finish_request_metrics(
                    metrics,
                    cuda_events,
                    latency_s=latency_s,
                    completion_started_at=completion_started_at,
                    outcome=installed.outcome.value,
                    result_next_action_index=installed.next_action_index,
                )
                return

            staged = self._queue.stage_chunk(
                policy_actions,
                post_policy_actions,
                request_id=request.request_id,
                reset_epoch=request.reset_epoch,
                task_epoch=request.task_epoch,
                task=request.task,
            )
            if staged.outcome is StageOutcome.DEADLINE_MISS:
                with self._request_lock:
                    self._stats = replace(self._stats, deadline_misses=self._stats.deadline_misses + 1)
            elif staged.outcome is StageOutcome.STALE:
                with self._request_lock:
                    self._stats = replace(self._stats, stale_results=self._stats.stale_results + 1)

            measured_steps = latency_to_steps(latency_s, self._fps)
            logger.debug(
                "predictive_async request=%d outcome=%s raw_required=%s planned=%s "
                "cap_exceeded=%s measured=%d late=%d next=%d",
                request.request_id,
                staged.outcome.value,
                None if request.delay_plan is None else request.delay_plan.raw_required_delay_steps,
                None if request.delay_plan is None else request.delay_plan.planned_delay_steps,
                None if request.delay_plan is None else request.delay_plan.prediction_cap_exceeded,
                measured_steps,
                staged.late_steps,
                self._queue.next_action_index,
            )
            self._finish_request_metrics(
                metrics,
                cuda_events,
                latency_s=latency_s,
                completion_started_at=completion_started_at,
                outcome=staged.outcome.value,
                result_next_action_index=staged.next_action_index,
                late_steps=staged.late_steps,
                consumed_steps_at_stage=(
                    staged.next_action_index - request.plan.next_action_index
                    if request.plan is not None and staged.outcome is not StageOutcome.STALE
                    else None
                ),
            )
        except Exception as caught_error:
            error = caught_error
            if request.startup_phase is not None:
                if metrics is not None and latency_s is not None and metrics["total_chunk_s"] is None:
                    self._complete_request_metrics(
                        metrics,
                        cuda_events,
                        latency_s=latency_s,
                        completion_started_at=completion_started_at,
                        outcome=metrics["outcome"],
                        result_next_action_index=metrics["result_next_action_index"],
                    )
                with self._request_lock:
                    if self._startup_interruption_reason is not None:
                        error = RuntimeError(self._startup_interruption_reason)
                        if metrics is not None:
                            metrics["failed_phase"] = "startup_interrupted"
                            metrics["startup_gate_outcome"] = "error"
                    self._latch_startup_failure_locked(str(error))
                if metrics is not None and metrics["startup_gate_outcome"] is None:
                    metrics["startup_gate_outcome"] = "error"
            if metrics is not None:
                metrics.update(
                    event="request_error",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    underflow_total=self._stats.underflows,
                )
                self._publish_request_metrics(metrics)
            if request.startup_phase is not None:
                self._signal_startup_failure()
                if error is not caught_error:
                    raise error from caught_error
            raise
