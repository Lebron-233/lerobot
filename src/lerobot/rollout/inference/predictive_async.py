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

if TYPE_CHECKING:
    from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.005
_JOIN_TIMEOUT_S = 3.0
_MAX_CONSECUTIVE_ERRORS = 5


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
    kind: Literal["warmup", "bootstrap", "planned"]
    observation: dict[str, Any]
    requested_at: float
    reset_epoch: int
    task: str
    task_epoch: int
    plan: TakeoverPlan | None = None
    delay_plan: DelayPlan | None = None


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
        if not use_torch_compile:
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

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
        self._policy_active.set()

    def reset(self) -> None:
        """Clear episode state while retaining lifetime action and request indices."""
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
        changed = super().set_task(task)
        if not changed:
            return False
        _, task_epoch = self.task_snapshot
        self._queue.invalidate_task(task_epoch)
        with self._request_lock:
            pending = self._pending_request
            current_task_epoch = self._queue.task_epoch
            if pending is not None and pending.task_epoch < current_task_epoch:
                self._pending_request = None
                self._request_ready.clear()
        return True

    def notify_observation(self, obs: dict) -> None:
        """Atomically pair this control-tick observation with a takeover plan."""
        if not self._policy_active.is_set() or self._shutdown_event.is_set():
            return
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
            request = self._make_request_locked(copy(obs), task=task, task_epoch=task_epoch)
            if request is None:
                return
            self._pending_request = request
            self._stats = replace(
                self._stats,
                requests_started=self._stats.requests_started + 1,
                bootstrap_requests=(self._stats.bootstrap_requests + (request.kind == "bootstrap")),
                planned_requests=self._stats.planned_requests + (request.kind == "planned"),
            )
            self._request_ready.set()

    def _make_request_locked(
        self, observation: dict[str, Any], *, task: str, task_epoch: int
    ) -> _InferenceRequest | None:
        request_id = self._request_id
        now = time.perf_counter()
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
                delay_plan=delay_plan,
            )

        if available == 0:
            if self._fallback_mode == "discard":
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
        if creation.plan is None:
            return None
        self._request_id += 1
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
        result = self._queue.get_with_task()
        if result.post_policy_action is None:
            with self._request_lock:
                self._stats = replace(self._stats, underflows=self._stats.underflows + 1)
            return None
        if result.task is None:
            raise RuntimeError("ScheduledActionQueue returned an action without task provenance")
        self._set_dispatched_task(result.task)
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
            self._ready_event.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()

    def _run_request(self, request: _InferenceRequest) -> None:
        batch = build_dataset_frame(self._hw_features, request.observation, prefix="observation")
        batch = prepare_observation_for_inference(batch, self._device, request.task, self._robot.robot_type)
        batch["task"] = [request.task]
        batch = self._preprocessor(batch)

        with torch.inference_mode():
            predict_kwargs: dict[str, Any] = {}
            warmup_token_override = request.kind == "warmup" and self._warmup_completed > 0
            if request.kind == "planned" or warmup_token_override:
                token_policy: Any = self._policy
                predicted_request = request.kind == "planned" and self._context_mode == "predicted"
                if predicted_request and {
                    key for key in batch if key.startswith("observation.images.")
                } != set(POLICY_CAMERA_KEYS):
                    raise ValueError("A predicted planned observation requires exactly camera1 and camera2")
                images, image_masks = token_policy.prepare_images(batch)
                if predicted_request and (len(images) != 2 or len(image_masks) != 2):
                    raise ValueError("prepare_images must preserve both frozen camera streams")
                image_tokens, image_token_masks = token_policy.model.encode_image_tokens(images, image_masks)
                future_tokens = image_tokens
                if predicted_request:
                    plan = request.plan
                    assert plan is not None
                    if plan.planned_delay_steps > 0:
                        # Queue snapshots keep the configured runtime cap. Only
                        # the frozen predictor boundary extends them to eight rows.
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
                        prediction = self._future_latent_predictor(
                            image_tokens,
                            image_token_masks,
                            committed_actions,
                            committed_mask,
                            state,
                            delay_steps,
                        )
                        future_tokens = tuple(
                            (tokens.float() + delta.float()).to(tokens.dtype)
                            for tokens, delta in zip(image_tokens, prediction.delta_tokens, strict=True)
                        )
                predict_kwargs.update(
                    future_image_tokens=future_tokens,
                    future_image_token_masks=image_token_masks,
                )
            actions = self._policy.predict_action_chunk(batch, **predict_kwargs)
        policy_actions = actions.squeeze(0).clone()
        post_policy_actions = self._postprocessor(actions).squeeze(0)
        # CUDA launches asynchronously.  Publish the queue entry, sample latency,
        # and mark compile warmup ready only after the policy and postprocessor
        # really completed; asynchronous failures then stay on the worker path.
        _synchronize_policy_device(self._device)
        latency_s = time.perf_counter() - request.requested_at

        if request.kind == "warmup":
            with self._request_lock:
                self._warmup_completed += 1
                if self._warmup_completed >= self._compile_warmup_inferences:
                    self._ready_event.set()
            return

        self._latency_tracker.add(latency_s)
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
