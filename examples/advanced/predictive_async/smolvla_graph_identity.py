"""Experimental Graph/identity adapter; the factory and default engines stay unchanged."""

import threading
from contextlib import contextmanager

import torch
from smolvla_graph_runtime import SmolVLAGraphRuntime

from lerobot.rollout.inference.predictive_async import (
    PredictiveAsyncInferenceEngine,
    _synchronize_policy_device,
)


class SmolVLAGraphIdentityEngine(PredictiveAsyncInferenceEngine):
    """Keep model state on the existing worker and publish only independent CPU chunks."""

    def __init__(self, **kwargs):
        if kwargs.get("context_mode", "identity") != "identity" or kwargs.get("use_torch_compile", False):
            raise ValueError("This experiment supports identity without torch.compile only")
        super().__init__(**kwargs)
        self.runtime = None
        self.owner_thread = None
        self._owner_reset_pending = False
        self.lifecycle = []
        self.graph_released = False
        self.original_sampler_restored = False
        self.worker_joined = False

    def _make_graph_runtime(self):
        return SmolVLAGraphRuntime(self._policy.model)

    def _reset_owned(self):
        self.runtime._check_owner()
        self.runtime.release_graph()
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self.lifecycle.append({"event": "owner_reset", "thread": threading.get_ident()})

    @contextmanager
    def _worker_resources(self):
        self.owner_thread = threading.get_ident()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type, enabled=getattr(self._policy.config, "use_amp", False)
            ),
        ):
            self.runtime = self._make_graph_runtime()
            try:
                with self.runtime:
                    self._reset_owned()
                    self.lifecycle.append({"event": "owner_enter", "thread": self.owner_thread})
                    try:
                        yield
                    finally:
                        # A reset during capture/startup may terminate startup. GPU work has
                        # returned now, so its deferred processor reset still belongs here.
                        if self._owner_reset_pending:
                            self._reset_owned()
                            self._owner_reset_pending = False
            finally:
                self.original_sampler_restored = self._policy.model.sample_actions == self.runtime.original
                _synchronize_policy_device(self._device)
                self.graph_released = self.runtime.graph is None
                self.lifecycle.append(
                    {
                        "event": "owner_exit",
                        "thread": threading.get_ident(),
                        "sampler_restored": self.original_sampler_restored,
                        "graph_released": self.graph_released,
                    }
                )

    def _request_error_is_fatal(self, request):
        return True

    def _run_request(self, request):
        with self._request_lock:
            reset_pending = self._owner_reset_pending
            self._owner_reset_pending = False
        if reset_pending:
            self._reset_owned()
        # The startup probe also advances reset_epoch. Same text in a different
        # task/reset generation must not revive an obsolete capture.
        self.runtime.begin_episode("graph", (request.reset_epoch, request.task_epoch, request.task))
        return super()._run_request(request)

    def _prepare_queue_actions(self, actions, metrics):
        policy_actions, post_actions = super()._prepare_queue_actions(actions, metrics)
        policy_cpu = policy_actions.detach().to(device="cpu", copy=True)
        post_cpu = post_actions.detach().to(device="cpu", copy=True)
        _synchronize_policy_device(self._device)
        if not torch.isfinite(policy_cpu).all() or not torch.isfinite(post_cpu).all():
            raise ValueError("Graph identity queue actions must be finite")
        return policy_cpu, post_cpu

    def reset(self):
        # Invalidate CPU state immediately. Never wait for the owner with a queue
        # or request lock held; already launched GPU work is allowed to finish.
        with self._request_lock:
            interrupted, cancelled = self._interrupt_startup_locked("reset")
            _, task_epoch = self.task_snapshot
            self._reset_epoch += 1
            self._queue.reset(self._reset_epoch, task_epoch=task_epoch)
            self._owner_reset_pending = True
            self._pending_request = None
            self._request_ready.clear()
        self._discard_task_change()
        self._finish_startup_interruption(interrupted, cancelled)

    def stop(self):
        with self._request_lock:
            # A completed in-flight result can no longer install into the stopped
            # adapter, including bootstrap results without a takeover plan.
            self._reset_epoch += 1
            _, task_epoch = self.task_snapshot
            self._queue.reset(self._reset_epoch, task_epoch=task_epoch)
            self._shutdown_event.set()
        super().stop()
        self.worker_joined = self._worker is None and self.owner_thread is not None
