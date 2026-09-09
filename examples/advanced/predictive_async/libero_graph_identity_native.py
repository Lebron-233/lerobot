"""One fixed twenty-episode Graph serialized/identity-async native comparison."""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import libero_reference_qualification as reference
import numpy as np
import torch
from libero_graph_native_equivalence import FLAGS, POLICY, VLM
from libero_reference_smoke import (
    CAMERA_KEYS,
    action_outside_bounds,
    load_runtime,
    terminal_reason,
    write_json,
)
from smolvla_graph_identity import SmolVLAGraphIdentityEngine
from smolvla_graph_runtime import SmolVLAGraphRuntime
from validate_smolvla_graph_worker import UNTRACKED_DOCS, AuditedProcessor, CallAudit, cpu, equal

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.feature_utils import build_dataset_frame

REPO = reference.REPO
PYTHON = "/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python"
PREPARATION = REPO / "outputs/smolvla_graph_identity_native_preparation_a6966f38"
MANIFEST = REPO / "docs/experiments/SMOLVLA_GRAPH_IDENTITY_NATIVE_MANIFEST.json"
CONDITIONS = ("graph_serialized", "graph_identity_async")
CONTROL = {
    "fps": 20,
    "queue_threshold": 30,
    "latency_quantile": 0.9,
    "latency_window": 50,
    "delay_safety_margin_steps": 1,
    "min_prediction_delay": 0,
    "max_prediction_delay": 8,
    "committed_guard_steps": 2,
    "max_late_steps": 2,
    "context_mode": "identity",
    "fallback_mode": "identity",
    "use_torch_compile": False,
}
LIMITS = {
    "episodes": (1, 20),
    "settling": (10, 200),
    "measurement": (280, 5600),
    "model": (160, 3200),
    "capture": (2, 40),
}
E_FIELDS = (
    "native_closed_loop_contract_passed",
    "native_multirow_consumption_observed",
    "native_replanning_takeover_observed",
    "native_model_control_overlap_observed",
    "paired_scheduling_comparison_complete",
)


def fixed_manifest():
    rows = []
    for task, name in enumerate(reference.TASK_NAMES):
        for condition in CONDITIONS if task % 2 == 0 else CONDITIONS[::-1]:
            rows.append(
                {
                    "ordinal": len(rows),
                    "pair_index": task,
                    "task_id": task,
                    "task_name": name,
                    "initial_state_id": 41,
                    "environment_seed": 940041 + 100 * task,
                    "policy_seed": 950041 + 100 * task,
                    "condition": condition,
                    "control": dict(CONTROL),
                    "limits": {k: v[0] for k, v in LIMITS.items()},
                    "ready_wall_slots": 1200,
                    "startup_seconds": 30,
                    "request_seconds": 15,
                    "native_call_seconds": 30,
                }
            )
    return {
        "suite": "libero_object",
        "task_order_index": 0,
        "rows": rows,
        "policy_revision": reference.POLICY_REVISION,
        "vlm_revision": reference.VLM_REVISION,
        "assets_revision": reference.ASSETS_REVISION,
        "chunk_size": 50,
        "n_action_steps": 1,
        "num_steps": 10,
        "consume_policy": "queue_until_valid_takeover_or_exhaustion",
        "queue_pops_per_native_dispatch": 1,
        "late_policy": "whole_discard_any_late",
        "global_limits": {k: v[1] for k, v in LIMITS.items()},
        "reference_calls": 0,
        "soft_stop_seconds": 3570,
        "hard_stop_seconds": 3600,
        **FLAGS,
    }


def read_manifest():
    value = json.loads(MANIFEST.read_text())
    if value != fixed_manifest():
        raise ValueError("The fixed twenty-row E manifest changed")
    return value


class Budget:
    def __init__(self):
        self.total = Counter()
        self.episode = Counter()

    def begin_episode(self):
        self.episode = Counter()
        self.take("episodes")

    def check(self, kind):
        local, total = LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"E {kind} budget exhausted before dispatch")

    def take(self, kind):
        self.check(kind)
        self.episode[kind] += 1
        self.total[kind] += 1

    def call(self, reference=False):
        if reference:
            raise ValueError("E has no eager reference calls")
        self.take("model")


class Clock:
    now = staticmethod(time.perf_counter)
    sleep = staticmethod(time.sleep)


class Calls:
    """Small flushed intent/return journal, including the owned process watchdog input."""

    def __init__(self, path, clock=None):
        self.file = path.open("x")
        self.clock = clock or Clock()
        self.lock = threading.Lock()
        self.next_id = 0
        self.records = []
        self.logging_seconds = 0.0

    def emit(self, event, **values):
        with self.lock:
            started = self.clock.now()
            record = {"event": event, "timestamp": started, **values}
            self.file.write(json.dumps(record, allow_nan=False) + "\n")
            self.file.flush()
            self.records.append(record)
            self.logging_seconds += self.clock.now() - started

    def start(self, kind, ordinal, limit=30, **values):
        with self.lock:
            identifier = self.next_id
            self.next_id += 1
        self.emit("call_intent", call_id=identifier, kind=kind, ordinal=ordinal, limit=limit, **values)
        return identifier

    def call(self, kind, ordinal, function, *, details=None):
        identifier = self.start(kind, ordinal, **(details or {}))
        started = self.clock.now()
        try:
            value = function()
        except BaseException:
            self.emit("call_error", call_id=identifier, exception=traceback.format_exc())
            raise
        ended = self.clock.now()
        elapsed = ended - started
        self.emit("call_return", call_id=identifier, elapsed=elapsed)
        if elapsed > 30:
            raise TimeoutError(f"Native call {kind} returned after its 30-second limit")
        return value

    def close(self):
        self.file.close()


class Metrics:
    def __init__(self):
        self.events = []
        self.last_get = None
        self.closed = False

    def emit(self, event):
        event = dict(event)
        self.events.append(event)
        if event["event"] == "queue_get":
            self.last_get = event

    def close(self):
        self.closed = True


def observation(raw, language, index, returned_at):
    """Exactly the existing native transform, then independent CPU worker inputs."""
    batch = reference.checked_observation(raw, language)
    obs = {f"state_{i}": float(batch[OBS_STATE][0, i]) for i in range(8)}
    for key in CAMERA_KEYS:
        obs[key.rsplit(".", 1)[1]] = (
            (batch[key][0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy().copy()
        )
    obs.update(e_observation_index=index, e_observation_returned_at=returned_at)
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
        **{k: {"dtype": "video", "shape": obs[k.rsplit(".", 1)[1]].shape} for k in CAMERA_KEYS},
    }
    rebuilt = prepare_observation_for_inference(
        build_dataset_frame(features, obs, "observation"), torch.device("cpu")
    )
    if not all(equal(rebuilt[k], batch[k]) for k in (OBS_STATE, *CAMERA_KEYS)):
        raise ValueError("Native CPU observation reconstruction changed the original transform")
    robot = raw["robot_state"]
    record = {
        "index": index,
        "returned_at": returned_at,
        "state": cpu(batch[OBS_STATE]),
        "raw_pixels": cpu(raw["pixels"]),
        "eef_quaternion_xyzw": cpu(robot["eef"]["quat"]),
        "raw_eef_position": cpu(robot["eef"]["pos"]),
        "raw_gripper_qpos": cpu(robot["gripper"]["qpos"]),
        "worker_observation": cpu(obs),
    }
    return obs, features, record


def initial_difference(left, right):
    for key in ("state", "eef_quaternion_xyzw", "raw_eef_position", "raw_gripper_qpos"):
        if not equal(left[key], right[key]):
            return key
    for camera in ("image", "image2"):
        if not equal(left["raw_pixels"][camera], right["raw_pixels"][camera]):
            return camera
    return None


class NativeRuntime(SmolVLAGraphRuntime):
    def __init__(self, model, engine):
        super().__init__(model)
        self.engine = engine

    def _capture(self, inputs):
        if self.engine.control_t0 is not None:
            raise RuntimeError("E attempted a capture during measured native control")
        self.engine.budget.take("capture")
        return super()._capture(inputs)


class NativeAudit(CallAudit):
    def __init__(self, engine, policy, budget):
        super().__init__(policy, budget)
        self.engine = engine

    def record(self, name):
        super().record(name)
        if name == "policy":
            self.engine.current["model_started_at"] = time.perf_counter()

    def attach(self):
        super().attach()
        original = self.policy.predict_action_chunk

        def timed(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                self.engine.current["model_returned_at"] = time.perf_counter()

        self.policy.predict_action_chunk = timed


class NativeEngine(SmolVLAGraphIdentityEngine):
    """Only owner seed, completion notification and read-only experiment evidence."""

    def __init__(self, policy, pre, post, features, spec, budget, calls, *, device="cuda"):
        self.spec, self.budget, self.calls = spec, budget, calls
        self.audit = NativeAudit(self, policy, budget)
        self.metrics = Metrics()
        self.finished = threading.Condition()
        self.completed = set()
        self.rows, self.arrays = [], {}
        self.accepted = {}
        self.current = None
        self.control_t0 = None
        self.active_calls = self.peak_active_calls = 0
        super().__init__(
            policy=policy,
            preprocessor=AuditedProcessor(pre, self.audit, "pre"),
            postprocessor=AuditedProcessor(post, self.audit, "post"),
            hw_features=features,
            robot_wrapper=SimpleNamespace(robot_type="libero"),
            task=spec["task_name"].replace("_", " "),
            device=device,
            metrics_sink=self.metrics,
            **CONTROL,
        )

    def _make_graph_runtime(self):
        self.audit.attach()
        torch.manual_seed(self.spec["policy_seed"])
        return NativeRuntime(self._policy.model, self)

    @contextmanager
    def _worker_resources(self):
        try:
            with super()._worker_resources():
                yield
        finally:
            self.audit.detach()

    def _run_request(self, request):
        self.current = {
            "request_id": request.request_id,
            "kind": request.kind,
            "startup_phase": request.startup_phase,
            "reset_epoch": request.reset_epoch,
            "task_epoch": request.task_epoch,
            "observation_index": request.observation["e_observation_index"],
            "observation_returned_at": request.observation["e_observation_returned_at"],
            "requested_at": request.requested_at,
            "started_at": time.perf_counter(),
            "owner_thread": threading.get_ident(),
        }
        self.rows.append(self.current)
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        if self.active_calls != 1:
            raise RuntimeError("More than one model request is active")
        self.current["watch_id"] = self.calls.start(
            "model_request", self.spec["ordinal"], limit=15, request_id=request.request_id
        )
        before = self.audit.counts.copy()
        if request.plan is not None:
            plan = request.plan
            if plan.committed_policy_actions.device.type != "cpu":
                raise ValueError("Controller plan is not on CPU")
            self.current["plan"] = {
                k: getattr(plan, k) for k in ("next_action_index", "takeover_index", "planned_delay_steps")
            }
            self.arrays[f"prefix_{request.request_id}"] = cpu(asdict(plan))
        try:
            super()._run_request(request)
        except BaseException:
            self.current["exception"] = traceback.format_exc()
            raise
        finally:
            self.current["api_counts"] = dict(self.audit.counts - before)

    def _prepare_queue_actions(self, actions, metrics):
        policy, post = super()._prepare_queue_actions(actions, metrics)
        if (
            policy.shape != (50, 7)
            or post.shape != (50, 7)
            or policy.device.type != "cpu"
            or post.device.type != "cpu"
            or policy.data_ptr() == post.data_ptr()
        ):
            raise ValueError("E publication requires independent CPU 50x7 policy/post chunks")
        self.current.update(self.runtime.metadata)
        self.current.update(
            cpu_chunks=True,
            device_completion_barrier=True,
            model_cpu_completed_at=time.perf_counter(),
            language_length=self.runtime.latest_inputs[4].shape[1],
        )
        self.arrays[f"request_{self.current['request_id']}"] = {
            "inputs": cpu(self.runtime.latest_inputs),
            "noise": cpu(self.runtime.latest_noise),
            "full_chunk": cpu(self.runtime.latest),
            "policy_chunk": cpu(policy),
            "post_chunk": cpu(post),
        }
        return policy, post

    def _request_finished(self, request):
        try:
            self.current["finished_at"] = time.perf_counter()
            if not self.current.get("exception"):
                counts = self.current["api_counts"]
                if any(counts.get(k, 0) != 1 for k in ("policy", "prepare_images", "vision", "noise")):
                    raise ValueError("E request encoding/noise counts differ")
                if self.current["replay_count"] != 1:
                    raise ValueError("E request did not perform exactly one Graph replay")
            self.calls.emit("call_return", call_id=self.current["watch_id"])
        except BaseException:
            self.current["exception"] = traceback.format_exc()
            raise
        finally:
            self.active_calls -= 1
            with self.finished:
                self.completed.add(request.request_id)
                self.finished.notify_all()

    def check_failure(self):
        if self.failed:
            raise RuntimeError(self.failure_traceback)
        row = self.current
        if row is not None and row.get("exception"):
            raise RuntimeError(row["exception"])
        if row is not None and "finished_at" not in row and time.perf_counter() - row["requested_at"] > 15:
            raise TimeoutError("E model request exceeded 15 seconds")

    def wait_request(self, request_id, timeout=15):
        # No queue/request lock is held while waiting for this exact request.
        with self.finished:
            done = self.finished.wait_for(lambda: request_id in self.completed or self.failed, timeout)
        if not done:
            raise TimeoutError(f"E request {request_id} did not finish within {timeout}s")
        self.check_failure()


def notify(engine, obs):
    with engine._request_lock:
        before = engine._request_id
    engine.notify_observation(obs)
    with engine._request_lock:
        after = engine._request_id
    if after != before:
        engine.accepted[before] = {
            "request_id": before,
            "accepted_at": time.perf_counter(),
            "observation_index": obs["e_observation_index"],
        }
        return before
    return None


def startup(engine, obs, clock):
    started = clock.now()
    engine.start()
    engine.resume()
    for _ in range(3):
        remaining = 30 - (clock.now() - started)
        if remaining <= 0:
            raise TimeoutError("E startup exceeded 30 seconds")
        request_id = notify(engine, obs)
        if request_id is None:
            raise RuntimeError("Original startup did not issue its registered request")
        engine.wait_request(request_id, timeout=min(15, remaining))
    engine.check_failure()
    if not engine.ready or clock.now() - started > 30:
        raise RuntimeError("Original three-stage startup did not become ready within 30s")
    if [row["startup_phase"] for row in engine.rows] != ["cold_temporary", "probe", "fresh_warmed"]:
        raise ValueError("Original startup phase sequence changed")
    return clock.now() - started


class NativeSession:
    def __init__(self, spec, budget, calls, factory):
        self.spec, self.budget, self.calls, self.factory = spec, budget, calls, factory
        self.env = None
        self.segment = "settling"
        self.returned = Counter()
        self.dispatch = None
        self.native_steps = []

    def create_reset(self):
        ordinal = self.spec["ordinal"]
        self.env = self.calls.call("environment_factory", ordinal, lambda: self.factory(self.spec))
        self.calls.call("environment_ensure_and_initial_reset", ordinal, self.env.unwrapped._ensure_env)
        native = self.env.unwrapped._env
        original_step = native.step

        def native_step(action):
            self.budget.take(self.segment)
            command = np.asarray(action).copy()
            action_outside_bounds(command)
            details = {
                "segment": self.segment,
                "action": command.tolist(),
                "number": self.budget.episode[self.segment],
            }
            if self.segment == "measurement":
                if self.dispatch is None or not np.array_equal(command, self.dispatch["command"]):
                    raise ValueError("Native command differs from the single queue dispatch")
                details.update(
                    slot=self.dispatch["slot"],
                    action_index=self.dispatch["action_index"],
                    observation_index=self.dispatch["observation_index"],
                )

            def invoke_native():
                started_at = self.calls.clock.now()
                transition = original_step(action)
                returned_at = self.calls.clock.now()
                self.returned[self.segment] += 1
                self.native_steps.append(
                    {
                        **details,
                        "started_at": started_at,
                        "returned_at": returned_at,
                        "native_done": bool(transition[2]),
                    }
                )
                return transition

            return self.calls.call("native_step", ordinal, invoke_native, details=details)

        native.step = native_step
        for name in ("reset", "seed", "set_init_state"):
            original = getattr(native, name)

            def wrapped(*args, _original=original, _name=name, **kwargs):
                return self.calls.call("native_" + _name, ordinal, lambda: _original(*args, **kwargs))

            setattr(native, name, wrapped)
        raw, _ = self.calls.call(
            "environment_reset", ordinal, lambda: self.env.reset(seed=self.spec["environment_seed"])
        )
        if (
            self.returned["settling"] != 10
            or self.env.unwrapped.init_state_id != self.spec["initial_state_id"] + 1
        ):
            raise ValueError("Original native reset row or ten settling calls changed")
        self.segment = "measurement"
        return raw

    def step(self, dispatch):
        self.budget.check("measurement")
        self.dispatch = dispatch
        return self.calls.call(
            "environment_step", self.spec["ordinal"], lambda: self.env.step(dispatch["command"])
        )

    def close(self):
        if self.env is None:
            return None
        self.calls.call("environment_close", self.spec["ordinal"], self.env.close)
        if self.env.unwrapped._env is not None:
            raise RuntimeError("Native environment close was not confirmed")
        return True


def slot_at(now, t0):
    return max(0, math.floor((now - t0) * 20))


def measured_control(engine, native, spec, obs, records, *, clock=None):
    clock = clock or Clock()
    t0 = clock.now()
    engine.control_t0 = t0
    next_slot = 0
    dispatches, gets, opportunities, blocks = [], [], [], []
    result = {
        "t0": t0,
        "dispatches": dispatches,
        "gets": gets,
        "opportunities": opportunities,
        "blocks": blocks,
        "success": False,
        "first_success_action": None,
        "terminal_reason": None,
    }
    try:
        while next_slot < spec["ready_wall_slots"] and len(dispatches) < spec["limits"]["measurement"]:
            clock.sleep(max(0, t0 + next_slot / 20 - clock.now()))
            slot = max(next_slot, slot_at(clock.now(), t0))
            if slot >= spec["ready_wall_slots"]:
                break
            engine.check_failure()
            request_id = notify(engine, obs)
            opportunities.append(
                {
                    "slot": slot,
                    "timestamp": clock.now(),
                    "request_id": request_id,
                    "observation_index": obs["e_observation_index"],
                    "observation_age": clock.now() - obs["e_observation_returned_at"],
                }
            )
            if request_id is not None and spec["condition"] == "graph_serialized":
                wait_start = clock.now()
                engine.wait_request(request_id)
                blocks.append({"kind": "serialized_wait", "start": wait_start, "end": clock.now()})
            # Never notify again on the unchanged observation after a crossing wait.
            slot = max(slot, slot_at(clock.now(), t0))
            if slot >= spec["ready_wall_slots"]:
                break
            engine.check_failure()
            native.budget.check("measurement")
            action = engine.get_action(None)
            get = dict(engine.metrics.last_get)
            get.update(slot=slot, controller_time=clock.now())
            gets.append(get)
            if action is None:
                get.update(
                    queue_epoch=engine.queue.reset_epoch,
                    task_epoch=engine.queue.task_epoch,
                    queue_available=engine.queue.available_steps(),
                    requests_started=engine.stats.requests_started,
                    observation_index=obs["e_observation_index"],
                )
                next_slot = max(slot, slot_at(clock.now(), t0)) + 1
                continue
            if action.device.type != "cpu":
                raise ValueError("Controller received a non-CPU action")
            command = action.detach().numpy().copy()
            action_outside_bounds(command)
            slot = max(slot, slot_at(clock.now(), t0))
            if slot >= spec["ready_wall_slots"]:
                get["not_dispatched"] = "wall_slot_limit"
                break
            engine.check_failure()
            dispatch = {
                "slot": slot,
                "action_index": get["action_index"],
                "queue_get_outcome": get["outcome"],
                "command": command,
                "observation_index": obs["e_observation_index"],
                "observation_age": clock.now() - obs["e_observation_returned_at"],
                "dispatched_at": clock.now(),
                "jitter_seconds": clock.now() - (t0 + slot / 20),
            }
            dispatches.append(dispatch)
            returned = native.step(dispatch)
            returned_at = clock.now()
            raw, reward, terminated, truncated, info = returned
            blocks.append({"kind": "env_busy", "start": dispatch["dispatched_at"], "end": returned_at})
            dispatch.update(
                returned_at=returned_at,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                success=bool(info.get("is_success", False)),
                native_step_index=len(dispatches),
                returned_observation_index=len(dispatches),
            )
            obs, _, record = observation(
                raw, spec["task_name"].replace("_", " "), len(dispatches), returned_at
            )
            records.append(record)
            result.update(
                success=bool(info.get("is_success", False)),
                terminal_reason=terminal_reason(terminated, truncated, info),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            if result["success"]:
                result["first_success_action"] = len(dispatches)
            if result["terminal_reason"] is not None:
                break
            next_slot = slot + 1
        if result["terminal_reason"] is None:
            result["terminal_reason"] = (
                "action_limit" if len(dispatches) >= spec["limits"]["measurement"] else "wall_slot_limit"
            )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
        result["terminal_reason"] = "technical_failure"
        raise
    finally:
        result["ended_at"] = clock.now()
        engine.control_result = result
    return result


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        **{f"p{p}": values[math.ceil(p / 100 * len(values)) - 1] for p in (50, 95, 99)},
        "max": values[-1],
    }


def slot_accounting(control, limit):
    t0, end = control["t0"], control["ended_at"]
    count = min(limit, max(1, math.ceil((end - t0) * 20)))
    dispatched = {row["slot"]: row for row in control["dispatches"]}
    if len(dispatched) != len(control["dispatches"]) or any(k >= limit for k in dispatched):
        raise ValueError("Multiple dispatches in a wall slot or dispatch beyond the wall limit")
    underflows = {row["slot"] for row in control["gets"] if row["outcome"] == "underflow"}
    rows = []
    for slot in range(count):
        coverage = Counter()
        for block in control["blocks"]:
            coverage[block["kind"]] += max(
                0, min(t0 + (slot + 1) / 20, block["end"]) - max(t0 + slot / 20, block["start"])
            )
        reason = (
            "dispatch"
            if slot in dispatched
            else "underflow"
            if slot in underflows
            else max(coverage, key=coverage.get)
            if coverage and max(coverage.values()) > 0
            else "scheduler_miss"
        )
        rows.append({"slot": slot, "outcome": reason, "blocked_seconds": dict(coverage)})
    longest = run = 0
    for row in rows:
        run = 0 if row["outcome"] == "dispatch" else run + 1
        longest = max(longest, run)
    times = [t0, *[d["dispatched_at"] for d in control["dispatches"]], end]
    return {
        "slots": rows,
        "counts": dict(Counter(row["outcome"] for row in rows)),
        "total_slots": count,
        "no_action_slots": count - len(dispatched),
        "longest_no_action_slot_run": longest,
        "longest_dispatch_gap_seconds": max(b - a for a, b in zip(times, times[1:], strict=False)),
        "measured_wall_seconds": end - t0,
        "wall_window_overshoot_seconds": max(0, end - t0 - limit / 20),
    }


def audit_episode(engine, native, control):
    captures = engine.runtime.captures
    if len(captures) != 2 or any(
        c["status"] != "captured"
        or c["eager_setup_calls"] != 1
        or c["side_stream_warmup_calls"] != 3
        or c["capture_calls"] != 1
        or len(c["projection_shapes"]) != 10
        or c["owner_thread"] != engine.owner_thread
        for c in captures
    ):
        raise ValueError("E startup capture counts or owner differ from the registered contract")
    if engine.budget.episode["model"] != len(engine.rows) or any(
        row["owner_thread"] != engine.owner_thread for row in engine.rows
    ):
        raise ValueError("E model dispatch count or request owner differs")
    terminals = [e for e in engine.metrics.events if e["event"] in ("chunk_request", "request_error")]
    by_id = {e["request_id"]: e for e in terminals}
    if len(by_id) != len(terminals) or len(by_id) != len(engine.rows):
        raise ValueError("Requests do not have unique terminal records")
    cancelled = sorted(set(engine.accepted) - set(by_id))
    if (
        engine.stats.requests_started != len(engine.accepted)
        or len(cancelled) > 1
        or (cancelled and cancelled != [max(engine.accepted)])
    ):
        raise ValueError("Accepted requests do not reconcile with worker terminals and stop cancellation")
    rows = {row["request_id"]: row for row in engine.rows}
    boots, staged = {}, {}
    for row in engine.rows:
        terminal = by_id[row["request_id"]]
        row["terminal"] = terminal
        if row.get("exception") or terminal["event"] == "request_error":
            raise RuntimeError("Model request failed")
        if not row["cpu_chunks"] or not row["device_completion_barrier"]:
            raise ValueError("CPU publication evidence missing")
        if terminal["outcome"] == "installed" and row["startup_phase"] != "cold_temporary":
            index = terminal["result_next_action_index"]
            if index in boots:
                raise ValueError("Ambiguous bootstrap activation")
            boots[index] = row["request_id"]
        if terminal["outcome"] in ("staged_early", "staged_on_time"):
            index = terminal["takeover_index"]
            if index in staged:
                raise ValueError("Ambiguous planned activation")
            staged[index] = row["request_id"]
    expected_index, active = 0, None
    sources, consumed = {}, Counter()
    for get in control["gets"]:
        index = get["action_index"]
        if index != expected_index:
            raise ValueError("Queue index advanced without a successful get or skipped a row")
        if get["outcome"] == "underflow":
            continue
        if get["outcome"] == "takeover":
            active = (staged[index], index)
        elif index in boots:
            active = (boots[index], index)
        if active is None:
            raise ValueError("Queue action has no activation source")
        request_id, first_index = active
        offset = index - first_index
        if not 0 <= offset < 50 or rows[request_id]["model_cpu_completed_at"] > get["timestamp_s"]:
            raise ValueError("Queue source row is out of range or was not completed")
        sources[index] = (request_id, offset)
        consumed[request_id] += 1
        expected_index += 1
    measurements = [row for row in native.native_steps if row["segment"] == "measurement"]
    if len(measurements) != len(control["dispatches"]):
        raise ValueError("Native intent/return does not match controller dispatches")
    takeovers = 0
    sent = Counter()
    for dispatch, step in zip(control["dispatches"], measurements, strict=True):
        request_id, offset = sources[dispatch["action_index"]]
        chunk = engine.arrays[f"request_{request_id}"]["post_chunk"]
        if not equal(torch.from_numpy(dispatch["command"]), chunk[offset]) or not np.array_equal(
            dispatch["command"], np.asarray(step["action"], dtype=dispatch["command"].dtype)
        ):
            raise ValueError("Dispatched/native action differs from its identified chunk row")
        if (
            step["action_index"] != dispatch["action_index"]
            or step["observation_index"] != dispatch["observation_index"]
        ):
            raise ValueError("Native call provenance differs from the queue dispatch")
        dispatch.update(source_request_id=request_id, source_row_offset=offset)
        sent[request_id] += 1
        if dispatch["queue_get_outcome"] == "takeover":
            if offset != 0:
                raise ValueError("Planned takeover did not start at row zero")
            takeovers += 1
    overlaps = []
    for request in engine.rows:
        if request["startup_phase"] is not None:
            continue
        for step in measurements:
            start = max(request["model_started_at"], step["started_at"])
            end = min(request["model_returned_at"], step["returned_at"])
            if end > start:
                overlaps.append(
                    {
                        "request_id": request["request_id"],
                        "native_step_index": step["number"],
                        "start": start,
                        "end": end,
                        "seconds": end - start,
                    }
                )
    if engine.spec["condition"] == "graph_serialized" and overlaps:
        raise ValueError("Serialized model/native calls overlapped")
    chunks = [
        {
            "request_id": i,
            "startup_phase": row["startup_phase"],
            "outcome": by_id[i]["outcome"],
            "queue_rows_consumed": consumed[i],
            "native_rows_sent": sent[i],
            "unconsumed_rows": 50 - consumed[i],
            "unsent_rows": 50 - sent[i],
        }
        for i, row in rows.items()
    ]
    multirow = any(
        b["source_request_id"] == a["source_request_id"]
        and b["source_row_offset"] == a["source_row_offset"] + 1
        for a, b in zip(control["dispatches"], control["dispatches"][1:], strict=False)
    )
    timing = {
        "startup_request_seconds": [
            r["terminal"]["total_chunk_s"] for r in engine.rows if r["startup_phase"] is not None
        ],
        "measured_request_seconds": [
            r["terminal"]["total_chunk_s"] for r in engine.rows if r["startup_phase"] is None
        ],
        "model_api_seconds": [r["model_returned_at"] - r["model_started_at"] for r in engine.rows],
        "native_step_seconds": [s["returned_at"] - s["started_at"] for s in measurements],
        "serialized_wait_seconds": [
            b["end"] - b["start"] for b in control["blocks"] if b["kind"] == "serialized_wait"
        ],
        "dispatch_observation_age_seconds": [d["observation_age"] for d in control["dispatches"]],
        "request_observation_age_seconds": [
            r["requested_at"] - r["observation_returned_at"] for r in engine.rows
        ],
        "dispatch_jitter_seconds": [d["jitter_seconds"] for d in control["dispatches"]],
    }
    return {
        "source_audit_passed": True,
        "cancelled_before_worker_at_stop": cancelled,
        "multirow_observed": multirow,
        "planned_takeovers": takeovers,
        "new_feedback_used_by_later_requests": any(r["observation_index"] > 0 for r in engine.rows),
        "chunks": chunks,
        "chunk_consumption_distribution": dict(Counter(c["native_rows_sent"] for c in chunks)),
        "discarded_chunks": sum(
            c["outcome"] in ("deadline_miss", "stale", "probe_discarded")
            or c["startup_phase"] == "cold_temporary"
            for c in chunks
        ),
        "unconsumed_rows": sum(c["unconsumed_rows"] for c in chunks),
        "overlaps": overlaps,
        "peak_model_inflight": engine.peak_active_calls,
        "timing_samples": timing,
        "timings": {k: quantiles(v) for k, v in timing.items()},
    }


def run_episode(
    spec,
    directory,
    policy,
    pre,
    post,
    factory,
    budget,
    calls,
    paired_initial=None,
    *,
    engine_class=NativeEngine,
    clock=None,
    device="cuda",
):
    clock = clock or Clock()
    directory.mkdir()
    started = clock.now()
    write_json(directory / "started.json", {"spec": spec, "started_at": started})
    budget.begin_episode()
    result = {
        "spec": spec,
        "status": "technical_failure",
        "initial_pair_exact": None,
        "environment_closed": None,
        "worker_joined": None,
        "graph_released": None,
        "original_sampler_restored": None,
        "first_failure": None,
    }
    native = NativeSession(spec, budget, calls, factory)
    engine = initial = control = None
    records = []
    pending_at_stop = None
    logging_start = calls.logging_seconds
    try:
        raw = native.create_reset()
        returned_at = clock.now()
        obs, features, initial = observation(raw, spec["task_name"].replace("_", " "), 0, returned_at)
        records.append(initial)
        if paired_initial is not None:
            difference = initial_difference(paired_initial, initial)
            result["initial_pair_exact"] = difference is None
            if difference is not None:
                result["initial_difference"] = difference
                raise ValueError(f"First paired initial observation difference: {difference}")
        engine = engine_class(policy, pre, post, features, spec, budget, calls, device=device)
        result["startup_seconds"] = startup(engine, obs, clock)
        control = measured_control(engine, native, spec, obs, records, clock=clock)
        result["status"] = "completed"
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if engine is not None:
            control = getattr(engine, "control_result", control)
            with engine._request_lock:
                pending = engine._pending_request
                if pending is not None:
                    pending_at_stop = {
                        "request_id": pending.request_id,
                        "kind": pending.kind,
                        "observation_index": pending.observation["e_observation_index"],
                        "plan": None if pending.plan is None else cpu(asdict(pending.plan)),
                    }
            engine.stop()
            for key in ("worker_joined", "graph_released", "original_sampler_restored"):
                result[key] = getattr(engine, key)
            result["worker_failed"] = engine.failed
            result["metrics_closed"] = engine.metrics.closed
            if (
                not all(
                    result[k]
                    for k in (
                        "worker_joined",
                        "graph_released",
                        "original_sampler_restored",
                        "metrics_closed",
                    )
                )
                or engine.failed
            ):
                result["status"] = "technical_failure"
                result["first_failure"] = (
                    result["first_failure"] or engine.failure_traceback or "Worker cleanup was not confirmed"
                )
        if engine is None or engine.worker_joined:
            try:
                result["environment_closed"] = native.close()
            except BaseException:
                result["status"] = "technical_failure"
                result["first_failure"] = result["first_failure"] or traceback.format_exc()
            if engine is not None:
                result.update(
                    requests=engine.rows,
                    accepted_requests=list(engine.accepted.values()),
                    metrics=engine.metrics.events,
                    stats=asdict(engine.stats),
                    captures=list(engine.runtime.captures) if engine.runtime else [],
                    owner_lifecycle=engine.lifecycle,
                    owner_thread=engine.owner_thread,
                    controller_thread=threading.get_ident(),
                    api_counts=dict(engine.audit.counts),
                    api_events=engine.audit.events,
                )
            if control is not None:
                result.update(
                    {
                        k: control[k]
                        for k in ("success", "first_success_action", "terminal_reason", "t0", "ended_at")
                    }
                )
                try:
                    result["wall"] = slot_accounting(control, spec["ready_wall_slots"])
                    if result["status"] == "completed":
                        result["audit"] = audit_episode(engine, native, control)
                except BaseException:
                    result["status"] = "technical_failure"
                    result["first_failure"] = result["first_failure"] or traceback.format_exc()
            serialization_start = clock.now()
            torch.save(
                {
                    "observations": records,
                    "requests": {} if engine is None else engine.arrays,
                    "control": cpu(control),
                    "pending_at_stop": pending_at_stop,
                },
                directory / "arrays.pt",
            )
            result["serialization_seconds"] = clock.now() - serialization_start
        else:
            result["arrays_not_serialized"] = "Worker exit unconfirmed; shared arrays remain unsaved"
        result.update(
            native_steps=native.native_steps,
            native_returned=dict(native.returned),
            budget=dict(budget.episode),
            logging_seconds=calls.logging_seconds - logging_start,
            episode_wall_seconds=clock.now() - started,
        )
        write_json(directory / "result.json", result)
    return result, initial


def summarize(records, manifest):
    completed = [r for r in records if r["status"] == "completed"]
    async_rows = [r for r in completed if r["spec"]["condition"] == "graph_identity_async"]
    serial_rows = [r for r in completed if r["spec"]["condition"] == "graph_serialized"]
    pairs = []
    for pair in range(10):
        rows = [r for r in completed if r["spec"]["pair_index"] == pair]
        if len(rows) != 2:
            continue
        values = {
            r["spec"]["condition"]: {
                "success": r["success"],
                "first_success_action": r["first_success_action"],
                "terminal_reason": r["terminal_reason"],
                "measured_actions": r["native_returned"]["measurement"],
                "wall_seconds": r["wall"]["measured_wall_seconds"],
                "no_action_slots": r["wall"]["no_action_slots"],
                "longest_dispatch_gap_seconds": r["wall"]["longest_dispatch_gap_seconds"],
                "planned_takeovers": r["audit"]["planned_takeovers"],
                "chunk_consumption_distribution": r["audit"]["chunk_consumption_distribution"],
            }
            for r in rows
        }
        a, b = values["graph_identity_async"], values["graph_serialized"]
        pairs.append(
            {
                "pair_index": pair,
                "initial_exact": any(r["initial_pair_exact"] is True for r in rows),
                "conditions": values,
                "async_minus_serialized": {
                    k: a[k] - b[k]
                    for k in (
                        "measured_actions",
                        "wall_seconds",
                        "no_action_slots",
                        "longest_dispatch_gap_seconds",
                        "planned_takeovers",
                    )
                },
            }
        )
    complete = len(completed) == 20 and len(pairs) == 10 and all(p["initial_exact"] for p in pairs)
    result = {
        "status": "completed" if complete else "technical_failure",
        "episodes_scheduled": 20,
        "episodes_completed": len(completed),
        "pairs": pairs,
        "native_closed_loop_contract_passed": complete,
        "native_multirow_consumption_observed": any(r["audit"]["multirow_observed"] for r in async_rows),
        "native_replanning_takeover_observed": any(
            r["audit"]["planned_takeovers"] > 0 and r["audit"]["new_feedback_used_by_later_requests"]
            for r in async_rows
        ),
        "native_model_control_overlap_observed": bool(async_rows)
        and bool(serial_rows)
        and any(r["audit"]["overlaps"] for r in async_rows)
        and all(not r["audit"]["overlaps"] for r in serial_rows),
        "paired_scheduling_comparison_complete": complete,
        "conditions": {},
        "manifest": manifest,
        "first_failure": next((r.get("first_failure") for r in records if r.get("first_failure")), None),
        **FLAGS,
    }
    for condition in CONDITIONS:
        group = [r for r in completed if r["spec"]["condition"] == condition]
        samples = {}
        for r in group:
            for key, values in r["audit"]["timing_samples"].items():
                samples.setdefault(key, []).extend(values)
        result["conditions"][condition] = {
            "episodes_completed": len(group),
            "successes": sum(r["success"] for r in group),
            "measured_actions": sum(r["native_returned"]["measurement"] for r in group),
            "measured_wall_seconds": sum(r["wall"]["measured_wall_seconds"] for r in group),
            "startup_seconds": sum(r["startup_seconds"] for r in group),
            "wall_slots": dict(sum((Counter(r["wall"]["counts"]) for r in group), Counter())),
            "planned_takeovers": sum(r["audit"]["planned_takeovers"] for r in group),
            "model_native_overlaps": sum(len(r["audit"]["overlaps"]) for r in group),
            "chunk_consumption_distribution": dict(
                sum((Counter(r["audit"]["chunk_consumption_distribution"]) for r in group), Counter())
            ),
            "timings": {key: quantiles(values) for key, values in samples.items()},
        }
    return result


def require_source(head):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines()
    if actual != head or any(line[:3] != "?? " or line[3:] not in UNTRACKED_DOCS for line in status):
        raise ValueError("E execution source must equal its registered clean HEAD")


def request_shutdown(_signal, _frame):
    raise TimeoutError("E supervisor requested shutdown")


def worker(args):
    require_source(args.execution_head)
    manifest = read_manifest()
    budget = Budget()
    calls = Calls(args.output / "calls.jsonl")
    records = []
    outcome = {"execution_head": args.execution_head, **dict.fromkeys(E_FIELDS, False)}
    try:
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("Registered GPU changed")
        start = time.perf_counter()
        policy, pre, post, report = load_runtime(POLICY, VLM)
        outcome.update(model_load_seconds=time.perf_counter() - start, policy_load=report)
        factory = reference.make_native_env_factory()
        paired_initial = None
        for spec in manifest["rows"]:
            print(f"START E {spec['ordinal']:02d} task={spec['task_id']} {spec['condition']}", flush=True)
            result, initial = run_episode(
                spec,
                args.output / f"episode_{spec['ordinal']:03d}",
                policy,
                pre,
                post,
                factory,
                budget,
                calls,
                paired_initial,
            )
            records.append(result)
            print(
                f"END E {spec['ordinal']:02d} {result['status']} success={result.get('success')} measured={result['native_returned'].get('measurement', 0)}",
                flush=True,
            )
            if result["status"] != "completed":
                break
            paired_initial = initial if spec["ordinal"] % 2 == 0 else None
        outcome.update(summarize(records, manifest))
    except BaseException:
        outcome.update(status="technical_failure", first_failure=traceback.format_exc())
    finally:
        outcome.update(
            budget=dict(budget.total),
            episode_statuses=[
                {
                    "ordinal": spec["ordinal"],
                    "status": records[spec["ordinal"]]["status"]
                    if spec["ordinal"] < len(records)
                    else "not_run",
                }
                for spec in manifest["rows"]
            ],
            calls_logging_seconds=calls.logging_seconds,
        )
        calls.close()
        write_json(args.output / "worker_result.json", outcome)
    return 0 if outcome.get("native_closed_loop_contract_passed") else 2


def read_call_journal(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.endswith("}")]


def supervise(args):
    require_source(args.execution_head)
    manifest = read_manifest()
    if Path(sys.executable) != Path(PYTHON):
        raise ValueError("E must use the unchanged model interpreter")
    expected = REPO / "outputs" / f"smolvla_graph_identity_native_{args.execution_head[:8]}"
    if args.output != expected or args.output.exists():
        raise ValueError("E output path must be the new exclusive registered directory")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("E registration/readback differs")
    args.output.mkdir()
    for path in PREPARATION.iterdir():
        if path.is_file():
            shutil.copyfile(path, args.output / path.name)
    command = [
        PYTHON,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--execution-head",
        args.execution_head,
        "--output",
        str(args.output),
        "--worker",
    ]
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    stop_reason, signaled_at, forced = None, None, False
    cursor = 0
    pending = {}
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"E single queue pid={child.pid} head={args.execution_head} hard_limit=3600s", flush=True)
        try:
            while child.poll() is None:
                now = time.perf_counter()
                journal_path = args.output / "calls.jsonl"
                if journal_path.exists():
                    with journal_path.open() as journal:
                        journal.seek(cursor)
                        while line := journal.readline():
                            if not line.endswith("\n"):
                                break
                            cursor = journal.tell()
                            event = json.loads(line)
                            if event["event"] == "call_intent":
                                pending[event["call_id"]] = event
                            elif event["event"] in ("call_return", "call_error"):
                                pending.pop(event["call_id"], None)
                expired = [event for event in pending.values() if now - event["timestamp"] > event["limit"]]
                if signaled_at is None and (expired or now - start >= 3570):
                    stop_reason = {"expired_calls": expired} if expired else {"outer_soft_limit": 3570}
                    os.killpg(child.pid, signal.SIGTERM)
                    signaled_at = now
                if now - start >= 3600 or (
                    signaled_at is not None and stop_reason.get("expired_calls") and now - signaled_at >= 5
                ):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            stop_reason = {"supervisor_exception": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        exit_code = child.wait()
    execution = {
        "execution_head": args.execution_head,
        "command": command,
        "python": sys.executable,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - start,
        "child_pid": child.pid,
        "child_exit_code": exit_code,
        "child_exit_confirmed": True,
        "attempts": 1,
        "retries": 0,
        "stop_reason": stop_reason,
        "forced_termination": forced,
        "normal_exit": exit_code == 0 and stop_reason is None,
        "registration_url": registration["html_url"],
        "registration_readback_body_exact": True,
    }
    write_json(args.output / "execution.json", execution)
    records = []
    for spec in manifest["rows"]:
        directory = args.output / f"episode_{spec['ordinal']:03d}"
        path = directory / "result.json"
        records.append(
            json.loads(path.read_text())
            if path.exists()
            else {
                "spec": spec,
                "status": "started_return_unknown" if (directory / "started.json").exists() else "not_run",
            }
        )
    outcome = summarize(records, manifest)
    events = read_call_journal(args.output / "calls.jsonl")
    returns = {event["call_id"] for event in events if event["event"] == "call_return"}
    unknown = [
        event for event in events if event["event"] == "call_intent" and event["call_id"] not in returns
    ]
    outcome.update(
        execution_head=args.execution_head,
        execution=execution,
        episode_statuses=[{"ordinal": r["spec"]["ordinal"], "status": r["status"]} for r in records],
        unknown_calls=unknown,
        native_call_intents=dict(Counter(e["kind"] for e in events if e["event"] == "call_intent")),
        native_call_returns=dict(
            Counter(e["kind"] for e in events if e["event"] == "call_intent" and e["call_id"] in returns)
        ),
    )
    worker_path = args.output / "worker_result.json"
    if worker_path.exists():
        report = json.loads(worker_path.read_text())
        outcome["budget"] = report.get("budget", {})
        outcome["policy_load"] = report.get("policy_load")
        outcome["first_failure"] = outcome["first_failure"] or report.get("first_failure")
    if not execution["normal_exit"] or unknown:
        outcome["native_closed_loop_contract_passed"] = False
        outcome["paired_scheduling_comparison_complete"] = False
        outcome["status"] = "technical_failure"
        outcome["first_failure"] = outcome["first_failure"] or {
            "exit": exit_code,
            "stop_reason": stop_reason,
            "unknown_calls": unknown,
        }
    write_json(args.output / "result.json", outcome)
    print(json.dumps({k: outcome[k] for k in ("status", "episodes_completed", *E_FIELDS)}), flush=True)
    return 0 if outcome["native_closed_loop_contract_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    signal.signal(signal.SIGTERM, request_shutdown)
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
