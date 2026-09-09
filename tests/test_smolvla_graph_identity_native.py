"""E controller contracts on the real worker/queue, with CPU model and Env fixtures."""

import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "policies/rtc"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_graph_identity_native as e  # noqa: E402
import smolvla_graph_runtime as runtime_module  # noqa: E402
from profile_libero_cuda_graph_compute import invoke_graph  # noqa: E402
from test_scheduled_action_queue import (  # noqa: E402
    test_any_late_result_is_dropped_without_skipping_its_prefix as assert_whole_late_discard,
)
from test_smolvla_graph_identity import Policy, Processor  # noqa: E402
from test_smolvla_graph_native import FakeGraph  # noqa: E402


@pytest.fixture
def fake_graph(monkeypatch):
    class CPUPreparedGraph(FakeGraph):
        def __init__(self, model, original, inputs, projection, capture_record):
            for _ in range(3):
                invoke_graph(original, inputs)
                capture_record["side_stream_warmup_calls"] += 1
            capture_record["capture_calls"] += 1
            super().__init__(model, original, inputs, projection, capture_record)

    monkeypatch.setattr(runtime_module, "TokenGraph", CPUPreparedGraph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


class ControlledClock:
    def __init__(self):
        self.value = 1000.0
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.value

    def sleep(self, seconds):
        with self.lock:
            self.value += max(0, seconds)


class NativePolicy(Policy):
    def __init__(self, clock, condition, *, gate=False, fail=False):
        super().__init__()
        self.clock, self.condition, self.gate, self.fail = clock, condition, gate, fail
        self.calls = 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.active = False

    def predict_action_chunk(self, batch, **kwargs):
        self.calls += 1
        self.active = True
        try:
            if self.fail:
                raise ValueError("first CPU model error")
            if self.calls == 4 and self.gate:
                self.entered.set()
                if self.condition == "graph_serialized":
                    self.clock.sleep(0.13)
                else:
                    assert self.release.wait(3), "Async controller waited instead of reaching native Env"
            self.clock.sleep(0.001)
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self.active = False


class Post(Processor):
    def __call__(self, value):
        super().__call__(value)
        return value * 2 + 1


class FakeNative:
    def __init__(self, outer):
        self.outer = outer

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.outer.pixel = 0
        self.outer.measurements = 0
        return self.outer.raw()

    def set_init_state(self, state):
        return self.outer.raw()

    def step(self, action):
        env = self.outer
        if env.segment == "measurement":
            if env.options.get("native_error"):
                raise RuntimeError("first native error")
            if env.policy.gate and env.measurements == 21 and env.policy.condition == "graph_identity_async":
                assert env.policy.entered.wait(3)
                assert env.policy.active
                env.overlap_reached = True
                env.clock.sleep(0.005)
                env.policy.release.set()
            if env.policy.condition == "graph_serialized":
                assert not env.policy.active
            env.clock.sleep(env.options.get("step_seconds", 0.005))
        else:
            env.clock.sleep(0.001)
        env.pixel += 1
        return env.raw(), 0.0, False, {}


class FakeEnv:
    def __init__(self, spec, policy, clock, options):
        self.spec, self.policy, self.clock, self.options = spec, policy, clock, options
        self.unwrapped = self
        self._env = None
        self.init_state_id = spec["initial_state_id"]
        self.pixel = self.measurements = 0
        self.segment = "settling"
        self.overlap_reached = False
        self.images = {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in ("image", "image2")}

    def raw(self):
        for image in self.images.values():
            image.fill(self.pixel % 256)
        return {
            "pixels": self.images,
            "robot_state": {
                "eef": {
                    "pos": np.array([self.pixel / 1000, 0.0, 0.2]),
                    "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                },
                "gripper": {"qpos": np.array([0.02, 0.02])},
            },
        }

    def _ensure_env(self):
        if self._env is None:
            self._env = FakeNative(self)
            self._env.reset()

    def reset(self, seed):
        self._env.seed(seed)
        self._env.reset()
        self._env.set_init_state(self.init_state_id)
        self.init_state_id += 1
        for _ in range(10):
            self._env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        self.segment = "measurement"
        return self.raw(), {"is_success": False}

    def step(self, command):
        self.measurements += 1
        raw, reward, _, _ = self._env.step(command)
        success = self.measurements == self.options.get("success_at")
        truncated = self.measurements == self.options.get("truncate_at")
        return raw, reward, success, truncated, {"is_success": success, "done": success}

    def close(self):
        self._env = None


@pytest.fixture
def clock(monkeypatch):
    value = ControlledClock()
    monkeypatch.setattr(e.time, "perf_counter", value.now)
    return value


def episode(
    tmp_path,
    clock,
    condition="graph_serialized",
    *,
    gate=False,
    options=None,
    limit=28,
    slots=1200,
    engine_class=e.NativeEngine,
    fail_model=False,
    paired_initial=None,
    postprocessor=None,
):
    spec = copy.deepcopy(e.fixed_manifest()["rows"][0])
    spec["condition"] = condition
    spec["limits"]["measurement"] = limit
    spec["ready_wall_slots"] = slots
    policy = NativePolicy(clock, condition, gate=gate, fail=fail_model)
    environments = []

    def factory(row):
        env = FakeEnv(row, policy, clock, options or {})
        environments.append(env)
        return env

    calls = e.Calls(tmp_path / "calls.jsonl", clock)
    try:
        result, initial = e.run_episode(
            spec,
            tmp_path / "episode",
            policy,
            Processor(),
            postprocessor or Post(),
            factory,
            e.Budget(),
            calls,
            paired_initial,
            engine_class=engine_class,
            clock=clock,
            device="cpu",
        )
    finally:
        policy.release.set()
        calls.close()
    return result, initial, policy, environments, calls


@pytest.mark.parametrize("condition", e.CONDITIONS)
def test_same_real_engine_multiline_takeover_waiting_and_overlap(tmp_path, clock, fake_graph, condition):
    result, _, policy, envs, _ = episode(tmp_path, clock, condition, gate=True)
    assert result["status"] == "completed", result["first_failure"]
    assert result["worker_joined"] and result["graph_released"] and result["original_sampler_restored"]
    assert result["spec"]["control"] == e.CONTROL
    assert result["native_returned"] == {"settling": 10, "measurement": 28}
    assert result["budget"]["capture"] == 2
    assert result["audit"]["multirow_observed"]
    assert result["audit"]["planned_takeovers"] >= 1
    assert result["audit"]["new_feedback_used_by_later_requests"]
    assert all(row["cpu_chunks"] for row in result["requests"])
    arrays = torch.load(tmp_path / "episode/arrays.pt", weights_only=True)
    control = arrays["control"]
    assert len({row["slot"] for row in control["dispatches"]}) == 28
    assert [row["action_index"] for row in control["dispatches"]] == list(range(28))
    assert len(arrays["observations"]) == 29
    for chunk in arrays["requests"].values():
        if "post_chunk" in chunk:
            assert torch.equal(chunk["post_chunk"], chunk["policy_chunk"] * 2 + 1)
    if condition == "graph_serialized":
        assert result["wall"]["counts"].get("serialized_wait", 0) >= 2
        assert result["audit"]["overlaps"] == []
        requests = [op for op in control["opportunities"] if op["request_id"] == 3]
        assert len(requests) == 1
        first_after_wait = control["dispatches"][20]
        assert first_after_wait["slot"] > requests[0]["slot"]
    else:
        assert policy.release.is_set() and envs[0].overlap_reached
        assert result["audit"]["overlaps"]
        assert result["wall"]["counts"].get("serialized_wait", 0) == 0


def test_wait_uses_exact_request_and_holds_no_queue_or_request_lock(tmp_path, clock, fake_graph):
    class Inspected(e.NativeEngine):
        def wait_request(self, request_id, timeout=15):
            assert self._request_lock.acquire(blocking=False)
            self._request_lock.release()
            assert self.queue._lock.acquire(blocking=False)
            self.queue._lock.release()
            super().wait_request(request_id, timeout)
            assert request_id in self.completed

    result, *_ = episode(tmp_path, clock, gate=True, engine_class=Inspected)
    assert result["status"] == "completed", result["first_failure"]


def test_slow_env_skips_slots_without_catchup_and_wall_limit_is_normal(tmp_path, clock, fake_graph):
    result, *_ = episode(tmp_path, clock, options={"step_seconds": 0.12}, slots=6)
    assert result["status"] == "completed", result["first_failure"]
    assert result["terminal_reason"] == "wall_slot_limit"
    arrays = torch.load(tmp_path / "episode/arrays.pt", weights_only=True)
    assert [d["slot"] for d in arrays["control"]["dispatches"]] == [0, 2, 4]
    assert result["wall"]["total_slots"] == 6
    assert result["wall"]["no_action_slots"] == 3
    assert result["wall"]["counts"]["env_busy"] == 3


@pytest.mark.parametrize(
    "options,reason,success",
    [
        ({"success_at": 4, "truncate_at": 4}, "native_success", True),
        ({"truncate_at": 4}, "time_limit", False),
        ({}, "action_limit", False),
    ],
)
def test_success_has_priority_and_terminal_or_action_limit_stops_dispatch(
    tmp_path, clock, fake_graph, options, reason, success
):
    result, *_ = episode(tmp_path, clock, options=options, limit=4)
    assert result["status"] == "completed", result["first_failure"]
    assert result["terminal_reason"] == reason and result["success"] == success
    assert result["native_returned"]["measurement"] == 4


def test_none_keeps_index_and_feedback_without_native_steps(tmp_path, clock, fake_graph):
    spec = copy.deepcopy(e.fixed_manifest()["rows"][0])
    spec["condition"] = "graph_identity_async"
    spec["ready_wall_slots"] = 4
    policy = NativePolicy(clock, spec["condition"])
    calls = e.Calls(tmp_path / "calls.jsonl", clock)
    budget = e.Budget()
    budget.begin_episode()
    native = e.NativeSession(spec, budget, calls, lambda row: FakeEnv(row, policy, clock, {}))
    raw = native.create_reset()
    obs, features, initial = e.observation(raw, "A", 0, clock.now())
    engine = e.NativeEngine(policy, Processor(), Post(), features, spec, budget, calls, device="cpu")
    try:
        e.startup(engine, obs, clock)
        for _ in range(50):
            assert engine.get_action(None) is not None
        engine.pause()
        before = engine.queue.next_action_index
        records = [initial]
        control = e.measured_control(engine, native, spec, obs, records, clock=clock)
        assert control["dispatches"] == [] and len(control["gets"]) == 4
        assert all(g["outcome"] == "underflow" for g in control["gets"])
        assert engine.queue.next_action_index == before == 50
        assert native.returned["measurement"] == 0 and len(records) == 1
        assert {op["observation_index"] for op in control["opportunities"]} == {0}
    finally:
        engine.stop()
        native.close()
        calls.close()


def test_observations_own_both_images_and_preserve_exact_initial_pair(clock):
    spec = e.fixed_manifest()["rows"][0]
    env = FakeEnv(spec, NativePolicy(clock, "graph_serialized"), clock, {})
    raw = env.raw()
    obs, _, first = e.observation(raw, "A", 0, clock.now())
    _, _, same = e.observation(raw, "A", 0, clock.now())
    assert e.initial_difference(first, same) is None
    raw["pixels"]["image"].fill(23)
    raw["pixels"]["image2"].fill(24)
    assert not obs["image"].any() and not obs["image2"].any()
    _, _, changed = e.observation(raw, "A", 1, clock.now())
    assert e.initial_difference(first, changed) == "image"


def test_initial_pair_difference_stops_after_settling_before_model(tmp_path, clock, fake_graph):
    spec = e.fixed_manifest()["rows"][0]
    env = FakeEnv(spec, NativePolicy(clock, "graph_serialized"), clock, {})
    env.pixel = 10
    _, _, expected = e.observation(env.raw(), "A", 0, clock.now())
    expected["raw_pixels"]["image2"].fill_(123)
    result, _, policy, _, _ = episode(tmp_path, clock, paired_initial=expected)
    assert result["status"] == "technical_failure"
    assert result["initial_pair_exact"] is False and result["initial_difference"] == "image2"
    assert policy.calls == 0 and result["native_returned"] == {"settling": 10}
    assert result["environment_closed"]


def test_chunk_identity_does_not_depend_on_action_values(tmp_path, clock, fake_graph):
    class IdenticalPost(Processor):
        def __call__(self, value):
            super().__call__(value)
            return torch.zeros_like(value)

    result, *_ = episode(tmp_path, clock, gate=True, postprocessor=IdenticalPost())
    assert result["status"] == "completed", result["first_failure"]
    arrays = torch.load(tmp_path / "episode/arrays.pt", weights_only=True)
    dispatches = arrays["control"]["dispatches"]
    assert all(not row["command"].any() for row in dispatches)
    assert len({row["source_request_id"] for row in dispatches}) >= 2
    assert result["audit"]["planned_takeovers"] >= 1


def test_two_step_late_result_is_whole_discard():
    assert_whole_late_discard(2)


@pytest.mark.parametrize("kind", tuple(e.LIMITS))
def test_budget_rejects_next_dispatch_before_mutating_counts(kind):
    budget = e.Budget()
    local, total = e.LIMITS[kind]
    budget.episode[kind], budget.total[kind] = local, 0
    with pytest.raises(RuntimeError, match="before dispatch"):
        budget.take(kind)
    assert budget.episode[kind] == local and budget.total[kind] == 0
    budget.episode[kind], budget.total[kind] = 0, total
    with pytest.raises(RuntimeError, match="before dispatch"):
        budget.take(kind)
    assert budget.episode[kind] == 0 and budget.total[kind] == total


@pytest.mark.parametrize("kind", ["model", "measurement"])
def test_budget_intercepts_actual_model_or_native_call(tmp_path, clock, fake_graph, monkeypatch, kind):
    monkeypatch.setitem(e.LIMITS, kind, (0, 0))
    result, _, policy, _, calls = episode(tmp_path, clock)
    assert (
        result["status"] == "technical_failure"
        and "budget exhausted before dispatch" in result["first_failure"]
    )
    assert result["budget"].get(kind, 0) == 0
    assert result["native_returned"].get("measurement", 0) == 0
    if kind == "model":
        assert policy.calls == 0
    assert not any(
        r["event"] == "call_intent" and r["kind"] == "native_step" and r["segment"] == "measurement"
        for r in calls.records
    )


@pytest.mark.parametrize("failure", ["model", "nonfinite", "native", "slow_return"])
def test_first_error_stops_without_retry_and_preserves_native_return_status(
    tmp_path, clock, fake_graph, failure
):
    class Nonfinite(e.NativeEngine):
        def get_action(self, obs):
            value = super().get_action(obs)
            return value * float("nan")

    options = (
        {"native_error": True}
        if failure == "native"
        else {"step_seconds": 31}
        if failure == "slow_return"
        else {}
    )
    result, _, policy, _, calls = episode(
        tmp_path,
        clock,
        options=options,
        engine_class=Nonfinite if failure == "nonfinite" else e.NativeEngine,
        fail_model=failure == "model",
    )
    assert result["status"] == "technical_failure" and result["first_failure"]
    intents = [
        r
        for r in calls.records
        if r["event"] == "call_intent" and r["kind"] == "native_step" and r["segment"] == "measurement"
    ]
    assert len(intents) == (1 if failure in ("native", "slow_return") else 0)
    assert result["native_returned"].get("measurement", 0) == (1 if failure == "slow_return" else 0)
    assert result["worker_joined"] and result["environment_closed"]
    if failure == "model":
        assert policy.calls == 1
    if failure == "native":
        assert not any(
            r["event"] == "call_return" and r["call_id"] == intents[0]["call_id"] for r in calls.records
        )


def test_unconfirmed_join_never_serializes_shared_worker_arrays(tmp_path, clock, fake_graph):
    class Unconfirmed(e.NativeEngine):
        def stop(self):
            super().stop()
            self.worker_joined = False  # Exercise reporting of an unconfirmed cleanup.

    result, _, _, envs, _ = episode(tmp_path, clock, limit=2, engine_class=Unconfirmed)
    assert result["status"] == "technical_failure" and not result["worker_joined"]
    assert "arrays_not_serialized" in result and not (tmp_path / "episode/arrays.pt").exists()
    envs[0].close()


def test_fixed_manifest_rejects_expansion(tmp_path, monkeypatch):
    manifest = e.fixed_manifest()
    assert len(manifest["rows"]) == 20
    assert [row["initial_state_id"] for row in manifest["rows"]] == [41] * 20
    assert [row["condition"] for row in manifest["rows"][:4]] == [*e.CONDITIONS, *e.CONDITIONS[::-1]]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(e, "MANIFEST", path)
    assert e.read_manifest() == manifest
    manifest["rows"].append(manifest["rows"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="twenty-row"):
        e.read_manifest()


def test_actual_entry_import_only_and_help_without_cuda_or_model_work():
    directory = Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    source = "import runpy, sys, torch; sys.path.insert(0, sys.argv[1]); d = runpy.run_path(sys.argv[1] + '/libero_graph_identity_native.py', run_name='e_import_only'); assert len(d['fixed_manifest']()['rows']) == 20; assert not torch.cuda.is_initialized()"
    for command in (
        [sys.executable, "-c", source, str(directory)],
        [sys.executable, str(directory / "libero_graph_identity_native.py"), "--help"],
    ):
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
