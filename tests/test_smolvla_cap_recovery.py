"""E-L1 characterization and E-RCV1 contracts on the real NativeEngine/worker/queue."""

import copy
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_graph_cap_recovery_native as recovery  # noqa: E402
from test_smolvla_graph_identity_native import (  # noqa: E402
    ControlledClock,
    FakeEnv,
    NativePolicy,
    Post,
    Processor,
    e,
    fake_graph,  # noqa: F401
)

from lerobot.policies.rtc.latency_tracker import LatencyTracker  # noqa: E402
from lerobot.rollout.inference.latency_replay import compute_delay_plan, latency_to_steps  # noqa: E402

OLD_ACCEPTED = [
    0.2039953419007361,
    0.3028119750088081,
    0.21727405302226543,
    0.3505812670337036,
    0.957529005012475,
]
OLD_BOOTSTRAPS = [0.34502341505140066, 0.6467783220577985]


@pytest.fixture
def clock(monkeypatch):
    value = ControlledClock()
    monkeypatch.setattr(e.time, "perf_counter", value.now)
    return value


class TimedPolicy(NativePolicy):
    def __init__(self, clock, durations):
        super().__init__(clock, "graph_serialized")
        self.durations = durations
        self.block_call = self.fail_call = self.nan_call = None

    def predict_action_chunk(self, batch, **kwargs):
        index = self.calls
        if index == self.block_call:
            self.entered.set()
            assert self.release.wait(3), "CPU recovery probe was not released"
        if index == self.fail_call:
            raise ValueError("first controlled recovery failure")
        self.clock.sleep(self.durations.get(self.calls, 0.01) - 0.001)
        result = super().predict_action_chunk(batch, **kwargs)
        if index == self.nan_call:
            result[0, 0, 0] = float("nan")
        return result


def plan(engine, available=30):
    return compute_delay_plan(
        engine._latency_tracker,
        fps=20,
        latency_quantile=0.9,
        delay_safety_margin_steps=1,
        min_prediction_delay=0,
        max_prediction_delay=8,
        available_actions=available,
        committed_guard_steps=2,
    )


class CPUBudget(e.Budget):
    def check(self, kind):
        if kind == "measurement":
            if self.episode[kind] >= 2000:
                raise RuntimeError("Controlled CPU history construction exceeded 2000 fake steps")
        else:
            super().check(kind)


@contextmanager
def running_case(
    tmp_path,
    clock,
    *,
    durations=None,
    engine_class=e.NativeEngine,
    enabled=False,
    probe_limit=50,
    condition="graph_serialized",
):
    spec = copy.deepcopy(e.fixed_manifest()["rows"][1])
    spec["condition"] = condition
    if enabled:
        spec.update(recovery_policy=recovery.RECOVERY_POLICY, max_recovery_probes_per_episode=probe_limit)
        engine_class = recovery.RecoveryEngine
    policy = TimedPolicy(clock, durations or {})
    policy.condition = condition
    budget = CPUBudget()
    budget.begin_episode()
    calls = e.Calls(tmp_path / "calls.jsonl", clock)
    native = e.NativeSession(spec, budget, calls, lambda row: FakeEnv(row, policy, clock, {}))
    raw = native.create_reset()
    obs, features, _ = e.observation(raw, spec["task_name"].replace("_", " "), 0, clock.now())
    engine = engine_class(policy, Processor(), Post(), features, spec, budget, calls, device="cpu")
    case = SimpleNamespace(
        engine=engine, native=native, policy=policy, clock=clock, obs=obs, frame=0, trace=[]
    )
    try:
        e.startup(engine, obs, clock)
        assert engine.ready and [r["startup_phase"] for r in engine.rows] == [
            "cold_temporary",
            "probe",
            "fresh_warmed",
        ]
        yield case
    finally:
        history = list(engine._latency_tracker._values)
        engine.stop()
        closed = native.close()
        calls.close()
        evidence = {
            "requests": engine.rows,
            "metrics": engine.metrics.events,
            "trace": case.trace,
            "final_history": history,
            "native_steps": native.native_steps,
            "captures": engine.runtime.captures,
            "cleanup": {
                "worker_joined": engine.worker_joined,
                "graph_released": engine.graph_released,
                "sampler_restored": engine.original_sampler_restored,
                "metrics_closed": engine.metrics.closed,
                "environment_closed": closed,
            },
        }
        (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        assert all(evidence["cleanup"].values())


def consume(case, count):
    for _ in range(count):
        action = case.engine.get_action(None)
        assert action is not None
        get = case.engine.metrics.last_get
        raw, *_ = case.native.step(
            {
                "command": action.numpy().copy(),
                "slot": case.frame,
                "action_index": get["action_index"],
                "observation_index": case.frame,
            }
        )
        case.frame += 1
        case.obs, _, _ = e.observation(raw, case.engine.task_snapshot[0], case.frame, case.clock.now())


def submit(case):
    request_id = e.notify(case.engine, case.obs)
    if request_id is not None:
        case.engine.wait_request(request_id, timeout=3)
    return request_id


def slow_planned(case):
    consume(case, 20)
    assert submit(case) == 3
    assert case.engine.rows[-1]["kind"] == "planned"
    consume(case, 3)
    assert case.engine.metrics.last_get["outcome"] == "takeover"
    assert plan(case.engine).prediction_cap_exceeded


@pytest.mark.usefixtures("fake_graph")
def test_original_cap_latch_survives_two_fast_bootstraps(tmp_path, clock):
    with running_case(tmp_path, clock, durations={3: 1.0}) as case:
        engine = case.engine
        slow_planned(case)
        frozen = list(engine._latency_tracker._values)
        context = (engine._reset_epoch, engine.task_snapshot)
        for expected_id in (4, 5):
            consume(case, engine.queue.available_steps() - 30)
            before = engine.stats.requests_started
            assert submit(case) is None
            assert engine.metrics.events[-1]["decision"] == "cap_wait"
            assert engine.stats.requests_started == before
            consume(case, engine.queue.available_steps())
            assert submit(case) == expected_id
            event = engine.metrics.events[-1]
            assert event["request_kind"] == "bootstrap" and event["outcome"] == "installed"
            assert latency_to_steps(event["total_chunk_s"], 20) + 1 <= 8
            assert not event["latency_tracker_admitted"]
            assert list(engine._latency_tracker._values) == frozen
            assert plan(engine).prediction_cap_exceeded
            case.trace.append(
                {"bootstrap": expected_id, "history": frozen, "raw": plan(engine).raw_required_delay_steps}
            )
        assert engine.stats.planned_requests == 1 and not engine._request_in_flight
        assert context == (engine._reset_epoch, engine.task_snapshot)
        assert engine.peak_active_calls == 1 and not engine.failed
        assert len(engine.runtime.captures) == 2


@pytest.mark.usefixtures("fake_graph")
def test_positive_control_continues_planned_takeovers(tmp_path, clock):
    with running_case(tmp_path, clock) as case:
        for request_id in (3, 4, 5):
            consume(case, case.engine.queue.available_steps() - 30)
            assert submit(case) == request_id
            assert case.engine.rows[-1]["kind"] == "planned"
            consume(case, 3)
            assert case.engine.metrics.last_get["outcome"] == "takeover"
        assert case.engine.stats.planned_requests == 3
        assert len(case.engine._latency_tracker) == 4
        assert not plan(case.engine).prediction_cap_exceeded


def test_old_samples_recompute_raw16_without_inventing_bootstrap_admission(tmp_path):
    tracker = LatencyTracker(50)
    for value in OLD_ACCEPTED:
        tracker.add(value)
    estimate = tracker.percentile(0.9)
    assert estimate == 0.714749813079834
    assert latency_to_steps(estimate, 20) + 1 == 16
    # These two original bootstrap samples were excluded, including the individually fast first one.
    assert [latency_to_steps(v, 20) + 1 for v in OLD_BOOTSTRAPS] == [8, 14]
    assert list(tracker._values) == OLD_ACCEPTED
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {"history": OLD_ACCEPTED, "excluded": OLD_BOOTSTRAPS, "p90": estimate, "raw": 16}, indent=2
        )
    )


@pytest.mark.usefixtures("fake_graph")
def test_reset_and_task_boundaries_preserve_original_history(tmp_path, clock):
    with running_case(tmp_path, clock, durations={3: 1.0}) as case:
        slow_planned(case)
        engine = case.engine
        history = list(engine._latency_tracker._values)
        epoch, task = engine._reset_epoch, engine.task_snapshot
        engine.reset()
        assert engine._reset_epoch == epoch + 1 and engine.queue.available_steps() == 0
        assert list(engine._latency_tracker._values) == history
        assert engine.set_task("another task")
        assert engine.task_snapshot[1] == task[1] + 1
        assert list(engine._latency_tracker._values) == history
        assert engine.ready and engine._startup_phase == "complete"
        # A new NativeEngine, used by every native episode, constructs a new tracker.
        spec = case.native.spec
        fresh = e.NativeEngine(
            case.policy,
            Processor(),
            Post(),
            engine._hw_features,
            spec,
            case.native.budget,
            case.native.calls,
            device="cpu",
        )
        assert len(fresh._latency_tracker) == 0 and not fresh.ready
        fresh.stop()
        case.trace.append(
            {"reset_keeps_history": True, "task_change_keeps_history": True, "new_engine_empty_history": True}
        )


def make_planned(case, duration):
    consume(case, max(0, case.engine.queue.available_steps() - 30))
    case.policy.durations[case.policy.calls] = duration
    request_id = submit(case)
    assert request_id is not None and case.engine.rows[-1]["kind"] == "planned"
    consume(case, case.engine.rows[-1]["plan"]["planned_delay_steps"] + 1)
    assert case.engine.metrics.last_get["outcome"] == "takeover"
    return request_id


@pytest.mark.usefixtures("fake_graph")
def test_same_path_probe_preserves_active_staged_plan_and_admits_once(tmp_path, clock):
    with running_case(tmp_path, clock, durations={3: 1.0}, enabled=True) as case:
        engine = case.engine
        consume(case, 20)
        assert submit(case) == 3
        queue = engine.queue
        assert queue.has_staged_chunk()
        active, staged, pending = queue._active, queue._staged, queue._plan
        prefix = queue.peek_policy_actions(30)
        index, history = queue.next_action_index, list(engine._latency_tracker._values)
        assert submit(case) == 4
        assert queue._active is active and queue._staged is staged and queue._plan is pending
        assert queue.next_action_index == index and torch.equal(prefix, queue.peek_policy_actions(30))
        terminal = engine.metrics.events[-1]
        assert terminal["request_kind"] == "recovery_probe"
        assert terminal["outcome"] == "discarded_recovery_probe" and terminal["latency_tracker_admitted"]
        assert not terminal["policy_includes_vision"] and terminal["predictor_calls"] == 0
        assert terminal["plan_next_action_index"] is None and terminal["takeover_index"] is None
        assert list(engine._latency_tracker._values) == history + [terminal["total_chunk_s"]]
        row = engine.rows[-1]
        assert row["capture_id"] == 2 and row["replay_count"] == 1 and "plan" not in row
        assert row["cpu_chunks"] and row["device_completion_barrier"]
        assert all(row["api_counts"][k] == 1 for k in ("policy", "vision", "prepare_images", "noise", "post"))
        assert "prefix_4" not in engine.arrays and len(engine.runtime.captures) == 2
        consume(case, 3)
        assert engine.metrics.last_get["outcome"] == "takeover"
        assert torch.equal(
            engine.arrays["request_3"]["post_chunk"][0],
            torch.tensor(case.native.native_steps[-1]["action"], dtype=torch.float32),
        )


@pytest.mark.usefixtures("fake_graph")
def test_full_window_naturally_recovers_and_returns_to_planned_takeover(tmp_path, clock):
    with running_case(tmp_path, clock, enabled=True) as case:
        engine = case.engine
        # Fill the actual 50-member window through real completed planned requests.
        for _ in range(49):
            make_planned(case, 0.34)
        assert len(engine._latency_tracker) == 50 and not plan(engine).prediction_cap_exceeded
        for _ in range(6):
            make_planned(case, 0.4)
        assert len(engine._latency_tracker) == 50 and plan(engine).prediction_cap_exceeded
        high_history = list(engine._latency_tracker._values)
        consume(case, engine.queue.available_steps() - 30)
        case.policy.durations.clear()
        observations = []
        for attempt in range(1, 51):
            assert submit(case) is not None
            assert engine.rows[-1]["kind"] == "recovery_probe"
            observations.append(
                {
                    "attempt": attempt,
                    "raw": plan(engine).raw_required_delay_steps,
                    "history": list(engine._latency_tracker._values),
                }
            )
            if not plan(engine).prediction_cap_exceeded:
                break
        assert attempt == 45
        assert engine.stats.recovery_probe_requests == 45
        assert len(engine._latency_tracker) == 50
        assert sum(value > 0.35 for value in engine._latency_tracker._values) == 5
        # Slow samples remain; the original linear P90 now falls within cap naturally.
        planned = make_planned(case, 0.01)
        assert engine.rows[-1]["request_id"] == planned and engine.rows[-1]["kind"] == "planned"
        assert engine.metrics.last_get["outcome"] == "takeover"
        assert len(engine.runtime.captures) == 2 and not engine.failed
        case.trace.append(
            {
                "high_history": high_history,
                "recovery_samples": observations,
                "restored_planned_request": planned,
                "native_takeover_action_index": engine.metrics.last_get["action_index"],
            }
        )


@pytest.mark.usefixtures("fake_graph")
def test_sustained_slow_probes_exhaust_budget_without_false_recovery(tmp_path, clock):
    with running_case(tmp_path, clock, durations={3: 1.0}, enabled=True) as case:
        slow_planned(case)
        engine = case.engine
        consume(case, engine.queue.available_steps() - 30)
        for _ in range(50):
            case.policy.durations[case.policy.calls] = 0.5
            assert submit(case) is not None
            assert engine.rows[-1]["kind"] == "recovery_probe"
            assert plan(engine).prediction_cap_exceeded
        before = case.policy.calls
        assert engine.stats.recovery_probe_requests == 50
        assert submit(case) is None and case.policy.calls == before
        assert engine.metrics.events[-1]["decision"] == "cap_wait"
        history = list(engine._latency_tracker._values)
        assert len(history) == 50 and all(value == pytest.approx(0.5) for value in history)
        consume(case, engine.queue.available_steps())
        assert submit(case) is not None
        assert engine.rows[-1]["kind"] == "bootstrap"
        assert not engine.metrics.events[-1]["latency_tracker_admitted"]
        assert list(engine._latency_tracker._values) == history
        assert engine.stats.recovery_probe_requests == 50 and engine.stats.planned_requests == 1
        engine.reset()
        engine.set_task("task B")
        assert engine.stats.recovery_probe_requests == 50
        case.trace.append({"exhausted_at": 50, "raw_after": plan(engine).raw_required_delay_steps})


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("boundary", ["reset", "task", "stop"])
def test_inflight_probe_does_not_admit_after_context_invalidation(tmp_path, clock, boundary):
    with running_case(tmp_path, clock, durations={3: 1.0}, enabled=True) as case:
        slow_planned(case)
        engine = case.engine
        consume(case, engine.queue.available_steps() - 30)
        history = list(engine._latency_tracker._values)
        case.policy.block_call = case.policy.calls
        request_id = e.notify(engine, case.obs)
        assert case.policy.entered.wait(3)
        assert engine.current["kind"] == "recovery_probe"
        stopper = None
        try:
            if boundary == "reset":
                engine.reset()
            elif boundary == "task":
                engine.set_task("new task")
            else:
                stopper = threading.Thread(target=engine.stop)
                stopper.start()
                assert engine._shutdown_event.wait(2)
        finally:
            case.policy.release.set()
        engine.wait_request(request_id, timeout=3)
        if stopper is not None:
            stopper.join(3)
            assert not stopper.is_alive()
        assert list(engine._latency_tracker._values) == history
        terminal = next(
            v
            for v in engine.metrics.events
            if v.get("request_id") == request_id and v["event"] == "chunk_request"
        )
        assert terminal["outcome"] == "stale_recovery_probe" and not terminal["latency_tracker_admitted"]
        assert engine.stats.stale_results == 1 and not engine.failed
        assert len(engine.runtime.captures) == 2


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("failure", ["policy", "nonfinite"])
def test_first_recovery_error_is_fatal_and_not_admitted(tmp_path, clock, failure):
    with running_case(tmp_path, clock, durations={3: 1.0}, enabled=True) as case:
        slow_planned(case)
        engine = case.engine
        consume(case, engine.queue.available_steps() - 30)
        history = list(engine._latency_tracker._values)
        setattr(case.policy, "fail_call" if failure == "policy" else "nan_call", case.policy.calls)
        request_id = e.notify(engine, case.obs)
        with pytest.raises(RuntimeError, match="controlled recovery failure|must be finite"):
            engine.wait_request(request_id, timeout=3)
        assert list(engine._latency_tracker._values) == history
        engine.stop()
        before = engine.stats.requests_started
        engine.notify_observation(case.obs)
        assert engine.stats.requests_started == before
        assert not next(v for v in engine.metrics.events if v["event"] == "request_error")[
            "latency_tracker_admitted"
        ]


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("condition", e.CONDITIONS)
def test_original_controller_waits_for_serialized_probe_only(tmp_path, clock, condition):
    with running_case(tmp_path, clock, durations={3: 1.0}, enabled=True, condition=condition) as case:
        engine = case.engine
        consume(case, 20)
        assert submit(case) == 3
        assert plan(engine).prediction_cap_exceeded
        # The original controller starts with a live old active+staged queue and a cap latch.
        if condition == "graph_identity_async":
            case.policy.block_call = case.policy.calls
            original_step = case.native.env.step

            def step(command):
                assert case.policy.entered.wait(3), "Controller did not reach Env during the probe"
                case.policy.release.set()
                return original_step(command)

            case.native.env.step = step
        spec = copy.deepcopy(case.native.spec)
        spec["limits"]["measurement"] = 2
        records = []
        control = e.measured_control(engine, case.native, spec, case.obs, records, clock=clock)
        case.policy.release.set()
        engine.wait_request(4, timeout=3)
        assert engine.rows[4]["kind"] == "recovery_probe"
        assert len(control["dispatches"]) == 2
        waits = [b for b in control["blocks"] if b["kind"] == "serialized_wait"]
        assert bool(waits) == (condition == "graph_serialized")
        assert any(o["request_id"] == 4 for o in control["opportunities"])
        case.trace.append({"condition": condition, "blocks": control["blocks"]})


def test_fixed_four_case_manifest_and_budgets(tmp_path):
    manifest = recovery.fixed_manifest()
    assert [(r["task_id"], r["condition"]) for r in manifest["rows"]] == [
        (0, "graph_serialized"),
        (0, "graph_identity_async"),
        (2, "graph_identity_async"),
        (2, "graph_serialized"),
    ]
    assert [r["environment_seed"] for r in manifest["rows"]] == [940041, 940041, 940241, 940241]
    assert [r["policy_seed"] for r in manifest["rows"]] == [950041, 950041, 950241, 950241]
    assert all(
        r["control"]["max_prediction_delay"] == 8 and r["max_recovery_probes_per_episode"] == 50
        for r in manifest["rows"]
    )
    budget = recovery.RecoveryBudget()
    for _ in range(4):
        budget.begin_episode()
    with pytest.raises(RuntimeError, match="episodes budget"):
        budget.begin_episode()
    for kind, (local, _) in recovery.LIMITS.items():
        if kind != "episodes":
            budget.episode[kind] = local
            with pytest.raises(RuntimeError, match=kind + " budget"):
                budget.take(kind)


def test_actual_recovery_entry_import_and_help_without_inference():
    directory = Path(recovery.__file__).parent
    code = "import sys,torch; sys.path.insert(0,sys.argv[1]); import libero_graph_cap_recovery_native; assert not torch.cuda.is_initialized(); print('import-only without CUDA initialization')"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    imported = subprocess.run(
        [sys.executable, "-c", code, str(directory)], text=True, capture_output=True, env=env
    )
    assert imported.returncode == 0, imported.stderr
    help_result = subprocess.run(
        [sys.executable, str(recovery.__file__), "--help"], text=True, capture_output=True, env=env
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--execution-head" in help_result.stdout and "--output" in help_result.stdout
    assert "--worker" not in help_result.stdout and "--resume" not in help_result.stdout


@pytest.mark.usefixtures("fake_graph")
def test_failed_env_close_prevents_array_serialization(tmp_path, clock, monkeypatch):
    from test_smolvla_graph_identity_native import episode

    def failed_close(self):
        raise RuntimeError("controlled Env close failure")

    monkeypatch.setattr(FakeEnv, "close", failed_close)
    result, _, _, environments, _ = episode(tmp_path, clock, limit=1, engine_class=recovery.RecoveryEngine)
    assert result["status"] == "technical_failure"
    assert "controlled Env close failure" in result["first_failure"]
    assert result["worker_joined"] and result["graph_released"]
    assert result["environment_closed"] is not True
    assert "arrays_not_serialized" in result and not (tmp_path / "episode/arrays.pt").exists()
    environments[0]._env = None
