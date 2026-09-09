"""E-S1 boundary tests use the original gate, tracker, real worker and queue."""

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_graph_startup_boundary as s  # noqa: E402
from test_smolvla_graph_identity_native import (  # noqa: E402
    ControlledClock,
    FakeEnv,
    NativePolicy,
    Post,
    Processor,
    fake_graph,  # noqa: F401
)

from lerobot.policies.rtc.latency_tracker import LatencyTracker  # noqa: E402
from lerobot.rollout.inference.latency_replay import compute_delay_plan, latency_to_steps  # noqa: E402

OLD_METRICS = json.loads(s.OLD_REPORT.read_text())["first_failure"]["probe_metrics"]


@pytest.fixture
def clock(monkeypatch):
    value = ControlledClock()
    monkeypatch.setattr(s.time, "perf_counter", value.now)
    return value


def make_engine(tmp_path, clock):
    spec = s.fixed_manifest()["cases"][0]
    policy = NativePolicy(clock, spec["condition"])
    env = FakeEnv(spec, policy, clock, {})
    _, features, _ = s.e.observation(env.raw(), "A", 0, clock.now())
    calls = s.e.Calls(tmp_path / "gate_calls.jsonl", clock)
    engine = s.BoundaryEngine(
        policy, Processor(), Post(), features, spec, s.StartupBudget(), calls, device="cpu"
    )
    return engine, calls


def test_original_failure_sample_is_rejected_by_original_gate(tmp_path, clock):
    engine, calls = make_engine(tmp_path, clock)
    metrics = copy.deepcopy(OLD_METRICS)
    latency = metrics["total_chunk_s"]
    assert latency == metrics["cuda_completed_at_s"] - metrics["requested_at_s"]
    try:
        with pytest.raises(RuntimeError, match="requires 9 delay steps, exceeding runtime cap 8"):
            engine._validate_startup_probe(metrics, latency_s=latency, actions_finite=True)
        assert metrics["startup_gate_raw_required_delay_steps"] == 9
        assert metrics["startup_gate_outcome"] == "cap_exceeded"
        assert len(engine._latency_tracker) == 0
    finally:
        calls.close()


@pytest.mark.parametrize(
    "latency,required", [(0.349999, 8), (0.35, 8), (0.35000001, 8), (0.350001, 9), (0.4, 9)]
)
def test_original_rounding_and_unclamped_gate_boundary(tmp_path, clock, latency, required):
    engine, calls = make_engine(tmp_path, clock)
    metrics = copy.deepcopy(OLD_METRICS)
    try:
        if required > 8:
            with pytest.raises(RuntimeError, match="exceeding runtime cap 8"):
                engine._validate_startup_probe(metrics, latency_s=latency, actions_finite=True)
        else:
            engine._validate_startup_probe(metrics, latency_s=latency, actions_finite=True)
        assert metrics["startup_gate_raw_required_delay_steps"] == required
        assert required == latency_to_steps(latency, 20) + 1
        assert engine._max_prediction_delay == 8
    finally:
        calls.close()


def test_actual_tracker_quantile_units_window_and_clamp_order():
    tracker = LatencyTracker(maxlen=2)
    control = {
        k: s.e.CONTROL[k]
        for k in (
            "fps",
            "latency_quantile",
            "delay_safety_margin_steps",
            "min_prediction_delay",
            "max_prediction_delay",
            "committed_guard_steps",
        )
    }
    assert compute_delay_plan(tracker, available_actions=30, **control) is None
    tracker.add(0.1)
    assert len(tracker) == 1 and tracker.percentile(0.9) == float(np.float32(0.1))
    tracker.add(0.3)
    assert tracker.percentile(0.9) == pytest.approx(0.28, abs=1e-7)
    assert tracker.percentile(0.9) != 0.3  # Linear interpolation, not report nearest-rank.
    tracker.add(0.35)
    assert len(tracker) == 2 and list(tracker._values) == [0.3, 0.35]
    tracker.reset()
    tracker.add(OLD_METRICS["total_chunk_s"])
    plan = compute_delay_plan(tracker, available_actions=30, **control)
    assert plan.raw_required_delay_steps == 9
    assert plan.planned_delay_steps == 8 and plan.prediction_cap_exceeded
    assert plan.available_after_guard_steps == 28
    assert latency_to_steps(float(np.float32(0.05)), 20) == 1
    assert latency_to_steps(0.050001, 20) == 2


def run_case(
    tmp_path,
    clock,
    monkeypatch,
    *,
    probe_latency=None,
    fail_model=False,
    mismatch=False,
    engine_class=s.BoundaryEngine,
):
    class Policy(NativePolicy):
        def predict_action_chunk(self, batch, **kwargs):
            if self.calls == 1 and probe_latency is not None:
                clock.sleep(probe_latency - 0.001)
            return super().predict_action_chunk(batch, **kwargs)

    spec = s.fixed_manifest()["cases"][0]
    policy = Policy(clock, spec["condition"], fail=fail_model)
    expected = FakeEnv(spec, policy, clock, {})
    expected.pixel = 10
    _, _, initial = s.e.observation(expected.raw(), "A", 0, clock.now())
    if mismatch:
        initial["raw_pixels"]["image2"].fill_(32)
    envs = []

    def factory(row):
        env = FakeEnv(row, policy, clock, {})
        envs.append(env)
        return env

    seed_calls = []
    original_seed = torch.manual_seed

    def seed(value):
        seed_calls.append((value, threading.get_ident()))
        return original_seed(value)

    monkeypatch.setattr(torch, "manual_seed", seed)
    calls = s.e.Calls(tmp_path / "calls.jsonl", clock)
    try:
        result = s.run_startup_only(
            policy,
            Processor(),
            Post(),
            factory,
            spec,
            tmp_path / "case",
            calls,
            initial,
            clock=clock,
            device="cpu",
            engine_class=engine_class,
        )
    finally:
        calls.close()
    return result, policy, calls, envs, seed_calls


@pytest.mark.usefixtures("fake_graph")
def test_real_startup_passes_then_stops_without_get_or_measured_dispatch(tmp_path, clock, monkeypatch):
    result, policy, calls, _, seeds = run_case(tmp_path, clock, monkeypatch)
    assert result["status"] == "startup_passed", result["first_failure"]
    assert result["startup_gate_passed_this_run"] and result["startup_ready_this_run"]
    assert result["cleanup_confirmed"] and result["initial_exact"]
    assert result["native_returned"] == {"settling": 10}
    assert result["queue_get_count"] == 0 and result["measured_native_dispatches"] == 0
    assert policy.calls == result["budget"]["model"] == 3 and result["budget"]["capture"] == 2
    assert seeds == [(950341, result["owner_thread"])]
    assert [r["startup_phase"] for r in result["requests"]] == ["cold_temporary", "probe", "fresh_warmed"]
    assert [r["reset_epoch"] for r in result["requests"]] == [0, 0, 1]
    assert {r["observation_index"] for r in result["requests"]} == {0}
    before = [b for b in result["boundary_events"] if b["event"] == "before_original_request"]
    assert [len(b["tracker_samples_seconds"]) for b in before] == [0, 0, 1]
    assert before[2]["tracker_samples_seconds"] == [result["metrics"][1]["total_chunk_s"]]
    assert all(r["cpu_chunks"] and r["device_completion_barrier"] for r in result["requests"])
    assert all(
        c["eager_setup_calls"] == 1 and c["side_stream_warmup_calls"] == 3 and c["capture_calls"] == 1
        for c in result["captures"]
    )
    assert not any(v.get("kind") == "environment_step" for v in calls.records)
    assert (tmp_path / "case/arrays.pt").exists()


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("failure", ["gate", "model", "initial"])
def test_first_failure_has_no_followup_model_native_or_seed(tmp_path, clock, monkeypatch, failure):
    result, policy, _, _, seeds = run_case(
        tmp_path,
        clock,
        monkeypatch,
        probe_latency=OLD_METRICS["total_chunk_s"] if failure == "gate" else None,
        fail_model=failure == "model",
        mismatch=failure == "initial",
    )
    assert result["status"] == ("gate_rejected" if failure == "gate" else "technical_failure")
    assert result["cleanup_confirmed"] and result["first_failure"]
    assert result["native_returned"] == {"settling": 10}
    assert result["measured_native_dispatches"] == 0
    assert policy.calls == {"gate": 2, "model": 1, "initial": 0}[failure]
    assert len(seeds) == (0 if failure == "initial" else 1)
    if failure == "gate":
        assert result["required_steps_unclamped"] == 9
        assert result["startup_gate_passed_this_run"] is False
        assert result["budget"]["capture"] == 1 and result["queue_get_count"] == 0
        assert all(not event["tracker_samples_seconds"] for event in result["boundary_events"])


@pytest.mark.usefixtures("fake_graph")
def test_actual_main_dispatch_budget_stops_before_third_model(tmp_path, clock, monkeypatch):
    monkeypatch.setitem(s.LIMITS, "model", 2)
    result, policy, *_ = run_case(tmp_path, clock, monkeypatch)
    assert (
        result["status"] == "technical_failure"
        and "budget exhausted before dispatch" in result["first_failure"]
    )
    assert policy.calls == 2 and result["budget"]["model"] == 2
    assert result["native_returned"] == {"settling": 10} and result["cleanup_confirmed"]


def test_zero_measured_budget_blocks_underlying_native_call(tmp_path, clock):
    spec = s.fixed_manifest()["cases"][0]
    policy = NativePolicy(clock, spec["condition"])
    budget = s.StartupBudget()
    budget.begin_episode()
    calls = s.e.Calls(tmp_path / "calls.jsonl", clock)
    native = s.e.NativeSession(spec, budget, calls, lambda row: FakeEnv(row, policy, clock, {}))
    try:
        native.create_reset()
        with pytest.raises(RuntimeError, match="measurement budget exhausted before dispatch"):
            native.env._env.step(np.zeros(7))
        assert native.env.pixel == 10 and native.returned == {"settling": 10}
        assert not any(v.get("segment") == "measurement" for v in calls.records)
    finally:
        native.close()
        calls.close()


@pytest.mark.usefixtures("fake_graph")
def test_unconfirmed_join_does_not_save_arrays_or_report_cleanup(tmp_path, clock, monkeypatch):
    class Unconfirmed(s.BoundaryEngine):
        def stop(self):
            super().stop()
            self.worker_joined = False

    result, _, _, envs, _ = run_case(tmp_path, clock, monkeypatch, engine_class=Unconfirmed)
    assert result["status"] == "cleanup_failure" and not result["cleanup_confirmed"]
    assert not (tmp_path / "case/arrays.pt").exists()
    envs[0].close()


def test_fixed_single_case_rejects_expansion_and_parameter_changes(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    monkeypatch.setattr(s, "MANIFEST", path)
    original = s.fixed_manifest()
    path.write_text(json.dumps(original))
    assert s.read_manifest()["limits"] == {
        "episodes": 1,
        "settling": 10,
        "measurement": 0,
        "model": 3,
        "capture": 2,
    }
    for mutation in ("second_case", "cap", "seed"):
        changed = copy.deepcopy(original)
        if mutation == "second_case":
            changed["cases"].append(changed["cases"][0])
        elif mutation == "cap":
            changed["cases"][0]["control"]["max_prediction_delay"] = 9
        else:
            changed["cases"][0]["policy_seed"] += 1
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="one-case"):
            s.read_manifest()


def test_actual_entry_import_only_and_help_without_env_or_model():
    directory = Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    source = "import sys,torch; sys.path.insert(0,sys.argv[1]); import libero_graph_startup_boundary as s; assert len(s.fixed_manifest()['cases'])==1; assert not torch.cuda.is_initialized()"
    for command in (
        [sys.executable, "-c", source, str(directory)],
        [sys.executable, str(directory / "libero_graph_startup_boundary.py"), "--help"],
    ):
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
