"""E-RCV3 observability contracts; real CPU worker and queue, no native/GPU claim."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_graph_cap_trace_native as trace  # noqa: E402
import test_smolvla_cap_recovery as base  # noqa: E402
from test_smolvla_cap_recovery import fake_graph  # noqa: E402, F401

clock = base.clock


def cpu_engine(arm):
    class Engine(trace.TraceEngine):
        def __init__(self, policy, pre, post, features, spec, budget, calls, **kwargs):
            spec = copy.deepcopy(spec)
            spec.update(
                arm=arm,
                recovery_policy=base.recovery.RECOVERY_POLICY if arm == "candidate" else "disabled",
                host_pause_seconds=0.6,
            )
            super().__init__(policy, pre, post, features, spec, budget, calls, **kwargs)

    return Engine


def test_manifest_keeps_original_four_cases_limits_and_marks_independent_experiment():
    original, new = trace.stress.read_manifest(), trace.manifest()
    for key in ("rows", "limits", "policy_revision", "vlm_revision", "assets_revision"):
        assert new[key] == original[key]
    assert new["experiment"] == "E-RCV3-trace"
    assert new["predecessor_is_terminal"] and not new["replaces_or_completes_predecessor"]
    assert new["attempts"] == 1 and new["retries"] == 0


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("arm", ["disabled", "candidate"])
def test_phase_traces_preserve_original_latch_or_recovery(tmp_path, clock, monkeypatch, arm):
    monkeypatch.setattr(trace, "pause", clock.sleep)
    with base.running_case(tmp_path, clock, engine_class=cpu_engine(arm)) as case:
        engine = case.engine
        assert engine.pause_count == 0
        base.slow_planned(case)
        assert engine.pause_count == 1
        frozen = list(engine._latency_tracker._values)
        assert frozen[-1] == pytest.approx(0.61)
        base.consume(case, engine.queue.available_steps() - 30)
        if arm == "disabled":
            assert base.submit(case) is None
            base.consume(case, engine.queue.available_steps())
            assert base.submit(case) is not None
            assert engine.rows[-1]["kind"] == "bootstrap"
            assert list(engine._latency_tracker._values) == frozen
        else:
            for _ in range(50):
                if not base.plan(engine).prediction_cap_exceeded:
                    break
                assert base.submit(case) is not None
                assert engine.rows[-1]["kind"] == "recovery_probe"
            assert not base.plan(engine).prediction_cap_exceeded
            assert frozen[-1] in engine._latency_tracker._values
            assert base.submit(case) is not None
            assert engine.rows[-1]["kind"] == "planned"
            base.consume(case, engine.rows[-1]["plan"]["planned_delay_steps"] + 1)
            assert engine.metrics.last_get["outcome"] == "takeover"
        assert engine.pause_count == 1
    events = trace.read_events(tmp_path / "calls.jsonl")
    accounting = trace.journal_accounting(events)
    assert not accounting["unknown_calls"] and accounting["journal_consistent"]
    assert accounting["request_snapshots"] == len(engine.rows)
    pauses = [v for v in events if v.get("phase") == "host_pause"]
    assert [v["boundary"] for v in pauses] == ["started", "returned"]
    assert all(v["request_id"] == 3 for v in pauses)
    assert pauses[-1]["intervention"]["elapsed_seconds"] == pytest.approx(0.6)
    snapshots = [v for v in events if v["event"] == "request_snapshot"]
    assert all(v["terminal"] is not None for v in snapshots)
    assert snapshots[3]["terminal"]["latency_tracker_admitted"]
    assert snapshots[3]["row"]["recovery"]["history_after"]["samples_seconds"] == frozen
    cleanup = next(v for v in events if v["event"] == "engine_cleanup")
    assert all(
        cleanup[k] for k in ("worker_joined", "graph_released", "original_sampler_restored", "metrics_closed")
    )


def test_failed_phase_is_flushed_and_original_error_propagates(tmp_path):
    calls = base.e.Calls(tmp_path / "calls.jsonl")
    engine = object.__new__(trace.TraceEngine)
    engine.calls, engine.spec = calls, {"ordinal": 3}
    engine.trace_request = SimpleNamespace(request_id=3, reset_epoch=1, task_epoch=0)
    with pytest.raises(ValueError, match="original failure"), engine.traced("vision"):
        raise ValueError("original failure")
    calls.close()
    events = trace.read_events(tmp_path / "calls.jsonl")
    assert [v["boundary"] for v in events] == ["started", "error"]
    assert events[-1]["phase"] == "vision" and "original failure" in events[-1]["exception"]


def test_journal_distinguishes_error_from_unknown_and_rejects_duplicate_terminal():
    events = [
        {"event": "call_intent", "call_id": 0, "kind": "model_request"},
        {"event": "call_intent", "call_id": 1, "kind": "native_step", "segment": "measurement"},
        {"event": "call_error", "call_id": 0},
    ]
    result = trace.journal_accounting(events)
    assert [v["call_id"] for v in result["unknown_calls"]] == [1]
    assert result["call_errors"] == {"model_request": 1}
    assert result["measured_native_returns"] == 0
    events.append({"event": "call_return", "call_id": 1})
    assert trace.journal_accounting(events)["measured_native_returns"] == 1
    events.append({"event": "call_return", "call_id": 1})
    assert not trace.journal_accounting(events)["journal_consistent"]


def test_missing_worker_receipt_cannot_erase_timeout_or_report_zero_budget():
    declared = trace.manifest()
    result = trace.summarize([{"spec": v, "status": "not_run"} for v in declared["rows"]], declared)
    expired = {"event": "call_intent", "call_id": 1251, "kind": "model_request", "ordinal": 3}
    execution = {"stop_reason": {"expired_calls": [expired]}, "child_exit_code": -9}
    result = trace.finalize(result, execution, trace.journal_accounting([expired]), {})
    assert result["status"] == "technical_failure"
    assert result["first_failure"]["stop_reason"] == execution["stop_reason"]
    assert result["budget"] is None and not result["budget_complete"]
    assert not result["trace_four_episode_contract_passed"]
    assert not result["trace_mechanism_contrast_passed"]
    assert not result["natural_latency_recovery_demonstrated"]
    assert not result["baseline_qualified"] and result["risk_thresholds"] is None


@pytest.mark.usefixtures("fake_graph")
def test_actual_initial_checkpoint_before_model_without_extra_reset(tmp_path, clock):
    spec = copy.deepcopy(trace.manifest()["rows"][0])
    spec["limits"]["measurement"] = 2
    policy = base.NativePolicy(clock, "graph_serialized")
    calls = base.e.Calls(tmp_path / "calls.jsonl", clock)
    factory = trace.checkpoint_factory(lambda row: base.FakeEnv(row, policy, clock, {}), calls)
    result, initial = base.e.run_episode(
        spec,
        tmp_path / "episode_000",
        policy,
        base.Processor(),
        base.Post(),
        factory,
        base.e.Budget(),
        calls,
        engine_class=cpu_engine("disabled"),
        clock=clock,
        device="cpu",
    )
    calls.close()
    assert result["status"] == "completed", result["first_failure"]
    saved = torch.load(tmp_path / "episode_000/initial_checkpoint.pt", map_location="cpu", weights_only=False)
    assert base.e.initial_difference(saved, initial) is None
    events = trace.read_events(tmp_path / "calls.jsonl")
    checkpoint_index = next(i for i, v in enumerate(events) if v["event"] == "initial_checkpoint")
    first_model = next(i for i, v in enumerate(events) if v.get("kind") == "model_request")
    assert checkpoint_index < first_model
    assert sum(v["event"] == "initial_checkpoint" for v in events) == 1
    assert result["native_returned"] == {"settling": 10, "measurement": 2}
