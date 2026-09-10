"""New E-NAT1 boundaries only; existing recovery/trace contracts remain accepted."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_graph_natural_native as natural  # noqa: E402
import test_smolvla_cap_recovery as base  # noqa: E402
from test_smolvla_cap_recovery import fake_graph  # noqa: E402, F401

clock = base.clock


def test_fixed_twenty_cases_keep_seeds_controls_and_enable_same_recovery_in_both_arms():
    original, new = base.e.fixed_manifest(), natural.manifest()
    assert new["experiment"] == "E-NAT1" and new["intervention"] == "none"
    assert len(new["rows"]) == 20 and new["global_limits"] == original["global_limits"]
    assert new["attempts"] == 1 and new["retries"] == 0
    for before, after in zip(original["rows"], new["rows"], strict=True):
        assert all(after[key] == value for key, value in before.items())
        assert after["recovery_policy"] == base.recovery.RECOVERY_POLICY
        assert after["max_recovery_probes_per_episode"] == 50
        assert "host_pause_seconds" not in after
    for offset in range(0, 20, 2):
        left, right = new["rows"][offset : offset + 2]
        assert {left["condition"], right["condition"]} == set(base.e.CONDITIONS)
        for key in ("task_id", "initial_state_id", "environment_seed", "policy_seed", "control"):
            assert left[key] == right[key]


class CPUEngine(natural.NaturalEngine):
    def __init__(self, policy, pre, post, features, spec, budget, calls, **kwargs):
        spec = copy.deepcopy(spec)
        spec.update(recovery_policy=base.recovery.RECOVERY_POLICY, max_recovery_probes_per_episode=50)
        super().__init__(policy, pre, post, features, spec, budget, calls, **kwargs)


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("condition", base.e.CONDITIONS)
def test_real_cpu_worker_plans_without_injected_pause(tmp_path, clock, monkeypatch, condition):
    def forbidden_pause(_seconds):
        raise AssertionError("Natural experiment reached the stress injection")

    monkeypatch.setattr(natural.trace, "pause", forbidden_pause)
    with base.running_case(tmp_path, clock, engine_class=CPUEngine, condition=condition) as case:
        base.consume(case, 20)
        assert base.submit(case) == 3
        assert case.engine.rows[-1]["kind"] == "planned"
        assert not base.plan(case.engine).prediction_cap_exceeded
        assert case.engine.pause_count == 0
        base.consume(case, 3)
        assert case.engine.metrics.last_get["outcome"] == "takeover"
    events = natural.trace.read_events(tmp_path / "calls.jsonl")
    assert not any(event.get("phase") == "host_pause" for event in events)
    assert all("host_intervention" not in row for row in case.engine.rows)
    accounting = natural.trace.journal_accounting(events)
    assert accounting["journal_consistent"] and not accounting["unknown_calls"]
    assert accounting["request_snapshots"] == 4


@pytest.mark.usefixtures("fake_graph")
def test_real_cpu_slow_policy_keeps_recovery_without_host_pause(tmp_path, clock):
    with base.running_case(tmp_path, clock, durations={3: 1.0}, engine_class=CPUEngine) as case:
        base.slow_planned(case)
        slow_sample = list(case.engine._latency_tracker._values)[-1]
        base.consume(case, case.engine.queue.available_steps() - 30)
        for _ in range(50):
            if not base.plan(case.engine).prediction_cap_exceeded:
                break
            assert base.submit(case) is not None
            assert case.engine.rows[-1]["kind"] == "recovery_probe"
        assert not base.plan(case.engine).prediction_cap_exceeded
        assert slow_sample in case.engine._latency_tracker._values
        assert base.submit(case) is not None
        base.consume(case, case.engine.rows[-1]["plan"]["planned_delay_steps"] + 1)
        assert case.engine.metrics.last_get["outcome"] == "takeover"
        assert case.engine.pause_count == 0


def test_unknown_or_missing_worker_receipt_cannot_pass_new_contract():
    declared = natural.manifest()
    records = [{"spec": row, "status": "not_run"} for row in declared["rows"]]
    result = natural.summarize(records, declared)
    expired = {"event": "call_intent", "call_id": 1, "kind": "model_request", "ordinal": 0}
    execution = {"stop_reason": {"expired_calls": [expired]}, "child_exit_code": -9}
    result = natural.finalize(result, execution, natural.trace.journal_accounting([expired]), {})
    assert result["status"] == "technical_failure"
    assert not result["nat_native_closed_loop_contract_passed"]
    assert not result["nat_paired_scheduling_comparison_complete"]
    assert result["budget"] is None and result["unknown_calls"] == [expired]
    assert result["first_failure"]["stop_reason"] == execution["stop_reason"]
    assert not result["baseline_qualified"] and not result["predictor_benefit_tested"]
    assert not result["independent_cpu_audit_completed"]
