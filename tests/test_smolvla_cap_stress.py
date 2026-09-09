"""Diagnostic host intervention on the existing CPU NativeEngine worker fixture."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_graph_cap_stress_native as stress  # noqa: E402
import test_smolvla_cap_recovery as base  # noqa: E402
from test_smolvla_cap_recovery import fake_graph  # noqa: E402, F401

clock = base.clock


def cpu_engine(arm):
    class Engine(stress.StressEngine):
        def __init__(self, policy, pre, post, features, spec, budget, calls, **kwargs):
            spec = copy.deepcopy(spec)
            spec.update(
                arm=arm,
                recovery_policy=base.recovery.RECOVERY_POLICY if arm == "candidate" else "disabled",
                host_pause_seconds=0.6,
            )
            super().__init__(policy, pre, post, features, spec, budget, calls, **kwargs)

    return Engine


def test_fixed_stress_manifest_has_balanced_order_and_frozen_source_identity():
    manifest = stress.read_manifest()
    rows = manifest["rows"]
    assert [r["task_id"] for r in rows] == [0, 0, 2, 2]
    assert [r["arm"] for r in rows] == ["disabled", "candidate", "candidate", "disabled"]
    assert all(r["condition"] == "graph_identity_async" and r["host_pause_seconds"] == 0.6 for r in rows)
    assert all(
        r["control"] == base.e.CONTROL and r["limits"] == base.e.fixed_manifest()["rows"][0]["limits"]
        for r in rows
    )
    for left, right in (rows[:2], rows[2:]):
        assert all(
            left[k] == right[k] for k in ("task_name", "initial_state_id", "environment_seed", "policy_seed")
        )
    assert sum(r["max_recovery_probes_per_episode"] for r in rows if r["arm"] == "candidate") == 100


@pytest.mark.usefixtures("fake_graph")
@pytest.mark.parametrize("arm", ["disabled", "candidate"])
def test_one_host_pause_with_real_worker_history_and_recovery(tmp_path, clock, monkeypatch, arm):
    monkeypatch.setattr(stress, "pause", clock.sleep)
    with base.running_case(tmp_path, clock, engine_class=cpu_engine(arm)) as case:
        engine = case.engine
        assert engine.pause_count == 0
        base.slow_planned(case)
        assert engine.pause_count == 1
        first = engine.rows[-1]
        assert first["host_intervention"]["elapsed_seconds"] == pytest.approx(0.6)
        assert first["host_intervention"]["started_at"] >= first["model_cpu_completed_at"]
        assert engine.metrics.events[-1]["event"] == "queue_get"
        frozen = list(engine._latency_tracker._values)
        assert len(frozen) == 2 and frozen[-1] == pytest.approx(0.61)
        base.consume(case, engine.queue.available_steps() - 30)
        if arm == "disabled":
            assert base.submit(case) is None
            base.consume(case, engine.queue.available_steps())
            assert base.submit(case) is not None
            assert engine.rows[-1]["kind"] == "bootstrap"
            assert list(engine._latency_tracker._values) == frozen
            assert base.plan(engine).prediction_cap_exceeded
            assert engine.stats.planned_requests == 1 and engine.stats.recovery_probe_requests == 0
        else:
            probes = 0
            while base.plan(engine).prediction_cap_exceeded and probes < 50:
                assert base.submit(case) is not None
                probes += 1
                assert engine.rows[-1]["kind"] == "recovery_probe"
                assert engine.metrics.events[-1]["outcome"] == "discarded_recovery_probe"
                assert engine.metrics.events[-1]["latency_tracker_admitted"]
            assert 0 < probes < 50 and not base.plan(engine).prediction_cap_exceeded
            assert frozen[-1] in engine._latency_tracker._values
            assert base.submit(case) is not None
            assert engine.rows[-1]["kind"] == "planned"
            base.consume(case, engine.rows[-1]["plan"]["planned_delay_steps"] + 1)
            assert engine.metrics.last_get["outcome"] == "takeover"
        assert engine.pause_count == 1 and not engine.failed
        assert sum("host_intervention" in r for r in engine.rows) == 1
        assert engine.peak_active_calls == 1 and len(engine.runtime.captures) == 2


def test_empty_or_unstarted_queue_cannot_claim_recovery():
    manifest = stress.read_manifest()
    result = stress.summarize([{"spec": r, "status": "not_run"} for r in manifest["rows"]], manifest)
    assert result["episodes_completed"] == 0
    assert not result["stress_four_episode_contract_passed"]
    assert not result["stress_disabled_latch_both_observed"]
    assert not result["stress_candidate_recovery_both_observed"]
    assert not result["natural_latency_recovery_demonstrated"]
    assert not result["baseline_qualified"] and result["risk_thresholds"] is None


def test_missing_intervention_and_probes_cannot_supply_stress_evidence():
    result = stress.evidence({"spec": {"ordinal": 0, "arm": "candidate"}, "requests": []})
    assert not result["valid_host_pause_with_cap_exceedance"]
    assert not result["planned_takeover_after_recovery"]
    assert not result["disabled_latch_observed"]
