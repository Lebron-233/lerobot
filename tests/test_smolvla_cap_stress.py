"""Diagnostic host intervention on the existing CPU NativeEngine worker fixture."""

import copy
import sys
import threading
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


@pytest.mark.parametrize("seconds, expected_raw", [(0.35000001, 8), (0.3500001, 9)])
def test_fast_bootstrap_evidence_uses_original_integer_tolerance(seconds, expected_raw):
    history = {"samples_seconds": [0.01, 0.61], "raw_required_steps": 13}
    record = {
        "spec": {"ordinal": 0, "arm": "disabled"},
        "requests": [
            {
                "request_id": 3,
                "kind": "planned",
                "startup_phase": None,
                "model_cpu_completed_at": 1.0,
                "host_intervention": {"started_at": 1.0, "elapsed_seconds": 0.6},
                "terminal": {"latency_tracker_admitted": True},
                "recovery": {"history_after": history},
            },
            {
                "request_id": 4,
                "kind": "bootstrap",
                "startup_phase": None,
                "reset_epoch": 1,
                "task_epoch": 0,
                "terminal": {
                    "outcome": "installed",
                    "latency_tracker_admitted": False,
                    "total_chunk_s": seconds,
                },
                "recovery": {"history_after": history},
            },
        ],
    }
    result = stress.evidence(record)
    assert base.latency_to_steps(seconds, 20) + 1 == expected_raw
    assert result["disabled_latch_observed"] is (expected_raw <= 8)
    if expected_raw <= 8:
        assert result["fast_excluded_bootstrap_evidence"][0]["raw_required_steps"] == 8


@pytest.mark.usefixtures("fake_graph")
def test_async_host_pause_admits_late_sample_and_links_audited_native_row_zero(tmp_path, clock, monkeypatch):
    entered, release, consumed = threading.Event(), threading.Event(), threading.Event()

    def controlled_pause(seconds):
        clock.sleep(seconds)
        entered.set()
        assert release.wait(3), "Controller could not proceed while the model owner paused"

    monkeypatch.setattr(stress, "pause", controlled_pause)
    with base.running_case(
        tmp_path, clock, engine_class=cpu_engine("candidate"), condition="graph_identity_async"
    ) as case:
        engine = case.engine
        control = {"gets": [], "dispatches": [], "blocks": []}

        def consume(count):
            for _ in range(count):
                base.consume(case, 1)
                get = copy.deepcopy(engine.metrics.last_get)
                step = case.native.native_steps[-1]
                control["gets"].append(get)
                control["dispatches"].append(
                    {
                        **copy.deepcopy(case.native.dispatch),
                        "queue_get_outcome": get["outcome"],
                        "dispatched_at": step["started_at"],
                        "observation_age": 0.0,
                        "jitter_seconds": 0.0,
                    }
                )

        consume(20)
        assert base.e.notify(engine, case.obs) == 3
        controller_errors = []

        def advance_controller():
            try:
                consume(engine.rows[-1]["plan"]["planned_delay_steps"] + 1)
            except BaseException as error:
                controller_errors.append(error)
            finally:
                consumed.set()

        controller = threading.Thread(target=advance_controller)
        try:
            assert entered.wait(3)
            controller.start()
            assert consumed.wait(2), "Host pause held a lock needed by the controller"
        finally:
            release.set()
            if controller.ident is not None:
                controller.join(3)
        assert not controller.is_alive() and not controller_errors
        engine.wait_request(3, timeout=3)
        terminal = next(
            m for m in engine.metrics.events if m["event"] == "chunk_request" and m["request_id"] == 3
        )
        assert terminal["outcome"] == "deadline_miss" and terminal["late_steps"] > 0
        assert terminal["latency_tracker_admitted"] and terminal["total_chunk_s"] >= 0.61
        assert base.plan(engine).prediction_cap_exceeded

        for _ in range(50):
            if not base.plan(engine).prediction_cap_exceeded:
                break
            assert base.submit(case) is not None
            assert engine.rows[-1]["kind"] == "recovery_probe"
        assert not base.plan(engine).prediction_cap_exceeded
        planned_id = base.submit(case)
        assert planned_id is not None and engine.rows[-1]["kind"] == "planned"
        consume(engine.rows[-1]["plan"]["planned_delay_steps"] + 1)
        audit = base.e.audit_episode(engine, case.native, control)
        record = {
            "spec": engine.spec,
            "requests": engine.rows,
            "metrics": engine.metrics.events,
            "native_steps": case.native.native_steps,
            "audit": audit,
        }
        result = stress.evidence(record)
        assert result["valid_host_pause_with_cap_exceedance"] and result["planned_takeover_after_recovery"]
        chain = result["native_recovery_chains"][0]
        source = control["dispatches"][-1]
        assert chain["paused_request_id"] == 3
        assert chain["source_request_id"] == source["source_request_id"] == planned_id
        assert chain["source_row_offset"] == source["source_row_offset"] == 0
        assert chain["native_started_at"] == case.native.native_steps[-1]["started_at"]

        missing_native = copy.deepcopy(record)
        missing_native["native_steps"] = [
            s for s in missing_native["native_steps"] if s["number"] != chain["native_step_number"]
        ]
        assert not stress.evidence(missing_native)["planned_takeover_after_recovery"]
        other_epoch = copy.deepcopy(record)
        other_epoch["requests"][3]["task_epoch"] += 1
        assert not stress.evidence(other_epoch)["planned_takeover_after_recovery"]
        case.trace.append({"stress_evidence": result})
