"""Contracts for the frozen new-state probe; no GPU/model/native claims."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_frozen_newstate as study  # noqa: E402


def test_manifest_is_new_states_not_confirmation_or_old_test():
    rows = study.manifest()["rows"]
    assert [(r["task_id"], r["initial_state_id"]) for r in rows] == [
        (8, 42),
        (9, 42),
        (9, 43),
        (8, 43),
        (8, 44),
        (9, 44),
        (9, 45),
        (8, 45),
    ]
    assert all(r["condition"] == "graph_identity_async" for r in rows)
    assert len({(r["task_id"], r["initial_state_id"], r["policy_seed"]) for r in rows}) == 8
    assert study.manifest()["training_updates"] == 0


def test_mismatch_preserves_episode_delay_and_is_deterministic():
    samples = [
        {"ordinal": o, "delay": d, "request_id": rid}
        for o, d, rid in ((0, 3, 5), (0, 3, 3), (0, 4, 6), (1, 3, 3), (1, 3, 4))
    ]
    before = copy.deepcopy(samples)
    assert study.donor_indices(samples) == {1: 0, 0: 1, 3: 4, 4: 3}
    assert samples == before


def metric_row(ordinal, conditioned=0.8, no_action=0.9):
    values = {"identity": 1.0, "conditioned": conditioned, "no_action": no_action}
    return {
        "ordinal": ordinal,
        **{f"{arm}_{metric}": value for arm, value in values.items() for metric in study.METRICS},
    }


def test_macro_is_equal_episode_not_sample_weighted():
    rows = [metric_row(0, 0.2)] * 9 + [metric_row(1, 1.8)]
    metric = study.aggregate(rows)["metrics"]["conditioned_row0_oracle_mse"]
    assert metric["sample_mean"] == pytest.approx(0.36)
    assert metric["episode_macro"] == pytest.approx(1.0)
    assert not study.aggregate(rows)["frozen_action_increment_observed"]


def test_increment_requires_both_action_metrics_and_both_comparators():
    rows = [metric_row(i) for i in range(8)]
    assert study.aggregate(rows)["frozen_action_increment_observed"]
    rows[0]["conditioned_row0_oracle_mse"] = 2.0
    assert not study.aggregate(rows)["frozen_action_increment_observed"]
    assert not study.aggregate([])["frozen_action_increment_observed"]


@pytest.mark.parametrize("kind", list(study.LIMITS))
def test_native_budget_stops_before_over_budget_dispatch(kind):
    budget = study.Budget()
    budget.episode[kind] = study.LIMITS[kind][0]
    before = dict(budget.total)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.take(kind)
    assert dict(budget.total) == before


def test_pending_journal_preserves_unknown_not_zero(tmp_path):
    call = {"event": "call_intent", "call_id": 1, "kind": "model_request", "timestamp": 0.0, "limit": 15}
    (tmp_path / "calls.jsonl").write_text(json.dumps(call) + "\n")
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "started", "phase": "load", "at": 0.0, "limit": 60}) + "\n"
    )
    cursors, pending, active = {}, {}, {}
    study.update_pending(tmp_path, cursors, pending, active)
    assert pending == {1: call} and "load" in active
    with (tmp_path / "calls.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "call_return", "call_id": 1}) + "\n")
    study.update_pending(tmp_path, cursors, pending, active)
    assert pending == {} and "load" in active


def test_selection_is_first_twelve_not_metric_dependent(monkeypatch):
    pairs = [{"request_id": i} for i in range(20, 2, -1)]
    monkeypatch.setattr(study.pilot, "aligned_pairs", lambda record, arrays: (pairs, []))
    selected, excluded = study.choose_pairs({}, {})
    assert [p["request_id"] for p in selected] == list(range(3, 15))
    assert len(excluded) == 6
