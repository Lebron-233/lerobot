"""F-COV2 new identity, frozen checkpoint, outcome and budget contracts (CPU only)."""

import copy
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_coverage_heldout as study  # noqa: E402


def test_fixed_new_pairs_and_all_four_frozen_arms():
    m = study.manifest()
    assert [(r["task_id"], r["initial_state_id"]) for r in m["rows"]] == [(8, 48), (9, 48), (9, 49), (8, 49)]
    assert set(m["arms"]) == {
        "single_conditioned",
        "multi_conditioned",
        "single_no_action",
        "multi_no_action",
    }
    assert m["best_step"] == 72 and m["training_updates"] == 0
    assert all(r["condition"] == "graph_identity_async" and r["split"] == "heldout" for r in m["rows"])


def row(i, multi=0.8, strongest=0.7):
    values = {
        "identity": 1.0,
        "single_conditioned": 0.9,
        "multi_conditioned": multi,
        "single_no_action": strongest,
        "multi_no_action": 0.95,
    }
    return {
        "ordinal": i,
        "task": 8 + i % 2,
        "state": 48 + i // 2,
        **{a: dict.fromkeys(("latent", "row0", "chunk"), v) for a, v in values.items()},
    }


def test_primary_gate_does_not_hide_strongest_baseline():
    result = study.aggregate([row(i) for i in range(4)])
    assert result["heldout_primary_gate_passed"]
    assert not result["strongest_no_action_row0_better"]
    assert not result["heldout_all_comparators_better"]


def test_no_partial_or_identity_tie_can_pass():
    assert not study.aggregate([])["heldout_primary_gate_passed"]
    assert not study.aggregate([row(i) for i in range(3)])["heldout_primary_gate_passed"]
    assert not study.aggregate([row(i, multi=1.0) for i in range(4)])["heldout_primary_gate_passed"]


def test_chunk_regression_blocks_primary_gate():
    rows = [row(i) for i in range(4)]
    for r in rows:
        r["multi_conditioned"]["chunk"] = 1.1
    assert not study.aggregate(rows)["heldout_primary_gate_passed"]


def test_episode_macro_not_sample_weighted():
    rows = [row(0, multi=0.2)] * 9 + [row(1, multi=1.8)]
    metric = study.aggregate(rows)["metrics"]["multi_conditioned"]["row0"]
    assert metric["sample_mean"] == pytest.approx(0.36)
    assert metric["episode_macro"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field,value", [("best_step", 36), ("seed", 9), ("arm", "other"), ("nonzero_residual", False)]
)
def test_wrong_checkpoint_rejected(field, value):
    ck = {
        "arm": study.ARMS[0],
        "best_step": 72,
        "seed": 20260912,
        "config": asdict(study.pilot.config()),
        "nonzero_residual": True,
        "state_dict": {"up_projection.weight": torch.ones(1), "up_projection.bias": torch.zeros(1)},
    }
    study.validate_checkpoint(ck, study.ARMS[0])
    wrong = copy.deepcopy(ck)
    wrong[field] = value
    with pytest.raises(ValueError, match="Frozen"):
        study.validate_checkpoint(wrong, study.ARMS[0])


def test_budget_rejects_before_increment_and_native_after_four():
    counts = Counter(decoder=96)
    with pytest.raises(RuntimeError):
        study.take(counts, "decoder")
    assert counts["decoder"] == 96
    budget = study.Budget()
    for _ in range(4):
        budget.begin_episode()
    with pytest.raises(RuntimeError):
        budget.begin_episode()
    assert budget.total["episodes"] == 4


def test_first_four_selection_uses_existing_alignment(monkeypatch):
    pairs = [{"request_id": n} for n in (7, 3, 6, 5, 4)]
    monkeypatch.setattr(study.pilot, "aligned_pairs", lambda *_: (pairs, []))
    selected, excluded = study.cov.selected_pairs({}, {})
    assert [p["request_id"] for p in selected] == [3, 4, 5, 6]
    assert excluded == [{"request_id": 7, "reason": "fixed_first_four_complete"}]
