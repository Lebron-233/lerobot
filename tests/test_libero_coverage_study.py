"""CPU contracts for the fixed F-COV1 coverage comparison, not native tests."""

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_coverage_study as study  # noqa: E402


def pool():
    return [
        {"task": t, "initial_state_id": st, "request_id": i + 3, "split": "train"}
        for t in range(6)
        for st in (46, 48, 49)
        for i in range(4)
    ]


def test_manifest_protects_all_test_tasks_and_old_initial_states():
    m = study.manifest()
    assert len(m["rows"]) == 16
    assert [(r["task_id"], r["initial_state_id"]) for r in m["rows"]] == [(t, 48) for t in range(8)] + [
        (t, 49) for t in range(7, -1, -1)
    ]
    assert all(r["condition"] == "graph_identity_async" for r in m["rows"])
    assert {r["task_id"] for r in m["rows"] if r["split"] == "validation"} == {6, 7}
    assert all(
        r["environment_seed"] == 1000000 + 100 * r["task_id"] + r["initial_state_id"] for r in m["rows"]
    )


@pytest.mark.parametrize("arm", study.ARMS)
def test_fixed_equal_exposures_and_action_arms_share_order(arm):
    samples = pool()
    before = copy.deepcopy(samples)
    chosen = study.pool_for(samples, arm)
    order = study.schedule(chosen, arm)
    assert len(order) == 72
    assert Counter(chosen[i]["task"] for i in order) == dict.fromkeys(range(6), 12)
    counts = Counter((chosen[i]["task"], chosen[i]["initial_state_id"]) for i in order)
    assert set(counts.values()) == ({12} if arm.startswith("single") else {4})
    equivalent = arm.replace("no_action", "conditioned")
    assert order == study.schedule(study.pool_for(samples, equivalent), equivalent)
    assert before == samples


def test_missing_coverage_rejected_and_validation_not_training():
    with pytest.raises(ValueError, match="coverage"):
        study.pool_for([s for s in pool() if s["initial_state_id"] != 49], "multi_conditioned")
    validation = {"task": 6, "initial_state_id": 48, "request_id": 3, "split": "validation"}
    assert validation not in study.pool_for(pool() + [validation], "multi_conditioned")


def test_episode_macro_not_sample_weighted():
    rows = [{"task": 0, "state": 48, "latent": 1.0, "row0": 1.0, "chunk": 1.0}] * 4
    rows += [{"task": 1, "state": 48, "latent": 3.0, "row0": 3.0, "chunk": 3.0}]
    values = study.summarize_rows(rows)["row0"]
    assert values["episode_macro"] == 2.0 and values["sample_mean"] == 1.4


def test_identity_selection_cannot_pass_candidate_gate():
    checkpoints = {a: {"best_step": 0, "nonzero_residual": False} for a in study.ARMS}
    histories = {
        a: [{"step": 0, "metrics": {k: {"episode_macro": 1.0} for k in ("row0", "chunk")}}]
        for a in study.ARMS
    }
    assert not study.gate(checkpoints, histories)["validation_candidate_gate_passed"]


def test_budget_checked_before_increment():
    counts = Counter(decoder=900)
    with pytest.raises(RuntimeError):
        study.take(counts, "decoder")
    assert counts["decoder"] == 900
    budget = study.Budget()
    budget.total["episodes"] = 16
    with pytest.raises(RuntimeError):
        budget.begin_episode()


def test_joint_loss_scales_are_shared_with_parent():
    z, a = torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)
    value = study.parent.objective_loss(z, a, "joint_conditioned", {"latent": 2.0, "row0": 3.0})
    assert value.item() == 2.0
    value.backward()
    assert z.grad.item() == 0.5 and a.grad.item() == pytest.approx(1 / 3)
