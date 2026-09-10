"""CPU-only F-ACT1 split, objective and reporting contracts."""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_action_objective as study  # noqa: E402


def test_new_data_split_has_no_old_states_or_task_leakage():
    rows = study.manifest()["rows"]
    assert len(rows) == 12
    assert [(r["task_id"], r["initial_state_id"]) for r in rows] == [
        *[(t, 46) for t in range(8)],
        (8, 46),
        (9, 46),
        (9, 47),
        (8, 47),
    ]
    tasks = {s: {r["task_id"] for r in rows if r["split"] == s} for s in ("train", "validation", "test")}
    assert tasks == {"train": set(range(6)), "validation": {6, 7}, "test": {8, 9}}
    assert all(r["initial_state_id"] > 45 for r in rows)
    assert all(r["condition"] == "graph_identity_async" for r in rows)
    assert len({(r["task_id"], r["initial_state_id"], r["policy_seed"]) for r in rows}) == 12


def test_factorial_arms_and_fixed_budgets():
    assert set(study.ARMS) == {
        f"{loss}_{action}" for loss in ("token", "joint") for action in ("conditioned", "no_action")
    }
    assert study.UPDATES == 60
    assert study.LIMITS["episodes"] == (1, 12)
    budget = study.Budget()
    budget.begin_episode()
    budget.episode["model"] = 160
    with pytest.raises(RuntimeError, match="before dispatch"):
        budget.take("model")
    assert budget.episode["model"] == 160


def sample(split, z=2.0, a=3.0):
    return {
        "split": split,
        "inputs": (
            torch.full((1, 2, 3), z),
            torch.full((1, 2, 3), z),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        ),
        "future": (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),
        "archived_full_chunk": torch.full((1, 50, 32), a),
        "oracle": torch.zeros(1, 50, 32),
    }


def test_scales_use_only_train_not_validation_or_test():
    rows = [sample("train"), {"split": "validation"}, {"split": "test"}]
    assert study.train_scales(rows) == {"latent": 4.0, "row0": 9.0, "train_count": 1}


@pytest.mark.parametrize("arm", study.ARMS)
def test_objective_is_fixed_and_differentiable(arm):
    token = torch.tensor(2.0, requires_grad=True)
    frozen = torch.tensor(3.0, requires_grad=False)
    latent, action = token.square(), (token * frozen - 4).square()
    loss = study.objective_loss(latent, action, arm, {"latent": 4.0, "row0": 2.0})
    loss.backward()
    assert loss.item() == (3.0 if arm.startswith("joint") else 1.0)
    assert token.grad.item() == (7.0 if arm.startswith("joint") else 1.0)
    assert frozen.grad is None


def test_joint_cannot_silently_train_without_action_loss():
    with pytest.raises(ValueError, match="real decoder"):
        study.objective_loss(torch.tensor(1.0), None, "joint_conditioned", {"latent": 1, "row0": 1})


def metric_row(ordinal, joint=0.7, token=0.9, without=0.8):
    values = {
        "identity": 1.0,
        "token_conditioned": token,
        "token_no_action": 0.95,
        "joint_conditioned": joint,
        "joint_no_action": without,
    }
    return {
        "ordinal": ordinal,
        **{
            f"{a}_{m}": v
            for a, v in values.items()
            for m in ("latent_mse", "row0_oracle_mse", "chunk_oracle_mse")
        },
    }


def test_episode_macro_not_dominated_by_more_samples():
    rows = [metric_row(8, joint=0.2)] * 9 + [metric_row(9, joint=1.8)]
    value = study.macro(rows, "joint_conditioned_row0_oracle_mse")
    assert value["sample_mean"] == pytest.approx(0.36)
    assert value["episode_macro"] == pytest.approx(1.0)
    assert not study.aggregate(rows)["joint_candidate_gate_passed"]


def test_reporting_separates_training_objective_from_action_input_increment():
    rows = [metric_row(i) for i in range(8, 12)]
    assert study.aggregate(rows)["joint_candidate_gate_passed"]
    for r in rows:
        r["joint_no_action_row0_oracle_mse"] = 0.6
    report = study.aggregate(rows)
    assert report["action_objective_row0_improved"]
    assert not report["joint_action_input_increment_observed"]
    assert not report["joint_candidate_gate_passed"]
    assert not study.aggregate([])["joint_candidate_gate_passed"]


def test_no_action_is_only_action_ablation_with_real_predictor():
    torch.manual_seed(9)
    model = study.pilot.LightweightFutureLatentPredictor(study.pilot.config())
    row = sample("train")
    row.update(
        inputs=(
            torch.randn(1, 64, 960),
            torch.randn(1, 64, 960),
            torch.ones(1, 64, dtype=torch.bool),
            torch.ones(1, 64, dtype=torch.bool),
            torch.ones(1, 5, dtype=torch.long),
            torch.ones(1, 5, dtype=torch.bool),
            torch.zeros(1, 32),
            torch.randn(1, 50, 32),
        ),
        actions=torch.randn(1, 8, 7),
        mask=(torch.arange(8) < 3)[None],
        delay=3,
        future=(torch.zeros(1, 64, 960), torch.zeros(1, 64, 960)),
    )
    old = copy.deepcopy(row)
    seen = []
    handle = model.register_forward_pre_hook(lambda _m, args: seen.append(args[2].clone()))
    try:
        z1, _ = study.predict(model, row, "token_conditioned")
        z2, _ = study.predict(model, row, "joint_no_action")
    finally:
        handle.remove()
    assert torch.equal(seen[0], row["actions"])
    assert seen[1].count_nonzero() == 0
    for a, b in zip(z1, z2, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(row["actions"], old["actions"])


def test_pair_selection_uses_fixed_first_eight_not_values(monkeypatch):
    pairs = [{"request_id": r} for r in range(14, 2, -1)]
    monkeypatch.setattr(study.pilot, "aligned_pairs", lambda *_: (pairs, []))
    selected, excluded = study.chosen_pairs({}, {})
    assert [p["request_id"] for p in selected] == list(range(3, 11))
    assert len(excluded) == 4
