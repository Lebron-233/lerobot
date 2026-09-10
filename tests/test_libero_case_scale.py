"""F-SCL1 weighting and selection contracts; CPU only, no policy/Env load."""

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_case_scale as study  # noqa: E402


def test_floor_zero_error_and_mean_one():
    weights, floor = study.normalized_weights([0.0, 0.001, 0.1, 1.0], 0.1)
    assert floor == pytest.approx(0.01)
    assert sum(weights) / 4 == pytest.approx(1.0)
    assert weights[0] == weights[1] > weights[2] > weights[3]


@pytest.mark.parametrize("values", [[], [float("nan")], [-1.0]])
def test_invalid_training_errors_rejected(values):
    with pytest.raises(ValueError):
        study.normalized_weights(values, 0.1)


def test_global_and_case_objectives_change_only_action_coefficient():
    z, a = torch.tensor(6.0, requires_grad=True), torch.tensor(4.0, requires_grad=True)
    scales = {"latent": 3.0, "row0": 2.0}
    global_loss = study.objective(z, a, "global_conditioned", scales, 0.5)
    case_loss = study.objective(z, a, "case_conditioned", scales, 0.5)
    assert global_loss.item() == 4 and case_loss.item() == 3
    case_loss.backward()
    assert z.grad.item() == pytest.approx(1 / 3) and a.grad.item() == 0.25
    assert study.objective(z, a, "case_no_action", scales, 0.5).item() == 3


def test_weights_refuse_validation_data():
    with pytest.raises(ValueError, match="training data only"):
        study.weight_table([{"task": 6, "split": "validation"}], {"row0": 0.1})


def test_equal_episode_exposure_and_all_samples_once():
    pool = [
        {"task": t, "initial_state_id": s, "request_id": r}
        for t in range(6)
        for s in (46, 48, 49)
        for r in range(3, 7)
    ]
    schedule = study.cov.schedule(pool, "multi_conditioned")
    assert len(schedule) == study.UPDATES == 72
    assert set(schedule) == set(range(72))
    assert set(Counter((pool[i]["task"], pool[i]["initial_state_id"]) for i in schedule).values()) == {4}


def test_development_gate_distinguishes_identity_and_strongest_ablation():
    ck = {a: {"best_step": 72, "nonzero_residual": True} for a in study.ARMS}
    histories = {}
    scores = dict(zip(study.ARMS, (0.9, 0.8, 0.7, 0.85), strict=True))
    for a in study.ARMS:
        histories[a] = [
            {"step": 0, "metrics": {m: {"episode_macro": 1.0} for m in ("row0", "chunk")}},
            {"step": 72, "metrics": {m: {"episode_macro": scores[a]} for m in ("row0", "chunk")}},
        ]
    assert study.decision(ck, histories)["development_candidate_gate_passed"]
    assert not study.decision(ck, histories)["all_row0_comparators_better"]
    old = copy.deepcopy(ck)
    ck["case_conditioned"] = {"best_step": 0, "nonzero_residual": False}
    assert not study.decision(ck, histories)["development_candidate_gate_passed"]
    assert old["case_conditioned"]["best_step"] == 72


def test_empty_low_error_subgroup_is_not_zero_error():
    sample = {
        "task": 0,
        "initial_state_id": 46,
        "request_id": 3,
        "archived_full_chunk": torch.ones(1, 50, 32),
        "oracle": torch.zeros(1, 50, 32),
    }
    rows = [{"task": 0, "state": 46, "request_id": 3, "row0": 0.5}]
    result = study.low_error_summary(rows, [sample], 0.01)
    assert result["count"] == 0 and result["predicted_mean"] is None


def test_decoder_budget_rejects_before_dispatch():
    counts = Counter(decoder=899)
    study.cov.take(counts, "decoder")
    with pytest.raises(RuntimeError):
        study.cov.take(counts, "decoder")
    assert counts["decoder"] == 900
