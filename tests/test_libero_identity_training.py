"""Synthetic CPU tests only; no actual source tensors, models or Env loaded."""

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_identity_training as audit  # noqa: E402
import libero_identity_training as r  # noqa: E402


def rows():
    result = []
    for task in range(8):
        for state in ((46, 48, 49) if task < 6 else (48, 49)):
            for request in range(3, 7):
                values = {"identity": (2., 2.), "old_centered": (1.2, 1.6), "frozen": (1., 1.5),
                    "plain_true": (1.1, 1.4), "guarded_true": (0.8, 1.), "plain_zero": (2., 2.),
                    "guarded_zero": (2., 2.), "plain_mismatched": (1.8, 2.), "guarded_mismatched": (1.8, 2.)}
                if (task, state, request) == (4, 46, 5):
                    values.update(frozen=(6., 3.), plain_true=(5., 3.), guarded_true=(3., 2.))
                result.append({"key": [task, state, request], "split": "train" if task < 6 else "validation",
                    "delay": 3, "donor": [task, state, 3 if request == 6 else request+1],
                    "metrics": {a: {"row0": x, "chunk": y, "latent": 3.} for a, (x, y) in values.items()}})
    return result


def test_matched_fixed_contract():
    s = r.specification()
    assert s["updates_per_arm"] == 72 and s["checkpoint_selection"] == "fixed_final_72_no_validation_selection"
    assert s["limits"] == {"decoder": 930, "predictor": 1508, "backward": 144, "updates": 144}
    assert s["qualification_reads"] == s["new_env"] == 0
    assert r.STEPS == 72 and r.LR == 1e-4 and r.CAPTURES == 24


def test_soft_objective_formula_and_gradient():
    metrics = tuple(torch.tensor(x, requires_grad=True) for x in (6., 4., 5.))
    identity, scales, weight = {"row0": 2., "chunk": 3.}, {"latent": 3., "row0": 2., "chunk": 5.}, 0.5
    plain = r.objective(metrics, identity, scales, weight, "plain")
    guarded = r.objective(metrics, identity, scales, weight, "guarded")
    assert float(plain.detach()) == 3.
    assert float(guarded.detach()) == pytest.approx(5.4)
    guarded.backward()
    assert [float(x.grad) for x in metrics] == pytest.approx([1/3, 0.75, 0.4])
    score = dict(zip(("latent", "row0", "chunk"), (6., 4., 5.), strict=True))
    assert audit.loss_value(score, identity, scales, weight, "guarded") == pytest.approx(5.4)


def test_no_excess_hinge_when_better_than_identity():
    m = tuple(torch.tensor(x) for x in (3., 1., 1.))
    scales = {"latent": 3., "row0": 2., "chunk": 5.}
    i = {"row0": 2., "chunk": 3.}
    assert float(r.objective(m, i, scales, 0.5, "guarded")) == pytest.approx(1.45)
    with pytest.raises(ValueError):
        r.objective(m, i, scales, 1., "unknown")


def test_synthetic_positive_and_independent_decision():
    data = rows()
    assert r.statistics(data)["development_followup_supported"]
    assert audit.decision(data)


@pytest.mark.parametrize("case", ["row0_identity", "row0_frozen", "row0_mismatch", "chunk_frozen", "chunk_plain",
                                  "harm_frozen", "harm_plain", "val_harm", "majority", "episodes"])
def test_preregistered_gate_components(case):
    data = rows()
    val = [x for x in data if x["split"] == "validation"]
    if case == "row0_identity":
        for x in val:
            x["metrics"]["guarded_true"]["row0"] = 2.1
    elif case == "row0_frozen":
        for x in val:
            x["metrics"]["guarded_true"]["row0"] = 1.05
    elif case == "row0_mismatch":
        for x in val:
            x["metrics"]["guarded_mismatched"]["row0"] = 0.1
    elif case.startswith("chunk_"):
        arm = "frozen" if case == "chunk_frozen" else "plain_true"
        for x in val:
            x["metrics"][arm]["chunk"] = 0.9
    elif case.startswith("harm_"):
        arm = "frozen" if case == "harm_frozen" else "plain_true"
        for x in data:
            if x["key"] == [4, 46, 5]:
                x["metrics"][arm]["row0"] = 3.
    elif case == "val_harm":
        val[0]["metrics"]["guarded_true"]["row0"] = 2.2
    elif case == "majority":
        for i, x in enumerate(val):
            x["metrics"]["guarded_true"]["row0"] = 0. if i % 2 == 0 else 2.1
    else:
        for i, x in enumerate(val):
            x["metrics"]["guarded_true"]["row0"] = 0. if i < 8 else 2.1
    assert not r.statistics(data)["development_followup_supported"]
    assert not audit.decision(data)


def test_numpy_adamw_matches_cpu_arithmetic():
    value = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float32))
    optimizer = torch.optim.AdamW([value], lr=r.LR, weight_decay=r.WD, foreach=False)
    state = {}
    for i in range(4):
        previous = {"p": value.detach().clone()}
        grad = torch.tensor([0.02*(i+1), -0.1], dtype=torch.float32)
        value.grad = grad.clone()
        expected = audit.adamw_step(previous, {"p": grad}, state)["p"]
        optimizer.step()
        assert np.allclose(expected, value.detach().numpy(), rtol=1e-5, atol=1e-7)
    before = value.detach().clone()
    expected = audit.adamw_step({"p": before}, {}, state)
    assert np.array_equal(expected["p"], audit.array(before))


def test_zero_raw_and_precision_evidence():
    tokens = (torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    delta = (torch.full((1, 2, 3), 1e8), torch.full((1, 2, 3), 1e8))
    sample = {"inputs": tokens, "mask": torch.ones(1, 8, dtype=torch.bool)}
    actions = torch.zeros(1, 8, 7)
    ev = {"actions": actions, "mask": sample["mask"], "action_delta": delta, "zero_delta": delta,
          "raw": tokens, "visual": tokens}
    audit.evidence_check(ev, sample, actions, True)
    bad = copy.deepcopy(ev)
    bad["raw"] = tuple(x+1 for x in tokens)
    with pytest.raises(ValueError, match="Subtract-before-add"):
        audit.evidence_check(bad, sample, actions, True)


def test_budget_stops_before_extra_dispatch():
    counts = Counter(r.LIMITS)
    for name in r.LIMITS:
        with pytest.raises(ValueError, match="exhausted"):
            r.take(counts, name)
    assert counts == Counter(r.LIMITS)


def test_offline_metrics_detach_and_cpu_boundary(monkeypatch):
    visual = tuple(torch.ones(1, 2, 3, requires_grad=True) for _ in range(2))
    value = torch.ones(1, 50, 32, requires_grad=True)
    sample = {"inputs": (*visual, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)),
              "future": tuple(torch.zeros(1, 2, 3) for _ in range(2)), "oracle": torch.zeros(1, 50, 32)}
    original = r.p.metrics
    seen = []

    def cpu_only(v, out, source):
        assert all(x.device.type == "cpu" and not x.requires_grad for x in (*v, out))
        seen.append(True)
        return original(v, out, source)

    monkeypatch.setattr(r.p, "metrics", cpu_only)
    assert r.offline_metrics(visual, value, sample) == {"row0": 1., "chunk": 1., "latent": 1.}
    assert seen == [True]
    assert all(x.requires_grad for x in visual) and value.requires_grad


def test_cpu_only():
    assert not torch.cuda.is_initialized()
