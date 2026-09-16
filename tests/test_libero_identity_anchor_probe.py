"""Synthetic CPU tests only: no original data or models are read."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_identity_anchor_probe as a  # noqa: E402
import libero_identity_anchor_probe as p  # noqa: E402


def rows():
    result = []
    for split, tasks in (("train", range(6)), ("validation", (6, 7))):
        states = (46, 48, 49) if split == "train" else (48, 49)
        for task in tasks:
            for state in states:
                for request in range(4):
                    metrics = {arm: {"row0": 2.0, "chunk": 2.0, "latent": 5.0} for arm in p.ARMS}
                    metrics["identity_true"] = {"row0": 1.0, "chunk": 1.0, "latent": 6.0}
                    result.append({"split": split, "key": [task, state, request], "metrics": metrics})
    return result


def test_transplant_subtract_before_add_and_zero():
    identity = (torch.tensor([1.0]),)
    delta = (torch.tensor([1e8]),)
    assert torch.equal(p.transplant(identity, delta, delta)[0], identity[0])
    assert not torch.equal((identity[0] + delta[0]) - delta[0], identity[0])


def test_transplant_uses_identity_not_old_base():
    value = p.transplant((torch.tensor([2.0]),), (torch.tensor([1.0]),), (torch.tensor([0.25]),))
    assert value[0].item() == 2.75


def test_camera_mismatch_rejected():
    with pytest.raises(ValueError):
        p.transplant((torch.ones(1),), (), ())


def test_development_counts_and_independent_gate():
    result = p.summarize(rows())
    assert result["splits"]["train"]["metrics"]["identity"]["samples"] == 72
    assert result["splits"]["validation"]["metrics"]["identity"]["samples"] == 16
    assert result["development_followup_supported"] is True
    assert a.independent_decision(rows()) is True
    assert not result["independent_qualification_claimed"]


@pytest.mark.parametrize("control", ["identity", "centered", "identity_mismatched"])
def test_all_controls_required(control):
    data = rows()
    for r in data:
        r["metrics"][control]["row0"] = 0.5
    assert not p.summarize(data)["development_followup_supported"]
    assert not a.independent_decision(data)


def test_sample_majority_cannot_be_replaced_by_mean():
    data = rows()
    for i, r in enumerate(data):
        r["metrics"]["identity_true"]["row0"] = 0 if i % 4 == 0 else 2.1
    assert not p.summarize(data)["development_followup_supported"]
    assert not a.independent_decision(data)


def test_chunk_guard():
    data = rows()
    for r in data:
        r["metrics"]["identity_true"]["chunk"] = 3.0
    assert not p.summarize(data)["development_followup_supported"]
    assert not a.independent_decision(data)


def test_paired_mismatch_denominator():
    data = rows()
    for r in data[:3]:
        del r["metrics"]["identity_mismatched"]
    c = p.summarize(data)["splits"]["train"]["contrasts"]["identity_true_vs_identity_mismatched"]
    assert c["paired_samples"] == 69
    assert c["paired_episodes"] == 18


def test_numpy_metric_reduction():
    z = (torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    s = {"future": tuple(torch.zeros_like(v) for v in z), "oracle": torch.zeros(1, 50, 32),
         "inputs": (*z, torch.tensor([[True, False]]), torch.ones(1, 2, dtype=torch.bool))}
    output = torch.ones(1, 50, 32)
    assert a.score(z, output, s) == p.metrics(z, output, s) == {"row0": 1., "chunk": 1., "latent": 1.}


def test_exact_error_decomposition():
    old, new, oracle = torch.ones(1, 50, 32), torch.ones(1, 50, 32) * .5, torch.zeros(1, 50, 32)
    d = a.decomposition(old, new, oracle)
    assert d["row0_change"] == -.75
    assert d["cross_term"] == -1.
    assert d["perturbation_energy"] == .25
    assert sum(d["row0_dimension_contributions"]) == pytest.approx(-.75)


def test_auditor_rejects_metric_tamper():
    with pytest.raises(ValueError):
        a.compare({"row0": 1.0}, {"row0": 1.1})


def test_fixed_budget_and_no_qualification_sources():
    assert p.DECODERS == 88*5 + 85
    assert p.CAPTURES == 8
    assert all("qualification" not in str(v) for v in p.source_paths())
    assert not torch.cuda.is_initialized()
