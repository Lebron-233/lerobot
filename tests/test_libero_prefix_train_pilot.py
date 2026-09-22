"""Tiny-model matched-objective and deterministic population tests; no data/weights."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from libero_prefix_train_pilot import objective, schedule  # noqa: E402

from tests.test_smolvla_prefix_training import TinyModel  # noqa: E402


@pytest.mark.parametrize("count", [0, 3, 8])
def test_matched_targets_and_masked_prediction_gradient(count):
    torch.manual_seed(3)
    model = TinyModel()
    truth, noise = torch.randn(2, 50, 32), torch.randn(2, 50, 32)
    truth[..., 7:] = 0
    inputs = (None, None, None, None, torch.zeros(2, 32))
    t, counts, valid = torch.tensor([0.25, 0.75]), torch.full((2,), count), torch.ones(2, 50, dtype=torch.bool)
    modes = ["unconditioned", "correct_prefix", "mismatched_prefix"]
    values = [objective(model, inputs, truth, noise, t, counts, valid, mode, truth.flip(0)) for mode in modes]
    for loss, velocity, flow in values:
        g = torch.autograd.grad(loss, velocity, retain_graph=True)[0]
        assert not g[~flow.loss_mask].count_nonzero()
        assert int(flow.loss_mask.sum()) == 2 * (50 - count) * 7
        assert torch.equal(flow.target_velocity, noise - truth)
    if count == 0:
        assert all(torch.equal(values[0][1], v[1]) for v in values[1:])
    else:
        assert not torch.equal(values[0][1], values[1][1])
        assert not torch.equal(values[1][1], values[2][1])


def test_schedule_is_fixed_balanced_and_sealed_excluded():
    rows = {}
    for task in (0, 2, 6, 7):
        for role in ("train", "dev", "sealed"):
            rows[f"{task}-{role}"] = {"task": task, "split": role, "rows": 180,
                                     "packet": {"anchors": list(range(0, 180, 20))}}
    a, b = schedule(rows), schedule(rows)
    assert a == b and len(a["train"]) == 128 and len(a["dev"]) == 16
    assert all("train" in x["trajectory_id"] and 0 <= x["C"] <= 8 for x in a["train"])
    assert all("dev" in x["trajectory_id"] and "dev" in x["donor"] for x in a["dev"])
    assert all(sum(rows[x["trajectory_id"]]["task"] == t for x in a["train"]) == 32 for t in (0, 2, 6, 7))
