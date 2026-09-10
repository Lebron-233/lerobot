"""F-OPT1 source isolation, deterministic design, and numerical objective contracts."""

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_optimization_probe as study  # noqa: E402


def examples():
    return [
        {
            "task": t,
            "ordinal": t,
            "request_id": 3 + i,
            "initial_state_id": 46,
            "delay": 3,
            "split": "train" if t < 6 else "validation",
        }
        for t, n in enumerate((8, 8, 8, 7, 8, 8, 6, 6))
        for i in range(n)
    ]


def test_fixed_anchor_selection_and_validation_separation():
    rows = examples()
    before = copy.deepcopy(rows)
    anchors, val = study.select_development(list(reversed(rows)))
    assert [study.keysort(s) for s in anchors] == [(t, 3) for t in range(6)]
    assert len(val) == 12 and all(s["task"] in (6, 7) for s in val)
    assert rows == before


@pytest.mark.parametrize("change", ["test", "count", "duplicate", "state", "delay"])
def test_contaminated_or_changed_development_refused(change):
    rows = examples()
    if change == "test":
        rows[0].update(task=8, split="test")
    elif change == "count":
        rows.pop()
    elif change == "duplicate":
        rows[1] = dict(rows[0])
    elif change == "state":
        rows[0]["initial_state_id"] = 47
    else:
        rows[0]["delay"] = 4
    with pytest.raises(ValueError):
        study.select_development(rows)


def test_objective_and_action_control_differ_one_factor():
    a, b, c, d = study.ARMS.values()
    assert a["objective"] == b["objective"] and a["actions"] == b["actions"]
    assert a["lr"] == 10 * b["lr"] and b["lr"] == c["lr"] == d["lr"]
    assert {k: b[k] for k in ("lr", "objective")} == {k: d[k] for k in ("lr", "objective")}
    z, action = torch.tensor(6.0, requires_grad=True), torch.tensor(4.0, requires_grad=True)
    scales = {"latent": 3.0, "row0": 2.0}
    assert study.combined_loss(z, action, b, scales).item() == 4.0
    loss = study.combined_loss(z, action, c, scales)
    assert loss.item() == 2.0
    loss.backward()
    assert z.grad is None and action.grad.item() == 0.5


def test_dispatch_budget_and_equal_round_robin():
    assert study.UPDATES == 24
    assert Counter(i % 6 for i in range(study.UPDATES)) == dict.fromkeys(range(6), 4)
    counts = Counter(decoder=335)
    study.take(counts, "decoder")
    with pytest.raises(RuntimeError):
        study.take(counts, "decoder")
    assert counts["decoder"] == 336


def test_snapshot_does_not_alias_gradient_outputs():
    z = torch.ones(1, 2, 3, requires_grad=True)
    value = torch.ones(1, 50, 32, requires_grad=True)
    result = study.snapshot((z,), value, (z.sum(), value.sum(), value.mean()))
    with torch.no_grad():
        z.zero_()
        value.zero_()
    assert result["visual"][0].count_nonzero() == 6 and result["output"].sum() == 1600
    assert not result["visual"][0].requires_grad
