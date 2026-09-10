"""Verified mixed-delay preparation contracts; no policy, GPU, or Env."""

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_case_scale as study  # noqa: E402


def example(task, state, rid, ordinal):
    delay = 4 if (task, state, rid) in study.DELAY_FOUR_KEYS else 3
    actions = torch.zeros(1, 8, 7)
    actions[:, :delay] = 0.25
    return {
        "task": task,
        "initial_state_id": state,
        "request_id": rid,
        "ordinal": ordinal,
        "split": "train" if task < 6 else "validation",
        "delay": delay,
        "current_index": 89,
        "future_index": 89 + delay,
        "actions": actions,
        "mask": (torch.arange(8) < delay)[None],
    }


def fixtures():
    old = [example(t, 46, r + 3, t) for t, n in enumerate((8, 8, 8, 7, 8, 8, 6, 6)) for r in range(n)]
    new = [
        example(t, st, r, t if st == 48 else 15 - t) for t in range(8) for st in (48, 49) for r in range(3, 7)
    ]
    return old, new


def test_all_original_examples_retained_with_exact_mixed_delay_mapping():
    old, new = fixtures()
    before = copy.deepcopy(new)
    training, validation = study.select_data(old, new)
    assert Counter(s["delay"] for s in training) == {3: 69, 4: 3}
    assert Counter(s["delay"] for s in validation) == {3: 16}
    assert len(training) == 72 and len(validation) == 16
    for left, right in zip(new, before, strict=True):
        assert left["delay"] == right["delay"]
        assert torch.equal(left["actions"], right["actions"])
        assert torch.equal(left["mask"], right["mask"])


@pytest.mark.parametrize("change", ["delay", "mask", "future", "padding", "nan", "split"])
def test_bad_four_step_example_is_rejected(change):
    old, new = fixtures()
    s = next(s for s in new if (s["task"], s["initial_state_id"], s["request_id"]) == (3, 48, 6))
    if change == "delay":
        s["delay"] = 3
    elif change == "mask":
        s["mask"][0, 3] = False
    elif change == "future":
        s["future_index"] -= 1
    elif change == "padding":
        s["actions"][0, 4, 0] = 1
    elif change == "nan":
        s["actions"][0, 0, 0] = float("nan")
    else:
        s["split"] = "validation"
    with pytest.raises(ValueError):
        study.select_data(old, new)


def test_coherently_relabelled_extra_four_step_example_is_rejected():
    old, new = fixtures()
    s = new[0]
    s.update(delay=4, future_index=93, mask=(torch.arange(8) < 4)[None])
    with pytest.raises(ValueError, match="verified sample identity"):
        study.select_data(old, new)


def test_removed_four_step_sample_cannot_pass_by_changing_denominator():
    old, new = fixtures()
    new = [s for s in new if (s["task"], s["initial_state_id"], s["request_id"]) != (3, 48, 6)]
    with pytest.raises(ValueError, match="64 unique"):
        study.select_data(old, new)
