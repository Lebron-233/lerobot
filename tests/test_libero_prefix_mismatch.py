"""F-PFX1 donor identity and recipient-only intervention contracts; CPU only."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_prefix_mismatch as study  # noqa: E402


def sample(request, delay=3, task=0, state=48, split="train"):
    return {
        "task": task, "initial_state_id": state, "request_id": request, "split": split,
        "delay": delay, "mask": (torch.arange(8) < delay)[None],
        "actions": torch.full((1, 8, 7), float(request)),
        "future": object(), "inputs": object(), "oracle": object(),
    }


def test_cyclic_order_is_identity_based_and_preserves_singletons():
    samples = [sample(6, 4), sample(5), sample(3), sample(4), sample(3, task=6, split="validation")]
    donors = study.donor_map(samples)
    assert donors[(0, 48, 3)] == (0, 48, 4)
    assert donors[(0, 48, 4)] == (0, 48, 5)
    assert donors[(0, 48, 5)] == (0, 48, 3)
    assert donors[(0, 48, 6)] is None and donors[(6, 48, 3)] is None
    assert study.donor_map(samples[::-1]) == donors


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        study.donor_map([sample(3), sample(3)])


def test_intervention_replaces_only_actions_and_does_not_mutate_sources():
    recipient, donor = sample(3), sample(4)
    result = study.with_prefix(recipient, donor)
    assert torch.equal(result["actions"], donor["actions"])
    assert all(result[k] is recipient[k] for k in recipient if k != "actions")
    result["actions"].zero_()
    assert donor["actions"].sum() > 0 and recipient["actions"].sum() > 0


@pytest.mark.parametrize("field,value", [
    ("split", "validation"), ("task", 1), ("initial_state_id", 49), ("delay", 4), ("request_id", 3),
])
def test_illegal_donor_is_rejected(field, value):
    recipient, donor = sample(3), sample(4)
    donor[field] = value
    with pytest.raises(ValueError, match="different example"):
        study.with_prefix(recipient, donor)


def test_mask_change_rejected():
    recipient, donor = sample(3), sample(4)
    donor["mask"][0, 0] = False
    with pytest.raises(ValueError, match="mask"):
        study.with_prefix(recipient, donor)
