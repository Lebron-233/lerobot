"""L15 prefix causality, conditioning isolation and full-case eligibility."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
import collect_so101_incremental_vision as collection  # noqa: E402
import train_so101_incremental_vision as training  # noqa: E402


def pair_request(recorder):
    plan = SimpleNamespace(
        planned_delay_steps=7,
        next_action_index=20,
        takeover_index=27,
        committed_policy_actions=torch.arange(48).reshape(8, 6).float(),
        committed_mask=torch.arange(8) < 7,
    )
    batch = {
        key: torch.ones(1, 4)
        for key in (collection.OBS_LANGUAGE_TOKENS, collection.OBS_LANGUAGE_ATTENTION_MASK)
    }
    recorder.request(
        20,
        plan,
        batch,
        (torch.zeros(1, 2, 4),) * 2,
        (torch.ones(1, 2, dtype=torch.bool),) * 2,
        torch.zeros(1, 32),
        torch.ones(1, 32),
    )
    return plan


def test_future_target_is_not_read_before_seven_completed_physical_actions(monkeypatch):
    recorder = collection.CausalVisualRecorder()
    plan = pair_request(recorder)
    calls = []

    def prepared(*args):
        calls.append(True)
        return {}, (torch.ones(1, 2, 4),) * 2, (torch.ones(1, 2, dtype=torch.bool),) * 2, torch.ones(1, 32)

    monkeypatch.setattr(collection, "prepared_observation", prepared)
    for step in range(21, 27):
        recorder.observe({}, step, None, None)
    assert not calls and not recorder.pairs
    recorder.observe({}, 27, None, None)
    ticks = [
        {
            "tick": i,
            "dispatch": "completed",
            "normalized_action": plan.committed_policy_actions[i - 20].tolist(),
            "terminated": False,
            "truncated": False,
        }
        for i in range(20, 27)
    ]
    assert recorder.audit(ticks)["audited_old_actions"] == 7
    ticks[-1]["terminated"] = True
    with pytest.raises(ValueError, match="terminal/reset"):
        recorder.audit(ticks)


def test_incomplete_terminal_prefix_is_counted_but_never_a_target():
    recorder = collection.CausalVisualRecorder()
    pair_request(recorder)
    report = recorder.audit([])
    assert report["complete_pairs"] == 0 and report["censored_terminal_or_bound_prefixes"] == 1


def test_only_causal_predicted_state_is_supplied_to_conditional_decoder(monkeypatch):
    query = {
        "state": torch.zeros(1, 32),
        "predicted_state": torch.ones(1, 32),
        "future_state": torch.full((1, 32), 99.0),
    }
    seen = []
    monkeypatch.setattr(training, "action_outputs", lambda p, f, q, z: seen.append(q["state"]) or z)
    training.conditional_actions(None, None, query, torch.zeros(1))
    assert torch.equal(seen[0], query["predicted_state"])
    assert torch.equal(query["state"], torch.zeros(1, 32))


def test_average_gain_cannot_bypass_case_coverage():
    rows = [
        {"episode": e, "state_only_l1": 1.0, "parent_joint_l1": 1.1, "student_l1": 0.9}
        for e in range(6)
        for _ in range(12)
    ]
    assert training.summarize(rows, held_out=True)["stable_incremental_action_gate"]
    for index, row in enumerate(rows):
        row["student_l1"] = 1.01 if index % 2 else 0.1
    report = training.summarize(rows, held_out=True)
    assert report["reduction_percent"] > 0 and not report["stable_incremental_action_gate"]
