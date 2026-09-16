"""Synthetic CPU checks; no experiment data, models, GPU, or Env are opened."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"
sys.path.insert(0, str(EXAMPLES))
import audit_libero_action_qualification as audit  # noqa: E402
import libero_action_qualification as q  # noqa: E402
import report_libero_action_qualification as report  # noqa: E402


def fake_rows():
    rows = []
    for task, state in q.PAIRS:
        for request in range(4):
            scores = {a: {"row0": 2.0, "chunk": 2.0, "latent": 3.0} for a in q.ARMS}
            scores["centered_true"] = {"row0": 1.0, "chunk": 1.0, "latent": 4.0}
            rows.append({"key": [task, state, request], "delay": 2,
                         "donor": [task, state, (request + 1) % 4], "metrics": scores})
    return rows


def test_manifest_frozen_scope():
    m = q.manifest()
    assert len(m["rows"]) == 8
    assert [(r["task_id"], r["initial_state_id"]) for r in m["rows"]] == list(q.PAIRS)
    assert all(r["condition"] == "graph_identity_async" and r["split"] == "qualification" for r in m["rows"])
    assert len({r["environment_seed"] for r in m["rows"]}) == 8
    assert not m["predictor_controls_environment"]


def test_history_both_schemas_and_reuse_rejected(tmp_path):
    for i, wrapper in enumerate(("spec", "tuple")):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "started.json").write_text(json.dumps({wrapper: {"task_id": 6, "initial_state_id": 0}}))
    history = q.history_inventory(tmp_path)
    assert len(history) == 2
    with pytest.raises(ValueError, match="already used"):
        q.check_unused(history)


def test_unknown_history_fails_closed(tmp_path):
    (tmp_path / "started.json").write_text('{}')
    with pytest.raises(ValueError, match="Unknown"):
        q.history_inventory(tmp_path)


def test_source_scope_no_old_test_or_confirmation():
    paths = [str(p) for p in q.source_files()]
    assert not any("heldout" in p or "confirmation" in p or "test_labels" in p for p in paths)
    assert q.CHECKPOINTS["base"].name == "case_no_action.pt"


def test_checkpoint_metadata_not_reselected():
    from dataclasses import asdict
    saved = {"arm": "centered", "best_step": 72, "seed": 20260912, "config": asdict(q.pilot.config())}
    q.checkpoint_valid(saved, "centered")
    saved["best_step"] = 36
    with pytest.raises(ValueError):
        q.checkpoint_valid(saved, "centered")


def test_positive_gates_and_independent_decision():
    rows = fake_rows()
    result = q.aggregate(rows)
    assert result["heldout_primary_gate_passed"] and result["heldout_robustness_gate_passed"]
    assert audit.independent_gates(rows) == (True, True)


@pytest.mark.parametrize("control", ["base", "ordinary_true", "centered_mismatched"])
def test_every_mean_control_is_necessary(control):
    rows = fake_rows()
    for r in rows:
        r["metrics"][control]["row0"] = 0.5
    assert not q.aggregate(rows)["heldout_primary_gate_passed"]
    assert audit.independent_gates(rows) == (False, False)


def test_missing_mismatch_is_not_paired_against_full_true_denominator():
    rows = fake_rows()
    for r in rows[:4]:
        del r["metrics"]["centered_mismatched"]
        del r["metrics"]["ordinary_mismatched"]
        r["donor"] = None
    result = q.aggregate(rows)
    assert result["contrasts"]["centered_mismatched"]["paired_samples"] == 28
    assert result["contrasts"]["centered_mismatched"]["paired_episodes"] == 7
    assert not result["heldout_primary_gate_passed"]
    assert audit.independent_gates(rows) == (False, False)


def test_episode_macro_not_sample_macro():
    rows = fake_rows()
    rows = rows[:1] + rows[4:]
    rows[0]["metrics"]["base"]["row0"] = 10
    result = q.aggregate(rows)
    assert result["metrics"]["base"]["macro"]["row0"] == 3


def test_sample_regressions_block_robustness_without_hiding_mean_gain():
    rows = fake_rows()
    for i, r in enumerate(rows):
        r["metrics"]["centered_true"]["row0"] = 0.0 if i % 4 == 0 else 2.1
    result = q.aggregate(rows)
    assert result["heldout_primary_gate_passed"]
    assert not result["heldout_robustness_gate_passed"]
    assert audit.independent_gates(rows) == (True, False)


def test_concentrated_episode_benefit_blocks_leave_one_out():
    rows = fake_rows()
    for i, r in enumerate(rows):
        r["metrics"]["base"]["row0"] = 10 if i < 4 else 2
        r["metrics"]["centered_true"]["row0"] = 1 if i < 24 else 2.9
    result = q.aggregate(rows)
    assert result["contrasts"]["base"]["leave_one_episode_out"]
    # Inject net-negative other episodes while retaining six positive episodes.
    for i, r in enumerate(rows):
        if 4 <= i < 24:
            r["metrics"]["centered_true"]["row0"] = 1.99
        if i >= 24:
            r["metrics"]["centered_true"]["row0"] = 3
    result = q.aggregate(rows)
    assert not result["robustness_checks"]["all_leave_one_episode_out_positive"]


@pytest.mark.parametrize("control", ["base", "identity"])
def test_chunk_guard_cannot_be_replaced_with_token_metric(control):
    rows = fake_rows()
    for r in rows:
        r["metrics"][control]["chunk"] = 0.1
    assert not q.aggregate(rows)["heldout_primary_gate_passed"]


def test_offline_budget_exhausts_before_increment():
    for name, limit in q.LIMITS.items():
        counts = Counter({name: limit})
        with pytest.raises(RuntimeError, match="before dispatch"):
            q.take(counts, name)
        assert counts[name] == limit
    assert q.LIMITS["decoder"] == 48 + 7*32 + 2*32
    assert q.LIMITS["predictor"] == 64 + 7*32 + 3*32


def test_native_budget_has_local_and_global_guards():
    b = q.Budget()
    b.episode["measurement"] = 280
    with pytest.raises(RuntimeError):
        b.check("measurement")
    b.episode["measurement"] = 0
    b.total["measurement"] = 2240
    with pytest.raises(RuntimeError):
        b.check("measurement")


def test_independent_metric_reduction():
    visual = (torch.ones(1, 2, 3), torch.ones(1, 2, 3))
    sample = {"future": tuple(torch.zeros_like(v) for v in visual),
              "inputs": (*visual, torch.tensor([[True, False]]), torch.ones(1, 2, dtype=torch.bool))}
    output, oracle = torch.ones(1, 50, 32), torch.zeros(1, 50, 32)
    expected = {"row0": 1.0, "chunk": 1.0, "latent": 1.0}
    assert q.metrics(visual, output, sample, oracle) == expected
    assert audit.independent_metrics(visual, output, oracle, sample) == expected


def test_zero_and_subtract_before_add_regression():
    base, delta = (torch.tensor([1.0]),), (torch.tensor([1e8]),)
    assert torch.equal(q.acr.compose(base, delta, delta, "centered")[0], base[0])
    assert not torch.equal((base[0] + delta[0]) - delta[0], base[0])


@pytest.mark.parametrize("delay", [0, 1, 3, 4, 8])
def test_new_prefix_uses_actual_delay_not_old_identity_table(delay):
    sample = {"delay": delay, "current_index": 10, "future_index": 10 + delay,
              "actions": torch.zeros(1, 8, 7), "mask": (torch.arange(8) < delay)[None]}
    q.validate_prefix(sample)
    sample["future_index"] += 1
    with pytest.raises(ValueError, match="Future index"):
        q.validate_prefix(sample)


def test_nonfinite_metrics_fail_closed():
    visual = (torch.full((1, 2, 3), float("nan")), torch.ones(1, 2, 3))
    sample = {"future": tuple(torch.zeros_like(v) for v in visual),
              "inputs": (*visual, torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))}
    with pytest.raises(ValueError, match="Nonfinite"):
        q.metrics(visual, torch.zeros(1, 50, 32), sample, torch.zeros(1, 50, 32))


def test_auditor_rejects_numeric_tamper():
    with pytest.raises(ValueError):
        audit.compare_json({"row0": 1.0}, {"row0": 1.1})


def test_evaluator_synthetic_cpu_preserves_mismatch_denominators(tmp_path, monkeypatch):
    samples = []
    for request, delay in enumerate((2, 2, 3)):
        tokens = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
        mask = (torch.arange(8) < delay)[None]
        actions = torch.ones(1, 8, 7) * (request + 1) * mask.unsqueeze(-1)
        samples.append({"task": 6, "initial_state_id": 0, "request_id": request, "delay": delay,
                        "split": "qualification", "inputs": (*tokens, torch.ones(1, 2, dtype=torch.bool),
                        torch.ones(1, 2, dtype=torch.bool)), "future": tokens, "actions": actions,
                        "mask": mask, "archived_full_chunk": torch.zeros(1, 50, 32)})

    def fake_base(model, sample, counts):
        q.take(counts, "predictor")
        return tuple(v + 0.25 for v in sample["inputs"][:2])

    def fake_branch(model, sample, actions, counts):
        q.take(counts, "predictor")
        return tuple(torch.ones_like(v) * (0.125 + actions.sum() * 0.01) for v in sample["inputs"][:2])

    def fake_decode(runtime, sample, visual, output, counts, label):
        q.take(counts, "decoder")
        return torch.ones(1, 50, 32) * visual[0].mean()

    monkeypatch.setattr(q, "base_tokens", fake_base)
    monkeypatch.setattr(q, "branch", fake_branch)
    monkeypatch.setattr(q, "decode", fake_decode)
    counts = Counter()
    rows = q.evaluate(samples, dict.fromkeys(("base", "centered", "ordinary")), None, tmp_path, counts)
    assert counts["predictor"] == 7*3 + 3*2
    assert counts["decoder"] == 7*3 + 2*2
    assert counts["zero_exact"] == counts["identity_exact"] == 3
    assert rows[-1]["donor"] is None
    assert "centered_mismatched" not in rows[-1]["metrics"]
    assert len(list((tmp_path / "predictions").glob("*.pt"))) == 3


def test_report_includes_adverse_results_and_gates():
    summary = q.aggregate(fake_rows())
    text = report.render({"status": "completed", "execution_head": "abc", **summary},
                         {"independent_contract_accepted": True, "summary": summary, **summary})
    assert "最不利" in text and "基底−中心化" in text and "净收益份额" in text
    assert "不是部署延迟" in text and "task 6/7" in text
    assert "技术证据未被独立接纳" in report.render({"status": "technical_failure"}, {})


def test_preparation_readback_identity_and_digest(tmp_path, monkeypatch):
    head = "a" * 40
    prep, output, source = tmp_path / "prep", tmp_path / "out", tmp_path / "source"
    prep.mkdir()
    source.write_text("frozen")
    monkeypatch.setattr(q, "preparation_path", lambda _: prep)
    monkeypatch.setattr(q, "output_path", lambda _: output)
    monkeypatch.setattr(q, "source_files", lambda: [source])
    monkeypatch.setattr(q, "history_inventory", lambda _: [])
    (prep / "preparation.json").write_text(json.dumps({"execution_head": head, "manifest": q.manifest(),
        "source_hashes": {str(source): q.digest(source)}, "history": []}))
    body = f"F-ACQ1 {head} {output} {q.digest(prep / 'preparation.json')}"
    (prep / "registration.md").write_text(body)
    payload = {"id": 123, "body": body, "issue_url": "https://api.github.com/repos/Lebron-233/lerobot/issues/1"}
    (prep / "registration_readback.json").write_text(json.dumps(payload))
    q.validate_preparation(head, registration=True)
    payload["issue_url"] = "https://api.github.com/repos/Lebron-233/lerobot/issues/2"
    (prep / "registration_readback.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="registration"):
        q.validate_preparation(head, registration=True)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source bytes"):
        q.validate_preparation(head)


def test_empty_mismatch_fails_science_gate_without_invalid_averages():
    rows = fake_rows()
    for r in rows:
        del r["metrics"]["centered_mismatched"]
        del r["metrics"]["ordinary_mismatched"]
        r["donor"] = None
    summary = q.aggregate(rows)
    assert not summary["heldout_primary_gate_passed"]
    assert summary["contrasts"]["centered_mismatched"]["macro_benefit"] is None
    assert audit.independent_gates(rows) == (False, False)


def test_cpu_tests_do_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
