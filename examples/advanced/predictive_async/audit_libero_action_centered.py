"""F-ACR1 independent CPU audit; does not import a policy or run a predictor."""

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import audit_libero_prefix_mismatch as reductions
import torch

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "outputs/smolvla_prefix_mismatch_a6a6e997"
PREP = REPO / "outputs/smolvla_action_centered_preparation_11c81ae1"
ARMS = ("centered", "ordinary")


def equal(left, right, message):
    if not torch.equal(left, right):
        raise ValueError(message)


def close(left, right, message):
    if not math.isfinite(left) or abs(left - right) > 1e-7 + 1e-6 * abs(right):
        raise ValueError(f"{message}: {left} != {right}")


def candidate_gate(metrics, contrasts, best_step):
    c = metrics["centered"]["true"]
    return bool(
        best_step > 0
        and all(contrasts[name]["macro_direction"] == "better" for name in
                ("base_minus_centered", "mismatch_minus_centered", "ordinary_minus_centered"))
        and all(contrasts[name]["episode_directions"].get("better", 0) >= 3 for name in
                ("base_minus_centered", "mismatch_minus_centered"))
        and all(reductions.direction(metrics[name]["chunk"]["episode_macro"],
                                     c["chunk"]["episode_macro"]) != "worse" for name in ("base", "identity"))
    )


def audit(output):
    started, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    torch.set_num_threads(1)
    run = json.loads((output / "result.json").read_text())
    if run["status"] != "completed" or run["execution"]["child_exit_code"] != 0:
        raise ValueError("Run did not complete and exit successfully")
    if run["execution"]["stop_reason"] is not None or run["execution"]["active_at_exit"]:
        raise ValueError("Run was interrupted or has an unclosed phase")
    expected_counts = {"predictor_forwards": 1143, "decoder": 850, "gradient_decoder": 144,
                       "updates": 144, "backward": 144, "vla_loads": 1, "branch_initializations": 2,
                       "base_reference_exact": 88, "base_invariant_exact": 120}
    if any(run["counts"].get(k) != v for k, v in expected_counts.items()):
        raise ValueError("Formal budget differs")
    if any(run[k] != 0 for k in ("new_env", "new_native", "visual_encodings", "test_reads", "real_robot", "retries")):
        raise ValueError("Additional computation outside protocol")
    if run["attempts"] != 1 or not run["vla_frozen"] or not run["graph_released"]:
        raise ValueError("Freeze, cleanup or attempt contract failed")
    source = {}
    for path in reductions.LABELS:
        for s in torch.load(path, map_location="cpu", weights_only=False):
            if s["task"] not in range(8):
                raise ValueError("Non-development data was loaded")
            k = (s["task"], s["initial_state_id"], s["request_id"])
            if k in source:
                raise ValueError("Duplicate source identity")
            source[k] = s
    selected = []
    for task in range(8):
        for state in ((46, 48, 49) if task < 6 else (48, 49)):
            keys = sorted(k for k in source if k[:2] == (task, state))[:4]
            if len(keys) != 4:
                raise ValueError("Incomplete fixed episode")
            selected.extend(keys)
    source = {k: source[k] for k in selected}
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest != json.loads((PREP / "manifest.json").read_text()):
        raise ValueError("Preparation manifest changed")
    assignments = {tuple(r["key"]): r for r in manifest["rows"]}
    if set(assignments) != set(source) or len(source) != 88:
        raise ValueError("Selection differs from fixed source")
    groups = defaultdict(list)
    for k, s in sorted(source.items()):
        if s["split"] != ("train" if k[0] < 6 else "validation"):
            raise ValueError("Split changed")
        groups[(s["split"], k[0], k[1], s["delay"])].append(k)
    donors = {}
    for group in groups.values():
        for i, k in enumerate(group):
            donors[k] = group[(i + 1) % len(group)] if len(group) > 1 else None
            if assignments[k]["donor"] != (list(donors[k]) if donors[k] else None):
                raise ValueError("Donor rule differs")
    original = torch.load(BASE / "predictions.pt", map_location="cpu", weights_only=False)
    bases = {tuple(r["key"]): r for r in original if r["context"] == "no_action"}
    del original
    references = torch.load(output / "base_references.pt", map_location="cpu", weights_only=False)
    if len(references) != 88 or {tuple(r["key"]) for r in references} != set(source):
        raise ValueError("Base replay coverage differs")
    for r in references:
        equal(r["output"], bases[tuple(r["key"])]["output"], "Base replay not exact")
    weights = json.loads((output / "training_weights.json").read_text())
    if weights != json.loads((PREP / "weights.json").read_text()):
        raise ValueError("Case weights or scales changed")
    weight_by_key = {tuple(r["key"]): r["case_weight"] for r in weights["rows"]}
    initial = [torch.load(output / f"initial_{a}.pt", map_location="cpu", weights_only=True) for a in ARMS]
    if initial[0].keys() != initial[1].keys():
        raise ValueError("Initial parameter names differ")
    for k in initial[0]:
        equal(initial[0][k], initial[1][k], "Branch initialization differs")
    comparisons, audited_rows = 0, 0

    def check_row(r, arm):
        nonlocal comparisons, audited_rows
        k, context = tuple(r["key"]), r["context"]
        s, base = source[k], bases[k]
        if r["arm"] != arm or r["split"] != s["split"]:
            raise ValueError("Arm or split metadata changed")
        actions = (torch.zeros_like(s["actions"]) if context == "zero" else
                   source[donors[k]]["actions"] if context == "mismatched" and donors[k] else
                   s["actions"] if context == "true" else None)
        if actions is None:
            raise ValueError("Illegal context or singleton donor")
        equal(r["actual_actions"], actions, "Actual action input differs")
        equal(r["actual_mask"], s["mask"], "Actual prefix mask differs")
        for b, a, z, raw, visual in zip(base["raw_visual"], r["action_delta"], r["zero_delta"],
                                      r["raw_visual"], r["visual"], strict=True):
            if not torch.isfinite(raw).all():
                raise ValueError("Nonfinite visual tokens")
            expected = b + (a - z if arm == "centered" else a)
            equal(raw, expected, "Saved centered/ordinary composition differs")
            equal(visual, raw.to(torch.bfloat16).float(), "Quantization boundary changed")
            if arm == "ordinary" and torch.count_nonzero(z):
                raise ValueError("Ordinary branch unexpectedly subtracts a zero-action prediction")
            if arm == "centered" and context == "zero":
                equal(raw, b, "Centered null-action invariant failed")
        if r["output"].shape != (1, 50, 32) or not torch.isfinite(r["output"]).all():
            raise ValueError("Invalid complete output")
        if arm == "centered" and context == "zero":
            equal(r["output"], base["output"], "Centered null-action output differs")
        m = reductions.metrics(r["visual"], r["output"], s)
        for name in m:
            close(r["metrics"][name], m[name], f"Metric {k}/{arm}/{context}/{name}")
            comparisons += 1
        audited_rows += 1
        return {"key": list(k), **m}

    reports, contrasts, selected_steps, by_arm = {}, {}, {}, {}
    for arm in ARMS:
        history = json.loads((output / f"history_{arm}.json").read_text())
        if [h["step"] for h in history] != [0, 36, 72]:
            raise ValueError("Validation schedule changed")
        curve_arrays = {}
        for step, entry in zip((0, 36, 72), history, strict=True):
            arrays = torch.load(output / f"assessment_val_{arm}_{step}.pt", map_location="cpu", weights_only=False)
            if len(arrays) != 16 or {tuple(r["key"]) for r in arrays} != {k for k in source if k[0] >= 6}:
                raise ValueError("Validation coverage changed")
            metrics = reductions.summary([check_row(r, arm) for r in arrays])
            for name in metrics:
                close(entry["metrics"][name]["episode_macro"], metrics[name]["episode_macro"], "History macro")
                comparisons += 1
            if step == 0:
                for r in arrays:
                    equal(r["output"], bases[tuple(r["key"])]["output"], "Step-zero output changed")
            curve_arrays[step] = {tuple(r["key"]): r for r in arrays}
        chosen = min(history, key=lambda h: h["metrics"]["row0"]["episode_macro"])["step"]
        checkpoint = torch.load(output / f"{arm}.pt", map_location="cpu", weights_only=False)
        if checkpoint["best_step"] != chosen or checkpoint["seed"] != 20260912:
            raise ValueError("Checkpoint reselection or seed change")
        selected_weights = torch.load(output / f"weights_{arm}_{chosen}.pt", map_location="cpu", weights_only=True)
        for k, v in checkpoint["state_dict"].items():
            equal(v, selected_weights[k], "Saved checkpoint is not selected validation snapshot")
        selected_steps[arm] = chosen
        updates = torch.load(output / f"training_{arm}.pt", map_location="cpu", weights_only=False)
        if len(updates) != 72:
            raise ValueError("Wrong training update count")
        for i, r in enumerate(updates):
            task, sweep = i % 6, i // 6
            state = (46, 48, 49)[sweep % 3]
            expected = sorted(k for k in source if k[:2] == (task, state))[sweep // 3]
            if tuple(r["key"]) != expected or r["step"] != i + 1 or r["context"] != "true":
                raise ValueError("Training sequence changed")
            if r["weight"] != weight_by_key[expected] or not math.isfinite(r["gradient_norm"]):
                raise ValueError("Training weight or finite-gradient contract differs")
            m = check_row(r, arm)
            objective = m["latent"] / weights["scales"]["latent"] + r["weight"] * m["row0"] / weights["scales"]["row0"]
            close(r["objective"], objective, "Independent training objective")
            comparisons += 1
        del updates
        arrays = torch.load(output / f"assessment_selected_{arm}.pt", map_location="cpu", weights_only=False)
        expected = {(k, c) for k in source for c in ("true", "zero", "mismatched") if c != "mismatched" or donors[k]}
        if len(arrays) != 261 or {(tuple(r["key"]), r["context"]) for r in arrays} != expected:
            raise ValueError("Selected assessment coverage differs")
        grouped = defaultdict(list)
        for r in arrays:
            k = tuple(r["key"])
            grouped[r["split"], r["context"]].append(check_row(r, arm))
            if k[0] >= 6 and r["context"] == "true":
                equal(r["output"], curve_arrays[chosen][k]["output"], "Selected validation replay differs")
                for a, b in zip(r["visual"], curve_arrays[chosen][k]["visual"], strict=True):
                    equal(a, b, "Selected validation tokens differ")
        by_arm[arm] = grouped
        del arrays
    for split in ("train", "validation"):
        keys = [k for k in source if source[k]["split"] == split]
        eligible = {k for k in keys if donors[k]}
        base_rows = [{"key": list(k), **reductions.metrics(bases[k]["visual"], bases[k]["output"], source[k])} for k in keys]
        identity_rows = [{"key": list(k), **reductions.metrics(source[k]["inputs"][:2], source[k]["archived_full_chunk"], source[k])} for k in keys]
        reports[split] = {"base": reductions.summary(base_rows), "identity": reductions.summary(identity_rows)}
        for arm in ARMS:
            reports[split][arm] = {c: reductions.summary(by_arm[arm][split, c]) for c in ("true", "zero", "mismatched")}
        centered = by_arm["centered"][split, "true"]
        paired = [r for r in centered if tuple(r["key"]) in eligible]
        contrasts[split] = {
            "base_minus_centered": reductions.comparison(base_rows, centered),
            "mismatch_minus_centered": reductions.comparison(by_arm["centered"][split, "mismatched"], paired),
            "ordinary_minus_centered": reductions.comparison(by_arm["ordinary"][split, "true"], centered),
            "ordinary_zero_minus_true": reductions.comparison(by_arm["ordinary"][split, "zero"], by_arm["ordinary"][split, "true"]),
        }
        reports[split]["mismatch_eligible_centered_true"] = reductions.summary(paired)
    active, phases = {}, Counter()
    for line in (output / "events.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev["event"] == "started":
            if ev["phase"] in active:
                raise ValueError("Duplicate active phase")
            active[ev["phase"]] = ev
        elif ev["event"] == "returned":
            if ev["seconds"] > active.pop(ev["phase"])["limit"]:
                raise ValueError("Phase deadline exceeded")
        else:
            raise ValueError("Error or unknown phase event")
        phases[ev["event"]] += 1
    captures = json.loads((output / "offline_captures.json").read_text())
    if active or len(captures) != run["counts"]["offline_captures"] or len(captures) > 64:
        raise ValueError("Phase or graph capture accounting differs")
    gate = candidate_gate(reports["validation"], contrasts["validation"], selected_steps["centered"])
    result = {
        "experiment": "F-ACR1", "execution_head": run["execution_head"], "independent_contract_accepted": True,
        "development_candidate_gate_passed": gate, "selected_steps": selected_steps,
        "numeric_comparisons": comparisons, "audited_prediction_rows": audited_rows,
        "metrics": reports, "contrasts": contrasts, "counts": run["counts"], "phase_counts": dict(phases),
        "capture_internal_calls": {name: sum(c[name] for c in captures) for name in
                                   ("eager_setup_calls", "side_stream_warmup_calls", "capture_calls")},
        "execution": run["execution"], "audit_started_at_utc": utc, "audit_seconds": time.perf_counter() - started,
        "cuda_initialized": torch.cuda.is_initialized(), "audit_model_forwards": 0,
        "baseline_qualified": False, "realtime_qualified": False, "predictor_benefit_tested": False,
        "risk_thresholds": None, "old_confirmation": "not_started_untouched", "test_reads": 0,
    }
    if torch.cuda.is_initialized():
        raise ValueError("Independent audit initialized CUDA")
    reductions.dump(output / "independent_audit.json", result)
    print(json.dumps({k: result[k] for k in ("independent_contract_accepted", "development_candidate_gate_passed",
                                            "selected_steps", "numeric_comparisons", "counts")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args().output.resolve())
