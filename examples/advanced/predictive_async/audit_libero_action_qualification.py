"""CPU-only F-ACQ1 evidence audit. No model forward, Env, CUDA, or metric selection."""

import argparse
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import libero_action_qualification as q
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b, message):
    q.pilot.require_equal(a, b, message)


def close(a, b, message):
    require(math.isfinite(float(a)) and math.isfinite(float(b))
            and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7), message)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def independent_metrics(visual, output, oracle, sample):
    # Deliberately separate from the runner's reduction.
    error = output[0, :, :7].to(torch.float64) - oracle[0, :, :7].to(torch.float64)
    first = math.fsum(float(x) ** 2 for x in error[0]) / 7
    chunk = float(torch.sum(error * error)) / 350
    sums, sizes = [], []
    for index in range(2):
        mask = sample["inputs"][index + 2].bool()
        delta = visual[index].double()[mask] - sample["future"][index].double()[mask]
        sums.append(float(torch.sum(delta * delta)))
        sizes.append(delta.numel())
    require(sum(sizes) > 0, "Empty visual token support")
    return {"row0": first, "chunk": chunk, "latent": math.fsum(sums) / sum(sizes)}


def independent_gates(rows):
    """Recompute decisions without calling runner aggregate or tolerance helpers."""
    def macro(arm, metric, subset):
        groups = defaultdict(list)
        for r in subset:
            groups[tuple(r["key"][:2])].append(r["metrics"][arm][metric])
        return {k: math.fsum(v) / len(v) for k, v in groups.items()}

    def average(values):
        return math.fsum(values) / len(values)

    conditions = [{tuple(r["key"][:2]) for r in rows} == set(q.PAIRS)]
    contrasts = {}
    for control in ("base", "ordinary_true", "centered_mismatched"):
        subset = [r for r in rows if control in r["metrics"]]
        if not subset:
            return False, False
        a, b = macro("centered_true", "row0", subset), macro(control, "row0", subset)
        av, bv = average(list(a.values())), average(list(b.values()))
        conditions.append(bv - av > 1e-7 + 1e-6 * abs(bv))
        contrasts[control] = {k: b[k] - a[k] for k in a}
        if control in ("base", "centered_mismatched"):
            conditions.append(sum(b[k] - a[k] > 1e-7 + 1e-6 * abs(b[k]) for k in a) >= 6)
        if control == "centered_mismatched":
            conditions.append(len(a) == 8)
    for control in ("base", "identity"):
        a = average(list(macro("centered_true", "chunk", rows).values()))
        b = average(list(macro(control, "chunk", rows).values()))
        conditions.append(a <= b + 1e-7 + 1e-6 * abs(b))
    primary = all(conditions)
    wins = sum(r["metrics"]["base"]["row0"] - r["metrics"]["centered_true"]["row0"] >
               1e-7 + 1e-6 * abs(r["metrics"]["base"]["row0"]) for r in rows)
    benefits = contrasts["base"]
    base = average(list(macro("base", "row0", rows).values()))
    leave_out = len(benefits) == 8 and all(
        average([v for j, v in benefits.items() if j != k]) > 1e-7 + 1e-6 * abs(base) for k in benefits)
    return primary, primary and wins > len(rows) / 2 and leave_out


def compare_json(a, b, path="summary"):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), f"Keys differ: {path}")
        for k in a:
            compare_json(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f"Length differs: {path}")
        for i, value in enumerate(a):
            compare_json(value, b[i], f"{path}[{i}]")
    elif isinstance(a, float):
        close(a, b, path)
    else:
        require(a == b, f"Value differs: {path}")


def audit(output):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "Audit entered with CUDA initialized")
    result = json.loads((output / "result.json").read_text())
    head = result["execution_head"]
    require(output == q.output_path(head), "Output identity differs")
    require(result["status"] == "completed" and result["first_failure"] is None, "Run not completed")
    execution = result["execution"]
    require(execution["child_exit_code"] == 0 and execution["exit_confirmed"] is True
            and not execution["forced_termination"] and execution["stop_reason"] is None
            and not execution["pending_calls_at_exit"] and not execution["active_phases_at_exit"],
            "Exit or pending-call evidence failed")
    require(result["attempts"] == 1 and result["retries"] == 0, "Retry budget differs")
    require(result["graph_released"] and result["vla_frozen"] and result["frozen_weights_unchanged"], "Freeze/Graph receipt failed")
    for name in ("training_updates", "backward", "real_robot", "old_test_reads"):
        require(result[name] == 0, f"Forbidden action: {name}")
    for name in ("baseline_qualified", "realtime_qualified", "predictor_benefit_tested", "predictor_controls_environment"):
        require(result[name] is False, f"Qualification flag changed: {name}")
    require(result["risk_thresholds"] is None and result["old_confirmation"] == "not_started_untouched", "Safety scope changed")
    prepared = json.loads((q.preparation_path(head) / "preparation.json").read_text())
    require(prepared["source_hashes"] == {str(p): q.digest(p) for p in q.source_files()}, "Source bytes changed")
    compare_json(json.loads((output / "manifest.json").read_text()), prepared["manifest"], "manifest")
    q.check_unused(prepared["history"])
    old_history = [r for r in q.history_inventory(q.e.REPO / "outputs")
                   if not r["path"].startswith(output.name + "/")]
    require(old_history == prepared["history"], "Historical inputs changed or concurrent collection occurred")
    registration = json.loads((q.preparation_path(head) / "registration_readback.json").read_text())
    body = (q.preparation_path(head) / "registration.md").read_text()
    require(registration.get("body") == body and all(v in body for v in
            (head, str(output), q.digest(q.preparation_path(head) / "preparation.json"))), "Registration differs")

    before, after = load(output / "frozen_checkpoints_before.pt"), load(output / "frozen_weights_after.pt")
    for arm, path in q.CHECKPOINTS.items():
        original = load(path)
        q.checkpoint_valid(before[arm], arm)
        require(original["state_dict"].keys() == before[arm]["state_dict"].keys() == after[arm].keys(), "Weight names differ")
        for name, value in original["state_dict"].items():
            exact(value, before[arm]["state_dict"][name], "Initial checkpoint differs")
            exact(value, after[arm][name], "Frozen weights changed")

    anchors = load(output / "anchor_replay.pt")
    _, validation = q.pfx.load_data()
    require([tuple(r["key"]) for r in anchors] == sorted(q.pfx.key(s) for s in validation), "Anchor identities differ")
    old_base = {tuple(r["key"]): r for r in load(q.acr.BASE / "predictions.pt") if r["context"] == "no_action"}
    refs = {arm: {tuple(r["key"]): r for r in load(q.SOURCE / f"assessment_selected_{arm}.pt")
                  if r["context"] == "true"} for arm in ("centered", "ordinary")}
    for r in anchors:
        key = tuple(r["key"])
        for a, b in zip(r["base_raw"], old_base[key]["raw_visual"], strict=True):
            exact(a, b, "Anchor raw base differs")
        exact(r["base_output"], old_base[key]["output"], "Anchor base output differs")
        for arm in refs:
            for field in ("raw_visual", "visual"):
                for a, b in zip(r[arm][field], refs[arm][key][field], strict=True):
                    exact(a, b, f"Anchor {arm} {field} differs")
            exact(r[arm]["output"], refs[arm][key]["output"], "Anchor output differs")

    samples = load(output / "aligned_cache.pt")
    by_key = {q.pfx.key(s): s for s in samples}
    require(len(by_key) == len(samples) and 8 <= len(samples) <= 32, "Sample count/uniqueness differs")
    native_keys, native_totals, selections = set(), Counter(), []
    for spec in prepared["manifest"]["rows"]:
        directory = output / f"episode_{spec['ordinal']:03d}"
        record, arrays = json.loads((directory / "result.json").read_text()), load(directory / "arrays.pt")
        require(record["spec"] == spec and record["status"] == "completed" and record["first_failure"] is None, "Native identity/status differs")
        for flag in ("environment_closed", "worker_joined", "graph_released", "original_sampler_restored", "metrics_closed"):
            require(record[flag] is True, f"Native cleanup failed: {flag}")
        pairs, excluded = q.cov.selected_pairs(record, arrays)
        selections.append({"ordinal": spec["ordinal"], "requests": [p["request_id"] for p in pairs], "excluded": excluded})
        for pair in pairs:
            key = (spec["task_id"], spec["initial_state_id"], pair["request_id"])
            native_keys.add(key)
            s = by_key[key]
            require(s["split"] == "qualification" and s["ordinal"] == spec["ordinal"], "Sample split/ordinal differs")
            for field in ("current_index", "future_index", "delay"):
                require(s[field] == pair[field], f"Alignment differs: {field}")
            for field in ("actions", "mask"):
                exact(s[field], pair[field], f"Committed prefix differs: {field}")
            q.validate_prefix(s)
            for a, b in zip(s["inputs"], pair["cached"]["inputs"], strict=True):
                exact(a, b, "Cached model input differs from native")
            exact(s["archived_full_chunk"], pair["cached"]["full_chunk"], "Archived native output differs")
        for kind, (maximum, _) in q.NATIVE_LIMITS.items():
            value = record["budget"].get(kind, 0)
            require(isinstance(value, int) and 0 <= value <= maximum, f"Episode budget exceeded: {kind}")
            native_totals[kind] += value
        require(record["budget"]["episodes"] == 1, "Episode creation count differs")
        require(len(record["captures"]) == record["budget"]["capture"], "Native capture count differs")
    require(native_keys == set(by_key), "Native selection is not exactly the aligned cache")
    compare_json(selections, json.loads((output / "selections.json").read_text()), "selections")
    donors = q.pfx.donor_map(samples)
    expected_paths = {f"{k[0]}_{k[1]}_{k[2]}.pt" for k in by_key}
    require({p.name for p in (output / "predictions").glob("*.pt")} == expected_paths, "Prediction coverage differs")
    rows, numerical = [], 0
    for key, s in sorted(by_key.items()):
        r = load(output / "predictions" / f"{key[0]}_{key[1]}_{key[2]}.pt")
        require(r["key"] == list(key) and r["delay"] == s["delay"], "Prediction key/delay differs")
        require(r["donor"] == (list(donors[key]) if donors[key] else None), "Mismatch donor differs")
        expected_arms = set(q.ARMS) - ({"centered_mismatched", "ordinary_mismatched"} if donors[key] is None else set())
        require(set(r["metrics"]) == expected_arms and set(r["outputs"]) == expected_arms | {"oracle"}, "Arm coverage differs")
        exact(r["outputs"]["identity"], s["archived_full_chunk"], "Full identity replay differs")
        for a, b in zip(r["visual"]["oracle"], s["future"], strict=True):
            exact(a, b, "Oracle future tokens differ")
        for a, b in zip(r["visual"]["identity"], s["inputs"][:2], strict=True):
            exact(a, b, "Identity tokens differ")
        for a, b in zip(r["base_raw"], r["visual"]["base"], strict=True):
            exact(a.to(torch.bfloat16).float(), b, "Base precision boundary differs")
        for name, evidence in r["residuals"].items():
            arm, context = name.split("_", 1)
            used = q.pfx.with_prefix(s, by_key[donors[key]]) if context == "mismatched" else s
            expected_actions = torch.zeros_like(s["actions"]) if context == "zero" else used["actions"]
            exact(evidence["actual_actions"], expected_actions, "Actual residual action differs")
            exact(evidence["actual_mask"], s["mask"], "Actual residual mask differs")
            for index in range(2):
                a, z, base = evidence["action_delta"][index], evidence["zero_delta"][index], r["base_raw"][index]
                expected = base + (a - z) if arm == "centered" else base + a
                exact(expected, evidence["raw_visual"][index], "Subtract-before-add algebra differs")
                exact(expected.to(torch.bfloat16).float(), evidence["visual"][index], "BF16 boundary differs")
                exact(evidence["visual"][index], r["visual"][name][index], "Decoded visual evidence differs")
                if name == "centered_zero":
                    exact(a, z, "Centered zero branch mismatch")
                    exact(expected, base, "Centered zero raw tokens differ")
        exact(r["outputs"]["centered_zero"], r["outputs"]["base"], "Centered zero full output differs")
        scores = {}
        for arm in expected_arms:
            require(r["outputs"][arm].shape == (1, 50, 32) and torch.isfinite(r["outputs"][arm]).all(), "Invalid decoder output")
            scores[arm] = independent_metrics(r["visual"][arm], r["outputs"][arm], r["outputs"]["oracle"], s)
            for metric, value in scores[arm].items():
                close(value, r["metrics"][arm][metric], "Saved metric differs")
                numerical += 1
        rows.append({"key": list(key), "delay": s["delay"], "donor": r["donor"], "metrics": scores})
    logged = [json.loads(line) for line in (output / "metric_rows.jsonl").read_text().splitlines()]
    compare_json(rows, logged, "metric_rows")
    summary = q.aggregate(rows)
    for field, value in summary.items():
        compare_json(value, result[field], field)
    primary, robust = independent_gates(rows)
    require(primary == result["heldout_primary_gate_passed"] and robust == result["heldout_robustness_gate_passed"], "Independent decision differs")
    n, m = len(rows), sum(v is not None for v in donors.values())
    expected = {"encoding": n + 8, "decoder": 48 + 7*n + 2*m, "predictor": 64 + 7*n + 3*m,
                "anchor_exact": 16, "current_exact": 8, "identity_exact": n, "zero_exact": n,
                "vla_loads": 1, "predictor_loads": 3}
    require(result["expected_counts"] == expected, "Expected counts changed")
    for k, v in expected.items():
        require(result["counts"][k] == v and (k not in q.LIMITS or v <= q.LIMITS[k]), f"Count differs: {k}")
    require(result["episodes_completed"] == 8 and result["native_budget"]["episodes"] == 8, "Native count differs")
    require(dict(native_totals) == result["native_budget"], "Native totals differ from per-episode budgets")
    for kind, (_, maximum) in q.NATIVE_LIMITS.items():
        require(0 <= result["native_budget"].get(kind, 0) <= maximum, f"Native budget exceeded: {kind}")
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    active, starts, returns = {}, 0, 0
    for event in events:
        name = event["phase"]
        if event["event"] == "started":
            require(name not in active, "Duplicate phase")
            active[name] = event["limit"]
            starts += 1
        else:
            require(event["event"] == "returned" and name in active, "Missing/errored phase")
            require(event["seconds"] <= active.pop(name), "Phase exceeded deadline")
            returns += 1
    require(not active and starts == returns, "Unclosed phases")
    accounting = q.natural.trace.journal_accounting(q.natural.trace.read_events(output / "calls.jsonl"))
    require(accounting == result["accounting"] and accounting["journal_consistent"]
            and not accounting["call_errors"] and not accounting["unknown_calls"], "Native call accounting differs")
    captures = json.loads((output / "offline_captures.json").read_text())
    require(len(captures) == result["counts"]["offline_captures"] <= 8, "Offline capture budget differs")
    require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    return {"independent_contract_accepted": True, "heldout_primary_gate_passed": primary,
            "heldout_robustness_gate_passed": robust, "samples": n, "mismatched_samples": m,
            "numerical_comparisons": numerical, "phases_closed": starts, "cuda_initialized": False,
            "audit_model_forwards": 0, "new_env": 0, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "independent_audit.json").exists():
        raise ValueError("Audit already exists; do not overwrite")
    start = time.perf_counter()
    try:
        result = audit(output)
        result["first_failure"] = None
    except BaseException:
        result = {"independent_contract_accepted": False, "heldout_primary_gate_passed": False,
                  "heldout_robustness_gate_passed": False, "first_failure": traceback.format_exc()}
    result["audit_wall_seconds"] = time.perf_counter() - start
    q.e.write_json(output / "independent_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}), flush=True)
    return 0 if result["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
