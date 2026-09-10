"""Independent CPU reductions of F-PFX1 arrays; no predictor or policy imports."""

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
PREP = REPO / "outputs/smolvla_prefix_mismatch_preparation_ece0d933"
LABELS = (
    REPO / "outputs/smolvla_action_objective_1666c067/development_labels.pt",
    REPO / "outputs/smolvla_coverage_6249b03d/new_labels.pt",
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def direction(control, treatment):
    delta = control - treatment
    tolerance = 1e-7 + 1e-6 * abs(control)
    return "better" if delta > tolerance else "worse" if delta < -tolerance else "tie"


def metrics(visual, output, sample):
    numerator, denominator = 0.0, 0
    for z, target, mask in zip(visual, sample["future"], sample["inputs"][2:4], strict=True):
        difference = (z.double() - target.double()).square()
        numerator += float(difference[mask].sum())
        denominator += int(mask.sum()) * z.shape[-1]
    actions = (output[..., :7].double() - sample["oracle"][..., :7].double()).square()
    return {"latent": numerator / denominator, "row0": float(actions[:, 0].mean()),
            "chunk": float(actions.mean())}


def summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row['key'][0]}/{row['key'][1]}"].append(row)
    result = {}
    for metric in ("latent", "row0", "chunk"):
        per_episode = {k: math.fsum(r[metric] for r in v) / len(v) for k, v in grouped.items()}
        result[metric] = {
            "samples": len(rows), "episodes": len(grouped), "per_episode": per_episode,
            "episode_macro": math.fsum(per_episode.values()) / len(per_episode),
            "sample_mean": math.fsum(r[metric] for r in rows) / len(rows),
        }
    return result


def comparison(control_rows, true_rows):
    controls = {tuple(r["key"]): r for r in control_rows}
    if set(controls) != {tuple(r["key"]) for r in true_rows}:
        raise ValueError("Paired comparison denominators differ")
    c, t = summary(control_rows), summary(true_rows)
    per_sample = [{"key": r["key"], "delta": controls[tuple(r["key"])]["row0"] - r["row0"],
                   "direction": direction(controls[tuple(r["key"])]["row0"], r["row0"])}
                  for r in true_rows]
    per_episode = {k: {"delta": value - t["row0"]["per_episode"][k],
                       "direction": direction(value, t["row0"]["per_episode"][k])}
                   for k, value in c["row0"]["per_episode"].items()}
    return {
        "row0_control_minus_true": c["row0"]["episode_macro"] - t["row0"]["episode_macro"],
        "macro_direction": direction(c["row0"]["episode_macro"], t["row0"]["episode_macro"]),
        "episode_directions": dict(Counter(v["direction"] for v in per_episode.values())),
        "sample_directions": dict(Counter(v["direction"] for v in per_sample)),
        "per_episode": per_episode, "per_sample": per_sample,
    }


def token_change(left, right, masks):
    diffs = torch.cat([(a.double() - b.double())[mask].flatten()
                       for a, b, mask in zip(left, right, masks, strict=True)])
    changed = int(torch.count_nonzero(diffs))
    return {"mse": float(diffs.square().mean()), "max_abs": float(diffs.abs().max()),
            "changed_elements": changed, "valid_elements": diffs.numel(),
            "changed_fraction": changed / diffs.numel()}


def audit(output):
    started, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    torch.set_num_threads(1)
    manifest = json.loads((PREP / "manifest.json").read_text())
    run = json.loads((output / "result.json").read_text())
    if run["status"] != "completed" or run["execution"]["child_exit_code"] != 0:
        raise ValueError("Only a completed, exited run can be accepted")
    if run["execution"]["stop_reason"] is not None or run["execution"]["active_at_exit"]:
        raise ValueError("Unclosed or interrupted execution")
    expected_counts = {"vla_loads": 1, "predictor_loads": 2, "predictor_forwards": 261,
                       "decoder": 261, "selected_replay_exact": 176}
    if any(run["counts"].get(k) != v for k, v in expected_counts.items()):
        raise ValueError("Actual formal budget differs")
    if any(run[k] != 0 for k in ("updates", "backward", "visual_encodings", "new_env",
                                 "new_native", "test_reads", "real_robot", "retries")):
        raise ValueError("Forbidden additional computation")
    if not all(run[k] is True for k in ("graph_released", "vla_frozen", "frozen_models_unchanged")):
        raise ValueError("Freeze or cleanup contract failed")
    loaded = []
    for path in LABELS:
        loaded.extend(torch.load(path, map_location="cpu", weights_only=False))
    if any(s["task"] not in range(8) for s in loaded):
        raise ValueError("Non-development label entered the computation")
    source = {(s["task"], s["initial_state_id"], s["request_id"]): s for s in loaded}
    assignments = {tuple(r["key"]): r for r in manifest["rows"]}
    # Reconstruct the protocol independently of the runner's donor_map/select_data.
    selected = []
    for task in range(8):
        for state in ((46, 48, 49) if task < 6 else (48, 49)):
            group = sorted(k for k in source if k[:2] == (task, state))[:4]
            if len(group) != 4:
                raise ValueError("Incomplete fixed first-four episode")
            selected.extend(group)
    if set(selected) != set(assignments) or len(assignments) != 88:
        raise ValueError("Registered selection is not the fixed common data")
    groups = defaultdict(list)
    for k in sorted(selected):
        s = source[k]
        groups[(s["split"], s["task"], s["initial_state_id"], s["delay"])].append(k)
    for group in groups.values():
        for i, k in enumerate(group):
            expected = list(group[(i + 1) % len(group)]) if len(group) > 1 else None
            if assignments[k]["donor"] != expected:
                raise ValueError("Actual donor is not the identity-only cycle")
    arrays = torch.load(output / "predictions.pt", map_location="cpu", weights_only=False)
    indexed, values, comparisons = {}, defaultdict(list), 0
    for row in arrays:
        k, context = tuple(row["key"]), row["context"]
        if (k, context) in indexed or k not in assignments:
            raise ValueError("Unknown or duplicate prediction")
        assignment, sample = assignments[k], source[k]
        if any(row[field] != assignment[field] for field in ("key", "split", "delay", "donor")):
            raise ValueError("Prediction identity metadata changed")
        if context == "true":
            expected_actions = sample["actions"]
        elif context == "no_action":
            expected_actions = torch.zeros_like(sample["actions"])
        elif context == "mismatched" and assignment["donor"] is not None:
            expected_actions = source[tuple(assignment["donor"])]["actions"]
        else:
            raise ValueError("Invalid context or singleton prediction")
        if not torch.equal(row["actual_actions"], expected_actions) or not torch.equal(row["actual_mask"], sample["mask"]):
            raise ValueError("Actual action input or mask differs from the intervention")
        if row["output"].shape != (1, 50, 32) or not torch.isfinite(row["output"]).all():
            raise ValueError("Invalid full decoder output")
        for raw, visual in zip(row["raw_visual"], row["visual"], strict=True):
            if not torch.isfinite(raw).all() or not torch.equal(raw.to(torch.bfloat16).float(), visual):
                raise ValueError("Saved quantization boundary differs")
        reduced = metrics(row["visual"], row["output"], sample)
        for name, value in reduced.items():
            if abs(value - row["metrics"][name]) > 1e-7 + 1e-6 * abs(value):
                raise ValueError(f"Independent metric differs: {k}/{context}/{name}")
            comparisons += 1
        indexed[k, context] = row
        values[row["split"], context].append({"key": list(k), **reduced})
    expected = {(k, c) for k, a in assignments.items() for c in ("true", "no_action", "mismatched")
                if c != "mismatched" or a["donor"] is not None}
    if set(indexed) != expected or len(arrays) != 261:
        raise ValueError("Incomplete prediction/context coverage")
    for k, assignment in assignments.items():
        s = source[k]
        values[assignment["split"], "identity"].append({
            "key": list(k), **metrics(s["inputs"][:2], s["archived_full_chunk"], s)})
    reports, contrasts, sensitivity = {}, {}, {}
    for split in ("train", "validation"):
        eligible = {k for k, a in assignments.items() if a["split"] == split and a["donor"] is not None}
        paired = {c: [r for r in values[split, c] if tuple(r["key"]) in eligible]
                  for c in ("true", "mismatched", "no_action", "identity")}
        reports[split] = {
            "all": {c: summary(values[split, c]) for c in ("true", "no_action", "identity")},
            "eligible": {c: summary(paired[c]) for c in paired},
        }
        contrasts[split] = {
            "mismatch_minus_true": comparison(paired["mismatched"], paired["true"]),
            "paired_no_action_minus_true": comparison(paired["no_action"], paired["true"]),
            "all_no_action_minus_true": comparison(values[split, "no_action"], values[split, "true"]),
        }
        rows = []
        for k in sorted(eligible):
            true, wrong, s = indexed[k, "true"], indexed[k, "mismatched"], source[k]
            masks = s["inputs"][2:4]
            action_difference = (true["actual_actions"] - wrong["actual_actions"])[s["mask"]].double()
            output_difference = (true["output"][..., :7] - wrong["output"][..., :7]).double()
            rows.append({
                "key": list(k), "prefix_mse": float(action_difference.square().mean()),
                "raw_tokens": token_change(true["raw_visual"], wrong["raw_visual"], masks),
                "quantized_tokens": token_change(true["visual"], wrong["visual"], masks),
                "output_row0_mse_between_inputs": float(output_difference[:, 0].square().mean()),
                "output_chunk_mse_between_inputs": float(output_difference.square().mean()),
            })
        sensitivity[split] = {
            "eligible": len(rows), "prefix_changed": sum(r["prefix_mse"] > 0 for r in rows),
            "raw_token_changed": sum(r["raw_tokens"]["changed_elements"] > 0 for r in rows),
            "quantized_token_changed": sum(r["quantized_tokens"]["changed_elements"] > 0 for r in rows),
            "row0_changed": sum(r["output_row0_mse_between_inputs"] > 0 for r in rows),
            "rows": rows,
        }
    validation, contrast = reports["validation"]["all"], contrasts["validation"]
    stable = all(contrast[c]["macro_direction"] == "better" and
                 contrast[c]["episode_directions"].get("better", 0) >= 3
                 for c in ("mismatch_minus_true", "all_no_action_minus_true"))
    stable = stable and all(direction(validation[c]["chunk"]["episode_macro"],
                                      validation["true"]["chunk"]["episode_macro"]) != "worse"
                            for c in ("identity", "no_action"))
    active, phase_counts = {}, Counter()
    for line in (output / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "started":
            if event["phase"] in active:
                raise ValueError("Repeated active phase")
            active[event["phase"]] = event
            phase_counts["started"] += 1
        elif event["event"] == "returned":
            original = active.pop(event["phase"])
            if event["seconds"] > original["limit"]:
                raise ValueError("Phase exceeded its fixed deadline")
            phase_counts["returned"] += 1
        else:
            raise ValueError("Error or unknown event")
    if active:
        raise ValueError("Unclosed phase")
    captures = json.loads((output / "offline_captures.json").read_text())
    if len(captures) != run["counts"]["offline_captures"] or len(captures) > 64:
        raise ValueError("Graph capture count differs")
    result = {
        "experiment": "F-PFX1", "execution_head": run["execution_head"],
        "independent_contract_accepted": True, "numeric_comparisons": comparisons,
        "metrics": reports, "contrasts": contrasts, "sensitivity": sensitivity,
        "stable_development_action_utility": bool(stable),
        "manifest_counts": manifest["counts"], "eligible_counts": manifest["eligible"],
        "singleton_keys": [r["key"] for r in manifest["rows"] if r["donor"] is None],
        "formal_counts": run["counts"], "phase_counts": dict(phase_counts),
        "capture_records": captures, "execution": run["execution"],
        "audit_started_at_utc": utc, "audit_seconds": time.perf_counter() - started,
        "cuda_initialized": torch.cuda.is_initialized(), "new_model_forward_in_audit": 0,
        "updates": 0, "new_native": 0, "test_reads": 0,
        "baseline_qualified": False, "realtime_qualified": False,
        "predictor_benefit_tested": False, "risk_thresholds": None,
        "old_confirmation": "not_started_untouched",
    }
    if torch.cuda.is_initialized():
        raise ValueError("Independent audit initialized CUDA")
    dump(output / "independent_audit.json", result)
    print(json.dumps({"independent_contract_accepted": True, "numeric_comparisons": comparisons,
                      "stable_development_action_utility": bool(stable),
                      "validation_contrasts": contrasts["validation"],
                      "validation_metrics": reports["validation"]["all"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args().output.resolve())
