"""F-PSD1 independent CPU provenance, Euler arithmetic and action-error audit.

Does not execute the Transformer, preprocessing pipeline, CUDA or an Env.
"""

import argparse
import json
import math
import traceback
from pathlib import Path

import libero_prefix_sampling as r
import numpy as np
import torch
from smolvla_prefix_training import episode_action_window
from verify_smolvla_prefix_training import tensor_sha


def check(ok, message):
    if not ok:
        raise ValueError(message)


def exact(a, b, message):
    check(
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a, b),
        message,
    )


def metrics(pred, target, valid, count):
    a, b = pred.numpy().astype(np.float64)[0, :, :7], target.numpy().astype(np.float64)[0, :, :7]
    check(
        a.shape == b.shape == (50, 7) and np.isfinite(a).all() and np.isfinite(b).all(), "Metric shape/finite"
    )
    live = valid.numpy()[0]
    check(live.dtype == bool and live.shape == (50,) and live[count], "Metric valid first action")
    err = np.square(a - b)
    suffix = live & (np.arange(50) >= count)
    short = suffix & (np.arange(50) < count + 10)
    return {
        "block": float(err[live].mean()),
        "suffix": float(err[suffix].mean()),
        "first": float(err[count].mean()),
        "short": float(err[short].mean()),
    }


def inspect_trace(row, ref):
    c = row["condition_count"]
    trace = row["trace"]
    if row["mode"] == "native_unconditioned":
        check(not trace and c == 0, "Native trace/conditioning unexpected")
        return 0
    check(len(trace) == 10, "Trace must contain ten steps")
    x = ref["noise"].clone()
    clean = torch.zeros_like(x)
    if c:
        clean[:, :c, :7] = row["condition_prefix"]
    for step, event in enumerate(trace):
        if c:
            x[:, :c] = clean[:, :c]
        check(event["x"].shape == event["velocity"].shape == (1, 50, 32), "Trace tensor shape")
        check(torch.isfinite(event["x"]).all() and torch.isfinite(event["velocity"]).all(), "Nonfinite trace")
        # Distinct CPU/GPU mul-add execution is an arithmetic check, not a model re-execution.
        torch.testing.assert_close(event["x"], x, rtol=1e-6, atol=1e-6)
        if c:
            exact(event["x"][:, :c], clean[:, :c], "Clean prefix changed inside integration")
        expected_time = torch.full((1, 50) if c else (1,), 1.0 + step * (-1.0 / 10), dtype=torch.float32)
        if c:
            expected_time[:, :c] = 0
        exact(event["time"], expected_time, "Per-action Euler time changed")
        x = event["x"] + (-1.0 / 10) * event["velocity"]
        if c:
            x[:, :c] = clean[:, :c]
    torch.testing.assert_close(row["full"], x, rtol=1e-6, atol=1e-6)
    if c:
        exact(row["full"][:, :c], clean[:, :c], "Returned prefix changed")
    return 10


def reduce_rows(scored):
    table = {}
    for label in r.LABELS:
        table[label] = {}
        for mode in r.MODES:
            rows = [v for v in scored if v["checkpoint"] == label and v["mode"] == mode]
            check(len(rows) == 16, "Scored population differs")
            group = {}
            for space in ("normalized", "command"):
                group[space] = {}
                for metric in ("block", "suffix", "first", "short"):
                    ids = sorted({v["trajectory_id"] for v in rows})
                    check(len(ids) == 4, "Trajectory population differs")
                    per = {}
                    for identity in ids:
                        vals = [v[space][metric] for v in rows if v["trajectory_id"] == identity]
                        check(len(vals) == 4, "Unequal trajectory denominators")
                        per[identity] = float(np.mean(vals))
                    group[space][metric] = {"mean": float(np.mean(list(per.values()))), "per_trajectory": per}
            table[label][mode] = group
    candidate = table["prefix"]["correct_prefix"]["normalized"]
    primary = ("suffix", "first")

    def better(label, mode):
        return all(candidate[k]["mean"] < table[label][mode]["normalized"][k]["mean"] for k in primary)

    per_gains = {
        k: {
            identity: table["ordinary"]["native_unconditioned"]["normalized"][k]["per_trajectory"][identity]
            - v
            for identity, v in candidate[k]["per_trajectory"].items()
        }
        for k in primary
    }
    gates = {
        "both_metrics_better_than_ordinary_unconditioned": better("ordinary", "native_unconditioned"),
        "both_metrics_better_than_frozen_unconditioned": better("frozen", "native_unconditioned"),
        "at_least_three_trajectories_each_primary_metric": all(
            sum(v > 0 for v in per_gains[k].values()) >= 3 for k in primary
        ),
        "both_metrics_better_than_ordinary_same_prefix": better("ordinary", "correct_prefix"),
        "both_metrics_better_than_frozen_same_prefix": better("frozen", "correct_prefix"),
        "both_metrics_correct_better_than_mismatch": better("prefix", "mismatched_prefix"),
    }
    paired = []
    for case in range(16):
        c = next(
            v
            for v in scored
            if v["checkpoint"] == "prefix" and v["mode"] == "correct_prefix" and v["case"] == case
        )
        b = next(
            v
            for v in scored
            if v["checkpoint"] == "ordinary" and v["mode"] == "native_unconditioned" and v["case"] == case
        )
        paired.append(
            {
                "case": case,
                "trajectory_id": c["trajectory_id"],
                "task": c["task"],
                "anchor": c["anchor"],
                "C": c["C"],
                "ordinary": b["normalized"],
                "prefix": c["normalized"],
                "candidate_minus_ordinary": {k: c["normalized"][k] - b["normalized"][k] for k in primary},
            }
        )
    return {
        "metric_table": table,
        "trajectory_gain_vs_ordinary": per_gains,
        "paired_cases": paired,
        "development_gates": gates,
        "decoded_action_followup_supported": all(gates.values()),
    }


def audit(out):
    check(not torch.cuda.is_initialized(), "CPU audit only")
    result = json.loads((out / "result.json").read_text())
    saved, registration = r.validate(result["head"])
    check(
        out == r.paths(result["head"])[1] and result["registration_id"] == registration,
        "Output/registration identity",
    )
    check(
        result["status"] == "completed"
        and result["first_failure"] is None
        and result["decodes_started"] == result["decodes_returned"] == result["vision_encodes"] == 192
        and result["denoise_steps"] == 1920
        and result["saved_rows"] == 192
        and result["model_loads"] == result["attempts"] == 1
        and result["retries"] == 0
        and result["base_projections_restored"]
        and result["nonprojection_weights_unchanged"]
        and result["sources_unchanged"],
        "Formal execution incomplete",
    )
    check(
        all(
            result[k] == 0
            for k in ("optimizer_updates", "backwards", "new_env", "sealed_evaluations", "graph_captures")
        ),
        "Scope changed",
    )
    check(r.sha(out / "evaluation.pt") == result["evidence_sha256"], "Evidence digest changed")
    evidence = torch.load(out / "evaluation.pt", map_location="cpu", weights_only=True)
    refs, rows = evidence["references"], evidence["rows"]
    meta, schedule = r.dev_contract()
    packets = {k: dict(np.load(r.COHORT / v["packet"]["path"], allow_pickle=False)) for k, v in meta.items()}
    prior = torch.load(r.PRIOR_OUT / "evaluation.pt", map_location="cpu", weights_only=True)
    check(set(refs) == set(range(16)) and len(rows) == 192, "Evidence population")
    for case, ref in refs.items():
        spec = schedule[case]
        check(ref["spec"] == spec, "Reference schedule changed")
        packet = packets[spec["trajectory_id"]]
        raw, valid = episode_action_window(torch.from_numpy(packet["actions"]), spec["anchor"])
        exact(ref["valid"], valid[None], "Terminal validity changed")
        check(
            valid[spec["C"]] and ref["truth"].shape == ref["noise"].shape == (1, 50, 32),
            "Target/noise dimensions",
        )
        torch.testing.assert_close(ref["post_truth"][0, valid], raw[valid], rtol=1e-5, atol=1e-5)
        check(not ref["truth"][..., 7:].count_nonzero(), "Target padding not zero")
        previous = next(
            v
            for v in prior
            if v["checkpoint"] == "frozen" and v["mode"] == "unconditioned" and v["case"] == case
        )
        exact(
            ref["noise"] - ref["truth"],
            previous["target"],
            "Noise/normalization differs from frozen pilot target",
        )
        expected_mask = (
            valid[None, :, None]
            & (torch.arange(50)[None, :, None] >= spec["C"])
            & (torch.arange(32)[None, None, :] < 7)
        )
        exact(expected_mask, previous["mask"], "Evaluation target/mask population changed")
    # Independently verify the exact absolute parameter population recorded for each checkpoint.
    initial = torch.load(r.PRIOR_OUT / "initial_projections.pt", map_location="cpu", weights_only=True)
    for label in r.LABELS:
        projection = (
            initial
            if label == "frozen"
            else torch.load(r.PRIOR_OUT / f"{label}_checkpoint.pt", map_location="cpu", weights_only=True)[
                "projections"
            ]
        )
        check(
            evidence["projection_hashes"][label] == {k: tensor_sha(v) for k, v in projection.items()},
            "Loaded projection identity differs",
        )
    seen, scored, trace_steps = {}, [], 0
    for index, row in enumerate(rows):
        spec = schedule[row["case"]]
        key = (row["checkpoint"], row["mode"], row["case"])
        check(
            key not in seen and row["call_id"] == index and all(row[k] == v for k, v in spec.items()),
            "Duplicate/order/schedule mismatch",
        )
        seen[key] = row
        ref = refs[row["case"]]
        check(row["input_fingerprint"] == ref["fingerprint"], "Input fingerprint differs")
        check(
            row["full"].shape == (1, 50, 32)
            and row["processed"].shape == (1, 50, 7)
            and torch.isfinite(row["full"]).all()
            and torch.isfinite(row["processed"]).all(),
            "Output shape/finite",
        )
        c = spec["C"] if row["mode"] in ("correct_prefix", "mismatched_prefix") else 0
        check(row["condition_count"] == c, "Condition count differs")
        if c:
            prefix = ref["truth"][:, :c, :7] if row["mode"] == "correct_prefix" else ref["donor_prefix"]
            exact(row["condition_prefix"], prefix, "Prefix source differs")
        else:
            check(row["condition_prefix"] is None, "Unexpected prefix payload")
        trace_steps += inspect_trace(row, ref)
        normal = metrics(row["full"], ref["truth"], ref["valid"], spec["C"])
        for k, v in normal.items():
            check(math.isclose(v, row["metrics"][k], rel_tol=1e-5, abs_tol=1e-6), "Action metric mismatch")
        if row["mode"] != "calibration_c0":
            scored.append(
                {
                    "checkpoint": row["checkpoint"],
                    "mode": row["mode"],
                    **spec,
                    "task": meta[spec["trajectory_id"]]["task"],
                    "normalized": normal,
                    "command": metrics(row["processed"], ref["post_truth"], ref["valid"], spec["C"]),
                }
            )
    check(
        set(seen)
        == {
            (label, mode, case)
            for label in r.LABELS
            for mode in (*r.MODES, "calibration_c0")
            for case in range(16)
        },
        "Final coverage",
    )
    for label in r.LABELS:
        for case in range(16):
            native, cal = seen[(label, "native_unconditioned", case)], seen[(label, "calibration_c0", case)]
            for field in ("full", "processed"):
                exact(native[field], cal[field], "C0 native equivalence failed")
    intents, returned = {}, set()
    for v in map(json.loads, (out / "calls.jsonl").read_text().splitlines()):
        cid = v["call_id"]
        if v["event"] == "intent":
            check(cid not in intents, "Duplicate call intent")
            intents[cid] = v
        else:
            check(
                v["event"] == "return" and cid in intents and cid not in returned and 0 < v["seconds"] <= 30,
                "Call return identity/deadline",
            )
            returned.add(cid)
    check(set(intents) == returned == set(range(192)) and trace_steps == 1440, "Call/trace closure")
    return {
        "independent_contract_accepted": True,
        "decodes": 192,
        "vision_encodes": 192,
        "denoise_steps": 1920,
        "cpu_euler_steps_checked": trace_steps,
        "c0_pairs_exact": 48,
        "c0_arrays_exact": 96,
        "scored_outputs": 144,
        "four_trajectory_cases": 16,
        "calls_closed": 192,
        **reduce_rows(scored),
        "per_case_metrics": scored,
        "optimizer_updates": 0,
        "new_env": 0,
        "sealed_evaluations": 0,
        "audit_model_forwards": 0,
        "deployment_qualified": False,
        "first_failure": None,
        "cuda_initialized": False,
        "limitations": [
            "Four already seen development trajectories; expert prefixes are offline conditioning.",
            "Full block error includes hard-copied prefix; decisions use suffix and first unexecuted action instead.",
            "CPU arithmetic audit does not recompute Transformer, postprocessing, random CUDA generator or GPU timing.",
            "C0 scalar path equivalence does not qualify conditional Graph execution or live task retention.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output.resolve()
    torch.set_num_threads(1)
    check(not (out / "independent_audit.json").exists(), "Unique audit required")
    try:
        value = audit(out)
    except BaseException:
        value = {"independent_contract_accepted": False, "first_failure": traceback.format_exc()}
    r.write(out / "independent_audit.json", value)
    print(
        json.dumps(
            {k: v for k, v in value.items() if k not in ("per_case_metrics", "metric_table", "paired_cases")}
        ),
        flush=True,
    )
    return 0 if value["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
