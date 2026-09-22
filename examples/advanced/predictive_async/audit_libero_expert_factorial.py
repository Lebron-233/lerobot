"""F-EAT1 CPU audit: recorded provenance, matched targets and decoded actions.

Does not rerun the Transformer, compute parameter gradients, replay all AdamW
steps, or measure GPU timing. Main gates are fixed before formal execution.
"""

import argparse
import json
import math
from pathlib import Path

import libero_expert_factorial as r
import numpy as np
import torch
from audit_libero_prefix_sampling import check, exact, inspect_trace, metrics
from verify_smolvla_prefix_training import tensor_sha

PRIMARY = ("suffix", "first")


def reduce_rows(scored):
    table = {}
    for label in r.LABELS:
        table[label] = {}
        for mode in r.MODES:
            selected = [v for v in scored if v["checkpoint"] == label and v["mode"] == mode]
            check(
                len(selected) == 16 and {v["case"] for v in selected} == set(range(16)), "Scored population"
            )
            table[label][mode] = {}
            for space in ("normalized", "command"):
                table[label][mode][space] = {}
                for metric in ("block", "suffix", "first", "short"):
                    per = {}
                    for identity in sorted({v["trajectory_id"] for v in selected}):
                        vals = [v[space][metric] for v in selected if v["trajectory_id"] == identity]
                        check(len(vals) == 4, "Per-trajectory denominator")
                        per[identity] = float(np.mean(vals))
                    check(len(per) == 4, "Trajectory count")
                    table[label][mode][space][metric] = {
                        "mean": float(np.mean(list(per.values()))),
                        "per_trajectory": per,
                    }
    candidate = table["expert_prefix"]["correct_prefix"]

    def better(label, mode):
        return all(
            candidate["normalized"][k]["mean"] < table[label][mode]["normalized"][k]["mean"] for k in PRIMARY
        )

    def command_not_worse(label, mode):
        return all(
            candidate["command"][k]["mean"] <= table[label][mode]["command"][k]["mean"]
            for k in (*PRIMARY, "short")
        )

    interactions = {}
    per_gain = {}
    for metric in PRIMARY:

        def value(label, key=metric):
            return table[label]["correct_prefix"]["normalized"][key]["mean"]

        projection_increment = value("projection_ordinary") - value("projection_prefix")
        expert_increment = value("expert_ordinary") - value("expert_prefix")
        interactions[metric] = {
            "projection_prefix_training_gain": projection_increment,
            "expert_prefix_training_gain": expert_increment,
            "difference_in_gains": expert_increment - projection_increment,
        }
        per_gain[metric] = {
            identity: table["projection_prefix"]["correct_prefix"]["normalized"][metric]["per_trajectory"][
                identity
            ]
            - v
            for identity, v in candidate["normalized"][metric]["per_trajectory"].items()
        }
    gates = {
        "better_than_projection_prefix": better("projection_prefix", "correct_prefix"),
        "better_than_expert_ordinary_same_prefix": better("expert_ordinary", "correct_prefix"),
        "better_than_frozen_same_prefix": better("frozen", "correct_prefix"),
        "better_than_projection_ordinary_unconditioned": better(
            "projection_ordinary", "native_unconditioned"
        ),
        "at_least_three_trajectories_each_metric": all(
            sum(v > 0 for v in values.values()) >= 3 for values in per_gain.values()
        ),
        "correct_better_than_mismatch": better("expert_prefix", "mismatched_prefix"),
        "positive_interaction_both_metrics": all(v["difference_in_gains"] > 0 for v in interactions.values()),
        "command_not_worse_than_projection_ordinary_unconditioned": command_not_worse(
            "projection_ordinary", "native_unconditioned"
        ),
        "command_not_worse_than_projection_prefix": command_not_worse("projection_prefix", "correct_prefix"),
    }
    return {
        "metric_table": table,
        "prefix_training_interaction": interactions,
        "per_trajectory_gain_vs_projection_prefix": per_gain,
        "development_gates": gates,
        "expert_adaptation_followup_supported": all(gates.values()),
    }


def audit(out):
    check(not torch.cuda.is_initialized(), "CPU audit only")
    result = json.loads((out / "result.json").read_text())
    saved, registration = r.validate(result["head"])
    check(out == r.paths(result["head"])[1] and result["registration_id"] == registration, "Output identity")
    check(result["status"] == "completed" and result["first_failure"] is None, "Run incomplete")
    check(
        result["updates"] == dict.fromkeys(r.ARMS, 128)
        and result["training_forwards"] == result["backwards"] == 512,
        "Train population",
    )
    check(
        result["decodes"] == result["saved_rows"] == 320
        and result["denoise_steps"] == 3200
        and result["vision_encodes"] == 832,
        "Decode population",
    )
    check(result["model_loads"] == result["attempts"] == 1 and result["retries"] == 0, "Attempt count")
    check(all(result[k] == 0 for k in ("new_env", "sealed_evaluations", "graph_captures")), "Unexpected work")
    check(
        result["base_restored"] and result["frozen_parameters_unchanged"] and result["sources_unchanged"],
        "Restoration",
    )
    check(r.sha(out / "evidence.pt") == result["evidence_sha256"], "Evidence SHA")
    evidence = torch.load(out / "evidence.pt", map_location="cpu", weights_only=True)
    log = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
    check(len(log) == 512, "Training log length")
    schedule = saved["specification"]["schedule"]
    for step, spec in enumerate(schedule["train"]):
        matched = [log[i * 128 + step] for i in range(4)]
        for label, row in zip(r.ARMS, matched, strict=True):
            check(
                row["arm"] == label and all(row[k] == v for k, v in spec.items()), "Train schedule mismatch"
            )
            check(
                math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm_before_clip"]),
                "Nonfinite train statistics",
            )
            check(all(math.isfinite(v) for v in row["gradient_norms"].values()), "Nonfinite gradient norms")
        check(
            all(
                all(v[k] == matched[0][k] for k in ("target_sha256", "mask_sha256", "coordinates"))
                for v in matched[1:]
            ),
            "Four arms not matched",
        )
    initial = torch.load(out / "initial_projections.pt", map_location="cpu", weights_only=True)
    old_initial = torch.load(
        r.sampling.PRIOR_OUT / "initial_projections.pt", map_location="cpu", weights_only=True
    )
    check(set(initial) == set(old_initial) and len(initial) == 8, "Projection population")
    for n, p in initial.items():
        exact(p, old_initial[n], "Wrong original starting weights")
    states = {}
    checkpoint_report = {}
    for label in r.ARMS:
        path = out / f"{label}_checkpoint.pt"
        check(r.sha(path) == result["checkpoints"][label], "Checkpoint SHA")
        s = torch.load(path, map_location="cpu", weights_only=True)
        check(
            s["version"] == r.VERSION
            and s["arm"] == label
            and s["updates"] == 128
            and not s["deployment_ready"],
            "Checkpoint version",
        )
        check(
            s["base_policy"] == str(r.e.POLICY)
            and s["data_manifest_sha256"] == r.sha(r.COHORT / "manifest.json"),
            "Base/data identity",
        )
        check(set(s["projections"]) == set(initial), "Projection keys")
        all_params = {**s["projections"], **s["adapters"]}
        expert = label.startswith("expert_")
        check(len(s["adapters"]) == (64 if expert else 0), "Adapter count")
        if expert:
            check(len(s["adapter_metadata"]) == 32, "Module count")
            check(
                [v["global_layer"] for v in s["adapter_metadata"]]
                == [i for i in range(0, 32, 2) for _ in range(2)],
                "Actual layer mapping",
            )
            expected_keys = set()
            for m in s["adapter_metadata"]:
                check(
                    m["rank"] == 8 and m["alpha"] == 8.0 and m["adapter_dtype"] == "torch.float32",
                    "Adapter config",
                )
                for part, shape in [("lora_A", (8, m["in_features"])), ("lora_B", (m["out_features"], 8))]:
                    key = m["path"] + "." + part
                    expected_keys.add(key)
                    check(tuple(s["adapters"][key].shape) == shape, "Adapter shape")
                    if part == "lora_B":
                        check(not s["initial_adapters"][key].count_nonzero(), "Nonzero initial B")
            check(set(s["adapters"]) == set(s["initial_adapters"]) == expected_keys, "Adapter paths")
        check(
            all(torch.isfinite(p).all() and p.dtype == torch.float32 for p in all_params.values()),
            "Trainable parameter dtype/finite",
        )
        opt = s["optimizer"]
        check(len(opt["state"]) == len(all_params), "Optimizer parameter count")
        group = opt["param_groups"][0]
        check(
            len(opt["param_groups"]) == 1
            and group["lr"] == 2.5e-6
            and tuple(group["betas"]) == (0.9, 0.95)
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0,
            "Optimizer settings",
        )
        for pid, p in zip(group["params"], all_params.values(), strict=True):
            state = opt["state"][pid]
            check(float(state["step"]) == 128, "Optimizer steps")
            for key in ("exp_avg", "exp_avg_sq"):
                check(state[key].shape == p.shape and torch.isfinite(state[key]).all(), "Optimizer state")
        check(
            result["resources"][label]["trainable_parameters"] == sum(p.numel() for p in all_params.values()),
            "Parameter count report",
        )
        check(result["resources"][label]["peak_allocated"] <= 8 * 2**30, "Memory bound")
        check(
            evidence["parameter_hashes"][label] == {n: tensor_sha(p) for n, p in all_params.items()},
            "Loaded evaluation checkpoint",
        )
        checkpoint_report[label] = {
            "sha256": r.sha(path),
            "parameter_tensors": len(all_params),
            "changed_projection_tensors": sum(
                not torch.equal(p, initial[n]) for n, p in s["projections"].items()
            ),
            "changed_adapter_tensors": sum(
                not torch.equal(p, s["initial_adapters"][n]) for n, p in s["adapters"].items()
            ),
        }
        states[label] = s
    check(
        all(
            torch.equal(p, states["expert_prefix"]["initial_adapters"][n])
            for n, p in states["expert_ordinary"]["initial_adapters"].items()
        ),
        "LoRA arm initialization mismatch",
    )
    for a, b in [("projection_ordinary", "expert_ordinary"), ("projection_prefix", "expert_prefix")]:
        x, y = evidence["first_training"][a], evidence["first_training"][b]
        exact(x["velocity"], y["velocity"], "Zero-B initial velocity")
        check(x["loss"] == y["loss"], "Initial loss")
        for n, v in x["projection_gradients"].items():
            exact(v, y["projection_gradients"][n], "Initial projection gradient")
    check(
        evidence["parameter_hashes"]["frozen"] == {n: tensor_sha(v) for n, v in initial.items()},
        "Frozen evaluation weights",
    )
    refs, rows = evidence["references"], evidence["rows"]
    check(set(refs) == set(range(16)) and len(rows) == 320, "Evaluation coverage")
    prior_velocity = torch.load(r.sampling.PRIOR_OUT / "evaluation.pt", map_location="cpu", weights_only=True)
    for case, ref in refs.items():
        spec = schedule["dev"][case]
        check(
            ref["spec"] == spec and ref["truth"].shape == ref["noise"].shape == (1, 50, 32),
            "Reference identity",
        )
        old = next(
            v
            for v in prior_velocity
            if v["checkpoint"] == "frozen" and v["mode"] == "unconditioned" and v["case"] == case
        )
        exact(ref["noise"] - ref["truth"], old["target"], "Original target normalization/noise")
        expected = (
            ref["valid"][..., None]
            & (torch.arange(50)[None, :, None] >= spec["C"])
            & (torch.arange(32)[None, None, :] < 7)
        )
        exact(expected, old["mask"], "Original suffix mask")
    seen, scored, trace_steps = {}, [], 0
    for i, row in enumerate(rows):
        spec = schedule["dev"][row["case"]]
        key = (row["checkpoint"], row["mode"], row["case"])
        check(
            key not in seen and row["call_id"] == 512 + i and all(row[k] == v for k, v in spec.items()),
            "Decode order/identity",
        )
        seen[key] = row
        ref = refs[row["case"]]
        check(row["input_fingerprint"] == ref["fingerprint"], "Condition inputs changed")
        c = spec["C"] if row["mode"] in ("correct_prefix", "mismatched_prefix") else 0
        check(row["condition_count"] == c, "C changed")
        if c:
            exact(
                row["condition_prefix"],
                ref["truth"][:, :c, :7] if row["mode"] == "correct_prefix" else ref["donor_prefix"],
                "Prefix source",
            )
        else:
            check(row["condition_prefix"] is None, "C0 prefix payload")
        trace_steps += inspect_trace(row, ref)
        normal = metrics(row["full"], ref["truth"], ref["valid"], spec["C"])
        check(
            all(math.isclose(v, row["metrics"][k], rel_tol=1e-5, abs_tol=1e-6) for k, v in normal.items()),
            "Metric recomputation",
        )
        if row["mode"] != "calibration_c0":
            scored.append(
                {
                    "checkpoint": row["checkpoint"],
                    "mode": row["mode"],
                    **spec,
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
        "Final decoded coverage",
    )
    for label in r.LABELS:
        for case in range(16):
            for k in ("full", "processed"):
                exact(
                    seen[label, "native_unconditioned", case][k],
                    seen[label, "calibration_c0", case][k],
                    "C0 equivalence",
                )
    intents, returns = {}, set()
    for event in map(json.loads, (out / "calls.jsonl").read_text().splitlines()):
        cid = event["call_id"]
        if event["event"] == "intent":
            check(cid not in intents, "Duplicate intent")
            intents[cid] = event
        else:
            check(
                event["event"] == "return"
                and cid in intents
                and cid not in returns
                and 0 < event["seconds"] <= 30,
                "Invalid return",
            )
            returns.add(cid)
    check(set(intents) == returns == set(range(832)) and trace_steps == 2400, "Call/trace closure")
    for row in log:
        event = intents[row["call_id"]]
        check(
            event["kind"] == "train" and event["label"] == row["arm"] and event["index"] == row["step"],
            "Training intent match",
        )
    for row in rows:
        event = intents[row["call_id"]]
        check(
            event["kind"] == "decode"
            and event["label"] == row["checkpoint"]
            and event["index"] == row["case"]
            and event["mode"] == row["mode"],
            "Decode intent match",
        )
    check(not torch.cuda.is_initialized(), "Audit initialized CUDA")
    return {
        "independent_contract_accepted": True,
        "updates": result["updates"],
        "matched_four_arm_targets": 128,
        "decodes": 320,
        "scored_outputs": 240,
        "vision_encodes": 832,
        "denoise_steps": 3200,
        "c0_pairs_exact": 80,
        "cpu_euler_steps_checked": trace_steps,
        "calls_closed": 832,
        "checkpoint_report": checkpoint_report,
        "resources": result["resources"],
        **reduce_rows(scored),
        "per_case_metrics": scored,
        "audit_model_forwards": 0,
        "sealed_evaluations": 0,
        "deployment_qualified": False,
        "limits": [
            "Known12train/4dev; no independent qualification or policy-prefix recovery labels.",
            "Single rank/seed/update budget. Not parameter-count or FLOP matched.",
            "CPU audit does not derive Transformer gradients or all intermediate optimizer steps.",
            "No new Graph, merge, environment, or live task-retention evaluation.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    out = p.parse_args().output.resolve()
    torch.set_num_threads(1)
    value = audit(out)
    r.write(out / "independent_audit.json", value)
    print(
        json.dumps({k: v for k, v in value.items() if k not in ("metric_table", "per_case_metrics")}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
