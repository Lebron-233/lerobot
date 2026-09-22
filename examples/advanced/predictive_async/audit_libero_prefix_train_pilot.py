"""Independent CPU reductions for F-PTP1; no model, gradient, CUDA or Env run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from libero_prefix_train_pilot import (
    ARMS,
    MODES,
    UPDATES,
    VERSION,
    corpus,
    paths,
    schedule,
    sha,
    sources,
    write,
)
from verify_smolvla_prefix_training import tensor_sha


def reduce_eval(evaluation):
    values = {}
    keys = set()
    for row in evaluation:
        key = (row["checkpoint"], row["mode"], row["case"])
        if key in keys:
            raise ValueError("Duplicate evaluation identity")
        keys.add(key)
        v, target, mask = (row[k].numpy() for k in ("velocity", "target", "mask"))
        if v.shape != (1, 50, 32) or target.shape != v.shape or mask.shape != v.shape or mask.dtype != bool:
            raise ValueError("Evaluation shape/type mismatch")
        if not np.isfinite(v).all() or not np.isfinite(target).all() or not mask.any():
            raise ValueError("Nonfinite evaluation")
        count = row["C"]
        if mask[:, :count].any() or mask[..., 7:].any() or int(mask.sum()) != row["coordinates"]:
            raise ValueError("Suffix mask differs")
        error = v.astype(np.float64) - target.astype(np.float64)
        loss = float((error[mask] ** 2).mean())
        if not math.isclose(loss, row["loss"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Independent loss differs")
        group = values.setdefault(row["checkpoint"], {}).setdefault(row["mode"], {})
        group.setdefault(row["trajectory_id"], []).append(loss)
    if keys != {(label, mode, i) for label in ("frozen", *ARMS) for mode in MODES for i in range(16)}:
        raise ValueError("Evaluation population differs")
    result = {}
    for label, modes in values.items():
        result[label] = {}
        for mode, trajectories in modes.items():
            if len(trajectories) != 4 or any(len(v) != 4 for v in trajectories.values()):
                raise ValueError("Trajectory-equal denominator differs")
            per = {key: float(np.mean(v)) for key, v in trajectories.items()}
            result[label][mode] = {"n_trajectories": 4, "n_cases": 16, "mean": float(np.mean(list(per.values()))), "per_trajectory": per}
    candidate = result["prefix"]["correct_prefix"]
    baseline = result["ordinary"]["unconditioned"]
    comparisons = {key: baseline["per_trajectory"][key] - value for key, value in candidate["per_trajectory"].items()}
    gates = {"better_than_matched_ordinary": candidate["mean"] < baseline["mean"],
             "better_than_frozen_prefix": candidate["mean"] < result["frozen"]["correct_prefix"]["mean"],
             "at_least_three_trajectories_better": sum(v > 0 for v in comparisons.values()) >= 3,
             "correct_better_than_mismatched": candidate["mean"] < result["prefix"]["mismatched_prefix"]["mean"]}
    return {"loss_table": result, "trajectory_gain_vs_ordinary": comparisons,
            "development_gates": gates, "offline_followup_supported": all(gates.values())}


def audit(out):
    result = json.loads((out / "result.json").read_text())
    prep, expected_out = paths(result["head"])
    saved = json.loads((prep / "preparation.json").read_text())
    if out != expected_out or saved["sources"] != sources() or saved["schedule"] != schedule(corpus()):
        raise ValueError("Source or output identity differs")
    if (result["status"] != "completed" or result["first_failure"] is not None
        or result["updates"] != dict.fromkeys(ARMS, UPDATES)
        or result["forwards_started"] != result["forwards_returned"] or result["forwards_returned"] != 400
        or result["backwards"] != 256 or result["evaluation_rows"] != 144
        or result["attempts"] != 1 or result["retries"] != 0 or result["model_loads"] != 1
        or not result["base_parameters_restored"] or not result["frozen_parameters_unchanged"]
        or any(result[k] != 0 for k in ("new_env", "sealed_evaluations", "graph_captures"))):
        raise ValueError("Execution scope incomplete")
    train = [json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()]
    if len(train) != 256:
        raise ValueError("Training log population")
    for i, spec in enumerate(saved["schedule"]["train"]):
        a, b = train[i], train[i + UPDATES]
        for row, arm in [(a, "ordinary"), (b, "prefix")]:
            if row["arm"] != arm or any(row[k] != v for k, v in spec.items()):
                raise ValueError("Training schedule changed")
            if not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm_before_clip"]):
                raise ValueError("Training metric nonfinite")
        for key in ("target_sha256", "mask_sha256", "coordinates"):
            if a[key] != b[key]:
                raise ValueError("Training arms have different targets/denominators")
    initial = torch.load(out / "initial_projections.pt", map_location="cpu", weights_only=True)
    checkpoints = {}
    for arm in ARMS:
        path = out / f"{arm}_checkpoint.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["interface_version"] != VERSION or state["updates"] != UPDATES or state["deployment_ready"]:
            raise ValueError("Checkpoint contract mismatch")
        params = state["projections"]
        if set(params) != set(initial) or len(params) != 8:
            raise ValueError("Wrong trainable tensors")
        changed = []
        for name, p in params.items():
            if p.shape != initial[name].shape or not torch.isfinite(p).all():
                raise ValueError("Invalid trained parameter")
            changed.append(not torch.equal(p, initial[name]))
        opt = state["optimizer"]
        if len(opt["state"]) != 8 or any(float(v["step"]) != UPDATES for v in opt["state"].values()):
            raise ValueError("Optimizer final step differs")
        if not all(changed):
            raise ValueError("Expected all eight projections to update")
        checkpoints[arm] = {"sha256": sha(path), "changed_tensors": sum(changed),
                            "parameter_sha256": {n: tensor_sha(p) for n, p in params.items()}}
    evaluation = torch.load(out / "evaluation.pt", map_location="cpu", weights_only=True)
    for row in evaluation:
        spec = saved["schedule"]["dev"][row["case"]]
        if any(row[k] != v for k, v in spec.items()):
            raise ValueError("Development identity changed")
    for case in range(16):
        selected = [r for r in evaluation if r["case"] == case]
        for row in selected[1:]:
            if not torch.equal(row["target"], selected[0]["target"]) or not torch.equal(row["mask"], selected[0]["mask"]):
                raise ValueError("Evaluation comparators have different targets")
    return {"independent_contract_accepted": True, "updates": result["updates"],
            "complete_forwards": 400, "matched_training_target_pairs": 128,
            "evaluation_losses_recomputed": 144, "checkpoints": checkpoints,
            **reduce_eval(evaluation), "training_convergence_proven": False, "deployment_qualified": False,
            "new_env": 0, "sealed_evaluations": 0, "audit_model_forwards": 0,
            "limitations": ["Four development trajectories; correct expert prefix is offline conditioning, not an online recovery guarantee.",
                            "Only projection parameters trained; no full-expert optimizer experiment.",
                            "CPU audit recomputes losses but not Transformer, gradients or all intermediate AdamW updates.",
                            "No conditional ten-step sampler, new-checkpoint Graph or task-retention evaluation in this pilot."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output.resolve()
    torch.set_num_threads(1)
    value = audit(out)
    if torch.cuda.is_initialized():
        raise ValueError("CPU audit initialized CUDA")
    write(out / "independent_audit.json", value)
    print(json.dumps(value), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
