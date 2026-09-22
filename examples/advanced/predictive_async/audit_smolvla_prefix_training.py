"""Independent CPU audit of F-PTI1 saved forward/gradient arrays; no model calls."""

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(output):
    require(not torch.cuda.is_initialized(), "CPU audit required")
    r = json.loads((output / "result.json").read_text())
    require(r["status"] == "completed" and r["first_failure"] is None, "Formal diagnostic incomplete")
    require(r["forwards_started"] == r["forwards_returned"] == r["backwards"] == 5, "Forward/backward budget")
    require(r["optimizer_updates"] == r["new_env"] == r["graph_captures"] == 0, "Scope changed")
    require(
        r["requires_grad_restored"] and r["projection_parameters_unchanged"] and r["all_sources_unchanged"],
        "Unrestored state",
    )
    require(r["attempts"] == 1 and r["retries"] == 0, "Unexpected repeated execution")
    require(sha(output / "evidence.pt") == r["evidence_sha256"], "Evidence bytes changed")
    rows = torch.load(output / "evidence.pt", map_location="cpu", weights_only=True)
    labels = ("native_c0", "interface_c0", "interface_c3", "interface_c8", "terminal_c1")
    require(tuple(rows) == labels, "Case population changed")
    checks = []
    for label, c in zip(labels, (0, 0, 3, 8, 1), strict=True):
        x = rows[label]
        v, target = x["velocity"].numpy().astype(np.float64), x["target"].numpy().astype(np.float64)
        require(v.shape == target.shape == (1, 50, 32), "Output shape")
        valid_rows = 3 if label == "terminal_c1" else 50
        mask = np.zeros((1, 50, 32), dtype=bool)
        mask[:, c:valid_rows, :7] = True
        require(np.array_equal(mask, x["loss_mask"].numpy()), "Loss mask differs")
        require(int(x["counts"][0]) == c and int(x["valid_steps"].sum()) == valid_rows, "Counts differ")
        loss = float(np.mean(((v - target) ** 2)[mask]))
        require(
            math.isclose(loss, float(x["loss"]), rel_tol=2e-6, abs_tol=2e-6),
            "Independent scalar loss differs",
        )
        expected_grad = np.zeros_like(v)
        expected_grad[mask] = 2 * (v - target)[mask] / mask.sum()
        actual_grad = x["prediction_gradient"].numpy()
        require(
            np.allclose(expected_grad, actual_grad, rtol=2e-6, atol=2e-6),
            "Direct prediction gradient differs",
        )
        require(not np.count_nonzero(actual_grad[~mask]), "Masked coordinate gradient is nonzero")
        require(
            np.array_equal(x["noisy_actions"][:, :c].numpy(), x["actions"][:, :c].numpy()),
            "Clean prefix differs",
        )
        require(not np.count_nonzero(x["flow_times"][:, :c].numpy()), "Prefix time differs")
        require(np.all(x["flow_times"][:, c:].numpy() == 0.5), "Suffix time differs")
        require(len(x["parameter_gradients"]) == 8, "Projection gradient population")
        for grad in x["parameter_gradients"].values():
            require(
                bool(torch.isfinite(grad).all() and grad.abs().sum() > 0),
                "Nonfinite/zero projection gradient",
            )
        checks.append(
            {
                "label": label,
                "C": c,
                "valid_rows": valid_rows,
                "coordinates": int(mask.sum()),
                "independent_loss": loss,
            }
        )
    first, second = rows[labels[0]], rows[labels[1]]
    for key in ("elementwise", "velocity", "loss", "prediction_gradient"):
        require(torch.equal(first[key], second[key]), "C0 exact mismatch: " + key)
    for key in first["parameter_gradients"]:
        require(
            torch.equal(first["parameter_gradients"][key], second["parameter_gradients"][key]),
            "C0 gradient mismatch: " + key,
        )
    return {
        "independent_contract_accepted": True,
        "real_checkpoint_forwards": 5,
        "c0_exact_arrays": 4,
        "c0_exact_parameter_gradient_tensors": 8,
        "cases": checks,
        "optimizer_updates": 0,
        "new_env": 0,
        "audit_model_forwards": 0,
        "cuda_initialized": False,
        "training_convergence_claimed": False,
        "task_retention_claimed": False,
        "full_dataset_time_contract_accepted": False,
        "limitations": [
            "One demonstration; numerical diagnostic, not task evaluation.",
            "Projection gradients only; no full expert-parameter optimizer test.",
            "CPU audit does not recompute the Transformer or parameter gradients.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    output = p.parse_args().output.resolve()
    dest = output / "independent_audit.json"
    require(not dest.exists(), "Do not overwrite an audit")
    torch.set_num_threads(1)
    try:
        result = audit(output)
    except BaseException:
        result = {"independent_contract_accepted": False, "first_failure": traceback.format_exc()}
    with dest.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)
    return 0 if result["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
