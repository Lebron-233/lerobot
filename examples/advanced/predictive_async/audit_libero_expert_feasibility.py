"""Independent F-EAF1 CPU arithmetic/parameter/gradient-record audit, no model."""

import argparse
import json
import math
from pathlib import Path

import libero_expert_feasibility as r
import torch


def check(ok, message):
    if not ok:
        raise ValueError(message)


def audit(out):
    result = json.loads((out / "result.json").read_text())
    saved, registration = r.validate(result["head"])
    check(result["status"] == "completed" and result["first_failure"] is None, "Run incomplete")
    check(
        result["registration_id"] == registration
        and result["forwards"] == result["backwards"] == result["optimizer_updates"] == 6,
        "Scope",
    )
    check(result["model_loads"] == result["attempts"] == 1 and result["retries"] == 0, "Attempt scope")
    check(all(result[k] == 0 for k in ("new_env", "sealed_evaluations", "graph_captures")), "Unexpected work")
    check(r.sha(out / "evidence.pt") == result["evidence_sha256"], "Evidence hash")
    evidence = torch.load(out / "evidence.pt", map_location="cpu", weights_only=True)
    stages = {}
    for label in r.STAGES:
        record = evidence[label]
        check(len(record["steps"]) == 3, "Step count")
        check(record["peak_allocated"] <= 8 * 2**30, "Memory gate")
        check(set(record["initial"]) == set(record["final"]), "Parameter set")
        for name, value in record["final"].items():
            check(
                value.shape == record["initial"][name].shape and torch.isfinite(value).all(),
                "Parameter finite",
            )
        check(len(record["optimizer"]["state"]) == len(record["final"]), "Optimizer parameter count")
        check(all(float(v["step"]) == 3 for v in record["optimizer"]["state"].values()), "Optimizer step")
        for step in record["steps"]:
            mask = step["mask"]
            check(mask.shape == step["velocity"].shape == step["target"].shape == (1, 50, 32), "Shape")
            expected = (torch.arange(50)[None, :, None] >= 3) & (torch.arange(32)[None, None, :] < 7)
            check(torch.equal(mask, expected), "Loss mask")
            loss = float((step["velocity"].double() - step["target"].double()).square()[mask].mean())
            check(math.isclose(loss, step["loss"], rel_tol=1e-5, abs_tol=1e-6), "Loss recomputation")
            check(all(math.isfinite(v) for v in step["gradient_norms"].values()), "Gradient norm finite")
        stages[label] = {k: record[k] for k in ("peak_allocated", "peak_reserved", "trainable_parameters")}
        stages[label]["changed_parameter_tensors"] = sum(
            not torch.equal(v, record["initial"][n]) for n, v in record["final"].items()
        )
        stages[label]["forward_mean_s"] = sum(s["forward_s"] for s in record["steps"]) / 3
        stages[label]["backward_mean_s"] = sum(s["backward_s"] for s in record["steps"]) / 3
    a, b = evidence["projection"]["steps"][0], evidence["expert"]["steps"][0]
    check(torch.equal(a["velocity"], b["velocity"]) and a["loss"] == b["loss"], "Zero identity")
    check(
        all(torch.equal(v, b["projection_gradients"][n]) for n, v in a["projection_gradients"].items()),
        "Projection gradient exact",
    )
    check(
        len(evidence["expert"]["hits"]) == 32 and all(v == 3 for v in evidence["expert"]["hits"].values()),
        "Actual target calls",
    )
    for suffix, step, positive in [("lora_A", 0, False), ("lora_B", 0, True), ("lora_A", 1, True)]:
        values = [
            v for n, v in evidence["expert"]["steps"][step]["gradient_norms"].items() if n.endswith(suffix)
        ]
        check(
            len(values) == 32 and (all(v > 0 for v in values) if positive else not any(values)),
            "Adapter gradient record",
        )
    check(result["base_restored"] and result["sources_unchanged"], "Restoration")
    check(not torch.cuda.is_initialized(), "CPU only")
    return {
        "independent_contract_accepted": True,
        "technical_followup_supported": True,
        "model_forwards": 6,
        "optimizer_updates": 6,
        "zero_output_exact": True,
        "projection_gradient_arrays_exact": 8,
        "target_linears_called": 32,
        "stages": stages,
        "audit_model_forwards": 0,
        "new_env": 0,
        "limitations": [
            "Three updates are technical probes, not evidence of task improvement.",
            "CPU audit checks stored gradient records, not a second gradient derivation.",
            "No expert Graph/merge or live task qualification.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output.resolve()
    torch.set_num_threads(1)
    value = audit(out)
    r.write(out / "independent_audit.json", value)
    print(json.dumps(value), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
