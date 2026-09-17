"""Independent CPU reduction of fixed full-path timings and output evidence."""

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import libero_rtc_fullpath as r
import numpy as np
import torch


def audit(output):
    require = r.require
    require(not torch.cuda.is_initialized(), "Audit must stay CPU-only")
    result = json.loads((output / "result.json").read_text())
    require(result["status"] == "completed" and result["first_failure"] is None, "Worker did not complete")
    execution = result["execution"]
    require(execution["exit_code"] == 0 and execution["exit_confirmed"] and not execution["forced"]
            and not execution["active"] and not execution["pending"] and execution["stop_reason"] is None,
            "Execution did not close")
    prepared = r.validate(result["execution_head"])
    require(output == r.paths(result["execution_head"])[1], "Output identity differs")
    data = r.load(r.paths(result["execution_head"])[0] / "data.pt")
    require([tuple(s["key"]) for s in data] == r.EXPECTED_KEYS, "Development identities changed")
    require({p.name for p in output.glob("request_*.pt")} ==
            {f"request_{i:03d}.pt" for i in range(192)}, "Request coverage differs")
    rows, index = [], 0
    for sample_index, sample in enumerate(data):
        baseline = None
        for repeat in range(6):
            arms = ["base", "rtc"] if (sample_index+repeat) % 2 == 0 else ["rtc", "base"]
            for arm in arms:
                row = r.load(output / f"request_{index:03d}.pt")
                require((row["index"], row["key"], row["repeat"], row["arm"]) ==
                        (index, sample["key"], repeat, arm), "Order or repetition changed")
                require(row["prefix_present"] == (arm == "rtc" and repeat > 0), "Guidance scope changed")
                require(row["queue_wait_s"] is None, "Offline requests are not queue measurements")
                require(row["full_output"].shape == (1, 50, 32) and
                        row["output"].shape == row["processed"].shape == (1, 50, 7), "Output shapes differ")
                for key in ("full_output", "output", "processed"):
                    require(torch.isfinite(row[key]).all(), "Nonfinite saved output")
                require(torch.equal(row["output"], row["full_output"][..., :7]), "Unpadding differs")
                if arm == "base" or repeat == 0:
                    require(torch.equal(row["full_output"], sample["archived_full_chunk"]), "No-guidance output differs")
                if repeat == 0:
                    if baseline is None:
                        baseline = row["output"]
                    else:
                        require(torch.equal(baseline, row["output"]), "No-prefix pair differs")
                parts = row["components"]
                require(all(math.isfinite(v) and v >= 0 for v in parts.values()), "Invalid component timing")
                total = parts["preprocess_transfer_s"] + parts["public_predict_total_s"] + parts["postprocess_transfer_s"]
                require(math.isclose(total, row["complete_s"], rel_tol=1e-9, abs_tol=1e-9), "Full-path timing does not reconcile")
                require(total > 0 and total < 30, "Invalid request time")
                rows.append(row)
                index += 1
    logged = [json.loads(v) for v in (output / "timing.jsonl").read_text().splitlines()]
    require(logged == [{k: v for k, v in row.items() if not isinstance(v, torch.Tensor)} for row in rows], "Timing log differs")
    timing = {}
    for arm in ("base", "rtc"):
        selected = [v for v in rows if v["arm"] == arm and v["repeat"] > 0]
        times = np.sort(np.array([v["complete_s"] for v in selected], dtype=np.float64))
        require(len(times) == 80, "Fixed measurement denominator changed")
        quantiles = {name: float(times[int(np.ceil(80*p))-1]) for name, p in (("p50_s", .5), ("p95_s", .95), ("p99_s", .99))}
        timing[arm] = {"n": 80, **quantiles, "max_s": float(times[-1]), "mean_s": float(times.mean()),
                       "required_delay_steps": int(np.ceil(20*quantiles["p99_s"]))+1}
        for k, value in timing[arm].items():
            require(math.isclose(value, result["timing"][arm][k], rel_tol=1e-9, abs_tol=1e-10), "Independent latency reduction differs")
    gate = all(v["required_delay_steps"] <= 8 for v in timing.values())
    require(gate == result["latency_gate_passed"], "Independent latency decision differs")
    expected = {"vla_loads": 1, "chunks_started": 192, "vision_encodes": 192, "chunks_returned": 192,
                "input_exact": 192, "no_guidance_exact": 112, "rtc_steps": 960,
                "guided_vjps_returned": 800, "no_prefix_pair_exact": 16}
    require(result["counts"] == expected, "Request accounting differs")
    phases, active = Counter(), {}
    for event in map(json.loads, (output / "events.jsonl").read_text().splitlines()):
        phases[event["event"]] += 1
        if event["event"] == "started":
            require(event["phase"] not in active, "Duplicate active phase")
            active[event["phase"]] = event["limit"]
        else:
            require(event["event"] == "returned" and event["phase"] in active, "Unmatched phase")
            require(event["seconds"] <= active.pop(event["phase"]), "Phase deadline exceeded")
    require(not active and phases == {"started": 193, "returned": 193}, "Phase accounting differs")
    require(result["attempts"] == 1 and result["retries"] == 0 and result["vla_frozen"], "Scope differs")
    require(all(result[k] == 0 for k in ("training_updates", "new_env", "qualification_reads", "graph_captures")), "Forbidden action")
    require(prepared["sources"] == r.hashes() and not torch.cuda.is_initialized(), "Sources/audit device changed")
    return {"independent_contract_accepted": True, "latency_gate_passed": gate, "timing": timing,
            "requests_checked": 192, "no_guidance_exact": 112, "no_prefix_pair_exact": 16,
            "phases": dict(phases), "audit_model_forwards": 0, "cuda_initialized": False,
            "first_failure": None, "limitation": "CPU audit does not rerun model, vision encoding or VJPs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    r.require(not (output / "independent_audit.json").exists(), "Never overwrite an audit")
    try:
        checked = audit(output)
    except BaseException:
        checked = {"independent_contract_accepted": False, "first_failure": traceback.format_exc()}
    r.write(output / "independent_audit.json", checked)
    print(json.dumps(checked), flush=True)
    return 0 if checked["independent_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
