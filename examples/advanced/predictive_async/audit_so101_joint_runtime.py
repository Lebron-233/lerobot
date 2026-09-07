"""Close L13 accounting and observe present GPU sharing without controlling other jobs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from run_so101_task_comparison import describe_trial
from summarize_so101_runtime import read_rows, summarize


def resources() -> dict:
    """Snapshot only. Process basename avoids publishing another project's paths."""
    result = {"observed_at": datetime.now().astimezone().isoformat(), "monotonic_s": time.perf_counter()}
    for key, fields in (
        ("gpu", "--query-gpu=name,memory.total,memory.used,utilization.gpu"),
        ("compute_processes", "--query-compute-apps=pid,process_name,used_gpu_memory"),
    ):
        response = subprocess.run(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        rows = list(csv.reader(io.StringIO(response.stdout), skipinitialspace=True))
        if key == "compute_processes":
            rows = [[pid, Path(name).name, memory] for pid, name, memory in rows]
        result[key] = rows
    result["process_control_actions"] = 0
    return result


def audit(root: Path) -> dict:
    campaign = json.loads((root / "summary.json").read_text())
    if not campaign["complete"] or len(campaign["trials"]) != 6:
        raise ValueError("All six qualification conditions must finish before audit")
    expected = {(b, a) for b in (0, 1) for a in ("visual_only", "state_only", "joint")}
    observed = [(row["block"], row["arm"]) for row in campaign["trials"]]
    if set(observed) != expected or len(set(observed)) != 6:
        raise ValueError("Qualification has missing or repeated conditions")
    rows = []
    for declared in campaign["trials"]:
        folder = root / f"block{declared['block']:02d}_{declared['arm']}"
        saved = json.loads((folder / "result.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        ticks = read_rows(folder / "ticks.jsonl")
        setup = read_rows(folder / "setup_ticks.jsonl")
        events = read_rows(folder / "events.jsonl")
        trial = describe_trial(folder)
        runtime = summarize(folder)
        if saved["source_commit"] != campaign["source_commit"]:
            raise ValueError("Unexpected source in qualification")
        if saved["context_variant"] != declared["arm"]:
            raise ValueError("Recorded context differs from the executed variant")
        if (
            manifest["args"]["minimum_delay"] != 7
            or saved["environment"]["torch_num_threads"] != 1
            or saved["environment"]["control_fps"] != 30
        ):
            raise ValueError("The calibrated operating profile changed")
        for key in ("completed_actions", "technical_valid", "status"):
            if trial[key] != declared[key]:
                raise ValueError(f"Summary differs from actual dispatch/result records: {key}")
        planned = [e for e in events if e.get("request_kind") == "planned" and e["event"] == "chunk_request"]
        state_calls = sum(e.get("state_predictor_calls", 0) for e in planned)
        visual_calls = sum(e.get("visual_predictor_calls", e.get("predictor_calls", 0)) for e in planned)
        if (state_calls, visual_calls) != (
            declared["state_predictor_calls"],
            declared["visual_predictor_calls"],
        ):
            raise ValueError("Component call accounting changed")
        for event in planned:
            if (
                declared["arm"] in ("joint", "state_only")
                and event.get("future_state_source") != "learned_current_state_and_committed_prefix"
            ):
                raise ValueError("Future state provenance is not causal")
        worst = max(ticks, key=lambda r: r.get("work_s", 0))
        rows.append(
            {
                "block": declared["block"],
                "arm": declared["arm"],
                "seed": trial["seed"],
                "status": saved["status"],
                "completed_actions": trial["completed_actions"],
                "setup_actions": sum(t["dispatch"] == "completed" for t in setup),
                "qualified": declared["qualification_pass"],
                "state_predictor_calls": state_calls,
                "visual_predictor_calls": visual_calls,
                "takeovers": declared["takeovers"],
                "component_calls_match": True,
                "runtime_checks": runtime["checks"],
                "worst_tick": worst["tick"],
                "worst_work_ms": worst["work_s"] * 1000,
                "worst_env_step_ms": worst["server_timing_s"]["env_step_and_flags"] * 1000,
                "error_last_line": (saved.get("error") or "").splitlines()[-1:],
                "engine_stats": saved["engine_stats"],
                "cleanup_complete": saved["metrics_closed"]
                and saved["subprocess_returncode"] == 0
                and not saved.get("cleanup_error"),
            }
        )
    return {
        "kind": "closed_L13_qualification_read_only_audit",
        "source_commit": campaign["source_commit"],
        "rows": rows,
        "completed_actions": sum(r["completed_actions"] for r in rows),
        "setup_actions": sum(r["setup_actions"] for r in rows),
        "state_predictor_calls": sum(r["state_predictor_calls"] for r in rows),
        "visual_predictor_calls": sum(r["visual_predictor_calls"] for r in rows),
        "qualified_conditions": sum(r["qualified"] for r in rows),
        "all_cleanup_complete": all(r["cleanup_complete"] for r in rows),
        "task_cohort_started": False,
        "snapshot_is_not_historical_causality_evidence": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.qualification_root)
    result["post_run_resource_snapshots"] = []
    for _ in range(3):
        result["post_run_resource_snapshots"].append(resources())
        time.sleep(1)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
