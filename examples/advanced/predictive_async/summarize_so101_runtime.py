"""Summarize source-bound live engineering trials without reopening L6 test data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def statistics(values: list[float], *, scale: float = 1.0) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float) * scale
    return {
        "count": len(values),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def read_rows(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text().splitlines() if row]


def summarize(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    ticks, events = read_rows(path / "ticks.jsonl"), read_rows(path / "events.jsonl")
    gets = [e for e in events if e["event"] == "queue_get"]
    requests = [e for e in events if e["event"] in ("chunk_request", "request_error")]
    planned = [e for e in requests if e["request_kind"] == "planned"]
    completed = [t for t in ticks if t["dispatch"] == "completed"]
    finished = [t for t in ticks if "finished_at_s" in t]
    origin = ticks[0]["scheduled_start_s"] if ticks else None
    inflight_old_actions = 0
    for request in planned:
        if request.get("cuda_completed_at_s") is None:
            continue
        for get, tick in zip(gets, ticks, strict=False):
            if (
                request["requested_at_s"] <= get["timestamp_s"] < request["cuda_completed_at_s"]
                and get["outcome"] != "underflow"
                and get["action_index"] < request["takeover_index"]
                and tick["dispatch"] == "completed"
                and tick.get("finished_at_s", float("inf")) <= request["cuda_completed_at_s"]
            ):
                inflight_old_actions += 1
    checks = {
        "one_get_per_attempted_tick": len(gets) == len(ticks),
        "all_attempted_dispatches_completed": len(completed) == len(ticks),
        "sequential_action_indices": [e["action_index"] for e in gets] == list(range(len(ticks))),
        "request_terminals_complete": len({e["request_id"] for e in requests})
        == result["engine_stats"]["requests_started"],
        "request_terminals_unique": len({e["request_id"] for e in requests}) == len(requests),
        "no_request_errors": not any(e["event"] == "request_error" for e in requests),
        "no_full_start_slot_lost": all(t["start_lateness_s"] < 1 / 30 for t in ticks),
        "no_full_finish_slot_lost": all(
            t["finished_at_s"] - t["scheduled_start_s"] < 2 / 30 for t in finished
        ),
        "sink_closed": result["metrics_closed"],
        "simulator_clean_exit": result["subprocess_returncode"] == 0 and not result.get("cleanup_error"),
    }
    positive_delay = [e for e in planned if e.get("planned_delay_steps", 0) > 0]
    if result["mode"] == "predicted":
        checks["one_predictor_call_per_positive_delay_request"] = bool(positive_delay) and all(
            e.get("predictor_calls") == 1 for e in positive_delay
        )
    return {
        "artifact": str(path.resolve()),
        "source_commit": result["source_commit"],
        "mode": result["mode"],
        "seed": manifest["args"]["seed"],
        "policy_seed": manifest["args"]["policy_seed"],
        "requested_tick_bound": manifest["args"]["max_steps"],
        "status": result["status"],
        "success": result.get("success"),
        "timeout": result.get("timeout"),
        "error": result.get("error"),
        "completed_dispatches": len(completed),
        "measured_simulation_seconds": len(completed) / 30,
        "first_tick_to_last_finish_s": finished[-1]["finished_at_s"] - ticks[0]["started_at_s"]
        if finished
        else None,
        "scheduled_origin_s": origin,
        "work_ms": statistics([t["work_s"] for t in finished], scale=1000),
        "start_lateness_ms": statistics([t["start_lateness_s"] for t in ticks], scale=1000),
        "work_over_33ms_count": sum(t["work_s"] > 1 / 30 for t in finished),
        "engine_stats": result["engine_stats"],
        "queue_outcomes": dict(Counter(e["outcome"] for e in gets)),
        "planned_outcomes": dict(Counter(e.get("outcome") for e in planned)),
        "planned_delay_counts": dict(Counter(e.get("planned_delay_steps") for e in planned)),
        "planned_chunk_ms": statistics(
            [e["total_chunk_s"] for e in planned if e.get("total_chunk_s")], scale=1000
        ),
        "predictor_forward_host_ms": statistics(
            [
                e["phase_host_wall_s"]["predictor_forward"]
                for e in planned
                if e.get("phase_host_wall_s", {}).get("predictor_forward") is not None
            ],
            scale=1000,
        ),
        "planned_predictor_calls": sum(e.get("predictor_calls", 0) for e in planned),
        "old_action_physics_steps_completed_while_planned_inference_pending": inflight_old_actions,
        "checks": checks,
        "bounded_runtime_gate_pass": result["status"] != "technical_failure" and all(checks.values()),
        "not_a_task_benefit_claim": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.runs]
    summary = {
        "kind": "L7_live_engineering_cohort_no_fitting_or_L6_test_reuse",
        "runs": rows,
        "completed_dispatches": sum(r["completed_dispatches"] for r in rows),
        "planned_predictor_calls": sum(r["planned_predictor_calls"] for r in rows),
        "all_bounded_runtime_gates_pass": all(r["bounded_runtime_gate_pass"] for r in rows),
        "risk_thresholds": None,
        "task_benefit_established": False,
    }
    with args.output.open("x") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    for row in rows:
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "mode",
                        "seed",
                        "status",
                        "completed_dispatches",
                        "planned_predictor_calls",
                        "queue_outcomes",
                        "bounded_runtime_gate_pass",
                        "work_ms",
                        "start_lateness_ms",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
