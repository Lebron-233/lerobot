"""Read-only L14 endpoint, commitment and simulated-delay accounting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_so101_l11_results import outcome_evidence, read_rows


def audit_schedule(ticks: list[dict], events: list[dict], arm: str, policy_seed: int) -> dict:
    """Verify actual old actions, not just a declared future takeover index."""
    if any(row["dispatch"] != "completed" for row in ticks):
        return {
            "schedule_complete": False,
            "reason": "technical_partial_dispatch",
            "requests": len(events),
            "state_calls": sum(e.get("state_predictor_calls", 0) for e in events),
            "visual_calls": sum(e.get("visual_predictor_calls", 0) for e in events),
            "takeovers": sum(r.get("queue_outcome") == "takeover" for r in ticks),
        }
    n = len(ticks)
    for step, row in enumerate(ticks):
        if row["tick"] != step or row["sim_step"] != step or row["queue_action_index"] != step:
            raise ValueError("Controlled-delay simulation or action index skipped a step")
        if not math.isclose(row["sim_time_s"], step / 30, abs_tol=1e-9):
            raise ValueError("Wrong simulated time base")
    expected = list(range(0, n, 50)) if arm == "sync" else ([0] + list(range(20, n, 27)) if n else [])
    if [e["request_step"] for e in events] != expected:
        raise ValueError("Request cadence differs from the fixed controlled-delay schedule")
    executed_prefix, unused_prefix = 0, 0
    takeovers = []
    for i, event in enumerate(events):
        step = event["request_step"]
        if event["request_id"] != i or event["noise_seed"] != policy_seed * 100000 + step:
            raise ValueError("Request identity or paired flow-noise rule differs")
        if not event["physics_paused_during_inference"]:
            raise ValueError("L14 must not be described as concurrent wall-clock inference")
        scheduled = arm != "sync" and i > 0
        state_calls = int(scheduled and arm in ("state_only", "joint"))
        visual_calls = int(scheduled and arm in ("visual_only", "joint"))
        if (event["state_predictor_calls"], event["visual_predictor_calls"]) != (state_calls, visual_calls):
            raise ValueError("Component-call ablation differs from the declared arm")
        if not scheduled:
            if event["simulated_delay_steps"] != 0 or event["takeover_index"] is not None:
                raise ValueError("Bootstrap/reference acquired a delayed forecast")
            if event["outcome"] != "installed":
                raise ValueError("Unfinished initial/reference action request")
            continue
        if event["simulated_delay_steps"] != 7 or event["takeover_index"] != step + 7:
            raise ValueError("The seven-step takeover index changed")
        if event["outcome"] != "staged_for_future_index":
            raise ValueError("Unfinished scheduled action request")
        committed = event["committed_normalized_prefix"]
        if len(committed) != 7:
            raise ValueError("The declared committed prefix is not seven rows")
        old_ticks = ticks[step : min(step + 7, n)]
        for row, action in zip(old_ticks, committed, strict=False):
            if row["normalized_action"] != action or row["queue_outcome"] == "takeover":
                raise ValueError("Physics did not execute the declared old committed action")
        executed_prefix += len(old_ticks)
        unused_prefix += 7 - len(old_ticks)
        if step + 7 < n:
            takeovers.append(step + 7)
    actual = [r["tick"] for r in ticks if r["queue_outcome"] == "takeover"]
    if actual != takeovers:
        raise ValueError("Actual queue takeover indices do not match the withheld chunks")
    return {
        "schedule_complete": True,
        "requests": len(events),
        "takeovers": len(actual),
        "executed_old_prefix_actions": executed_prefix,
        "unused_prefix_rows_at_task_stop": unused_prefix,
        "state_calls": sum(e["state_predictor_calls"] for e in events),
        "visual_calls": sum(e["visual_predictor_calls"] for e in events),
    }


def audit(root: Path) -> dict:
    from run_so101_joint_study import ARMS, task_summary
    from run_so101_task_comparison import describe_trial

    saved = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not saved["complete"] or len(saved["trials"]) != 40:
        raise ValueError("Finish the forty registered conditions before the closed-cohort audit")
    keys = [(r["block"], r["arm"]) for r in saved["trials"]]
    if set(keys) != {(b, a) for b in range(8) for a in ARMS} or len(set(keys)) != 40:
        raise ValueError("Missing or repeated L14 conditions")
    rows, rebuilt = [], []
    for trial in saved["trials"]:
        folder = root / f"block{trial['block']:02d}_{trial['arm']}"
        result = json.loads((folder / "result.json").read_text())
        config = json.loads((folder / "manifest.json").read_text())
        if (
            result["source_commit"] != saved["source_commit"]
            or config["source_commit"] != saved["source_commit"]
        ):
            raise ValueError("Mixed execution sources")
        if result["realtime_required"] or config["realtime_required"] or saved["realtime_qualified"]:
            raise ValueError("Controlled-delay task evidence is not a wall-clock qualification")
        ticks, events = read_rows(folder / "ticks.jsonl"), read_rows(folder / "events.jsonl")
        evidence = outcome_evidence(ticks, result)
        for key in (
            "completed_actions",
            "technical_valid",
            "first_placement_success",
            "restricted_placement_time_s",
        ):
            if trial[key] != evidence[key]:
                raise ValueError(f"Saved endpoint differs from the pre-reset evidence: {folder.name}/{key}")
        if (
            config["args"]["seed"] != 20270510 + trial["block"]
            or config["args"]["policy_seed"] != 3410 + trial["block"]
        ):
            raise ValueError("Registered scene or policy seed differs")
        schedule = audit_schedule(ticks, events, trial["arm"], config["args"]["policy_seed"])
        for stored, audited in (
            ("state_predictor_calls", "state_calls"),
            ("visual_predictor_calls", "visual_calls"),
            ("takeovers", "takeovers"),
        ):
            if trial[stored] != schedule[audited]:
                raise ValueError("Published component calls or takeovers differ from raw records")
        setup = read_rows(folder / "setup_ticks.jsonl")
        setup_count = sum(r["dispatch"] == "completed" for r in setup)
        if setup_count != result["setup_actions"] or setup_count != 30:
            raise ValueError("The common thirty physical setup steps are incomplete")
        recomputed = describe_trial(folder)
        recomputed.update(block=trial["block"], arm=trial["arm"])
        rebuilt.append(recomputed)
        rows.append(
            {
                "block": trial["block"],
                "arm": trial["arm"],
                "seed": trial["seed"],
                "status": result["status"],
                **evidence,
                "schedule": schedule,
                "setup_actions": setup_count,
                "cleanup_complete": result["subprocess_returncode"] == 0
                and result["metrics_closed"]
                and not result.get("cleanup_error"),
            }
        )
    summary = task_summary(rebuilt)
    for key, value in summary.items():
        if saved[key] != value:
            raise ValueError(f"Saved summary differs from the reconstructed registered statistic: {key}")
    return {
        "kind": "closed_controlled_delay_endpoint_and_commitment_audit",
        "execution_source": saved["source_commit"],
        "manifest": manifest,
        "rows": rows,
        **summary,
        "completed_actions": sum(r["completed_actions"] for r in rows),
        "setup_actions": sum(r["setup_actions"] for r in rows),
        "state_calls": sum(r["schedule"].get("state_calls", 0) for r in rows),
        "visual_calls": sum(r["schedule"].get("visual_calls", 0) for r in rows),
        "takeovers": sum(r["schedule"].get("takeovers", 0) for r in rows),
        "executed_old_prefix_actions": sum(r["schedule"].get("executed_old_prefix_actions", 0) for r in rows),
        "all_cleanup_complete": all(r["cleanup_complete"] for r in rows),
        "wall_clock_realtime_qualified": False,
        "original_three_orange_task_benefit_tested": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.campaign)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "manifest")}, indent=2))
