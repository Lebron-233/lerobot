"""Read-only outcome accounting for the closed L11 cohort, not a new efficacy test."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from so101_task_evidence import PlacementTracker, placement_flags


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def outcome_evidence(ticks: list[dict], result: dict) -> dict:
    """Check the saved pre-reset endpoint and expose, rather than infer, release."""
    tracker = PlacementTracker()
    completed = [row for row in ticks if row["dispatch"] == "completed"]
    for row in completed:
        witness = row.get("task_transition_after_action")
        if witness is None or witness["native_success"] != row["terminated"]:
            raise ValueError("Missing or inconsistent pre-reset native witness")
        tracker.update(witness, row["tick"])
    valid = (
        result["status"] in ("terminal", "censored_step_limit", "task_subgoal_reached")
        and result["subprocess_returncode"] == 0
        and result["metrics_closed"]
        and not result.get("cleanup_error")
        and len(completed) == len(ticks)
    )
    success = bool(valid and tracker.first_settled_step is not None)
    evidence = {
        "completed_actions": len(completed),
        "technical_valid": valid,
        "first_placement_success": success,
        "restricted_placement_time_s": (tracker.first_settled_step + 1) / 30 if success else 120.0,
        "native_success": result.get("success"),
        "settled_objects_at_endpoint": [],
        "endpoint_is_not_a_contact_release_test": True,
    }
    if success:
        index = next(i for i, row in enumerate(completed) if row["tick"] == tracker.first_settled_step)
        endpoint = completed[index]["task_transition_after_action"]
        flags = [
            placement_flags(row["task_transition_after_action"]) for row in completed[index - 9 : index + 1]
        ]
        names = [name for name in flags[-1] if all(row[name] for row in flags)]
        initial = ticks[0]["task_diagnostics_before_action"]["object_positions_world"]
        evidence.update(
            settled_objects_at_endpoint=names,
            endpoint_gripper_radians=endpoint["joint_positions_radians"][-1],
            initial_object_plate_planar_distances_m={
                name: math.dist(initial[name][:2], initial["Plate"][:2]) for name in names
            },
            object_movement_to_endpoint_m={
                name: math.dist(initial[name], endpoint["oranges"][name]["position"]) for name in names
            },
            endpoint_object_speeds_m_s={
                name: math.sqrt(sum(v * v for v in endpoint["oranges"][name]["linear_velocity"]))
                for name in names
            },
        )
    return evidence


def audit(root: Path) -> dict:
    campaign = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not campaign["complete"] or len(campaign["trials"]) != 36:
        raise ValueError("Finish all 36 registered conditions before publishing the L11 audit")
    expected = {
        (seed, mode) for seed in manifest["environment_seeds"] for mode in ("sync", "identity", "predicted")
    }
    observed = [(row["seed"], row["mode"]) for row in campaign["trials"]]
    if set(observed) != expected or len(set(observed)) != 36:
        raise ValueError("The registered cohort has missing or repeated conditions")
    rows = []
    for trial in campaign["trials"]:
        folder = root / f"block{trial['block']:02d}_{trial['mode']}"
        result = json.loads((folder / "result.json").read_text())
        run_manifest = json.loads((folder / "manifest.json").read_text())
        ticks = read_rows(folder / "ticks.jsonl")
        events = read_rows(folder / "events.jsonl")
        evidence = outcome_evidence(ticks, result)
        for key in (
            "technical_valid",
            "first_placement_success",
            "restricted_placement_time_s",
            "native_success",
        ):
            if evidence[key] != trial[key]:
                raise ValueError(f"Campaign summary disagrees with saved endpoint: {folder.name}/{key}")
        if evidence["completed_actions"] != trial["completed_actions"]:
            raise ValueError("Action count mismatch")
        if result["source_commit"] != campaign["source_commit"]:
            raise ValueError("Mixed execution sources in the frozen cohort")
        environment = result["environment"]
        if environment["torch_num_threads"] != 1 or environment["control_fps"] != 30:
            raise ValueError("Runtime differs from the registered single-worker 30 Hz profile")
        if trial["mode"] != "sync" and run_manifest["args"]["minimum_delay"] != 7:
            raise ValueError("Asynchronous comparison arms did not share the delay floor")
        planned = [event for event in events if event.get("request_kind") == "planned"]
        takeovers = sum(
            event.get("event") == "queue_get" and event.get("outcome") == "takeover" for event in events
        )
        setup_path = folder / "setup_ticks.jsonl"
        setup = read_rows(setup_path) if setup_path.is_file() else []
        if sum(item["dispatch"] == "completed" for item in setup) != result["setup_physics_steps"]:
            raise ValueError("Reported setup physics steps lack matching dispatch evidence")
        row = {
            "block": trial["block"],
            "mode": trial["mode"],
            "seed": trial["seed"],
            "status": result["status"],
            **evidence,
            "setup_actions": sum(item["dispatch"] == "completed" for item in setup),
            "planned_predictor_calls": sum(event.get("predictor_calls", 0) for event in planned),
            "takeovers": takeovers,
            "cleanup_complete": result["subprocess_returncode"] == 0
            and result["metrics_closed"]
            and not result.get("cleanup_error"),
            "error_last_line": (result.get("error") or "").splitlines()[-1:],
        }
        rows.append(row)
    return {
        "kind": "read_only_closed_L11_endpoint_and_accounting_audit",
        "execution_source": campaign["source_commit"],
        "task": manifest["task"],
        "rows": rows,
        "aggregates": campaign["aggregates"],
        "paired_mean_difference_s": campaign["paired_mean_difference_s"],
        "paired_bootstrap_95_interval_s": campaign["paired_bootstrap_95_interval_s"],
        "first_half_mean_difference_s": campaign["first_half_mean_difference_s"],
        "second_half_mean_difference_s": campaign["second_half_mean_difference_s"],
        "paired_geometry_equal": campaign["paired_geometry_equal"],
        "reliable_reference_gate": campaign["reliable_reference_gate"],
        "stable_task_benefit_gate": campaign["stable_task_benefit_gate"],
        "benefit_on_reliable_reference": campaign["benefit_on_reliable_reference"],
        "completed_actions": sum(row["completed_actions"] for row in rows),
        "setup_actions": sum(row["setup_actions"] for row in rows),
        "planned_predictor_calls": sum(row["planned_predictor_calls"] for row in rows),
        "all_cleanup_complete": all(row["cleanup_complete"] for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "native_full_task_benefit_tested": False,
        "stable_action_oracle_benefit_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
