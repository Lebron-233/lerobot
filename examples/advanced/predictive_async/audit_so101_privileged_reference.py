"""Read-only L16 information-boundary and physical-prefix audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from run_so101_privileged_reference import paired_summary, prefix_geometry_differences
from so101_task_evidence import PlacementTracker


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def information_schedule(ticks: list[dict], events: list[dict], arm: str, policy_seed: int) -> dict:
    if [row["tick"] for row in ticks] != list(range(len(ticks))):
        raise ValueError("Measurement indices skipped or repeated")
    if any(row["sim_step"] != row["tick"] for row in ticks):
        raise ValueError("Queue indices do not align with actual simulator steps")
    if any(row["dispatch"] != "completed" for row in ticks):
        raise ValueError("The study contains an unfinished dispatch")
    expected_requests = list(range(20, len(ticks), 27))
    if [event["request_step"] for event in events] != expected_requests:
        raise ValueError("Controlled observation requests did not follow the fixed queue cadence")
    if [event["request_id"] for event in events] != list(range(1, len(events) + 1)):
        raise ValueError("Request identities are not unique and ordered")
    old_actions, generations, censored = 0, 0, 0
    expected_takeovers = []
    for event in events:
        start, target = event["request_step"], event["takeover_index"]
        if target != start + 7 or event["noise_seed"] != policy_seed * 100000 + start:
            raise ValueError("Request noise seed or simulated delay differs")
        if event["state_forecaster_calls"] != 1 or event["visual_forecaster_calls"] != int(arm == "student"):
            raise ValueError("A context component was not isolated as registered")
        count = max(0, min(7, len(ticks) - start))
        if event["committed_prefix"][:count] != [
            row["normalized_action"] for row in ticks[start : start + count]
        ]:
            raise ValueError("An actual old action differs from its commitment")
        old_actions += count
        privileged = arm in ("oracle_visual", "oracle_joint")
        if bool(event["privileged"]) != privileged:
            raise ValueError("Privileged generation has been labeled causal or conversely")
        target_observed = target < len(ticks)
        if privileged and not target_observed:
            if (
                event["generated_step"] is not None
                or event["outcome"] != "censored_before_future_observation"
            ):
                raise ValueError("A target-time oracle call was invented after the trajectory ended")
            censored += 1
            continue
        expected_generation = target if privileged else start
        expected_state = target if arm == "oracle_joint" else start
        if (
            event["generated_step"] != expected_generation
            or event["visual_observation_step"] != expected_generation
            or event["state_observation_step"] != expected_state
        ):
            raise ValueError(
                "Actual or predicted state/visual information crossed its permitted time boundary"
            )
        expected_outcome = "staged_on_time" if privileged else "staged_early"
        if event["outcome"] != expected_outcome:
            raise ValueError("Generation was not staged at its declared information boundary")
        generations += 1
        if target_observed:
            expected_takeovers.append(target)
    actual_takeovers = [row["tick"] for row in ticks if row["queue_outcome"] == "takeover"]
    if actual_takeovers != expected_takeovers:
        raise ValueError("A committed target did not take over at the actual scheduled step")
    return {
        "requests": len(events),
        "generations": generations,
        "privileged_generations": generations if arm.startswith("oracle") else 0,
        "censored_oracle_requests": censored,
        "takeovers": len(actual_takeovers),
        "actual_old_prefix_actions": old_actions,
        "information_boundary_pass": True,
    }


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    arms = manifest["arms"]
    if manifest["phase"] != "study" or not summary["complete"]:
        raise ValueError("Finish the entire L16 reference cohort before publishing its audit")
    expected = {(block, arm) for block in range(8) for arm in arms}
    observed = [(row["block"], row["arm"]) for row in summary["rows"]]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Registered outcomes are missing or duplicated")
    if summary["source_commit"] != manifest["source_commit"] or summary["realtime_qualified"]:
        raise ValueError("Mixed source or false realtime qualification")
    if not summary["shared_simulator_cleanup_complete"] or summary["shared_simulator_returncode"] != 0:
        raise ValueError("The actual shared simulator did not close normally")
    rebuilt, rows, setup_count = [], [], 0
    for block in range(8):
        bootstrap_dir = root / f"block{block:02d}_bootstrap"
        saved = torch.load(bootstrap_dir / "bootstrap.pt", map_location="cpu", weights_only=True)
        preparation = read_rows(bootstrap_dir / "setup_ticks.jsonl")
        if len(preparation) != 30 or any(row["dispatch"] != "completed" for row in preparation):
            raise ValueError("Shared-bootstrap preparation is not the recorded fixed thirty steps")
        setup_count += len(preparation)
        if saved["noise_seed"] != (3610 + block) * 100000:
            raise ValueError("Shared bootstrap noise seed changed")
        for arm in arms:
            folder = root / f"block{block:02d}_{arm}"
            trial = json.loads((folder / "control_result.json").read_text())
            recorded = next(row for row in summary["rows"] if row["block"] == block and row["arm"] == arm)
            if trial != recorded:
                raise ValueError("Per-arm terminal data and closed global report disagree")
            ticks, events = read_rows(folder / "ticks.jsonl"), read_rows(folder / "events.jsonl")
            preparation = read_rows(folder / "setup_ticks.jsonl")
            if len(preparation) != 30 or any(row["dispatch"] != "completed" for row in preparation):
                raise ValueError("Per-arm preparation is incomplete")
            setup_count += len(preparation)
            schedule = information_schedule(ticks, events, arm, 3610 + block)
            prefix = ticks[:27]
            if [row["normalized_action"] for row in prefix] != saved["normalized"][: len(prefix)].tolist():
                raise ValueError("The actual common bootstrap differed before the first takeover")
            if [row["action"] for row in prefix] != saved["physical"][: len(prefix)].tolist():
                raise ValueError("Common normalized and physical bootstrap commitments differ")
            if (
                ticks[0]["state"] != saved["geometry"]["state"]
                or ticks[0]["objects_before"] != saved["geometry"]["objects"]
            ):
                raise ValueError("Starting physical geometry differs from shared-bootstrap geometry")
            tracker = PlacementTracker()
            for row in ticks:
                witness = row["task_transition_after_action"]
                if witness is None or witness["native_success"] != row["terminated"]:
                    raise ValueError("Native pre-reset task evidence is missing or inconsistent")
                tracker.update(witness, row["tick"])
            valid = trial["status"] != "technical_failure"
            succeeded = bool(valid and tracker.first_settled_step is not None)
            cost = (tracker.first_settled_step + 1) / 30 if succeeded else 120.0
            if (
                trial["success"] != succeeded
                or trial["restricted_time_s"] != cost
                or trial["completed_actions"] != len(ticks)
            ):
                raise ValueError("Task endpoint/count differs from the actual pre-reset witnesses")
            if trial["privileged_generations"] != schedule["privileged_generations"]:
                raise ValueError("Actual oracle call count differs from the labeled request count")
            rebuilt.append(trial)
            rows.append({"block": block, "arm": arm, "actions": len(ticks), "success": succeeded, **schedule})
    recomputed = paired_summary(rebuilt, arms)
    for key, value in recomputed.items():
        if summary[key] != value:
            raise ValueError(f"Closed statistic differs from the original outcomes: {key}")
    differences = prefix_geometry_differences(root, rebuilt)
    if differences != summary["pre_takeover_geometry_differences"]:
        raise ValueError("Pre-takeover physical divergence was not reported as observed")
    if setup_count != summary["setup_actions"]:
        raise ValueError("Shared and per-arm preparation counts do not match")
    return {
        "kind": "closed_L16_privileged_information_and_physical_commitment_audit",
        "execution_source": summary["source_commit"],
        "arms": arms,
        "rows": rows,
        **recomputed,
        "completed_actions": sum(row["actions"] for row in rows),
        "setup_actions": setup_count,
        "takeovers": sum(row["takeovers"] for row in rows),
        "privileged_generations": sum(row["privileged_generations"] for row in rows),
        "actual_old_prefix_actions": sum(row["actual_old_prefix_actions"] for row in rows),
        "shared_simulator_exits": 1,
        "status_counts": dict(Counter(row["status"] for row in rebuilt)),
        "all_information_boundaries_verified": True,
        "pre_takeover_geometry_differences": differences,
        "not_full_three_orange_task": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.campaign)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "pre_takeover_geometry_differences")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
