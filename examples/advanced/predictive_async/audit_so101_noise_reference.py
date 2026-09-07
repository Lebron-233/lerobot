"""Read-only L17 noise, physical-prefix and original-task outcome accounting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from audit_so101_privileged_reference import information_schedule, read_rows
from run_so101_noise_reference import ARMS, outcomes, summarize


def native_components(witness: dict) -> tuple[int, bool]:
    """Describe the original box/rest components; no new success definition."""
    plate = torch.tensor(witness["plate_position"], dtype=torch.float32)
    count = 0
    for name in ("Orange001", "Orange002", "Orange003"):
        orange = torch.tensor(witness["oranges"][name]["position"], dtype=torch.float32)
        bounds = plate.new_tensor([0.10, 0.10, 0.07])
        count += int(((orange > plate - bounds) & (orange < plate + bounds)).all())
    degrees = torch.tensor(witness["joint_positions_radians"], dtype=torch.float32) / torch.pi * 180
    low = degrees.new_tensor([-30, -130, 60, 20, -30, -40])
    high = degrees.new_tensor([30, -70, 120, 80, 30, 20])
    return count, bool(((degrees > low) & (degrees < high)).all())


def native_predicate(witness: dict) -> bool:
    """Recompute the pinned PickOrange box+rest rule, not a release criterion."""
    count, rest = native_components(witness)
    return count == 3 and rest


def audit(root: Path) -> dict:
    saved = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not saved["complete"] or len(saved["rows"]) != 32 or manifest["phase"] != "study":
        raise ValueError("Finish the entire once-only cohort before closed auditing")
    if saved["source_commit"] != manifest["source_commit"] or not saved["shared_cleanup_complete"]:
        raise ValueError("Source identity or shared simulator cleanup is incomplete")
    keys = [(r["block"], r["arm"]) for r in saved["rows"]]
    if len(set(keys)) != 32 or set(keys) != {(b, a) for b in range(8) for a in ARMS}:
        raise ValueError("Missing or duplicate experimental conditions")
    reconstructed, audit_rows, setup_count, mismatches = [], [], 0, []
    prefixes, native_witnesses = {}, []
    for block in range(8):
        folder = root / f"block{block:02d}_bootstrap"
        bootstrap = torch.load(folder / "bootstrap.pt", map_location="cpu", weights_only=True)
        field = bootstrap["noise_field"]
        setup = read_rows(folder / "setup_ticks.jsonl")
        if len(setup) != 30 or any(t["dispatch"] != "completed" for t in setup):
            raise ValueError("Bootstrap setup is not thirty logged actions")
        setup_count += len(setup)
        for arm in ARMS:
            folder = root / f"block{block:02d}_{arm}"
            row = json.loads((folder / "result.json").read_text())
            recorded = next(r for r in saved["rows"] if (r["block"], r["arm"]) == (block, arm))
            if row != recorded or row["seed"] != 20270810 + block or row["policy_seed"] != 3710 + block:
                raise ValueError("Recorded outcome identity changed")
            ticks, events = read_rows(folder / "ticks.jsonl"), read_rows(folder / "events.jsonl")
            measured = outcomes(ticks, row)
            if any(row[k] != value for k, value in measured.items()):
                raise ValueError("Subgoal/native/technical endpoints differ from physical evidence")
            setup = read_rows(folder / "setup_ticks.jsonl")
            if len(setup) != 30 or any(t["dispatch"] != "completed" for t in setup):
                raise ValueError("Per-arm setup is incomplete")
            setup_count += len(setup)
            context = "oracle_visual" if arm.endswith("oracle") else "state_only"
            schedule = information_schedule(ticks, events, context, 3710 + block)
            for event in events:
                if arm.startswith("coupled"):
                    start = event["takeover_index"]
                    if (
                        event.get("noise_mode") != "absolute_action_index"
                        or event.get("noise_window_start") != start
                        or event.get("noise_first_row") != field[0, start].tolist()
                        or event.get("noise_last_row") != field[0, start + 49].tolist()
                    ):
                        raise ValueError("Recorded coupled noise differs from the fixed action-time field")
                elif any(k in event for k in ("noise_mode", "noise_window_start", "noise_first_row")):
                    raise ValueError("Fresh control silently used coupled noise")
            max_boxes, first_all_boxes, last_boxes, last_rest = 0, None, 0, False
            for tick in ticks:
                witness = tick["task_transition_after_action"]
                last_boxes, last_rest = native_components(witness)
                max_boxes = max(max_boxes, last_boxes)
                if last_boxes == 3 and first_all_boxes is None:
                    first_all_boxes = tick["tick"]
                if (last_boxes == 3 and last_rest) != tick["terminated"]:
                    mismatches.append({"block": block, "arm": arm, "tick": tick["tick"]})
            if row["native_success"]:
                native_witnesses.append(
                    {"block": block, "arm": arm, "witness": ticks[-1]["task_transition_after_action"]}
                )
            before = ticks[:27]
            if [t["normalized_action"] for t in before] != bootstrap["normalized"][: len(before)].tolist():
                raise ValueError("Common initial normalized commands changed")
            if [t["action"] for t in before] != bootstrap["physical"][: len(before)].tolist():
                raise ValueError("Common initial physical commands changed")
            if (
                ticks[0]["state"] != bootstrap["geometry"]["state"]
                or ticks[0]["objects_before"] != bootstrap["geometry"]["objects"]
            ):
                raise ValueError("Starting physical geometry changed")
            prefixes[(block, arm)] = ticks[:28]
            reconstructed.append(row)
            audit_rows.append(
                {
                    "block": block,
                    "arm": arm,
                    **measured,
                    **schedule,
                    "takeover_jump_mean_range_normalized_l2": row["takeover_jump_mean_range_normalized_l2"],
                    "native_component_diagnostics_not_new_endpoints": {
                        "max_simultaneous_box_occupancy": max_boxes,
                        "first_all_three_boxes_step": first_all_boxes,
                        "final_box_occupancy": last_boxes,
                        "final_rest_predicate": last_rest,
                    },
                }
            )
    recomputed = summarize(reconstructed)
    if any(saved[k] != value for k, value in recomputed.items()):
        raise ValueError("The frozen scene-level statistics changed during audit")
    divergence = []
    for (block, arm), ticks in prefixes.items():
        if arm == "fresh_state":
            continue
        ref = prefixes[(block, "fresh_state")]
        n = min(len(ticks), len(ref))
        divergence.append(
            {
                "block": block,
                "arm": arm,
                "observations_including_first_target": n,
                "max_state_abs_difference": float(
                    np.abs(
                        np.array([t["state"] for t in ticks[:n]]) - np.array([t["state"] for t in ref[:n]])
                    ).max()
                ),
                "max_object_abs_difference_m": max(
                    abs(a["objects_before"][name][i] - b["objects_before"][name][i])
                    for a, b in zip(ticks[:n], ref[:n], strict=True)
                    for name in a["objects_before"]
                    for i in range(3)
                ),
            }
        )
    if mismatches:
        raise ValueError(f"Independent native-rule reconstruction disagreed at {mismatches[:5]}")
    return {
        "kind": "closed_L17_read_only_noise_information_and_full_task_audit",
        "execution_source": saved["source_commit"],
        **recomputed,
        "rows": audit_rows,
        "completed_actions": sum(r["completed_actions"] for r in reconstructed),
        "setup_actions": setup_count,
        "takeovers": sum(r["takeovers"] for r in audit_rows),
        "actual_old_prefix_actions": sum(r["actual_old_prefix_actions"] for r in audit_rows),
        "privileged_generations": sum(r["privileged_generations"] for r in audit_rows),
        "status_counts": dict(Counter(r["status"] for r in reconstructed)),
        "native_rule_mismatches": len(mismatches),
        "native_terminal_witnesses": native_witnesses,
        "pre_first_takeover_divergence": divergence,
        "shared_simulator_exits": 1,
        "subgoal_is_not_full_task": True,
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
            {
                k: v
                for k, v in report.items()
                if k not in ("rows", "native_terminal_witnesses", "pre_first_takeover_divergence")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
