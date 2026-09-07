"""L8 frozen twelve-block first-placement evaluation; no automatic trial retries."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from so101_task_evidence import PlacementTracker


def describe_trial(folder: Path) -> dict:
    result = json.loads((folder / "result.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    ticks = [json.loads(line) for line in (folder / "ticks.jsonl").read_text().splitlines()]
    tracker = PlacementTracker()
    completed = [tick for tick in ticks if tick["dispatch"] == "completed"]
    for tick in completed:
        witness = tick["task_transition_after_action"]
        if witness is None or witness["native_success"] != tick["terminated"]:
            raise ValueError("Missing or mismatched pre-reset task witness")
        tracker.update(witness, tick["tick"])
    valid = (
        result["status"] in ("terminal", "censored_step_limit", "task_subgoal_reached")
        and result["subprocess_returncode"] == 0
        and result["metrics_closed"]
        and not result.get("cleanup_error")
        and len(completed) == len(ticks)
    )
    succeeded = valid and tracker.first_settled_step is not None
    cost = (tracker.first_settled_step + 1) / 30 if succeeded else 120.0
    projections = result.get("projection_records", [])
    first = ticks[0] if ticks else {}
    actions = np.array([tick["action"] for tick in completed], dtype=float)
    step_change = None
    if len(actions) > 1:
        scale = np.array([220, 200, 190, 190, 320, 100], dtype=float)
        step_change = float(np.linalg.norm(np.diff(actions / scale, axis=0), axis=1).mean())
    return {
        "folder": str(folder.resolve()),
        "source_commit": result["source_commit"],
        "mode": result["mode"],
        "seed": manifest["args"]["seed"],
        "policy_seed": manifest["args"]["policy_seed"],
        "status": result["status"],
        "technical_valid": bool(valid),
        "first_placement_success": bool(succeeded),
        "restricted_placement_time_s": cost,
        "native_success": result["success"],
        "completed_actions": len(completed),
        "max_settled": tracker.max_simultaneous,
        "last_settled": tracker.last_simultaneous,
        "projected_components": sum(row["projected_components"] for row in projections),
        "projection_max_native": max((row["max_native_adjustment"] for row in projections), default=0),
        "mean_normalized_action_step_l2_descriptive": step_change,
        "engine_stats": result.get("engine_stats"),
        "error": result.get("error"),
        "initial_state": first.get("state"),
        "initial_objects": first.get("task_diagnostics_before_action", {}).get("object_positions_world"),
        "initial_camera_poses": first.get("camera_world_poses_opengl"),
    }


def comparison_summary(rows: list[dict]) -> dict:
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode] for mode in ("sync", "identity", "predicted")
    }
    aggregates = {
        mode: {
            "trials": len(group),
            "successes": sum(row["first_placement_success"] for row in group),
            "technical_failures": sum(not row["technical_valid"] for row in group),
            "mean_restricted_placement_time_s": float(
                np.mean([row["restricted_placement_time_s"] for row in group])
            )
            if group
            else None,
        }
        for mode, group in by_mode.items()
    }
    pairs, geometry_equal = [], []
    for seed in sorted({row["seed"] for row in rows}):
        conditions = {row["mode"]: row for row in rows if row["seed"] == seed}
        if {"identity", "predicted"} <= conditions.keys():
            identity, predicted = conditions["identity"], conditions["predicted"]
            pairs.append(predicted["restricted_placement_time_s"] - identity["restricted_placement_time_s"])
            geometry_equal.append(
                all(
                    identity[key] is not None and identity[key] == predicted[key]
                    for key in (
                        "initial_state",
                        "initial_objects",
                        "initial_camera_poses",
                    )
                )
            )
    summary = {
        "aggregates": aggregates,
        "paired_differences_s": pairs,
        "paired_geometry_equal": geometry_equal,
    }
    if len(pairs) == 12:
        delta = np.array(pairs)
        rng = np.random.default_rng(2089)
        samples = delta[rng.integers(0, 12, size=(50000, 12))].mean(axis=1)
        interval = np.percentile(samples, [2.5, 97.5]).tolist()
        summary.update(
            paired_mean_difference_s=float(delta.mean()),
            paired_bootstrap_95_interval_s=interval,
            first_half_mean_difference_s=float(delta[:6].mean()),
            second_half_mean_difference_s=float(delta[6:].mean()),
        )
        summary["stable_task_benefit_gate"] = bool(
            all(geometry_equal)
            and interval[1] < 0
            and delta[:6].mean() < 0
            and delta[6:].mean() < 0
            and aggregates["predicted"]["successes"] >= aggregates["identity"]["successes"]
            and aggregates["predicted"]["technical_failures"] <= aggregates["identity"]["technical_failures"]
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("initial", "warmed"), default="initial")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the protocol before the frozen comparison")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    first_seed, first_policy_seed = (20261101, 2101) if args.cohort == "initial" else (20261201, 2401)
    order = list(itertools.permutations(("sync", "identity", "predicted"))) * 2
    random.Random(2088).shuffle(order)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "task": "pickorange_first_settled_v1",
        "orders": order,
        "cohort": args.cohort,
        "environment_seeds": list(range(first_seed, first_seed + 12)),
        "policy_seeds": list(range(first_policy_seed, first_policy_seed + 12)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    common = [
        str(Path(__file__).with_name("eval_leisaac_so101.py")),
        "--max-steps",
        "3600",
        "--episode-seconds",
        "120",
        "--sim-python",
        str(base / "leisaac-sim-venv/bin/python"),
        "--leisaac-root",
        str(base / "leisaac-source"),
        "--assets-root",
        str(
            base
            / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets"
        ),
        "--matched-snapshot",
        str(
            base
            / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
        ),
        "--sim-device",
        "cpu",
        "--device",
        "cuda:0",
        "--camera-backend",
        "standard",
        "--task-evidence",
        "--action-contract",
        "feasible_v1",
        "--stop-after-first-placement",
    ]
    rows = []
    if args.cohort == "warmed":
        common += ["--startup-profile", "warmed_v2"]
    for block, modes in enumerate(order):
        for mode in modes:
            folder = args.output / f"block{block:02d}_{mode}"
            command = [
                sys.executable,
                *common,
                "--mode",
                mode,
                "--seed",
                str(first_seed + block),
                "--policy-seed",
                str(first_policy_seed + block),
                "--output",
                str(folder),
            ]
            if mode == "predicted":
                command += ["--predictor", str(base / "artifacts/m54l6_predictor_training_v1/best.pt")]
            process = subprocess.run(
                command,
                cwd=repo,
                env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if not (folder / "result.json").exists():
                raise RuntimeError(process.stdout[-6000:])
            (folder / "launcher.log").write_text(process.stdout)
            row = describe_trial(folder)
            row["block"] = block
            rows.append(row)
            report = {
                "complete": len(rows) == 36,
                "source_commit": source,
                "trials": rows,
                **comparison_summary(rows),
            }
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "block",
                            "mode",
                            "status",
                            "first_placement_success",
                            "restricted_placement_time_s",
                            "completed_actions",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
