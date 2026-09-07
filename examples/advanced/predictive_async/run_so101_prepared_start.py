"""L19 fixed two-pose native capability development, no training or trial retries."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from run_so101_sync_capability import PROMPTS, describe
from so101_prepared_start import (
    DATASET_REVISION,
    EPISODE,
    MOTOR_STATE,
    PROFILE,
    prepared_degrees,
    prepared_joints,
    provenance,
)

POSES = ("zero", PROFILE)


def select_medoid(first_states: np.ndarray) -> int:
    if first_states.ndim != 2 or first_states.shape[1] != 6 or not np.isfinite(first_states).all():
        raise ValueError("Expected finite native motor first states, not actions or future frames")
    if len(first_states) == 0:
        raise ValueError("No training first states supplied")
    scaled = first_states.astype(np.float64) / np.array([200, 200, 200, 200, 200, 100])
    distance = np.abs(scaled[:, None] - scaled[None, :]).mean(axis=(1, 2))
    return int(distance.argmin())


def check_training_source(base: Path) -> dict:
    import pandas as pd

    snapshot = (
        base
        / "matched-candidate-cache/datasets--LightwheelAI--leisaac-pick-orange/snapshots"
        / DATASET_REVISION
    )
    info = json.loads((snapshot / "meta/info.json").read_text())
    tasks = [json.loads(line) for line in (snapshot / "meta/tasks.jsonl").read_text().splitlines()]
    if info["total_episodes"] != 60 or tasks != [{"task_index": 0, "task": PROMPTS["existing_literal"]}]:
        raise ValueError("The pinned published training source or literal task text differs")
    points = np.stack(
        [
            pd.read_parquet(
                snapshot / f"data/chunk-000/episode_{i:06d}.parquet", columns=["observation.state"]
            ).iloc[0]["observation.state"]
            for i in range(60)
        ]
    )
    selected = select_medoid(points)
    if selected != EPISODE or not np.array_equal(points[selected].astype(np.float64), MOTOR_STATE):
        raise ValueError("The frozen medoid does not match all sixty published first states")
    return {**provenance(), "checked_training_first_states": len(points)}


def describe_prepared(folder: Path, pose: str, source: str) -> dict:
    row = describe(folder, "existing_literal", source)
    result = json.loads((folder / "result.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["args"]["initial_pose"] != pose or result["environment"]["initial_pose"] != pose:
        raise ValueError("The actual initial-pose profile differs from its assigned condition")
    row["pose"] = pose
    if pose == PROFILE:
        environment = result["environment"]
        if environment.get("prepared_start") != provenance():
            raise ValueError("Prepared state provenance was not propagated to the simulator")
        if environment["initial_joint_positions_radians"] != prepared_joints():
            raise ValueError("Native reset configuration differs from the frozen medoid")
        expected = [*prepared_degrees()[:5], MOTOR_STATE[5]]
        if row["initial_state"] is not None and not np.allclose(
            row["initial_state"], expected, atol=1e-4, rtol=0
        ):
            raise ValueError("The measured task-start state is not the registered preparation")
    return row


def summarize(rows: list[dict]) -> dict:
    aggregates = {}
    for pose in POSES:
        group = [r for r in rows if r["pose"] == pose]
        success = sum(r["native_success"] for r in group)
        failures = sum(not r["technical_valid"] for r in group)
        aggregates[pose] = {
            "trials": len(group),
            "native_successes": success,
            "subgoal_successes": sum(r["subgoal_success"] for r in group),
            "technical_failures": failures,
            "mean_restricted_native_time_s": float(np.mean([r["native_time_s"] for r in group]))
            if group
            else None,
            "qualified_development": len(group) == 8 and success >= 7 and failures == 0,
        }
    differences, objects, front = [], [], []
    for block in range(8):
        group = {r["pose"]: r for r in rows if r["block"] == block}
        if set(group) == set(POSES):
            zero, prepared = group["zero"], group[PROFILE]
            differences.append(prepared["native_time_s"] - zero["native_time_s"])
            objects.append(
                zero["initial_objects"] is not None and zero["initial_objects"] == prepared["initial_objects"]
            )
            front.append(
                zero["initial_cameras"] is not None
                and prepared["initial_cameras"] is not None
                and zero["initial_cameras"]["front"] == prepared["initial_cameras"]["front"]
            )
    interval = None
    if len(differences) == 8:
        values = np.asarray(differences)
        rng = np.random.default_rng(3901)
        interval = np.percentile(values[rng.integers(0, 8, (50000, 8))].mean(axis=1), [2.5, 97.5]).tolist()
    return {
        "aggregates": aggregates,
        "paired_native_time_differences_s": differences,
        "descriptive_paired_bootstrap_95_s": interval,
        "paired_initial_object_positions_equal": objects,
        "paired_front_camera_equal": front,
        "joint_and_wrist_pose_intentionally_different": True,
        "forecasting_benefit_tested": False,
        "realtime_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the initialization protocol before dispatch")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    training_source = check_training_source(base)
    orders = [POSES] * 4 + [tuple(reversed(POSES))] * 4
    random.Random(3900).shuffle(orders)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "L19_two_pose_native_task_development",
        "source_commit": source,
        "orders": orders,
        "environment_seeds": list(range(20271010, 20271018)),
        "policy_seeds": list(range(3910, 3918)),
        "prepared_start_source": training_source,
        "task": PROMPTS["existing_literal"],
        "training": False,
        "forecasting": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    common = {
        "mode": "sync",
        "max-steps": 3600,
        "episode-seconds": 120,
        "sim-python": base / "leisaac-sim-venv/bin/python",
        "leisaac-root": base / "leisaac-source",
        "assets-root": base
        / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets",
        "matched-snapshot": base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a",
        "sim-device": "cpu",
        "device": "cuda:0",
        "camera-backend": "standard",
        "action-contract": "feasible_v1",
        "startup-profile": "warmed_v2",
        "sync-execution-steps": 50,
    }
    rows, error = [], None
    try:
        for block, order in enumerate(orders):
            for pose in order:
                folder = args.output / f"block{block:02d}_{pose}"
                values = {
                    **common,
                    "seed": 20271010 + block,
                    "policy-seed": 3910 + block,
                    "initial-pose": pose,
                    "output": folder,
                }
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("eval_leisaac_so101.py")),
                    "--task-evidence",
                ]
                for key, value in values.items():
                    command.extend([f"--{key}", str(value)])
                process = subprocess.run(
                    command,
                    cwd=repo,
                    env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if not (folder / "result.json").is_file():
                    raise RuntimeError(process.stdout[-6000:])
                (folder / "launcher.log").write_text(process.stdout)
                row = {"block": block, **describe_prepared(folder, pose, source)}
                rows.append(row)
                (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "block",
                                "pose",
                                "status",
                                "completed_actions",
                                "native_success",
                                "subgoal_success",
                                "max_native_box_occupancy",
                            )
                        }
                    ),
                    flush=True,
                )
    except Exception:
        error = traceback.format_exc()
    report = {
        **manifest,
        "complete": error is None and len(rows) == 16,
        "rows": rows,
        "global_error": error,
        "completed_actions": sum(r["completed_actions"] for r in rows),
        "setup_actions": sum(r["setup_actions"] for r in rows),
        "all_cleanup_complete": all(r["cleanup_complete"] for r in rows),
        **summarize(rows),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    return int(not report["complete"] or not report["all_cleanup_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
