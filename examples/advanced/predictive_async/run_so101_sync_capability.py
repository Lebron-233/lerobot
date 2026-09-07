"""L18 fixed zero-delay native-task capability diagnostic; no forecast or prompt search."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

from audit_so101_noise_reference import native_components, native_predicate
from audit_so101_privileged_reference import read_rows
from run_so101_noise_reference import outcomes

PROMPTS = {
    "existing_literal": "Grab orange and place into plate",
    "model_card_literal": "Pick up the orange and put it in the plate",
}


def describe(folder: Path, prompt: str, source: str) -> dict:
    result = json.loads((folder / "result.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    ticks = read_rows(folder / "ticks.jsonl")
    if (
        result["source_commit"] != source
        or result["mode"] != "sync"
        or result["realtime_required"]
        or result["candidate"]["task"] != PROMPTS[prompt]
        or result["candidate"].get("predictor") is not None
        or "engine_stats" in result
        or manifest["args"]["sync_execution_steps"] != 50
        or manifest["args"]["stop_after_first_placement"]
    ):
        raise ValueError("The registered unaugmented synchronous candidate changed")
    measured = outcomes(ticks, {**result, "native_success": result.get("success")})
    cleanup = result["subprocess_returncode"] == 0 and not result.get("cleanup_error")
    if not cleanup:
        measured.update(
            technical_valid=False,
            native_success=False,
            subgoal_success=False,
            native_time_s=120.0,
            subgoal_time_s=120.0,
        )
    boxes, mismatches = [], 0
    for tick in ticks:
        if tick["dispatch"] != "completed":
            continue
        witness = tick["task_transition_after_action"]
        mismatches += int(native_predicate(witness) != tick["terminated"])
        boxes.append(native_components(witness)[0])
    if mismatches:
        raise ValueError("Native full-task outcome differs from the pre-reset physical witness")
    first = ticks[0] if ticks else {}
    return {
        **measured,
        "source_commit": source,
        "prompt": prompt,
        "task_text": result["candidate"]["task"],
        "seed": manifest["args"]["seed"],
        "policy_seed": manifest["args"]["policy_seed"],
        "status": result["status"],
        "max_native_box_occupancy": max(boxes, default=0),
        "native_rule_mismatches": mismatches,
        "setup_actions": result.get("setup_physics_steps", 0),
        "simulator_returncode": result["subprocess_returncode"],
        "cleanup_complete": cleanup and result["metrics_closed"],
        "initial_state": first.get("state"),
        "initial_objects": first.get("task_diagnostics_before_action", {}).get("object_positions_world"),
        "initial_cameras": first.get("camera_world_poses_opengl"),
        "error": result.get("error"),
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for prompt in PROMPTS:
        group = [r for r in rows if r["prompt"] == prompt]
        successes = sum(r["native_success"] for r in group)
        failures = sum(not r["technical_valid"] for r in group)
        result[prompt] = {
            "trials": len(group),
            "native_successes": successes,
            "subgoal_successes": sum(r["subgoal_success"] for r in group),
            "technical_failures": failures,
            "mean_restricted_native_time_s": sum(r["native_time_s"] for r in group) / len(group)
            if group
            else None,
            "qualified_development": len(group) == 8 and successes >= 7 and failures == 0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the diagnostic before creating any new scene")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    orders = [tuple(PROMPTS)] * 4 + [tuple(reversed(PROMPTS))] * 4
    random.Random(3800).shuffle(orders)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "kind": "L18_zero_delay_full_task_development",
        "orders": orders,
        "prompts": PROMPTS,
        "environment_seeds": list(range(20270910, 20270918)),
        "policy_seeds": list(range(3810, 3818)),
        "forecasting": False,
        "realtime": False,
        "training": False,
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
            for prompt in order:
                folder = args.output / f"block{block:02d}_{prompt}"
                values = {
                    **common,
                    "seed": 20270910 + block,
                    "policy-seed": 3810 + block,
                    "output": folder,
                    "sync-task-text": PROMPTS[prompt],
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
                row = {"block": block, **describe(folder, prompt, source)}
                rows.append(row)
                (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "block",
                                "prompt",
                                "status",
                                "native_success",
                                "subgoal_success",
                                "completed_actions",
                                "max_native_box_occupancy",
                            )
                        }
                    ),
                    flush=True,
                )
    except Exception:
        error = traceback.format_exc()
    paired_geometry = []
    for block in range(8):
        group = [r for r in rows if r["block"] == block]
        if len(group) == 2:
            paired_geometry.append(
                all(
                    group[0][key] is not None and group[0][key] == group[1][key]
                    for key in ("initial_state", "initial_objects", "initial_cameras")
                )
            )
    report = {
        **manifest,
        "complete": error is None and len(rows) == 16,
        "rows": rows,
        "aggregates": summarize(rows),
        "global_error": error,
        "paired_initial_geometry_equal": paired_geometry,
        "completed_actions": sum(r["completed_actions"] for r in rows),
        "setup_actions": sum(r["setup_actions"] for r in rows),
        "all_cleanup_complete": all(r["cleanup_complete"] for r in rows),
        "predictor_benefit_tested": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    return int(not report["complete"] or not report["all_cleanup_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
