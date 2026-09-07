"""Frozen L13 qualification and five-arm task study, with no trial replacement."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_so101_task_comparison import describe_trial
from summarize_so101_runtime import read_rows, summarize

ARMS = ("sync", "identity", "visual_only", "state_only", "joint")


def task_summary(rows: list[dict]) -> dict:
    aggregate = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        aggregate[arm] = {
            "trials": len(group),
            "successes": sum(r["first_placement_success"] for r in group),
            "technical_failures": sum(not r["technical_valid"] for r in group),
            "mean_restricted_time_s": float(np.mean([r["restricted_placement_time_s"] for r in group]))
            if group
            else None,
        }
    contrasts = {}
    for baseline in ("identity", "visual_only", "state_only"):
        differences, geometry = [], []
        for block in range(8):
            group = {r["arm"]: r for r in rows if r["block"] == block}
            if "joint" not in group or baseline not in group:
                continue
            joint, other = group["joint"], group[baseline]
            differences.append(joint["restricted_placement_time_s"] - other["restricted_placement_time_s"])
            geometry.append(
                all(
                    joint[k] is not None and joint[k] == other[k]
                    for k in ("initial_state", "initial_objects", "initial_camera_poses")
                )
            )
        result = {"differences_s": differences, "geometry_equal": geometry, "stable_gate": False}
        if len(differences) == 8:
            values = np.array(differences)
            rng = np.random.default_rng(3301)
            samples = values[rng.integers(0, 8, size=(50000, 8))].mean(1)
            interval = np.percentile(samples, [2.5, 97.5]).tolist()
            result.update(
                mean_difference_s=float(values.mean()),
                bootstrap_95_s=interval,
                first_half_s=float(values[:4].mean()),
                second_half_s=float(values[4:].mean()),
            )
            result["stable_gate"] = bool(
                all(geometry)
                and interval[1] < 0
                and values[:4].mean() < 0
                and values[4:].mean() < 0
                and aggregate["joint"]["successes"] >= aggregate[baseline]["successes"]
                and aggregate["joint"]["technical_failures"] <= aggregate[baseline]["technical_failures"]
            )
        contrasts[baseline] = result
    reference = aggregate["sync"]
    reference_ok = (
        reference["trials"] == 8 and reference["successes"] >= 7 and reference["technical_failures"] == 0
    )
    return {
        "aggregates": aggregate,
        "contrasts": contrasts,
        "reliable_reference_gate": reference_ok,
        "benefit_on_reliable_reference": bool(
            len(rows) == 40
            and reference_ok
            and all(contrasts[b]["stable_gate"] for b in ("identity", "visual_only"))
        ),
    }


def execute(repo: Path, folder: Path, arm: str, seed: int, policy_seed: int, qualification: bool) -> dict:
    base = repo.parent
    mode = arm if arm in ("sync", "identity") else "predicted"
    values = {
        "mode": mode,
        "seed": seed,
        "policy-seed": policy_seed,
        "output": folder,
        "max-steps": 600 if qualification else 3600,
        "episode-seconds": 60 if qualification else 120,
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
    }
    if mode != "sync":
        values["minimum-delay"] = 7
    if mode == "predicted":
        values["predictor"] = base / "artifacts/m54l6_predictor_training_v1/best.pt"
    if arm in ("state_only", "joint"):
        values["context-variant"] = arm
        values["future-state-checkpoint"] = base / "artifacts/m54l12_state_development_v1/state_best.pt"
    command = [sys.executable, str(Path(__file__).with_name("eval_leisaac_so101.py")), "--task-evidence"]
    if not qualification:
        command.append("--stop-after-first-placement")
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
    row = describe_trial(folder)
    row["arm"] = arm
    result = json.loads((folder / "result.json").read_text())
    events = read_rows(folder / "events.jsonl")
    planned = [e for e in events if e.get("request_kind") == "planned" and e["event"] == "chunk_request"]
    row["state_predictor_calls"] = sum(e.get("state_predictor_calls", 0) for e in planned)
    row["visual_predictor_calls"] = sum(
        e.get("visual_predictor_calls", e.get("predictor_calls", 0)) for e in planned
    )
    row["takeovers"] = sum(e["event"] == "queue_get" and e.get("outcome") == "takeover" for e in events)
    row["setup_actions"] = result.get("setup_physics_steps", 0)
    if qualification:
        runtime = summarize(folder) if result.get("engine_stats") is not None else None
        stats = result.get("engine_stats", {})
        calls_valid = bool(planned) and all(
            e.get("state_predictor_calls", 0) == int(arm in ("state_only", "joint"))
            and e.get("visual_predictor_calls", e.get("predictor_calls", 0)) == int(arm != "state_only")
            for e in planned
        )
        row["qualification_pass"] = bool(
            runtime
            and runtime["bounded_runtime_gate_pass"]
            and calls_valid
            and row["takeovers"] > 0
            and (row["completed_actions"] == 600 or row["native_success"] is True)
            and all(
                stats.get(key) == 0 for key in ("underflows", "deadline_misses", "prediction_cap_exceeded")
            )
        )
        row["runtime"] = runtime
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qualification", "task"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit source before source-bound L13 execution")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    qualification = args.phase == "qualification"
    if not qualification and (
        args.qualification is None or not json.loads(args.qualification.read_text())["qualified"]
    ):
        parser.error("All six new-context runtime qualifications must pass before task outcomes")
    if qualification:
        orders = [("visual_only", "state_only", "joint"), ("joint", "state_only", "visual_only")]
        first_seed, first_policy = 20270330, 3230
    else:
        orders = [ARMS[i:] + ARMS[:i] for i in range(5)]
        reverse = tuple(reversed(ARMS))
        orders += [reverse[i:] + reverse[:i] for i in range(3)]
        random.Random(3300).shuffle(orders)
        first_seed, first_policy = 20270410, 3310
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "source_commit": source,
                "phase": args.phase,
                "orders": orders,
                "first_seed": first_seed,
                "first_policy_seed": first_policy,
                "state_model_source": "bf4e025dbf0cacf4777289b90498a8b70d4974fa",
                "state_epoch": 29,
                "visual_predictor": "L6_epoch3_frozen",
                "risk_thresholds": None,
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    report = {}
    for block, order in enumerate(orders):
        for arm in order:
            row = execute(
                repo,
                args.output / f"block{block:02d}_{arm}",
                arm,
                first_seed + block,
                first_policy + block,
                qualification,
            )
            row["block"] = block
            rows.append(row)
            report = {
                "source_commit": source,
                "phase": args.phase,
                "trials": rows,
                "complete": len(rows) == (6 if qualification else 40),
            }
            if qualification:
                report["qualified"] = len(rows) == 6 and all(r["qualification_pass"] for r in rows)
            else:
                report.update(task_summary(rows))
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "block",
                            "arm",
                            "status",
                            "completed_actions",
                            "first_placement_success",
                            "restricted_placement_time_s",
                            "qualification_pass",
                            "state_predictor_calls",
                            "visual_predictor_calls",
                        )
                    }
                ),
                flush=True,
            )
    return int(qualification and not report["qualified"])


if __name__ == "__main__":
    raise SystemExit(main())
