"""L20 bounded feedback-span capability test, without any forecasting module."""

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
from run_so101_sync_capability import describe, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the fixed feedback-span protocol before dispatch")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    previous = json.loads((base / "artifacts/m54l19b_native_rest_v1/summary.json").read_text())
    if (
        not previous["complete"]
        or not previous["all_cleanup_complete"]
        or previous["aggregate"]["qualified_development"]
    ):
        parser.error("Requires closed, failed native-rest qualification; never bypass a qualified baseline")
    orders = [(50, 10)] * 4 + [(10, 50)] * 4
    random.Random(4000).shuffle(orders)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "L20_synchronous_feedback_span_native_development",
        "source_commit": source,
        "orders": orders,
        "environment_seeds": list(range(20271110, 20271118)),
        "policy_seeds": list(range(4010, 4018)),
        "generated_chunk_steps": 50,
        "consumed_spans": [50, 10],
        "initial_pose": "zero",
        "forecasting": False,
        "realtime": False,
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
        "initial-pose": "zero",
    }
    rows, error = [], None
    try:
        for block, order in enumerate(orders):
            for span in order:
                folder = args.output / f"block{block:02d}_span{span}"
                values = {
                    **common,
                    "seed": 20271110 + block,
                    "policy-seed": 4010 + block,
                    "sync-execution-steps": span,
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
                row = {
                    "block": block,
                    "span": span,
                    **describe(folder, "existing_literal", source, execution_steps=span),
                }
                rows.append(row)
                (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "block",
                                "span",
                                "status",
                                "completed_actions",
                                "native_success",
                                "subgoal_success",
                            )
                        }
                    ),
                    flush=True,
                )
    except Exception:
        error = traceback.format_exc()
    aggregates = {
        str(span): summarize([r for r in rows if r["span"] == span])["existing_literal"] for span in (50, 10)
    }
    differences, matched = [], []
    for block in range(8):
        group = {r["span"]: r for r in rows if r["block"] == block}
        if set(group) == {10, 50}:
            differences.append(group[10]["native_time_s"] - group[50]["native_time_s"])
            matched.append(
                all(
                    group[10][k] is not None and group[10][k] == group[50][k]
                    for k in ("initial_state", "initial_objects", "initial_cameras")
                )
            )
    interval = None
    if len(differences) == 8:
        rng = np.random.default_rng(4001)
        interval = np.percentile(
            np.asarray(differences)[rng.integers(0, 8, (50000, 8))].mean(1), [2.5, 97.5]
        ).tolist()
    report = {
        **manifest,
        "complete": error is None and len(rows) == 16,
        "rows": rows,
        "aggregates": aggregates,
        "global_error": error,
        "paired_native_time_difference_s": differences,
        "descriptive_paired_bootstrap_95_s": interval,
        "paired_initial_geometry_equal": matched,
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
