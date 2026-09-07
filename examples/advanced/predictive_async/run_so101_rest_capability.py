"""L19b once-only native-rest contingency after a failed, closed L19 cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from leisaac_so101_contract import REST_POSE_DEG
from run_so101_sync_capability import describe, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the contingency before any native-rest dispatch")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    prerequisite = json.loads((base / "artifacts/m54l19_prepared_start_v1/summary.json").read_text())
    if (
        not prerequisite["complete"]
        or not prerequisite["all_cleanup_complete"]
        or prerequisite["aggregates"]["training_medoid_v1"]["qualified_development"]
    ):
        parser.error("Requires a fully closed L19 cohort with a failed medoid qualification")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "L19b_native_rest_development_not_a_cross_cohort_effect",
        "source_commit": source,
        "prerequisite_source": prerequisite["source_commit"],
        "initial_pose": "rest",
        "native_rest_degrees": list(REST_POSE_DEG),
        "environment_seeds": list(range(20271030, 20271038)),
        "policy_seeds": list(range(3930, 3938)),
        "forecasters_loaded": False,
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
        "sync-execution-steps": 50,
        "initial-pose": "rest",
    }
    rows, error = [], None
    try:
        for block in range(8):
            folder = args.output / f"block{block:02d}_rest"
            values = {
                **common,
                "seed": 20271030 + block,
                "policy-seed": 3930 + block,
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
            row = {"block": block, **describe(folder, "existing_literal", source)}
            result = json.loads((folder / "result.json").read_text())
            if result["environment"]["initial_pose"] != "rest":
                raise ValueError("The actual environment did not use native-rest initialization")
            if row["initial_state"] is not None and not np.allclose(
                row["initial_state"], [*REST_POSE_DEG[:5], 0.0], atol=1e-4, rtol=0
            ):
                raise ValueError("Measured native-rest task-start state differs")
            rows.append(row)
            (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("block", "status", "completed_actions", "native_success", "subgoal_success")
                    }
                ),
                flush=True,
            )
    except Exception:
        error = traceback.format_exc()
    aggregate = summarize(rows)["existing_literal"]
    report = {
        **manifest,
        "complete": error is None and len(rows) == 8,
        "rows": rows,
        "aggregate": aggregate,
        "global_error": error,
        "completed_actions": sum(r["completed_actions"] for r in rows),
        "setup_actions": sum(r["setup_actions"] for r in rows),
        "all_cleanup_complete": all(r["cleanup_complete"] for r in rows),
        "forecasting_benefit_tested": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    return int(not report["complete"] or not report["all_cleanup_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
