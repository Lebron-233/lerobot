"""Six predeclared L9 forty-second qualifications; no trial retries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from summarize_so101_runtime import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit source before the registered qualification")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    orders = [("identity", "predicted"), ("predicted", "identity"), ("identity", "predicted")]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {"source_commit": source, "orders": orders, "seeds": [20261222, 20261223, 20261224]}, indent=2
        )
    )
    common = {
        "max-steps": 1200,
        "episode-seconds": 60,
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
    runs = []
    for block, modes in enumerate(orders):
        for mode in modes:
            folder = args.output / f"block{block:02d}_{mode}"
            values = {
                **common,
                "mode": mode,
                "seed": 20261222 + block,
                "policy-seed": 2422 + block,
                "output": folder,
            }
            if mode == "predicted":
                values["predictor"] = base / "artifacts/m54l6_predictor_training_v1/best.pt"
            command = [
                sys.executable,
                str(Path(__file__).with_name("eval_leisaac_so101.py")),
                "--task-evidence",
            ]
            for key, value in values.items():
                command.extend([f"--{key}", str(value)])
            execution = subprocess.run(
                command,
                cwd=repo,
                env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if not (folder / "result.json").is_file():
                raise RuntimeError(execution.stdout[-6000:])
            (folder / "launcher.log").write_text(execution.stdout)
            row = summarize(folder)
            stats = row["engine_stats"]
            row["qualification_pass"] = bool(
                row["bounded_runtime_gate_pass"]
                and (row["completed_dispatches"] == 1200 or row["success"] is True)
                and all(
                    stats[key] == 0 for key in ("underflows", "deadline_misses", "prediction_cap_exceeded")
                )
            )
            runs.append(row)
            report = {
                "source_commit": source,
                "complete": len(runs) == 6,
                "runs": runs,
                "qualified": len(runs) == 6 and all(r["qualification_pass"] for r in runs),
            }
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "mode",
                            "seed",
                            "status",
                            "completed_dispatches",
                            "qualification_pass",
                            "work_ms",
                            "planned_predictor_calls",
                        )
                    }
                ),
                flush=True,
            )
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
