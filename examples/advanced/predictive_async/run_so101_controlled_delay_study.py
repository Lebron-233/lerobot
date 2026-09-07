"""L14 fixed simulated-delay task cohort, not real-time qualification."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from run_so101_joint_study import ARMS, task_summary
from run_so101_task_comparison import describe_trial
from summarize_so101_runtime import read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke = json.loads((args.smoke / "result.json").read_text())
    events = read_rows(args.smoke / "events.jsonl")
    if (
        smoke["status"] != "censored_step_limit"
        or smoke["steps"] != 100
        or smoke["realtime_required"]
        or smoke["subprocess_returncode"] != 0
        or not any(
            e.get("state_predictor_calls") == 1 and e.get("visual_predictor_calls") == 1 for e in events
        )
    ):
        parser.error("The registered joint smoke did not qualify")
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit source before the fixed cohort")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    orders = [ARMS[i:] + ARMS[:i] for i in range(5)]
    reverse = tuple(reversed(ARMS))
    orders += [reverse[i:] + reverse[:i] for i in range(3)]
    random.Random(3400).shuffle(orders)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "execution": "controlled_simulated_delay_NOT_realtime",
        "orders": orders,
        "environment_seeds": list(range(20270510, 20270518)),
        "policy_seeds": list(range(3410, 3418)),
        "delay_steps": 7,
        "state_model": "L12_epoch29",
        "visual_model": "L6_epoch3",
        "risk_thresholds": None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for block, order in enumerate(orders):
        for arm in order:
            folder = args.output / f"block{block:02d}_{arm}"
            command = [
                sys.executable,
                str(Path(__file__).with_name("eval_so101_controlled_delay.py")),
                "--arm",
                arm,
                "--seed",
                str(20270510 + block),
                "--policy-seed",
                str(3410 + block),
                "--output",
                str(folder),
            ]
            process = subprocess.run(
                command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            if not (folder / "result.json").is_file():
                raise RuntimeError(process.stdout[-6000:])
            (folder / "launcher.log").write_text(process.stdout)
            row = describe_trial(folder)
            row.update(arm=arm, block=block)
            requests = read_rows(folder / "events.jsonl")
            ticks = read_rows(folder / "ticks.jsonl")
            row["state_predictor_calls"] = sum(e["state_predictor_calls"] for e in requests)
            row["visual_predictor_calls"] = sum(e["visual_predictor_calls"] for e in requests)
            row["takeovers"] = sum(t.get("queue_outcome") == "takeover" for t in ticks)
            if row["source_commit"] != source:
                raise RuntimeError("Source changed during the fixed cohort")
            rows.append(row)
            report = {
                "source_commit": source,
                "execution": manifest["execution"],
                "trials": rows,
                "complete": len(rows) == 40,
                "realtime_qualified": False,
                **task_summary(rows),
            }
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "block",
                            "arm",
                            "status",
                            "completed_actions",
                            "first_placement_success",
                            "restricted_placement_time_s",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
