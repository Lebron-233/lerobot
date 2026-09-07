"""Read-only reconstruction of closed L12 cases; no model or test rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from validate_so101_joint_context import summarize_contexts


def audit(root: Path, cache: Path) -> dict:
    result = json.loads((root / "test_report.json").read_text())
    collection = json.loads((cache / "result.json").read_text())
    manifest = json.loads((cache / "manifest.json").read_text())
    rows = result["rows"]
    observed = [(r["episode"], r["anchor"], r["delay"]) for r in rows]
    expected = {(e, t, d) for e in range(6) for t in range(0, 600, 50) for d in (1, 4, 7, 8)}
    if len(observed) != 288 or len(set(observed)) != 288 or set(observed) != expected:
        raise ValueError("The prespecified case grid is incomplete or repeated")
    if manifest["environment_seeds"] != list(range(20270320, 20270326)):
        raise ValueError("Wrong independent scene cohort")
    if collection["error"] or collection["cleanup_error"] or collection["subprocess_returncode"] != 0:
        raise ValueError("Collection did not finish and clean up normally")
    completed = 0
    for e in range(6):
        ticks = [
            json.loads(line) for line in (cache / f"episode_{e:02d}_ticks.jsonl").read_text().splitlines()
        ]
        if len(ticks) != 600 or any(r["dispatch"] != "completed" for r in ticks):
            raise ValueError("A collection trajectory has missing physical steps")
        if [r["step"] for r in ticks] != list(range(600)):
            raise ValueError("Nonsequential collection control steps")
        completed += len(ticks)
    summary = summarize_contexts(rows, 288)
    if summary["contrasts"] != result["contrasts"] or summary["means_l1_25"] != result["means_l1_25"]:
        raise ValueError("Published test summary differs from raw fixed cases")
    by_delay = {}
    for d in (1, 4, 7, 8):
        group = [r for r in rows if r["delay"] == d]
        by_delay[str(d)] = summarize_contexts(group, 72)
    state_errors = {
        str(e): {
            key: float(np.mean([r[key] for r in rows if r["episode"] == e]))
            for key in ("persistence_state_mae", "predicted_state_mae")
        }
        for e in range(6)
    }
    return {
        "kind": "closed_L12_no_model_inference_audit",
        "execution_source": result["source_commit"],
        "coherent_teacher": result["teacher"],
        "state_epoch": result["state_epoch"],
        "state_parameters": result["state_parameters"],
        "collection_actions": completed,
        "collection_observations": sum(r["observations"] for r in collection["episodes"]),
        "collection_cleanup_complete": True,
        **summary,
        "by_delay": by_delay,
        "state_errors_by_episode": state_errors,
        "joint_stable_gate": all(summary["contrasts"][b]["stable_gate"] for b in ("identity", "visual_only")),
        "task_benefit_tested": False,
        "runtime_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report_root, args.cache)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
