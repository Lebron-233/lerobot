"""Read-only L15 collection and selection audit; no policy/model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from train_so101_incremental_vision import summarize


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_data(cache: Path) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text())
    result = json.loads((cache / "result.json").read_text())
    if not result["complete"] or result["error"] or result["cleanup_error"]:
        raise ValueError("Incomplete collection is not an eligible data cohort")
    expected = 8 if manifest["split"] == "development" else 6
    if len(result["episodes"]) != expected or result["subprocess_returncode"] != 0:
        raise ValueError("Collection did not close all registered conditions")
    rows = []
    for episode in range(expected):
        payload = torch.load(cache / f"episode_{episode:02d}.pt", map_location="cpu", weights_only=True)
        ticks = read_rows(cache / f"episode_{episode:02d}_ticks.jsonl")
        events = read_rows(cache / f"episode_{episode:02d}_events.jsonl")
        setup = read_rows(cache / f"episode_{episode:02d}_setup_ticks.jsonl")
        if [t["tick"] for t in ticks] != list(range(len(ticks))):
            raise ValueError("Physical action indices skipped or repeated")
        if any(t["dispatch"] != "completed" for t in ticks):
            raise ValueError("Unfinished physical actions cannot be hidden by complete pairs")
        planned = {e["request_step"]: e for e in events if e["simulated_delay_steps"] == 7}
        observed = [p["request_step"] for p in payload["pairs"]]
        terminal = bool(ticks[-1]["terminated"] or ticks[-1]["truncated"])
        expected_steps = [step for step in planned if step + 7 < len(ticks)]
        if observed != expected_steps or len(set(observed)) != len(observed):
            raise ValueError("Missing, duplicated or post-terminal target pairs")
        for pair in payload["pairs"]:
            start = pair["request_step"]
            request = planned[start]
            actual = [ticks[i]["normalized_action"] for i in range(start, start + 7)]
            if actual != request["committed_normalized_prefix"]:
                raise ValueError("The seven executed actions differ from their frozen commitment")
            if not torch.equal(torch.tensor(actual), pair["actions"][0, :7]):
                raise ValueError("Training prefix differs from physically executed actions")
            if pair["target_step"] != start + 7 or pair["delay"].item() != 7:
                raise ValueError("The training target delay changed")
            if not torch.equal(pair["valid"], torch.arange(8)[None] < 7):
                raise ValueError("An eighth future action entered the committed prefix")
            if any(ticks[i]["terminated"] or ticks[i]["truncated"] for i in range(start, start + 7)):
                raise ValueError("A target crosses automatic reset")
            if ticks[start + 7]["queue_outcome"] != "takeover":
                raise ValueError("The paired future observation was not at the planned takeover")
        if any(e["visual_predictor_calls"] != 0 for e in events):
            raise ValueError("A visual forecaster influenced state-only collection")
        if sum(t["dispatch"] == "completed" for t in setup) != 30:
            raise ValueError("Setup physics calls are not the fixed logged thirty")
        rows.append(
            {
                "episode": episode,
                "actions": len(ticks),
                "pairs": len(observed),
                "setup_actions": len(setup),
                "native_terminal": terminal,
                "native_success": bool(ticks[-1]["terminated"]),
                "censored_prefixes": len(planned) - len(observed),
            }
        )
    return {
        "source_commit": result["source_commit"],
        "split": manifest["split"],
        "rows": rows,
        "actions": sum(row["actions"] for row in rows),
        "pairs": sum(row["pairs"] for row in rows),
        "setup_actions": sum(row["setup_actions"] for row in rows),
        "all_causal_prefixes_match": True,
    }


def check_case_grid(rows: list[dict], pair_counts: dict[int, int]) -> None:
    expected = {(e, q, n) for e, count in pair_counts.items() for q in range(count) for n in range(2)}
    actual = [(row["episode"], row["query_index"], row["noise_index"]) for row in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Incomplete/duplicated noise-query grid changes the registered denominator")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--test-report", type=Path)
    args = parser.parse_args()
    data = audit_data(args.cache)
    selection = json.loads((args.training / "selection.json").read_text())
    history = json.loads((args.training / "training.json").read_text())
    balanced = selection.get("profile", "single") == "balanced4"
    updates, interval = (400, 50) if balanced else (1600, 200)
    if [row["update"] for row in history] != list(range(1, updates + 1)):
        raise ValueError("The fixed training budget did not complete exactly once")
    if balanced and any(len({case[0] for case in row["batch_cases"]}) != 4 for row in history):
        raise ValueError("Balanced updates did not combine four distinct training episodes")
    candidates = []
    for update in range(0, updates + 1, interval):
        metric = json.loads((args.training / f"validation_{update:04d}.json").read_text())
        check_case_grid(metric["cases"], {row["episode"]: row["pairs"] for row in data["rows"][6:]})
        if summarize(metric["cases"]) != metric:
            raise ValueError("Saved validation metric disagrees with all its cases")
        eligible = update > 0 and metric["validation_eligible"]
        candidates.append((not eligible, metric["equal_episode_means"]["student_l1"], update))
    expected = min(candidates)
    if selection["selected_update"] != expected[2] or selection["qualified"] != (not expected[0]):
        raise ValueError("Selection does not follow the preregistered eligible-first rule")
    report = {
        "kind": "l15_no_inference_closed_selection_audit",
        "development": data,
        "selected_update": selection["selected_update"],
        "profile": selection.get("profile", "single"),
        "qualified": selection["qualified"],
        "selection_matches_registered_rule": True,
        "zero_residual_action_difference": selection["zero_residual_max_action_difference"],
        "gradient_proof": selection["gradient_proof"],
        "selected_validation": {k: v for k, v in selection["selected_validation"].items() if k != "cases"},
        "test_audited": False,
    }
    if args.test_cache is not None or args.test_report is not None:
        if args.test_cache is None or args.test_report is None or not selection["qualified"]:
            raise ValueError("Independent test audit requires the qualified selection and both artifacts")
        test_data = audit_data(args.test_cache)
        test = json.loads(args.test_report.read_text())
        check_case_grid(test["cases"], {row["episode"]: row["pairs"] for row in test_data["rows"]})
        rebuilt = summarize(test["cases"], held_out=True)
        if any(test[key] != value for key, value in rebuilt.items()):
            raise ValueError("The test statistic changed after model evaluation")
        report.update(
            test_audited=True,
            test_data=test_data,
            test_summary={k: v for k, v in rebuilt.items() if k != "cases"},
        )
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
