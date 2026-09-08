"""Complete the original ten-step control for every tuple in the closed one-step screen."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import libero_reference_qualification as reference
import libero_single_step_native as single
import numpy as np
from diagnose_libero_reference_records import summary as timing_summary
from libero_reference_smoke import write_json

CANDIDATE_HEAD = "ee273bcecc89fb2c6fc1b092b2770d134ec32085"
BOOTSTRAP_SEED = 970001


def registered_control_tuples():
    return [
        {**spec, "tuple_id": spec["tuple_id"].replace("single_step_native/", "matched_ten_step_reference/")}
        for spec in single.registered_tuples()
    ]


def require_closed_candidate(path):
    summary = json.loads((path / "summary.json").read_text())
    registration = json.loads((path / "registration.json").read_text())
    records = json.loads((path / "tuple_accounting.json").read_text())
    if (
        summary["source_head"] != CANDIDATE_HEAD
        or summary["completed"] != 90
        or summary["technical_failure"] != 0
        or summary["worker_exit_code"] != 0
        or not summary["records_verified"]
        or registration["control"]["denoising"] != 1
        or registration["tuples"] != single.registered_tuples()
        or [row["tuple"] for row in records] != single.registered_tuples()
    ):
        raise ValueError("Expected the complete, closed fixed one-step cohort at its registered source")
    return summary, registration, records


def exact_mcnemar(single_only, ten_only):
    n = single_only + ten_only
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(single_only, ten_only) + 1)) / 2**n)


def paired_statistics(candidate, control):
    """Resample paired differences within each of the ten fixed task strata."""
    if [r["tuple"] for r in candidate] != single.registered_tuples():
        raise ValueError("Candidate tuple identities changed")
    if [r["tuple"] for r in control] != registered_control_tuples():
        raise ValueError("Control tuple identities changed")
    if any(r["status"] != "completed" or r.get("audit_errors") for r in candidate + control):
        return None
    a = np.array([int(row["success"]) for row in candidate])
    b = np.array([int(row["success"]) for row in control])
    single_only = int(np.count_nonzero((a == 1) & (b == 0)))
    ten_only = int(np.count_nonzero((a == 0) & (b == 1)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000)
    differences = a - b
    tasks = []
    for task in range(10):
        lo, hi = task * 9, (task + 1) * 9
        values = differences[lo:hi]
        draws += values[rng.integers(0, 9, size=(20000, 9))].mean(axis=1) / 10
        tasks.append(
            {
                "task_id": task,
                "single_step_successes": int(a[lo:hi].sum()),
                "ten_step_successes": int(b[lo:hi].sum()),
                "single_only": int(np.count_nonzero((a[lo:hi] == 1) & (b[lo:hi] == 0))),
                "ten_only": int(np.count_nonzero((a[lo:hi] == 0) & (b[lo:hi] == 1))),
                "scheduled_pairs": 9,
            }
        )
    return {
        "pairs": 90,
        "single_step_successes": int(a.sum()),
        "ten_step_successes": int(b.sum()),
        "both_success": int(np.count_nonzero((a == 1) & (b == 1))),
        "single_only": single_only,
        "ten_only": ten_only,
        "both_failure": int(np.count_nonzero((a == 0) & (b == 0))),
        "single_minus_ten_success_rate": float(differences.mean()),
        "paired_stratified_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": 20000,
        "exact_two_sided_mcnemar_p": exact_mcnemar(single_only, ten_only),
        "tasks": tasks,
        "inference_scope": "post_screen_matched_diagnostic_not_independent_confirmatory_test",
    }


def initial_observation(directory):
    with (directory / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "observation" and event["index"] == 0:
                return event
    raise ValueError(f"Missing initial observation: {directory}")


def compare_initial_observations(candidate, control):
    left, right = initial_observation(candidate), initial_observation(control)
    result = {
        "state_equal": left["state"] == right["state"],
        "quaternion_equal": left["eef_quaternion_xyzw"] == right["eef_quaternion_xyzw"],
    }
    with (
        np.load(candidate / left["cameras"], allow_pickle=False) as a,
        np.load(control / right["cameras"], allow_pickle=False) as b,
    ):
        for camera in ("image", "image2"):
            result[camera + "_equal"] = np.array_equal(a[camera], b[camera])
    result["all_equal"] = all(result.values())
    return result


def supervise(args):
    candidate_summary, candidate_registration, candidate = require_closed_candidate(args.candidate_output)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference.REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=reference.REPO, text=True).strip()
    if head != args.execution_head or dirty:
        raise RuntimeError("Expected the registered clean source HEAD")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("Use the fixed offline EGL launch contract")
    versions = {name: version(name) for name in candidate_registration["versions"]}
    if versions != candidate_registration["versions"]:
        raise RuntimeError("Installed dependency versions changed between conditions")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "registration.json",
        {
            "source_head": head,
            "started_at": datetime.now(UTC).isoformat(),
            "supervisor_pid": os.getpid(),
            "candidate_source_head": CANDIDATE_HEAD,
            "candidate_output": str(args.candidate_output),
            "policy_path": str(args.policy_path),
            "vlm_path": str(args.vlm_path),
            "versions": versions,
            "control": {"frequency": 20, "chunk": 50, "consume": 1, "denoising": 10, "limit": 280},
            "tuples": registered_control_tuples(),
        },
    )
    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
    ]
    start = time.perf_counter()
    launch_error, exit_code = None, None
    with (args.output / "worker.log").open("x") as log:
        try:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
                write_json(args.output / "worker_process.json", {"pid": process.pid, "command": command})
                exit_code = process.wait()
        except OSError as exc:
            launch_error = str(exc)
    process_result = {
        "worker_exit_code": exit_code,
        "launch_error": launch_error,
        "worker_wall_seconds": time.perf_counter() - start,
        "ended_at": datetime.now(UTC).isoformat(),
    }
    write_json(args.output / "process_exit.json", process_result)
    control = [
        reference.audit_tuple(reference.tuple_directory(args.output, spec), spec)
        for spec in registered_control_tuples()
    ]
    write_json(args.output / "tuple_accounting.json", control)
    starts = []
    latencies = []
    for left, right in zip(candidate, control, strict=True):
        left_path = reference.tuple_directory(args.candidate_output, left["tuple"])
        right_path = reference.tuple_directory(args.output, right["tuple"])
        if right["status"] == "completed":
            starts.append(
                {"ordinal": right["tuple"]["ordinal"], **compare_initial_observations(left_path, right_path)}
            )
        latencies.extend(
            event["inference_seconds"]
            for event in reference.read_events(right_path / "events.jsonl")
            if event["event"] == "action_prepared"
        )
    initial_equal = len(starts) == 90 and all(row["all_equal"] for row in starts)
    result = {
        "scope": "post_screen_all_90_matched_original_ten_step_control",
        "source_head": head,
        "candidate_source_head": CANDIDATE_HEAD,
        "scheduled": 90,
        "completed": sum(row["status"] == "completed" for row in control),
        "technical_failure": sum(row["status"] == "technical_failure" for row in control),
        "not_run": sum(row["status"] == "not_run" for row in control),
        "ten_step_successes": sum(row["success"] for row in control),
        "environment_closed": sum(row.get("environment_closed") is True for row in control),
        "physical_step_unknown_tuples": sum(row.get("physical_step_pending") == "unknown" for row in control),
        "native_measurement_returns": sum(row.get("native_measurement_returns", 0) for row in control),
        "native_settling_returns": sum(row.get("native_settling_returns", 0) for row in control),
        "records_verified": all(not row.get("audit_errors") for row in control),
        "initial_observations_all_equal": initial_equal,
        "initial_observation_comparisons": starts,
        "paired_statistics": paired_statistics(candidate, control)
        if initial_equal and exit_code == 0
        else None,
        "ten_step_policy_selection_seconds": timing_summary(latencies) if latencies else None,
        "single_step_policy_selection_seconds": candidate_summary["policy_selection_seconds"],
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
        "old_confirmation": "not_started_untouched",
        **process_result,
    }
    write_json(args.output / "summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "initial_observation_comparisons"}, indent=2),
        flush=True,
    )
    return 0 if result["paired_statistics"] is not None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return single.run_worker(args, denoising_steps=10, tuples=registered_control_tuples())
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
