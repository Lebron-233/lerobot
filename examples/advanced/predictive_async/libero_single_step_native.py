"""One fixed 50/1/1 candidate, all ten Object tasks, previously unused state IDs 41-49."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import libero_reference_qualification as reference
import numpy as np
import torch
from diagnose_libero_reference_records import summary as timing_summary
from libero_reference_smoke import load_runtime, write_json

STATE_IDS = tuple(range(41, 50))
N_PER_TASK = len(STATE_IDS)
N_TOTAL = 10 * N_PER_TASK
BOOTSTRAP_SEED = 960001


def registered_tuples():
    """Enumerate the entire single new cohort; no old confirmation IDs are accepted."""
    return [
        {
            "ordinal": task * N_PER_TASK + state - STATE_IDS[0],
            "tuple_id": f"single_step_native/task_{task:02d}/state_{state:02d}",
            "task_id": task,
            "task_name": name,
            "initial_state_id": state,
            "environment_seed": 940000 + 100 * task + state,
            "policy_seed": 950000 + 100 * task + state,
        }
        for task, name in enumerate(reference.TASK_NAMES)
        for state in STATE_IDS
    ]


def summarize(records, exit_code, head):
    """Screen against predeclared observed counts, without granting full qualification."""
    if [row["tuple"] for row in records] != registered_tuples():
        raise ValueError("Records do not match the complete registered 90-slot cohort")
    tasks = []
    for task in range(10):
        rows = records[task * N_PER_TASK : (task + 1) * N_PER_TASK]
        observed = [row for row in rows if row["status"] != "not_run"]
        successes = sum(row["success"] for row in rows)
        tasks.append(
            {
                "task_id": task,
                "task_name": reference.TASK_NAMES[task],
                "scheduled": N_PER_TASK,
                "completed": sum(row["status"] == "completed" for row in rows),
                "observed": len(observed),
                "successes": successes,
                "success_rate_scheduled": successes / N_PER_TASK,
                "wilson_95_observed": reference.wilson(successes, len(observed)),
                "failed_state_ids": [
                    row["tuple"]["initial_state_id"] for row in observed if not row["success"]
                ],
                "restricted_simulated_seconds_mean_observed": (
                    float(np.mean([row["restricted_simulated_seconds"] for row in observed]))
                    if observed
                    else None
                ),
            }
        )
    observed = [row for row in records if row["status"] != "not_run"]
    result = {
        "scope": "independent_fixed_candidate_native_capability_screen_not_full_qualification",
        "source_head": head,
        "scheduled": N_TOTAL,
        "completed": sum(row["status"] == "completed" for row in records),
        "observed": len(observed),
        "technical_failure": sum(row["status"] == "technical_failure" for row in records),
        "not_run": sum(row["status"] == "not_run" for row in records),
        "successes": sum(row["success"] for row in records),
        "timeouts": sum(
            row["status"] == "completed" and not row["success"] and row.get("truncated", False)
            for row in records
        ),
        "environment_closed": sum(row.get("environment_closed") is True for row in records),
        "physical_step_unknown_tuples": sum(row.get("physical_step_pending") == "unknown" for row in records),
        "native_measurement_returns": sum(row.get("native_measurement_returns", 0) for row in records),
        "native_settling_returns": sum(row.get("native_settling_returns", 0) for row in records),
        "worker_exit_code": exit_code,
        "records_verified": all(not row.get("audit_errors") for row in records),
        "tasks": tasks,
        "macro_bootstrap_95": None,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": 20000,
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
        "old_confirmation": "not_started_untouched",
    }
    result["macro_success_rate_scheduled"] = result["successes"] / N_TOTAL
    if len(observed) == N_TOTAL:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = np.zeros(20000)
        for task in range(10):
            values = np.array([int(row["success"]) for row in records[task * 9 : (task + 1) * 9]])
            draws += values[rng.integers(0, N_PER_TASK, size=(20000, N_PER_TASK))].mean(axis=1) / 10
        result["macro_bootstrap_95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    result["native_screen_passed"] = (
        result["completed"] == N_TOTAL
        and result["technical_failure"] == result["not_run"] == 0
        and result["successes"] >= 81
        and all(task["successes"] >= 8 for task in tasks)
        and result["records_verified"]
        and exit_code == 0
    )
    return result


def run_worker(args):
    torch.set_num_threads(1)
    policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
    report["saved_num_steps"] = policy.config.num_steps
    policy.config.num_steps = policy.model.config.num_steps = 1
    report["executed_num_steps"] = 1
    write_json(args.output / "policy_load.json", report)
    import gymnasium as gym
    from libero.libero import benchmark, get_assets_path

    from lerobot.envs.libero import LiberoEnv

    if Path(get_assets_path()).resolve().name != reference.ASSETS_REVISION:
        raise RuntimeError("Assets are not the fixed reference snapshot")
    suite = benchmark.get_benchmark_dict()["libero_object"](task_order_index=0)
    if tuple(task.name for task in suite.tasks) != reference.TASK_NAMES:
        raise RuntimeError("Native task suite order changed")

    def env_factory(spec):
        env = LiberoEnv(
            suite,
            spec["task_id"],
            "libero_object",
            obs_type="pixels_agent_pos",
            episode_index=spec["initial_state_id"],
            hard_reset=True,
            num_steps_wait=10,
            control_freq=20,
            control_mode="relative",
            observation_width=256,
            observation_height=256,
        )
        return gym.wrappers.TimeLimit(env, max_episode_steps=280)

    try:
        for spec in registered_tuples():
            print(f"START {spec['tuple_id']}", flush=True)
            result = reference.run_episode(
                spec,
                reference.tuple_directory(args.output, spec),
                policy,
                pre,
                post,
                env_factory,
                denoising_steps=1,
            )
            print(
                f"END {spec['tuple_id']} {result['status']} success={result['success']} "
                f"actions={result['measured_actions']} wall={result['wall_seconds']:.3f}",
                flush=True,
            )
            if result["status"] != "completed":
                return 2
        return 0
    finally:
        policy.config.num_steps = policy.model.config.num_steps = 10


def supervise(args):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference.REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=reference.REPO, text=True).strip()
    if head != args.execution_head or dirty:
        raise RuntimeError("Run only at the preregistered clean execution HEAD")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("The fixed offline EGL launch contract is required")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "registration.json",
        {
            "source_head": head,
            "started_at": datetime.now(UTC).isoformat(),
            "supervisor_pid": os.getpid(),
            "protocol": "LIBERO_SINGLE_STEP_NATIVE_PLAN.md",
            "policy_path": str(args.policy_path),
            "vlm_path": str(args.vlm_path),
            "policy_revision": reference.POLICY_REVISION,
            "vlm_revision": reference.VLM_REVISION,
            "assets_revision": reference.ASSETS_REVISION,
            "versions": {
                name: version(name)
                for name in ("lerobot", "hf-libero", "mujoco", "robosuite", "torch", "numpy", "transformers")
            },
            "control": {"frequency": 20, "chunk": 50, "consume": 1, "denoising": 1, "limit": 280},
            "tuples": registered_tuples(),
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
    launch_error = None
    exit_code = None
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
    records = [
        reference.audit_tuple(reference.tuple_directory(args.output, spec), spec, denoising_steps=1)
        for spec in registered_tuples()
    ]
    write_json(args.output / "tuple_accounting.json", records)
    result = summarize(records, exit_code, head)
    result.update(process_result)
    samples = []
    for spec in registered_tuples():
        path = reference.tuple_directory(args.output, spec) / "events.jsonl"
        samples.extend(
            event["inference_seconds"]
            for event in reference.read_events(path)
            if event["event"] == "action_prepared"
        )
    result["policy_selection_seconds"] = timing_summary(samples) if samples else None
    result["policy_selection_over_50ms"] = sum(value > 0.05 for value in samples)
    result["policy_selection_over_100ms"] = sum(value > 0.1 for value in samples)
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["native_screen_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return run_worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
