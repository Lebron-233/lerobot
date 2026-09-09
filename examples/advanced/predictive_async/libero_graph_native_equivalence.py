"""Fixed stage A / B supervisor; no task filters, retries, replacements or native stage C."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import libero_reference_qualification as reference
import numpy as np
import torch
from libero_graph_model_contract import compare_arrays
from libero_reference_smoke import load_runtime, write_json
from smolvla_graph_runtime import SmolVLAGraphRuntime

REPO = reference.REPO
MANIFEST = REPO / "docs/experiments/LIBERO_GRAPH_NATIVE_EQUIVALENCE_MANIFEST.json"
CACHE = REPO.parent / "libero-reference-cache"
SOURCE = REPO / "outputs/libero_single_step_native_ee273bce"
POLICY = CACHE / f"hub/models--HuggingFaceVLA--smolvla_libero/snapshots/{reference.POLICY_REVISION}"
VLM = CACHE / f"hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/{reference.VLM_REVISION}"
FLAGS = {
    "baseline_qualified": False,
    "realtime_qualified": False,
    "predictor_benefit_tested": False,
    "risk_thresholds": None,
    "old_confirmation": "not_started_untouched",
}


def fixed_manifest():
    rows = []
    for task, name in enumerate(reference.TASK_NAMES):
        for state in (41, 49):
            pair = 2 * task + int(state == 49)
            modes = ("eager", "graph") if pair % 2 == 0 else ("graph", "eager")
            for mode in modes:
                rows.append(
                    {
                        "ordinal": len(rows),
                        "pair_index": pair,
                        "sampler_mode": mode,
                        "tuple_id": f"graph_native/pair_{pair:02d}/{mode}",
                        "task_id": task,
                        "task_name": name,
                        "initial_state_id": state,
                        "environment_seed": 940000 + 100 * task + state,
                        "policy_seed": 950000 + 100 * task + state,
                    }
                )
    return {
        "suite": "libero_object",
        "task_order_index": 0,
        "chunk_size": 50,
        "n_action_steps": 1,
        "num_steps": 10,
        "maximum_measured_actions": 11200,
        "maximum_settling_actions": 400,
        "worker_limit_seconds": 7200,
        "tuples": rows,
    }


def read_manifest():
    actual = json.loads(MANIFEST.read_text())
    if actual != fixed_manifest():
        raise ValueError("The committed forty-episode manifest changed")
    return actual


def timing_summary(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean_seconds": float(np.mean(values)),
        "p50_seconds": values[math.ceil(0.50 * len(values)) - 1],
        "p95_seconds": values[math.ceil(0.95 * len(values)) - 1],
        "p99_seconds": values[math.ceil(0.99 * len(values)) - 1],
        "max_seconds": values[-1],
        "over_50ms": sum(v > 0.05 for v in values),
    }


def observation_difference(left_dir, left, right_dir, right):
    for key in ("state", "eef_quaternion_xyzw", "raw_eef_position", "raw_gripper_qpos"):
        if left[key] != right[key]:
            return key
    with (
        np.load(left_dir / left["cameras"], allow_pickle=False) as a,
        np.load(right_dir / right["cameras"], allow_pickle=False) as b,
    ):
        for key in ("image", "image2"):
            if not np.array_equal(a[key], b[key]):
                return key
    return None


def compare_pair(output, specs):
    by_mode = {s["sampler_mode"]: s for s in specs}
    directories = {m: reference.tuple_directory(output, s) for m, s in by_mode.items()}
    events = {m: reference.read_events(p / "events.jsonl") for m, p in directories.items()}
    groups = {
        m: {
            kind: [e for e in es if e["event"] == kind]
            for kind in (
                "observation",
                "sampler_request",
                "measurement_returned",
                "action_prepared",
                "request_timing",
            )
        }
        for m, es in events.items()
    }
    result = {"pair_index": specs[0]["pair_index"], "exact_equal": False, "compared_requests": 0}
    result["timings"] = {
        m: {
            k: timing_summary([r[k] for r in g["request_timing"]])
            for k in (
                "engineering_seconds",
                "selector_to_cpu_action_seconds",
                "preprocessing_seconds",
                "logging_seconds",
                "raw_to_action_wall_seconds",
            )
        }
        for m, g in groups.items()
    }
    result["first_request_seconds"] = {
        m: g["request_timing"][0]["engineering_seconds"] for m, g in groups.items()
    }
    result["steady_request_timings"] = {
        m: timing_summary([r["engineering_seconds"] for r in g["request_timing"]][1:])
        for m, g in groups.items()
    }
    a, b = groups["eager"], groups["graph"]

    def failure(kind, index, detail):
        result["first_divergence"] = {"kind": kind, "observation_index": index, "detail": detail}
        return result

    for index in range(min(len(a["observation"]), len(b["observation"]))):
        difference = observation_difference(
            directories["eager"], a["observation"][index], directories["graph"], b["observation"][index]
        )
        if difference:
            return failure(
                "environment_initial_reproducibility_difference"
                if index == 0
                else "environment_reproducibility_difference_after_equal_actions",
                index,
                difference,
            )
        if index >= min(len(a["sampler_request"]), len(b["sampler_request"])):
            break
        with (
            np.load(directories["eager"] / a["sampler_request"][index]["arrays"]) as left,
            np.load(directories["graph"] / b["sampler_request"][index]["arrays"]) as right,
        ):
            comparison = compare_arrays(left, right)
        for key, value in comparison.items():
            if not value["exact_equal"]:
                return failure(
                    "noise_sequence_difference" if key == "noise" else "runtime_equivalence_failure",
                    index,
                    {key: value},
                )
        if a["action_prepared"][index]["normalized_state"] != b["action_prepared"][index]["normalized_state"]:
            return failure("normalized_input_difference", index, "normalized_state")
        if a["measurement_returned"][index] != b["measurement_returned"][index]:
            return failure(
                "native_outcome_difference_after_equal_actions",
                index,
                {"eager": a["measurement_returned"][index], "graph": b["measurement_returned"][index]},
            )
        result["compared_requests"] += 1
    if len(a["sampler_request"]) != len(b["sampler_request"]):
        return failure(
            "episode_length_difference",
            result["compared_requests"],
            {m: len(g["sampler_request"]) for m, g in groups.items()},
        )
    finals = {m: json.loads((p / "result.json").read_text()) for m, p in directories.items()}
    result["outcomes"] = {
        m: {
            k: r[k]
            for k in (
                "success",
                "first_success_action",
                "terminal_reason",
                "measured_actions",
                "terminated",
                "truncated",
                "wall_seconds",
            )
        }
        for m, r in finals.items()
    }
    for key in (
        "success",
        "native_success_observed",
        "first_success_action",
        "terminal_reason",
        "terminated",
        "truncated",
        "measured_actions",
        "settling_actions",
    ):
        if finals["eager"][key] != finals["graph"][key]:
            return failure("episode_outcome_difference", result["compared_requests"], key)
    result["exact_equal"] = True
    return result


def require_source(head):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True
    ).strip()
    if actual != head or dirty:
        raise RuntimeError("Tracked source must equal the exact published execution HEAD")


def require_model_pass(head):
    path = REPO / f"outputs/libero_graph_model_{head[:8]}/result.json"
    result = json.loads(path.read_text())
    supervisor = json.loads((path.parent / "supervisor.json").read_text())
    if not (result["passed"] and result["execution_head"] == head and supervisor["worker_exit_code"] == 0):
        raise RuntimeError("Stage A must pass and close at this execution HEAD before any native dispatch")


def native_worker(args):
    require_model_pass(args.execution_head)
    manifest = read_manifest()
    outcome = {
        "execution_head": args.execution_head,
        "pairs": [],
        "original_sampler_restored": False,
        "graph_released": False,
        "stage": "load",
    }
    policy = original = runtime = None
    code = 2
    try:
        torch.set_num_threads(1)
        start = time.perf_counter()
        policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
        outcome["model_load_seconds"] = time.perf_counter() - start
        write_json(args.output / "policy_load.json", report)
        original = policy.model.sample_actions
        env_factory = reference.make_native_env_factory()
        with torch.inference_mode(), SmolVLAGraphRuntime(policy.model) as runtime:
            for pair_index in range(20):
                specs = manifest["tuples"][pair_index * 2 : pair_index * 2 + 2]
                for spec in specs:
                    runtime.begin_episode(spec["sampler_mode"], spec["task_id"])
                    outcome["stage"] = spec["tuple_id"]
                    print(
                        f"START {spec['tuple_id']} task={spec['task_id']} state={spec['initial_state_id']}",
                        flush=True,
                    )
                    directory = reference.tuple_directory(args.output, spec)
                    reference.run_episode(
                        spec, directory, policy, pre, post, env_factory, denoising_steps=10, runtime=runtime
                    )
                    record = reference.audit_tuple(
                        directory, spec, denoising_steps=10, sampler_mode=spec["sampler_mode"]
                    )
                    write_json(directory / "audited.json", record)
                    print(
                        f"END {spec['tuple_id']} {record['status']} success={record['success']} "
                        f"actions={record['measured_actions']}",
                        flush=True,
                    )
                    if record["status"] != "completed":
                        outcome["first_failure"] = record
                        return 2
                pair = compare_pair(args.output, specs)
                outcome["pairs"].append(pair)
                write_json(args.output / f"pair_{pair_index:02d}.json", pair)
                print(
                    f"PAIR {pair_index} exact={pair['exact_equal']} requests={pair['compared_requests']}",
                    flush=True,
                )
                if not pair["exact_equal"]:
                    outcome["first_failure"] = pair["first_divergence"]
                    return 2
            outcome["stage"] = "completed"
            code = 0
    except Exception:
        outcome["exception"] = traceback.format_exc()
    finally:
        outcome["original_sampler_restored"] = policy is not None and policy.model.sample_actions == original
        outcome["graph_released"] = runtime is not None and runtime.graph is None
        if runtime is not None:
            outcome["captures"] = runtime.captures
            outcome["control_noise_draws"] = runtime.noise_draws
            outcome["control_requests"] = runtime.control_requests
        outcome["worker_return_code"] = code
        write_json(args.output / "worker_result.json", outcome)
    return code


def summarize(output, head, exit_code):
    rows = []
    for spec in read_manifest()["tuples"]:
        directory = reference.tuple_directory(output, spec)
        audited = directory / "audited.json"
        rows.append(
            json.loads(audited.read_text())
            if audited.exists()
            else reference.audit_tuple(directory, spec, denoising_steps=10, sampler_mode=spec["sampler_mode"])
        )
    worker_path = output / "worker_result.json"
    worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
    result = {
        "execution_head": head,
        "scheduled": 40,
        "worker_exit_code": exit_code,
        **FLAGS,
        "pairs": worker.get("pairs", []),
        "worker": worker,
        "completed": sum(r["status"] == "completed" for r in rows),
        "technical_failure": sum(r["status"] == "technical_failure" for r in rows),
        "not_run": sum(r["status"] == "not_run" for r in rows),
        "environment_closed": sum(r.get("environment_closed") is True for r in rows),
        "measurement_returns": sum(r.get("native_measurement_returns", 0) for r in rows),
        "settling_returns": sum(r.get("native_settling_returns", 0) for r in rows),
        "physical_step_unknown_tuples": sum(r.get("physical_step_pending") == "unknown" for r in rows),
    }
    all_times = {m: [] for m in ("eager", "graph")}
    result["episode_timings"] = []
    for row in rows:
        if row["status"] == "not_run":
            continue
        spec = row["tuple"]
        es = reference.read_events(reference.tuple_directory(output, spec) / "events.jsonl")
        times = [e for e in es if e["event"] == "request_timing"]
        all_times[spec["sampler_mode"]].extend(times)
        result["episode_timings"].append(
            {
                "tuple": spec,
                "wall_seconds": row.get("wall_seconds"),
                "engineering": timing_summary([e["engineering_seconds"] for e in times]),
                "environment_step": timing_summary(
                    [e["seconds"] for e in es if e["event"] == "environment_timing"]
                ),
            }
        )
    result["request_timings"] = {
        m: {
            k: timing_summary([e[k] for e in es])
            for k in (
                "engineering_seconds",
                "selector_to_cpu_action_seconds",
                "preprocessing_seconds",
                "logging_seconds",
                "raw_to_action_wall_seconds",
            )
        }
        for m, es in all_times.items()
    }
    result["episode_mean_timings"] = {
        m: timing_summary(
            [
                e["engineering"]["mean_seconds"]
                for e in result["episode_timings"]
                if e["tuple"]["sampler_mode"] == m and e["engineering"]["n"]
            ]
        )
        for m in all_times
    }
    result["paired_episode_mean_differences"] = [
        {
            "pair_index": p["pair_index"],
            "graph_minus_eager_seconds": (
                p["timings"]["graph"]["engineering_seconds"]["mean_seconds"]
                - p["timings"]["eager"]["engineering_seconds"]["mean_seconds"]
            ),
            "eager_over_graph_mean_ratio": (
                p["timings"]["eager"]["engineering_seconds"]["mean_seconds"]
                / p["timings"]["graph"]["engineering_seconds"]["mean_seconds"]
            ),
        }
        for p in result["pairs"]
    ]
    result["cold_preparation"] = {
        "model_load_seconds": worker.get("model_load_seconds"),
        "per_task_captures": worker.get("captures", []),
        "graph_preparation_total_seconds": sum(c["preparation_seconds"] for c in worker.get("captures", [])),
        "includes_preceding_eager_setup": True,
    }
    result["native_graph_equivalence_passed"] = (
        result["completed"] == result["environment_closed"] == 40
        and result["technical_failure"] == result["not_run"] == 0
        and exit_code == 0
        and len(result["pairs"]) == 20
        and all(p["exact_equal"] for p in result["pairs"])
        and worker.get("original_sampler_restored") is True
        and worker.get("graph_released") is True
    )
    write_json(output / "tuple_accounting.json", rows)
    write_json(output / "summary.json", result)
    return result


def supervise(args):
    require_source(args.execution_head)
    read_manifest()
    expected = (
        REPO
        / f"outputs/libero_graph_{'model' if args.phase == 'model' else 'native'}_{args.execution_head[:8]}"
    )
    if args.output.resolve() != expected:
        raise ValueError("Use the fixed exclusive output directory for this phase and execution HEAD")
    if args.phase == "native":
        require_model_pass(args.execution_head)
    if not (
        os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        and os.environ.get("MUJOCO_GL") == os.environ.get("PYOPENGL_PLATFORM") == "egl"
        and "PYTHONPATH" not in os.environ
    ):
        raise RuntimeError("Use the registered offline dedicated environment and EGL launch")
    args.output.mkdir(parents=True, exist_ok=False)
    registration = {
        "execution_head": args.execution_head,
        "phase": args.phase,
        "started_at": datetime.now(UTC).isoformat(),
        "manifest": read_manifest(),
        "argv": sys.argv,
        "python": sys.executable,
        "source": str(args.source),
        "versions": {p: version(p) for p in ("torch", "transformers", "mujoco", "robosuite")},
        "gpu": subprocess.check_output(["nvidia-smi"], text=True),
        **FLAGS,
    }
    write_json(args.output / "registration.json", registration)
    child_command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
    ]
    start = time.perf_counter()
    with (args.output / "worker.log").open("x") as log:
        process = subprocess.Popen(
            child_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        write_json(
            args.output / "worker_started.json",
            {
                "pid": process.pid,
                "process_group": process.pid,
                "command": child_command,
                "limit_seconds": 7200,
            },
        )
        timeout = False
        try:
            code = process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    write_json(
        args.output / "supervisor.json",
        {
            "worker_exit_code": code,
            "resource_timeout": timeout,
            "wall_seconds": time.perf_counter() - start,
            "worker_wait_returned": True,
        },
    )
    if args.phase == "native":
        result = summarize(args.output, args.execution_head, code)
        return 0 if result["native_graph_equivalence_passed"] else 2
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("model", "native"))
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.policy_path, args.vlm_path, args.source = POLICY, VLM, SOURCE
    if args.worker:
        require_source(args.execution_head)

        def terminate(_signum, _frame):
            raise RuntimeError("Supervisor resource deadline: stop dispatch and close owned resources")

        signal.signal(signal.SIGTERM, terminate)
        if args.phase == "model":
            from libero_graph_model_contract import run

            return run(args)
        return native_worker(args)
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
