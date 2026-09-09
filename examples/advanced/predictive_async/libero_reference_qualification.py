"""Execute the registered LIBERO reference cohorts once, with native crash accounting."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from libero_reference_smoke import (
    ASSETS_REVISION,
    POLICY_REVISION,
    SETTLING_ACTIONS,
    VLM_REVISION,
    action_outside_bounds,
    array_record,
    load_runtime,
    policy_observation,
    terminal_reason,
    write_json,
)

from lerobot.utils.constants import ACTION, OBS_STATE

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "docs/experiments/LIBERO_REFERENCE_QUALIFICATION_TUPLES.json"
LIMIT = 280
FREQUENCY = 20
TASK_NAMES = tuple(
    f"pick_up_the_{item}_and_place_it_in_the_basket"
    for item in (
        "alphabet_soup",
        "cream_cheese",
        "salad_dressing",
        "bbq_sauce",
        "ketchup",
        "tomato_sauce",
        "butter",
        "milk",
        "chocolate_pudding",
        "orange_juice",
    )
)


def registered_manifest():
    cohorts = {}
    for phase, first, env_base, policy_base in (
        ("development", 1, 510000, 610000),
        ("confirmation", 21, 710000, 810000),
    ):
        cohorts[phase] = [
            {
                "ordinal": t * 20 + s - first,
                "tuple_id": f"{phase}/task_{t:02d}/state_{s:02d}",
                "task_id": t,
                "task_name": name,
                "initial_state_id": s,
                "environment_seed": env_base + 100 * t + s,
                "policy_seed": policy_base + 100 * t + s,
            }
            for t, name in enumerate(TASK_NAMES)
            for s in range(first, first + 20)
        ]
    return {
        "suite": "libero_object",
        "task_order_index": 0,
        "policy_revision": POLICY_REVISION,
        "vlm_revision": VLM_REVISION,
        "assets_revision": ASSETS_REVISION,
        "cohorts": cohorts,
    }


def read_manifest(path):
    actual = json.loads(path.read_text())
    expected = registered_manifest()
    if actual != expected:
        raise ValueError("Manifest differs from the registered tasks, order, states, seeds or identities")
    return actual


def tuple_directory(phase_directory, spec):
    return phase_directory / f"tuple_{spec['ordinal']:03d}"


class Journal:
    def __init__(self, path):
        self.file = path.open("x")

    def emit(self, event, **values):
        self.file.write(json.dumps({"event": event, **values}, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def seed_policy(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_for_tuple(env, policy, pre, post, spec, journal):
    policy.reset()
    pre.reset()
    post.reset()
    journal.emit(
        "reset_started", environment_seed=spec["environment_seed"], initial_state_id=spec["initial_state_id"]
    )
    raw, info = env.reset(seed=spec["environment_seed"])
    if len(policy._queues[ACTION]) != 0:
        raise RuntimeError("Action queue is not empty after episode reset")
    journal.emit(
        "reset_returned", next_initial_state_id=env.unwrapped.init_state_id, queue_length=0, info=info
    )
    seed_policy(spec["policy_seed"])
    journal.emit("policy_seed_set", policy_seed=spec["policy_seed"])
    return raw


def checked_observation(raw, language):
    robot = raw["robot_state"]
    values = np.concatenate((robot["eef"]["pos"], robot["eef"]["quat"], robot["gripper"]["qpos"]))
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite native state: {array_record(values)}")
    batch = policy_observation(raw, language)
    if batch[OBS_STATE].shape != (1, 8) or not torch.isfinite(batch[OBS_STATE]).all():
        raise ValueError(f"Invalid policy state: {array_record(batch[OBS_STATE])}")
    return batch


def save_observation(output, index, raw, language, journal, timing=None):
    """Keep both actual raw cameras losslessly, including the terminal observation."""
    start = time.perf_counter()
    batch = checked_observation(raw, language)
    converted = time.perf_counter()
    file = output / "observations" / f"{index:03d}.npz"
    np.savez_compressed(file, image=raw["pixels"]["image"], image2=raw["pixels"]["image2"])
    journal.emit(
        "observation",
        index=index,
        cameras=str(file.relative_to(output)),
        state=array_record(batch[OBS_STATE]),
        eef_quaternion_xyzw=array_record(raw["robot_state"]["eef"]["quat"]),
        **(
            {
                "raw_eef_position": array_record(raw["robot_state"]["eef"]["pos"]),
                "raw_gripper_qpos": array_record(raw["robot_state"]["gripper"]["qpos"]),
            }
            if timing is not None
            else {}
        ),
    )
    if timing is not None:
        timing.update(
            observation_conversion_seconds=converted - start,
            observation_logging_seconds=time.perf_counter() - converted,
        )
    return batch


def run_episode(spec, output, policy, pre, post, env_factory, *, denoising_steps=10, runtime=None):
    """Write a final result only after cleanup; a native exit leaves a started marker."""
    output.mkdir()
    (output / "observations").mkdir()
    start = time.perf_counter()
    write_json(output / "started.json", {"tuple": spec, "started_at": datetime.now(UTC).isoformat()})
    result = {
        "tuple": spec,
        "status": "technical_failure",
        "native_success_observed": None,
        "success": False,
        "terminated": None,
        "truncated": None,
        "terminal_reason": None,
        "measured_actions": 0,
        "settling_actions": 0,
        "environment_closed": None,
        "first_success_action": None,
    }
    journal = Journal(output / "events.jsonl")
    env = None
    hook = None
    native_counts = {"settling": 0, "measurement": 0}
    try:
        journal.emit("environment_construction_started")
        env = env_factory(spec)
        env.unwrapped._ensure_env()
        native_step = env.unwrapped._env.step
        segment = "settling"

        def logged_native_step(action):
            number = native_counts[segment] + 1
            journal.emit("native_step_started", segment=segment, number=number, action=array_record(action))
            transition = native_step(action)
            native_counts[segment] = number
            journal.emit(
                "native_step_returned", segment=segment, number=number, native_done=bool(transition[2])
            )
            return transition

        env.unwrapped._env.step = logged_native_step
        raw = reset_for_tuple(env, policy, pre, post, spec, journal)
        observation_returned_at = time.perf_counter()
        result["settling_actions"] = native_counts["settling"]
        if native_counts["settling"] != SETTLING_ACTIONS:
            raise RuntimeError("Native reset did not execute exactly ten settling actions")
        if env.unwrapped.init_state_id != spec["initial_state_id"] + 1:
            raise RuntimeError("Native initial-state row selection differs from the tuple")
        language = spec["task_name"].replace("_", " ")
        segment = "measurement"
        observation_timing = {} if runtime is not None else None
        batch = save_observation(output, 0, raw, language, journal, observation_timing)
        shapes = []
        if runtime is None or runtime.mode == "eager":
            hook = policy.model.action_out_proj.register_forward_hook(
                lambda _module, _args, value: shapes.append(list(value.shape))
            )
        with torch.inference_mode():
            for number in range(1, LIMIT + 1):
                processing_start = time.perf_counter()
                processed = pre(batch)
                if processed[OBS_STATE].shape != (1, 8) or not torch.isfinite(processed[OBS_STATE]).all():
                    raise ValueError(f"Invalid normalized state: {array_record(processed[OBS_STATE])}")
                shapes.clear()
                inference_start = time.perf_counter()
                normalized = policy.select_action(processed)
                action = post(normalized).detach().cpu().numpy()[0]
                action_ready_at = time.perf_counter()
                journal.emit(
                    "action_prepared",
                    number=number,
                    observation_index=number - 1,
                    normalized_state=array_record(processed[OBS_STATE]),
                    normalized_action=array_record(normalized),
                    action=array_record(action),
                    projection_shapes=shapes.copy(),
                    queue_length=len(policy._queues[ACTION]),
                    inference_seconds=action_ready_at - inference_start,
                )
                if runtime is not None:
                    from smolvla_graph_runtime import valid_sampler_evidence

                    if not valid_sampler_evidence(runtime.mode, shapes, runtime.metadata, runtime.captures):
                        raise RuntimeError("Sampler capture/replay evidence differs from the contract")
                    runtime.record_request(output, number, normalized, action, journal)
                    journal.emit(
                        "request_timing",
                        number=number,
                        preprocessing_seconds=inference_start - processing_start,
                        selector_to_cpu_action_seconds=action_ready_at - inference_start,
                        engineering_seconds=action_ready_at - observation_returned_at,
                        processing_seconds=(
                            action_ready_at
                            - processing_start
                            + observation_timing["observation_conversion_seconds"]
                        ),
                        raw_to_action_wall_seconds=action_ready_at - observation_returned_at,
                        logging_seconds=(
                            time.perf_counter()
                            - action_ready_at
                            + observation_timing["observation_logging_seconds"]
                        ),
                        **observation_timing,
                    )
                outside = action_outside_bounds(action)
                if not torch.isfinite(normalized).all():
                    raise ValueError("Non-finite normalized action")
                if (runtime is None and shapes != [[1, 50, 32]] * denoising_steps) or len(
                    policy._queues[ACTION]
                ) != 0:
                    raise RuntimeError(
                        f"The production selector did not preserve 50/1/{denoising_steps} consumption"
                    )
                environment_start = time.perf_counter()
                raw, reward, terminated, truncated, info = env.step(action)
                observation_returned_at = time.perf_counter()
                if runtime is not None:
                    journal.emit(
                        "environment_timing",
                        number=number,
                        seconds=observation_returned_at - environment_start,
                    )
                result.update(
                    measured_actions=number,
                    native_success_observed=bool(info["is_success"]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    terminal_reason=terminal_reason(terminated, truncated, info),
                )
                journal.emit(
                    "measurement_returned",
                    number=number,
                    reward=float(reward),
                    native_success=bool(info["is_success"]),
                    native_done=bool(info["done"]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    outside_box_components=outside,
                )
                batch = save_observation(output, number, raw, language, journal, observation_timing)
                if number % 40 == 0:
                    print(
                        f"STEP {spec['tuple_id']} {number}/{LIMIT} success={info['is_success']}", flush=True
                    )
                if result["native_success_observed"]:
                    result["first_success_action"] = number
                if result["terminal_reason"] is not None:
                    break
            if result["terminal_reason"] is None:
                raise RuntimeError("The registered TimeLimit did not terminate the episode")
        result["status"] = "completed"
    except Exception:
        result["exception"] = traceback.format_exc()
        journal.emit("python_exception", traceback=result["exception"])
    finally:
        result["settling_actions"] = native_counts["settling"]
        if runtime is not None:
            result["sampler_mode"] = runtime.mode
            result["captures"] = list(runtime.captures)
        if hook is not None:
            hook.remove()
        if env is not None:
            journal.emit("cleanup_started")
            try:
                env.close()
                result["environment_closed"] = env.unwrapped._env is None
                if not result["environment_closed"]:
                    raise RuntimeError("Native environment remains open")
            except Exception:
                result["environment_closed"] = False
                result["status"] = "technical_failure"
                result["close_exception"] = traceback.format_exc()
            journal.emit("cleanup_returned", environment_closed=result["environment_closed"])
        if runtime is not None:
            try:
                if result["status"] != "completed" and runtime.latest_noise is not None:
                    failure = {"noise": runtime.latest_noise.detach().cpu().numpy()}
                    if runtime.latest is not None:
                        failure["full_chunk"] = runtime.latest.detach().cpu().numpy()
                    np.savez_compressed(output / "first_failure_sampler.npz", **failure)
            except Exception:
                result["failure_snapshot_exception"] = traceback.format_exc()
            finally:
                runtime.reset_output()
        result["success"] = result["status"] == "completed" and result["native_success_observed"] is True
        result["wall_seconds"] = time.perf_counter() - start
        result["restricted_simulated_seconds"] = (
            result["first_success_action"] / FREQUENCY if result["success"] else LIMIT / FREQUENCY
        )
        journal.close()
        write_json(output / "result.json", result)
    return result


def make_native_env_factory():
    """Build the pinned full native suite without dispatching an episode."""
    import gymnasium as gym
    from libero.libero import benchmark, get_assets_path

    from lerobot.envs.libero import LiberoEnv

    if Path(get_assets_path()).resolve().name != ASSETS_REVISION:
        raise RuntimeError("Native assets differ from the pinned snapshot")
    suite = benchmark.get_benchmark_dict()["libero_object"](task_order_index=0)
    if tuple(task.name for task in suite.tasks) != TASK_NAMES:
        raise RuntimeError("Native complete task order differs from the manifest")

    def env_factory(spec):
        native = LiberoEnv(
            suite,
            spec["task_id"],
            "libero_object",
            obs_type="pixels_agent_pos",
            episode_index=spec["initial_state_id"],
            hard_reset=True,
            num_steps_wait=SETTLING_ACTIONS,
            control_freq=FREQUENCY,
            control_mode="relative",
            observation_width=256,
            observation_height=256,
        )
        return gym.wrappers.TimeLimit(native, max_episode_steps=LIMIT)

    return env_factory


def run_worker(args):
    """One policy load per cohort; every tuple owns and closes a fresh native environment."""
    phase_directory = args.output / args.phase
    manifest = read_manifest(args.manifest)
    torch.set_num_threads(1)
    policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
    write_json(phase_directory / "policy_load.json", report)
    env_factory = make_native_env_factory()

    for spec in manifest["cohorts"][args.phase]:
        print(f"START {spec['tuple_id']}", flush=True)
        result = run_episode(spec, tuple_directory(phase_directory, spec), policy, pre, post, env_factory)
        print(
            f"END {spec['tuple_id']} {result['status']} success={result['success']} "
            f"steps={result['measured_actions']} wall={result['wall_seconds']:.2f}s",
            flush=True,
        )
        if result["status"] != "completed":
            return 2
    return 0


def read_events(path):
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    return events


def audit_tuple(directory, spec, *, denoising_steps=10, sampler_mode="eager"):
    """Reconcile native calls, returned observations and the final cleanup record."""
    if not (directory / "started.json").exists():
        return {"tuple": spec, "status": "not_run", "success": False}
    errors = []
    try:
        events = read_events(directory / "events.jsonl")
    except (ValueError, OSError) as exc:
        events = []
        errors.append(f"Unreadable step journal: {exc}")
    prepared = [e for e in events if e["event"] == "action_prepared"]
    measured = [e for e in events if e["event"] == "measurement_returned"]
    observations = [e for e in events if e["event"] == "observation"]
    starts = [e for e in events if e["event"] == "native_step_started"]
    returns = [e for e in events if e["event"] == "native_step_returned"]
    settling = [e for e in returns if e["segment"] == "settling"]
    physical = [e for e in returns if e["segment"] == "measurement"]
    pending = [
        (e["segment"], e["number"])
        for e in starts
        if not any(r["segment"] == e["segment"] and r["number"] == e["number"] for r in returns)
    ]
    final_path = directory / "result.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
    else:
        result = {
            "tuple": spec,
            "status": "technical_failure",
            "success": False,
            "exception": "Worker exited after tuple start without a final cleanup record",
            "environment_closed": None,
            "native_success_observed": None,
            "terminated": None,
            "truncated": None,
            "terminal_reason": "worker_exit_without_final_record",
            "measured_actions": len(measured),
            "settling_actions": len(settling),
            "first_success_action": None,
            "wall_seconds": None,
            "restricted_simulated_seconds": 14.0,
        }
    if result["tuple"] != spec:
        errors.append("Final tuple identity differs from manifest")
    if result["status"] == "completed":
        n = len(measured)
        if not (1 <= n <= LIMIT and len(prepared) == len(physical) == n == result["measured_actions"]):
            errors.append("Measurement/action/native return counts disagree")
        if len(settling) != SETTLING_ACTIONS or result["settling_actions"] != SETTLING_ACTIONS or pending:
            errors.append("Settling count or outstanding native call differs from contract")
        if [e["number"] for e in measured] != list(range(1, n + 1)):
            errors.append("Measurement order differs from contract")
        if [e["index"] for e in observations] != list(range(n + 1)):
            errors.append("Observation indices do not cover reset through terminal return")
        if not all((directory / e["cameras"]).is_file() for e in observations):
            errors.append("Recorded observation file is missing")
        if result["environment_closed"] is not True:
            errors.append("Required cleanup not confirmed")
        if measured:
            last = measured[-1]
            if result["native_success_observed"] != last["native_success"]:
                errors.append("Native success disagrees with final measurement")
            if result["terminated"] != last["terminated"] or result["truncated"] != last["truncated"]:
                errors.append("Terminal flags disagree with final measurement")
            if not (last["native_success"] or last["terminated"] or last["truncated"]):
                errors.append("No native terminal or TimeLimit return")
            if any(e["native_success"] or e["terminated"] or e["truncated"] for e in measured[:-1]):
                errors.append("Actions continued after an episode terminal")
        actual_actions = [e for e in starts if e["segment"] == "measurement"]
        sampler_requests = {e["number"]: e for e in events if e["event"] == "sampler_request"}
        for e, actual in zip(prepared, actual_actions, strict=False):
            if sampler_mode == "graph":
                from smolvla_graph_runtime import valid_sampler_evidence

                projection_valid = valid_sampler_evidence(
                    sampler_mode,
                    e["projection_shapes"],
                    sampler_requests.get(e["number"]),
                    result.get("captures", []),
                )
            else:
                projection_valid = e["projection_shapes"] == [[1, 50, 32]] * denoising_steps
            if (
                e["action"] != actual["action"]
                or not projection_valid
                or e["queue_length"] != 0
                or e["observation_index"] != e["number"] - 1
            ):
                errors.append(f"Saved prediction, native action or 50/1/{denoising_steps} record disagrees")
                break
    if errors:
        result["status"] = "technical_failure"
        result["success"] = False
        result["restricted_simulated_seconds"] = 14.0
    result.update(
        audit_errors=errors,
        physical_step_pending="unknown" if pending else False,
        pending_native_calls=pending,
        native_measurement_returns=len(physical),
        native_settling_returns=len(settling),
        caller_measurement_records=len(measured),
        prepared_actions=len(prepared),
        out_of_box_actions=sum(bool(e["outside_box_components"]) for e in measured),
        out_of_box_components=sum(len(e["outside_box_components"]) for e in measured),
        inference_seconds=sum(e["inference_seconds"] for e in prepared),
    )
    return result


def wilson(successes, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def passes_gate(summary):
    return (
        summary["scheduled"] == summary["completed"] == 200
        and summary["technical_failure"] == summary["not_run"] == 0
        and summary["successes"] >= 180
        and len(summary["tasks"]) == 10
        and all(
            task["scheduled"] == task["completed"] == 20 and task["successes"] >= 16
            for task in summary["tasks"]
        )
        and summary["worker_exit_code"] == 0
        and summary["records_verified"]
    )


def summarize(records, phase, exit_code, source_head):
    tasks = []
    for t in range(10):
        rows = [r for r in records if r["tuple"]["task_id"] == t]
        observed = [r for r in rows if r["status"] != "not_run"]
        successes = sum(r["success"] for r in rows)
        tasks.append(
            {
                "task_id": t,
                "task_name": TASK_NAMES[t],
                "scheduled": 20,
                "completed": sum(r["status"] == "completed" for r in rows),
                "technical_failure": sum(r["status"] == "technical_failure" for r in rows),
                "not_run": sum(r["status"] == "not_run" for r in rows),
                "observed": len(observed),
                "successes": successes,
                "decision_success_rate": successes / 20,
                "observed_wilson_95": wilson(successes, len(observed)),
                "wilson_denominator": len(observed),
                "restricted_simulated_mean_seconds_observed": (
                    float(np.mean([r["restricted_simulated_seconds"] for r in observed]))
                    if observed
                    else None
                ),
                "tuple_wall_seconds_known": sum(r.get("wall_seconds") or 0 for r in observed),
                "tuple_wall_seconds_unknown": sum(r.get("wall_seconds") is None for r in observed),
            }
        )
    observed = [r for r in records if r["status"] != "not_run"]
    summary = {
        "phase": phase,
        "source_head": source_head,
        "scheduled": 200,
        "completed": sum(t["completed"] for t in tasks),
        "technical_failure": sum(t["technical_failure"] for t in tasks),
        "not_run": sum(t["not_run"] for t in tasks),
        "observed": len(observed),
        "successes": sum(t["successes"] for t in tasks),
        "timeouts": sum(
            r["status"] == "completed" and not r["success"] and bool(r["truncated"]) for r in observed
        ),
        "native_truncations_observed": sum(bool(r.get("truncated")) for r in observed),
        "environment_closed": sum(r.get("environment_closed") is True for r in observed),
        "environment_close_failed": sum(r.get("environment_closed") is False for r in observed),
        "environment_close_unknown": sum(r.get("environment_closed") is None for r in observed),
        "measured_actions_returned": sum(r.get("native_measurement_returns", 0) for r in observed),
        "caller_measurement_records": sum(r.get("caller_measurement_records", 0) for r in observed),
        "settling_actions_returned": sum(r.get("native_settling_returns", 0) for r in observed),
        "physical_step_unknown_tuples": sum(r.get("physical_step_pending") == "unknown" for r in observed),
        "out_of_box_actions": sum(r.get("out_of_box_actions", 0) for r in observed),
        "out_of_box_components": sum(r.get("out_of_box_components", 0) for r in observed),
        "inference_seconds": sum(r.get("inference_seconds", 0) for r in observed),
        "restricted_simulated_mean_seconds_observed": (
            float(np.mean([r["restricted_simulated_seconds"] for r in observed])) if observed else None
        ),
        "worker_exit_code": exit_code,
        "records_verified": not any(r.get("audit_errors") for r in observed),
        "tasks": tasks,
        "bootstrap_macro_95": None,
        "bootstrap_seed": 910001 if phase == "development" else 910002,
        "bootstrap_repetitions": 20000,
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
    }
    summary["decision_success_rate"] = summary["successes"] / 200
    if all(t["observed"] == 20 for t in tasks):
        rng = np.random.default_rng(summary["bootstrap_seed"])
        draws = np.zeros(20000)
        for t in range(10):
            values = np.array([int(r["success"]) for r in records if r["tuple"]["task_id"] == t])
            draws += values[rng.integers(0, 20, size=(20000, 20))].mean(axis=1) / 10
        summary["bootstrap_macro_95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    else:
        summary["bootstrap_not_computed_reason"] = "Incomplete observations; not_run slots are not samples"
    summary["passed"] = passes_gate(summary)
    return summary


def require_development(output, source_head):
    """Refuse before creating any confirmation directory or worker process."""
    summary = json.loads((output / "development/summary.json").read_text())
    if summary["source_head"] != source_head or not passes_gate(summary):
        raise RuntimeError("Development did not close and pass under this execution HEAD")
    return summary


def supervise(args):
    manifest = read_manifest(args.manifest)
    if args.phase == "confirmation":
        require_development(args.output, args.execution_head)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if (
        head != args.execution_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    ):
        raise RuntimeError("Execution requires the registered clean source HEAD")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("Use the registered offline EGL launch environment")
    output = args.output / args.phase
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "source_head": head,
            "phase": args.phase,
            "manifest": str(args.manifest),
            "policy_path": str(args.policy_path),
            "vlm_path": str(args.vlm_path),
            "started_at": datetime.now(UTC).isoformat(),
            "supervisor_pid": os.getpid(),
            "versions": {
                name: version(name)
                for name in (
                    "lerobot",
                    "hf-libero",
                    "mujoco",
                    "robosuite",
                    "numpy",
                    "torch",
                    "transformers",
                )
            },
        },
    )
    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        args.phase,
        "--output",
        str(args.output),
        "--manifest",
        str(args.manifest),
        "--policy-path",
        str(args.policy_path),
        "--vlm-path",
        str(args.vlm_path),
        "--execution-head",
        args.execution_head,
    ]
    start = time.perf_counter()
    with (output / "worker.log").open("x") as log:
        try:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
                write_json(output / "worker_process.json", {"pid": process.pid, "command": command})
                exit_code = process.wait()
            launch_error = None
        except OSError as exc:
            exit_code = None
            launch_error = str(exc)
    wall = time.perf_counter() - start
    write_json(
        output / "process_exit.json",
        {
            "worker_exit_code": exit_code,
            "launch_error": launch_error,
            "worker_wall_seconds": wall,
            "ended_at": datetime.now(UTC).isoformat(),
        },
    )
    records = [audit_tuple(tuple_directory(output, s), s) for s in manifest["cohorts"][args.phase]]
    write_json(output / "tuple_accounting.json", records)
    summary = summarize(records, args.phase, exit_code, head)
    summary["worker_wall_seconds"] = wall
    summary["launch_error"] = launch_error
    if not summary["observed"] and exit_code != 0:
        summary["campaign_blocker"] = (
            "Worker setup/launch failed before any tuple started; inspect worker.log"
        )
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return run_worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
