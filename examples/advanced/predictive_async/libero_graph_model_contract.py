"""Stage A: exactly ten recorded initial observations, two consecutive requests each."""

import time
import traceback

import libero_reference_qualification as reference
import numpy as np
import torch
from libero_reference_smoke import load_runtime, write_json
from libero_single_step_matched_control import initial_observation
from profile_libero_graph_recorded import recorded_batch
from smolvla_graph_runtime import SmolVLAGraphRuntime

from lerobot.utils.constants import ACTION


def compare_arrays(left, right):
    return {
        key: {
            "exact_equal": bool(
                left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key])
            ),
            "max_abs_difference": float(np.max(np.abs(left[key] - right[key]))),
        }
        for key in ("noise", "full_chunk", "normalized_action", "postprocessed_action")
    }


def run(args):
    result = {
        "execution_head": args.execution_head,
        "passed": False,
        "new_native_episodes": 0,
        "pairs": [],
        "task_boundary": [],
        "stage": "load",
    }
    original = policy = None
    runtime = None
    try:
        torch.set_num_threads(1)
        start = time.perf_counter()
        policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
        result["model_load_seconds"] = time.perf_counter() - start
        result["policy_load"] = report
        original = policy.model.sample_actions
        batches = []
        for task, name in enumerate(reference.TASK_NAMES):
            directory = args.source / f"tuple_{task * 9:03d}"
            batches.append(recorded_batch(directory, initial_observation(directory), name))

        def select(task):
            processed = pre(batches[task])
            normalized = policy.select_action(processed)
            action = post(normalized).detach().cpu().numpy().copy()
            if normalized.shape != (1, 7) or len(policy._queues[ACTION]) != 0:
                raise ValueError("The unchanged selector must consume exactly one of fifty actions")
            if not np.isfinite(action).all() or not torch.isfinite(normalized).all():
                raise ValueError("Nonfinite selected action")
            return runtime.snapshot(normalized, action)

        with torch.inference_mode(), SmolVLAGraphRuntime(policy.model) as runtime:
            for mode in ("eager", "graph"):
                result["stage"] = mode + "_continuous_sequence"
                reference.seed_policy(1009001)
                for task in range(10):
                    policy.reset()
                    pre.reset()
                    post.reset()
                    runtime.begin_episode(mode, task)
                    held = held_copy = None
                    for request in range(2):
                        arrays = select(task)
                        file = args.output / f"{mode}_task_{task:02d}_request_{request}.npz"
                        np.savez_compressed(file, **arrays)
                        if request == 0:
                            held, held_copy = runtime.latest, runtime.latest.clone()
                        elif not torch.equal(held, held_copy):
                            raise ValueError("A previous returned chunk changed on the next request")
                        if mode == "graph":
                            with np.load(
                                args.output / f"eager_task_{task:02d}_request_{request}.npz"
                            ) as eager:
                                comparison = compare_arrays(eager, arrays)
                            pair = {
                                "task_id": task,
                                "request": request,
                                "comparison": comparison,
                                "output_hold_unchanged": True,
                                **runtime.metadata,
                            }
                            result["pairs"].append(pair)
                            write_json(args.output / "progress.json", result)
                            if not all(v["exact_equal"] for v in comparison.values()):
                                result["first_failure"] = pair
                                raise ValueError("First unequal continuous-sequence pair")
                        print(f"MODEL {mode} task={task} request={request}", flush=True)
            # A separate contract on the same inputs; no native episode and no extra sample selection.
            result["stage"] = "task_boundary_0_1_0"
            for mode in ("eager", "graph"):
                reference.seed_policy(1009001)
                for ordinal, task in enumerate((0, 1, 0)):
                    policy.reset()
                    pre.reset()
                    post.reset()
                    runtime.begin_episode(mode, task)
                    arrays = select(task)
                    np.savez_compressed(args.output / f"boundary_{mode}_{ordinal}.npz", **arrays)
                    if mode == "graph":
                        with np.load(args.output / f"boundary_eager_{ordinal}.npz") as eager:
                            comparison = compare_arrays(eager, arrays)
                        row = {
                            "task_id": task,
                            "capture_id": runtime.metadata["capture_id"],
                            "comparison": comparison,
                        }
                        result["task_boundary"].append(row)
                        if not all(v["exact_equal"] for v in comparison.values()):
                            result["first_failure"] = row
                            raise ValueError("A-B-A task boundary mismatch")
            ids = [r["capture_id"] for r in result["task_boundary"]]
            if len(set(ids)) != 3:
                raise ValueError("Returning to task A reused an obsolete capture")
            result["captures"] = list(runtime.captures)
            result["control_requests"] = runtime.control_requests
            result["noise_draws"] = runtime.noise_draws
            if runtime.noise_draws != 46:
                raise ValueError("Unexpected control-noise sampling count")
        result["passed"] = True
        result["stage"] = "completed"
    except Exception:
        result["exception"] = traceback.format_exc()
    finally:
        result["original_sampler_restored"] = policy is not None and policy.model.sample_actions == original
        result["graph_released"] = runtime is not None and runtime.graph is None
        result["passed"] = (
            result["passed"] and result["original_sampler_restored"] and result["graph_released"]
        )
        write_json(args.output / "result.json", result)
    return 0 if result["passed"] else 2
