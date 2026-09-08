"""A fixed 10-versus-1 denoising compute probe, not a task qualification."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

from profile_libero_reference_compute import timing_summary


def check_projection_shapes(shapes, steps):
    if shapes != [[1, 50, 32]] * steps:
        raise ValueError(f"Expected {steps} full-chunk denoising projections, got {shapes}")


def run(args, result):
    import numpy as np
    import torch
    from libero_reference_smoke import load_runtime, policy_observation

    from lerobot.utils.constants import ACTION

    torch.set_num_threads(1)
    policy, pre, post, load_report = load_runtime(args.policy_path, args.vlm_path)
    result["strict_load"] = load_report
    if policy.config is not policy.model.config:
        raise RuntimeError("Unexpected separate policy/model configurations")
    result["device_name"] = torch.cuda.get_device_name()
    grid = np.indices((256, 256), dtype=np.uint16)
    image = np.stack((grid[0], grid[1], (grid[0] + grid[1]) % 256), axis=-1).astype(np.uint8)
    raw = {
        "pixels": {"image": image, "image2": np.ascontiguousarray(255 - image)},
        "robot_state": {
            "eef": {"pos": np.zeros(3), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
            "gripper": {"qpos": np.zeros(2)},
        },
    }
    batch = policy_observation(raw, "pick up the alphabet soup and place it in the basket")
    projections = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _module, _args, output: projections.append(list(output.shape))
    )
    original_sample = policy.model.sample_actions

    def select(steps, seed, phases=None):
        policy.config.num_steps = steps
        policy.reset()
        pre.reset()
        post.reset()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        processed = pre(batch)
        projections.clear()
        if phases is not None:

            def profiled(*positional, **keywords):
                return original_sample(*positional, **keywords, timings=phases)

            policy.model.sample_actions = profiled
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            normalized = policy.select_action(processed)
            action = post(normalized).detach().cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            policy.model.sample_actions = original_sample
        check_projection_shapes(projections, steps)
        if normalized.shape != (1, 7) or action.shape != (1, 7) or len(policy._queues[ACTION]):
            raise ValueError("Changed selector shape or action consumption")
        if not torch.isfinite(normalized).all() or not torch.isfinite(action).all():
            raise ValueError("Nonfinite diagnostic action")
        return {
            "seconds": elapsed,
            "projection_count": len(projections),
            "normalized_action": normalized.detach().cpu().tolist(),
            "action": action.tolist(),
        }

    try:
        with torch.inference_mode():
            result["pairs"] = []
            for index in range(25):
                seed = 930000 + index
                order = [10, 1] if index % 2 == 0 else [1, 10]
                measurements = {str(steps): select(steps, seed) for steps in order}
                left = np.asarray(measurements["10"]["action"])
                right = np.asarray(measurements["1"]["action"])
                result["pairs"].append(
                    {
                        "index": index,
                        "seed": seed,
                        "warmup": index < 5,
                        "order": order,
                        "measurements": measurements,
                        "action_max_abs_difference": float(np.max(np.abs(left - right))),
                        "action_exact_equal": bool(np.array_equal(left, right)),
                    }
                )
            measured = result["pairs"][5:]
            result["timings"] = {
                str(steps): timing_summary([row["measurements"][str(steps)]["seconds"] for row in measured])
                for steps in (10, 1)
            }
            result["exceeding_50ms"] = {
                str(steps): sum(row["measurements"][str(steps)]["seconds"] > 0.05 for row in measured)
                for steps in (10, 1)
            }
            result["phase_diagnostics"] = {}
            for steps in (10, 1):
                phases = {}
                sample = select(steps, 930024, phases)
                reference = result["pairs"][-1]["measurements"][str(steps)]
                if (
                    sample["action"] != reference["action"]
                    or sample["normalized_action"] != reference["normalized_action"]
                ):
                    raise RuntimeError("Phase instrumentation changed the same-mode action")
                result["phase_diagnostics"][str(steps)] = {"phases": phases, "measurement": sample}
            result["status"] = "completed"
    finally:
        hook.remove()
        policy.model.sample_actions = original_sample
        policy.config.num_steps = 10
        result["restored_num_steps"] = policy.config.num_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if head != args.execution_head or dirty:
        raise ValueError("Expected clean registered source")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "started",
        "execution_head": head,
        "scope": "synthetic_compute_only_changed_sampler_not_equivalent_controller",
        "input": "coordinate_ramp_dual_cameras_zero_position_gripper_identity_quaternion",
        "task_outcomes_measured": 0,
        "qualification_data_read": False,
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
    }
    try:
        run(args, result)
    except Exception:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        (args.output / "result.json").write_text(text)
        print(text, flush=True)


if __name__ == "__main__":
    main()
