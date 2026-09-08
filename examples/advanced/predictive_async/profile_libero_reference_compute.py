"""Compare an unchanged LIBERO selector with removal of its current-image encoder."""

import argparse
import json
import math
import statistics
import subprocess
import time
import traceback
from pathlib import Path


def timing_summary(samples):
    """Describe all positive samples; p95 uses the empirical nearest rank."""
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timing samples must be nonempty, finite and positive")
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "empirical_p95_seconds": ordered[math.ceil(0.95 * len(samples)) - 1],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def token_metadata(value):
    """Retain the actual native return structure without assuming camera count."""
    if isinstance(value, (list, tuple)):
        return [token_metadata(x) for x in value]
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def run_profile(args, result):
    import numpy as np
    import torch
    from libero_reference_smoke import load_runtime, policy_observation

    torch.set_num_threads(1)
    policy, pre, post, load_report = load_runtime(args.policy_path, args.vlm_path)
    result["strict_policy_load"] = load_report
    model = policy.model
    original_encode = model.encode_image_tokens
    capture = {}

    def capture_once(*positional, **keywords):
        tokens = original_encode(*positional, **keywords)
        capture.update(args=positional, kwargs=keywords, tokens=tokens)
        return tokens

    def cached_encode(*_positional, **_keywords):
        return capture["tokens"]

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

    def select(seed, encoder):
        policy.reset()
        pre.reset()
        post.reset()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        processed = pre(batch)
        model.encode_image_tokens = encoder
        torch.cuda.synchronize()
        start = time.perf_counter()
        normalized = policy.select_action(processed)
        action = post(normalized).detach().cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not torch.isfinite(normalized).all() or not np.isfinite(action).all():
            raise ValueError("Non-finite diagnostic action")
        return elapsed, normalized.detach().cpu().clone(), action.copy()

    try:
        with torch.inference_mode():
            select(919999, capture_once)
            if not capture:
                raise RuntimeError("The ordinary selector did not call encode_image_tokens")
            result["token_return_metadata"] = token_metadata(capture["tokens"])
            result["device_name"] = torch.cuda.get_device_name()
            result["paired_samples"] = []
            for index in range(25):
                seed = 920000 + index
                encoders = {"native": original_encode, "cached": cached_encode}
                order = ("native", "cached") if index % 2 == 0 else ("cached", "native")
                selected = {name: select(seed, encoders[name]) for name in order}
                left, right = selected["native"], selected["cached"]
                exact = torch.equal(left[1], right[1]) and np.array_equal(left[2], right[2])
                row = {
                    "pair_index": index,
                    "seed": seed,
                    "warmup": index < 5,
                    "order": list(order),
                    "native_seconds": left[0],
                    "cached_seconds": right[0],
                    "normalized_max_abs_difference": float((left[1] - right[1]).abs().max()),
                    "postprocessed_max_abs_difference": float(np.max(np.abs(left[2] - right[2]))),
                    "exact_equal": exact,
                }
                result["paired_samples"].append(row)
                if not exact:
                    raise RuntimeError(f"Current-token substitution changed pair {index}")
            model.encode_image_tokens = original_encode
            encoder_samples = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                original_encode(*capture["args"], **capture["kwargs"])
                torch.cuda.synchronize()
                encoder_samples.append(time.perf_counter() - start)
            measured = [x for x in result["paired_samples"] if not x["warmup"]]
            result["timings"] = {
                "native": timing_summary([x["native_seconds"] for x in measured]),
                "cached": timing_summary([x["cached_seconds"] for x in measured]),
                "encoder": timing_summary(encoder_samples),
            }
            result["encoder_samples_seconds"] = encoder_samples
            result["exact_equal_pairs_including_warmup"] = sum(
                x["exact_equal"] for x in result["paired_samples"]
            )
            result["cached_measured_samples_exceeding_50ms"] = sum(
                x["cached_seconds"] > 0.05 for x in measured
            )
            result["status"] = "completed"
    finally:
        model.encode_image_tokens = original_encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != args.execution_head:
        raise ValueError("Execution HEAD differs from the registered source")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "started",
        "execution_head": head,
        "input_kind": "synthetic_coordinate_ramps_and_zero_state",
        "task_outcomes_measured": 0,
        "qualification_data_read": False,
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
    }
    try:
        run_profile(args, result)
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
