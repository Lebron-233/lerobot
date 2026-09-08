"""Keep fresh vision and ten denoising steps; graph only the prefix/action sampler."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path


def flat_inputs(tokens, masks, language, language_mask, state, noise):
    if len(tokens) != 2 or len(masks) != 2 or noise is None:
        raise ValueError("This fixed two-camera probe requires explicit noise")
    return (*tokens, *masks, language, language_mask, state, noise)


def copy_inputs(static, current):
    """Refresh every graph input in place; never leave old state or noise in a replay."""
    if len(static) != len(current):
        raise ValueError("Graph input count changed")
    for target, source in zip(static, current, strict=True):
        if (target.shape, target.dtype, target.device) != (source.shape, source.dtype, source.device):
            raise ValueError("Graph input shape, dtype or device changed")
        target.copy_(source)


def synthetic_raw(index):
    import numpy as np

    grid = np.indices((256, 256), dtype=np.uint16)
    base = np.stack((grid[0], grid[1], (grid[0] + grid[1]) % 256), axis=-1)
    return {
        "pixels": {
            "image": ((base + 7 * index) % 256).astype(np.uint8),
            "image2": ((255 - base + 11 * index) % 256).astype(np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([0.001 * index, -0.002 * index, 0.25 + 0.001 * (index % 5)]),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            "gripper": {"qpos": np.array([0.03 + 0.0001 * index, -0.03 - 0.0001 * index])},
        },
    }


def run(args, result):
    import numpy as np
    import torch
    from libero_reference_qualification import TASK_NAMES
    from libero_reference_smoke import load_runtime, policy_observation
    from profile_libero_reference_compute import timing_summary

    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    torch.set_num_threads(1)
    policy, pre, post, load_report = load_runtime(args.policy_path, args.vlm_path)
    result["strict_policy_load"] = load_report
    result["device_name"] = torch.cuda.get_device_name()
    result["source_module"] = str(Path(__import__(type(policy).__module__, fromlist=["__file__"]).__file__))
    if (policy.config.chunk_size, policy.config.n_action_steps, policy.config.num_steps) != (50, 1, 10):
        raise ValueError("Expected the unmodified ten-step reference")
    if policy.config.compile_model:
        raise ValueError("No compiler or matmul precision change belongs in this probe")
    original_sampler = policy.model.sample_actions
    original_encoder = policy.model.encode_image_tokens
    projection_shapes, vision_counts, last = [], [], {}
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _m, _a, output: projection_shapes.append(list(output.shape))
    )

    def counted_encoder(*positional, **keywords):
        vision_counts.append(1)
        return original_encoder(*positional, **keywords)

    def eager_sampler(*positional, **keywords):
        output = original_sampler(*positional, **keywords)
        last["full"] = output
        return output

    def fixture(index):
        policy.reset()
        pre.reset()
        post.reset()
        batch = policy_observation(synthetic_raw(index), TASK_NAMES[index % 10].replace("_", " "))
        processed = pre(batch)
        torch.manual_seed(980000 + index)
        torch.cuda.manual_seed_all(980000 + index)
        noise = policy.model.sample_noise((1, 50, 32), policy.prepare_state(processed).device)
        return processed, noise

    graph = None
    try:
        with torch.inference_mode():
            processed, noise = fixture(0)
            images, image_masks = policy.prepare_images(processed)
            tokens, masks = original_encoder(images, image_masks)
            current = flat_inputs(
                tokens,
                masks,
                processed[OBS_LANGUAGE_TOKENS],
                processed[OBS_LANGUAGE_ATTENTION_MASK],
                policy.prepare_state(processed),
                noise,
            )
            static = tuple(tensor.clone() for tensor in current)
            result["input_metadata"] = [
                {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)} for t in static
            ]

            def captured_work():
                return original_sampler(
                    None,
                    None,
                    static[4],
                    static[5],
                    static[6],
                    noise=static[7],
                    future_image_tokens=(static[0], static[1]),
                    future_image_token_masks=(static[2], static[3]),
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            setup_start = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    captured_work()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            result["stage"] = "capture"
            projection_shapes.clear()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_output = captured_work()
            result["capture_projection_shapes"] = projection_shapes.copy()
            if projection_shapes != [[1, 50, 32]] * 10:
                raise RuntimeError("The captured workload does not contain the original ten projections")
            torch.cuda.synchronize()
            result["warmup_and_capture_seconds"] = time.perf_counter() - setup_start

            def graph_sampler(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
                if kwargs:
                    raise ValueError("This probe does not accept additional runtime features")
                fresh_tokens, fresh_masks = policy.model.encode_image_tokens(images, img_masks)
                copy_inputs(
                    static, flat_inputs(fresh_tokens, fresh_masks, lang_tokens, lang_masks, state, noise)
                )
                graph.replay()
                # The ordinary selector/postprocessor must not own the graph's output buffer.
                output = static_output.clone()
                last["full"] = output
                return output

            def select(index, sampler):
                processed, noise = fixture(index)
                projection_shapes.clear()
                vision_counts.clear()
                policy.model.sample_actions = sampler
                torch.cuda.synchronize()
                start = time.perf_counter()
                normalized = policy.select_action(processed, noise=noise)
                action = post(normalized).detach().cpu().numpy().copy()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                full = last["full"].detach().cpu().clone()
                normalized = normalized.detach().cpu().clone()
                if full.shape != (1, 50, 32) or normalized.shape != (1, 7):
                    raise RuntimeError("The sampler or selector shape changed")
                if not torch.isfinite(full).all() or not np.isfinite(action).all():
                    raise RuntimeError("Nonfinite output")
                if len(vision_counts) != 1 or len(policy._queues[ACTION]) != 0:
                    raise RuntimeError("Fresh full vision and one-action consumption must be preserved")
                if sampler is eager_sampler and projection_shapes != [[1, 50, 32]] * 10:
                    raise RuntimeError("The ordinary path is no longer the ten-step reference")
                return elapsed, full, normalized, action

            policy.model.encode_image_tokens = counted_encoder
            result["stage"] = "paired_measurements"
            result["pairs"] = []
            tensor_records = {}
            for index in range(25):
                order = ["eager", "graph"] if index % 2 == 0 else ["graph", "eager"]
                samplers = {"eager": eager_sampler, "graph": graph_sampler}
                selected = {name: select(index, samplers[name]) for name in order}
                left, right = selected["eager"], selected["graph"]
                equal = (
                    torch.equal(left[1], right[1])
                    and torch.equal(left[2], right[2])
                    and np.array_equal(left[3], right[3])
                )
                row = {
                    "index": index,
                    "seed": 980000 + index,
                    "synthetic_task_id": index % 10,
                    "warmup": index < 5,
                    "order": order,
                    "eager_seconds": left[0],
                    "graph_seconds": right[0],
                    "full_padded_chunk_max_abs_difference": float((left[1] - right[1]).abs().max()),
                    "normalized_selected_max_abs_difference": float((left[2] - right[2]).abs().max()),
                    "action_max_abs_difference": float(np.max(np.abs(left[3] - right[3]))),
                    "exact_equal": equal,
                }
                result["pairs"].append(row)
                for name, sample in selected.items():
                    tensor_records[f"{index:02d}_{name}_chunk"] = sample[1].numpy()
                    tensor_records[f"{index:02d}_{name}_action"] = sample[3]
                np.savez_compressed(args.output / "paired_outputs.npz", **tensor_records)
                if not equal:
                    raise RuntimeError(f"Exact equality failed at pair {index}; no speed interpretation")
            measured = result["pairs"][5:]
            result["timings"] = {
                name: timing_summary([row[name + "_seconds"] for row in measured])
                for name in ("eager", "graph")
            }
            result["graph_samples_over_50ms"] = sum(row["graph_seconds"] > 0.05 for row in measured)
            result["exact_equal_pairs"] = sum(row["exact_equal"] for row in result["pairs"])
            result["status"] = "completed"
            result["stage"] = "closed"
    finally:
        policy.model.sample_actions = original_sampler
        policy.model.encode_image_tokens = original_encoder
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if (
        head != args.execution_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    ):
        raise RuntimeError("Expected the registered clean source")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "started",
        "stage": "load",
        "execution_head": head,
        "scope": "synthetic_fresh_vision_ten_step_graph_compute_not_native_or_async_qualification",
        "new_rollouts": 0,
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
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
