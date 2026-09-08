"""Test CUDA launch-overhead removal without changing the frozen ten-step sampler."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from libero_reference_smoke import load_runtime, policy_observation, write_json
from profile_libero_reference_compute import timing_summary

from lerobot.utils.constants import ACTION


def copy_inputs(destination, source):
    """Refresh every fixed-address input, including language, masks, state and noise."""
    if len(destination) != len(source):
        raise ValueError("Graph input count changed")
    for target, value in zip(destination, source, strict=True):
        if (target.shape, target.dtype, target.device) != (value.shape, value.dtype, value.device):
            raise ValueError("Graph input shape, dtype or device changed")
        target.copy_(value)


def check_capture_shapes(shapes):
    if shapes != [[1, 50, 32]] * 10:
        raise ValueError("Capture must contain all ten original 50-action denoising projections")


def pack(images, masks, tokens, token_masks, state, noise):
    if len(images) != 2 or len(masks) != 2 or noise is None:
        raise ValueError("This bounded profile requires two cameras and explicit fresh noise")
    return (*images, *masks, tokens, token_masks, state, noise)


def invoke(original, values):
    return original(values[:2], values[2:4], values[4], values[5], values[6], noise=values[7])


def invoke_graph(original, values):
    # These tokens were freshly encoded from this call's images, not predicted or reused across calls.
    return original(
        None,
        None,
        values[4],
        values[5],
        values[6],
        noise=values[7],
        future_image_tokens=tuple(values[:2]),
        future_image_token_masks=tuple(values[2:4]),
    )


class EagerSampler:
    """Retain the full padded output for comparison outside the timed selector."""

    def __init__(self, original):
        self.original = original
        self.inputs = None
        self.latest = None

    def __call__(self, images, masks, tokens, token_masks, state, noise=None):
        self.inputs = pack(images, masks, tokens, token_masks, state, noise)
        self.latest = invoke(self.original, self.inputs)
        return self.latest


class GraphSampler:
    """A single experiment-local graph; not installed into the production runtime."""

    def __init__(self, model, original, inputs, projection):
        self.encode = model.encode_image_tokens
        tokens, masks = self.encode(inputs[:2], inputs[2:4])
        current = pack(tokens, masks, *inputs[4:])
        self.inputs = tuple(value.detach().clone() for value in current)
        self.visual_encoding_calls = 1
        self.replay_calls = 0
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                invoke_graph(original, self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.capture_projection_shapes = []
        hook = projection.register_forward_hook(
            lambda _module, _args, value: self.capture_projection_shapes.append(list(value.shape))
        )
        try:
            with torch.cuda.graph(self.graph, stream=stream):
                self.latest = invoke_graph(original, self.inputs)
        finally:
            hook.remove()
        check_capture_shapes(self.capture_projection_shapes)

    def __call__(self, images, masks, tokens, token_masks, state, noise=None):
        image_tokens, image_masks = self.encode(images, masks)
        self.visual_encoding_calls += 1
        copy_inputs(self.inputs, pack(image_tokens, image_masks, tokens, token_masks, state, noise))
        self.graph.replay()
        self.replay_calls += 1
        return self.latest


def fixture(index):
    """Four predetermined synthetic inputs; no episode, dataset or saved image is read."""
    grid = np.indices((256, 256), dtype=np.uint16)
    image = np.stack((grid[0], grid[1], (grid[0] + grid[1]) % 256), axis=-1).astype(np.uint8)
    image = np.ascontiguousarray(np.roll(image, 17 * index, axis=1))
    other = np.ascontiguousarray(np.roll(255 - image, 11 * index, axis=0))
    positions = [(0, 0, 0), (0.1, -0.1, 0.3), (-0.15, 0.02, 0.25), (0.03, 0.2, 0.18)]
    gaps = [0.0, 0.02, 0.04, 0.03]
    quaternions = [(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0.382683432365, 0.923879532511), (1, 0, 0, 0)]
    objects = ["alphabet soup", "cream cheese", "tomato sauce", "orange juice"]
    raw = {
        "pixels": {"image": image, "image2": other},
        "robot_state": {
            "eef": {"pos": np.asarray(positions[index]), "quat": np.asarray(quaternions[index])},
            "gripper": {"qpos": np.array([gaps[index], -gaps[index]])},
        },
    }
    return policy_observation(raw, f"pick up the {objects[index]} and place it in the basket")


def compare(left, right):
    names = ("full_padded_chunk", "selected_normalized_action", "postprocessed_action")
    return {
        name: {
            "exact_equal": bool(np.array_equal(a, b)),
            "max_abs_difference": float(np.max(np.abs(a - b))),
        }
        for name, a, b in zip(names, left[1:], right[1:], strict=True)
    }


def run(args, result):
    torch.set_num_threads(1)
    policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
    if policy.config.compile_model or policy.config.num_steps != 10:
        raise ValueError("Use the unchanged eager ten-step checkpoint configuration")
    result["policy_load"] = report
    original = policy.model.sample_actions
    eager = EagerSampler(original)
    samplers = {"eager": eager}
    batches = [fixture(index) for index in range(4)]

    def select(mode, index, seed):
        policy.reset()
        pre.reset()
        post.reset()
        processed = pre(batches[index])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sampler = samplers[mode]
        policy.model.sample_actions = sampler
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Fresh noise generation and all graph input copies are inside the timed call.
        noise = policy.model.sample_noise((1, 50, 32), policy.model.state_proj.weight.device)
        normalized = policy.select_action(processed, noise=noise)
        action = post(normalized).detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        full = sampler.latest.detach().cpu().numpy().copy()
        selected = normalized.detach().cpu().numpy().copy()
        if full.shape != (1, 50, 32) or selected.shape != (1, 7) or len(policy._queues[ACTION]) != 0:
            raise ValueError("The original chunk/selection contract changed")
        if not all(np.isfinite(value).all() for value in (full, selected, action)):
            raise ValueError("Nonfinite diagnostic output")
        return elapsed, full, selected, action

    try:
        with torch.inference_mode():
            result["stage"] = "initial_eager"
            initial = select("eager", 0, 979999)
            result["stage"] = "graph_capture"
            start = time.perf_counter()
            graph = GraphSampler(policy.model, original, eager.inputs, policy.model.action_out_proj)
            samplers["graph"] = graph
            result["capture_setup_seconds"] = time.perf_counter() - start
            result["capture_projection_shapes"] = graph.capture_projection_shapes
            result["fixed_inputs"] = [
                {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)} for x in graph.inputs
            ]
            result["stage"] = "initial_replay_equality"
            result["initial_replay_comparison"] = compare(initial, select("graph", 0, 979999))
            if not all(value["exact_equal"] for value in result["initial_replay_comparison"].values()):
                raise RuntimeError("Initial graph replay changed the original ten-step output")
            result["pairs"] = []
            result["stage"] = "paired_measurements"
            previous = None
            changed_outputs = 0
            for index in range(29):
                warmup = index < 5
                fixture_index = (index if warmup else index - 5) % 4
                seed = 980000 + index
                eager_first = index % 2 == 0 if warmup else (fixture_index + (index - 5) // 4) % 2 == 0
                order = ["eager", "graph"] if eager_first else ["graph", "eager"]
                values = {name: select(name, fixture_index, seed) for name in order}
                comparisons = compare(values["eager"], values["graph"])
                row = {
                    "pair_index": index,
                    "fixture_index": fixture_index,
                    "seed": seed,
                    "warmup": warmup,
                    "order": order,
                    "eager_seconds": values["eager"][0],
                    "graph_seconds": values["graph"][0],
                    "comparison": comparisons,
                }
                result["pairs"].append(row)
                np.savez_compressed(
                    args.output / f"pair_{index:03d}.npz",
                    eager_chunk=values["eager"][1],
                    graph_chunk=values["graph"][1],
                    eager_selected=values["eager"][2],
                    graph_selected=values["graph"][2],
                    eager_action=values["eager"][3],
                    graph_action=values["graph"][3],
                )
                if not all(value["exact_equal"] for value in comparisons.values()):
                    raise RuntimeError(f"Graph equality failed on pair {index}; no equivalence claim")
                if previous is not None and not np.array_equal(previous, values["eager"][1]):
                    changed_outputs += 1
                previous = values["eager"][1]
            measured = result["pairs"][5:]
            result["timings"] = {
                mode: timing_summary([row[f"{mode}_seconds"] for row in measured])
                for mode in ("eager", "graph")
            }
            result["graph_samples_exceeding_50ms"] = sum(row["graph_seconds"] > 0.05 for row in measured)
            result["native_output_changes_between_pairs"] = changed_outputs
            result["graph_visual_encoding_calls_including_setup"] = graph.visual_encoding_calls
            result["graph_replay_calls"] = graph.replay_calls
            result["graph_memory"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            result["status"] = "completed"
            result["stage"] = "completed"
    finally:
        policy.model.sample_actions = original
        result["original_sampler_restored"] = True
        result["final_num_steps"] = policy.config.num_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if head != args.execution_head or dirty:
        raise RuntimeError("Expected clean preregistered source")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "started",
        "execution_head": head,
        "scope": "synthetic_compute_only_full_ten_step_cuda_graph",
        "graph_region": "prefix_and_all_ten_denoising_steps; fresh_visual_encoding_outside_graph_each_call",
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
        write_json(args.output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
