"""Compare eager and ten-step graph actions on all 90 closed one-step episodes' recorded states."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from libero_reference_smoke import load_runtime, policy_observation, write_json
from libero_single_step_native import registered_tuples
from profile_libero_cuda_graph_compute import EagerSampler, GraphSampler, compare
from profile_libero_reference_compute import timing_summary

from lerobot.utils.constants import ACTION, OBS_STATE

CANDIDATE_HEAD = "ee273bcecc89fb2c6fc1b092b2770d134ec32085"


def recorded_batch(directory, observation, task_name):
    values = observation["state"]["values"]
    with np.load(directory / observation["cameras"], allow_pickle=False) as cameras:
        raw = {
            "pixels": {key: cameras[key].copy() for key in ("image", "image2")},
            "robot_state": {
                "eef": {
                    "pos": np.asarray(values[:3]),
                    "quat": np.asarray(observation["eef_quaternion_xyzw"]["values"]),
                },
                "gripper": {"qpos": np.asarray(values[6:8])},
            },
        }
    batch = policy_observation(raw, task_name.replace("_", " "))
    expected = torch.tensor(values, dtype=batch[OBS_STATE].dtype).reshape(1, 8)
    if not torch.equal(batch[OBS_STATE].cpu(), expected):
        raise ValueError("Reconstructed model-input state differs from the saved observation")
    return batch


def fixed_jobs(source):
    registration = json.loads((source / "registration.json").read_text())
    if registration["source_head"] != CANDIDATE_HEAD or registration["tuples"] != registered_tuples():
        raise ValueError("Use the original closed one-step screen, never the matched-control output")
    jobs = []
    for spec in registered_tuples():
        directory = source / f"tuple_{spec['ordinal']:03d}"
        record = json.loads((directory / "result.json").read_text())
        if record["tuple"] != spec or record["status"] != "completed":
            raise ValueError("The closed source cohort is incomplete")
        n = record["measured_actions"]
        observations = {}
        with (directory / "events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "observation":
                    observations[event["index"]] = event
        for stage, index in (("initial", 0), ("midpoint", n // 2), ("terminal", n)):
            jobs.append((directory, spec, stage, observations[index]))
    return jobs


def run(args, result):
    jobs = fixed_jobs(args.source)
    torch.set_num_threads(1)
    policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
    result["policy_load"] = report
    original = policy.model.sample_actions
    eager = EagerSampler(original)
    samplers = {"eager": eager}

    def select(mode, batch, seed):
        policy.reset()
        pre.reset()
        post.reset()
        processed = pre(batch)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sampler = samplers[mode]
        policy.model.sample_actions = sampler
        torch.cuda.synchronize()
        start = time.perf_counter()
        noise = policy.model.sample_noise((1, 50, 32), policy.model.state_proj.weight.device)
        normalized = policy.select_action(processed, noise=noise)
        action = post(normalized).detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        full = sampler.latest.detach().cpu().numpy().copy()
        normalized = normalized.detach().cpu().numpy().copy()
        if full.shape != (1, 50, 32) or normalized.shape != (1, 7) or len(policy._queues[ACTION]) != 0:
            raise ValueError("Full ten-step selector shape/queue contract changed")
        if not all(np.isfinite(value).all() for value in (full, normalized, action)):
            raise ValueError("Nonfinite model output")
        return elapsed, full, normalized, action

    try:
        with torch.inference_mode():
            directory, spec, _, obs = jobs[0]
            batch = recorded_batch(directory, obs, spec["task_name"])
            select("eager", batch, 989999)
            graph = GraphSampler(policy.model, original, eager.inputs, policy.model.action_out_proj)
            samplers["graph"] = graph
            result["capture_projection_shapes"] = graph.capture_projection_shapes
            result["warmup_pairs"] = []
            for index in range(5):
                a = select("eager", batch, 989990 + index)
                b = select("graph", batch, 989990 + index)
                comparisons = compare(a, b)
                result["warmup_pairs"].append(comparisons)
                if not all(value["exact_equal"] for value in comparisons.values()):
                    raise RuntimeError("Recorded-input warmup equality failed")
            result["pairs"] = []
            for index, (directory, spec, stage, obs) in enumerate(jobs):
                batch = recorded_batch(directory, obs, spec["task_name"])
                seed = 990000 + index
                order = ("eager", "graph") if index % 2 == 0 else ("graph", "eager")
                selected = {mode: select(mode, batch, seed) for mode in order}
                a, b = selected["eager"], selected["graph"]
                comparisons = compare(a, b)
                result["pairs"].append(
                    {
                        "index": index,
                        "source_tuple": spec["tuple_id"],
                        "task_id": spec["task_id"],
                        "stage": stage,
                        "observation_index": obs["index"],
                        "seed": seed,
                        "order": list(order),
                        "eager_seconds": a[0],
                        "graph_seconds": b[0],
                        "comparison": comparisons,
                    }
                )
                np.savez_compressed(
                    args.output / f"pair_{index:03d}.npz",
                    eager_chunk=a[1],
                    graph_chunk=b[1],
                    eager_action=a[3],
                    graph_action=b[3],
                )
                if not all(value["exact_equal"] for value in comparisons.values()):
                    raise RuntimeError(f"Recorded-input equality failed on pair {index}")
                if (index + 1) % 30 == 0:
                    print(f"RECORDED_PAIRS {index + 1}/270 exact", flush=True)
            result["timings"] = {
                mode: timing_summary([row[f"{mode}_seconds"] for row in result["pairs"]])
                for mode in ("eager", "graph")
            }
            result["speed_ratio_of_means"] = (
                result["timings"]["eager"]["mean_seconds"] / result["timings"]["graph"]["mean_seconds"]
            )
            result["graph_samples_over_50ms"] = sum(row["graph_seconds"] > 0.05 for row in result["pairs"])
            result["exact_measured_pairs"] = len(result["pairs"])
            result["exact_full_chunk_scalar_values"] = len(result["pairs"]) * 50 * 32
            result["fresh_visual_encodings_including_setup"] = graph.visual_encoding_calls
            result["graph_replay_calls"] = graph.replay_calls
            result["status"] = "completed"
    finally:
        policy.model.sample_actions = original
        result["original_sampler_restored"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "policy-path", "vlm-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != args.execution_head or subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("Require the registered clean source")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "started",
        "execution_head": head,
        "source": str(args.source),
        "source_execution_head": CANDIDATE_HEAD,
        "scope": "recorded_observation_pointwise_equivalence_not_closed_loop_success",
        "new_native_episodes": 0,
        "matched_control_records_read": False,
        "old_confirmation_read": False,
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
        print(json.dumps({k: v for k, v in result.items() if k not in ("pairs", "warmup_pairs")}, indent=2))


if __name__ == "__main__":
    main()
