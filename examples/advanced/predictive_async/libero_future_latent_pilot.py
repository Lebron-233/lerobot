"""F-LAT1-r1: native-worker preprocessing for the frozen offline pilot.

No Env is created. Future observations are supervision/privileged references,
never predictor inputs. Both action comparisons keep the original current state.
"""

import argparse
import importlib.metadata
import json
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_identity_native as e
import numpy as np
import torch
from smolvla_graph_runtime import SmolVLAGraphRuntime

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor

SOURCE = e.REPO / "outputs/smolvla_graph_natural_2d672b5e"
PREPARATION = e.REPO / "outputs/smolvla_libero_future_latent_r1_preparation_c5950d51"
PREPARED_PAIRS = e.REPO / "outputs/smolvla_libero_future_latent_preparation_195ff5fa/prepared_pairs.pt"
ARMS = ("conditioned", "no_action")
SEED = 20260910
UPDATES = 200


def split_for(task):
    if task not in range(10):
        raise ValueError("Only the fixed ten tasks are supported")
    return "train" if task < 6 else "validation" if task < 8 else "test"


def config():
    return FutureLatentConfig(
        token_dim=960,
        action_dim=7,
        state_dim=32,
        rank=16,
        action_hidden_dim=64,
        state_hidden_dim=32,
        delay_embedding_dim=16,
        fusion_hidden_dim=64,
        max_cameras=2,
        risk_head=False,
    )


def tensor(value):
    return torch.as_tensor(value).detach().cpu()


def require_equal(left, right, description):
    if not torch.equal(tensor(left), tensor(right)):
        raise ValueError(description)


def aligned_pairs(record, arrays):
    """New target-alignment check, not another audit of the old whole experiment."""
    observations = {v["index"]: v for v in arrays["observations"]}
    dispatches = {v["action_index"]: v for v in arrays["control"]["dispatches"]}
    native = {v["action_index"]: v for v in record["native_steps"] if v["segment"] == "measurement"}
    pairs, excluded = [], []
    for request in record["requests"]:
        if request["kind"] != "planned":
            continue
        rid = request["request_id"]
        prefix = arrays["requests"][f"prefix_{rid}"]
        start, end = prefix["next_action_index"], prefix["takeover_index"]
        delay = prefix["planned_delay_steps"]
        if request["observation_index"] != start or end != start + delay or not 1 <= delay <= 8:
            raise ValueError("Observation/committed-prefix/takeover indices are not aligned")
        if end not in observations or any(i not in native for i in range(start, end)):
            excluded.append({"request_id": rid, "reason": "episode_ended_before_complete_target"})
            continue
        mask = tensor(prefix["committed_mask"]).bool()
        if not delay <= len(mask) <= 8 or int(mask.sum()) != delay:
            raise ValueError("Committed mask length differs from the delay")
        expected_mask = torch.arange(len(mask)) < delay
        require_equal(mask, expected_mask, "Committed mask does not describe the full delay")
        actions = tensor(prefix["committed_policy_actions"])
        posts = tensor(prefix["committed_post_policy_actions"])
        for offset, index in enumerate(range(start, end)):
            dispatched = dispatches[index]
            source = arrays["requests"][f"request_{dispatched['source_request_id']}"]
            row = dispatched["source_row_offset"]
            require_equal(
                actions[offset], source["policy_chunk"][row], "Normalized prefix differs from source"
            )
            require_equal(posts[offset], dispatched["command"], "Committed post action was not dispatched")
            require_equal(posts[offset], native[index]["action"], "Committed post action differs from native")
        future = observations[end]
        if native[end - 1]["returned_at"] > future["returned_at"]:
            raise ValueError("Future observation precedes the last committed native return")
        if end in native and future["returned_at"] > native[end]["started_at"]:
            raise ValueError("Future observation was obtained after the first takeover action")
        padded = torch.zeros((1, 8, 7), dtype=actions.dtype)
        padded[0, :delay] = actions[:delay]
        pairs.append(
            {
                "request_id": rid,
                "current_index": start,
                "future_index": end,
                "delay": delay,
                "actions": padded,
                "mask": (torch.arange(8) < delay).unsqueeze(0),
                "current_observation": observations[start],
                "future_observation": future,
                "cached": arrays["requests"][f"request_{rid}"],
            }
        )
    return pairs, excluded


def worker_batch(saved, language, device):
    """Use the exact native worker path, including uint8 /255 on its device."""
    observation = {
        k: v.numpy().copy() if isinstance(v, torch.Tensor) else v
        for k, v in saved["worker_observation"].items()
    }
    features = {
        e.OBS_STATE: {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
        **{
            key: {"dtype": "video", "shape": observation[key.rsplit(".", 1)[1]].shape}
            for key in e.CAMERA_KEYS
        },
    }
    batch = e.build_dataset_frame(features, observation, prefix="observation")
    batch = e.prepare_observation_for_inference(batch, torch.device(device), language, "libero")
    batch["task"] = [language]
    return batch


@contextmanager
def phase(output, name, limit=120):
    with (output / "events.jsonl").open("a") as stream:
        start = time.perf_counter()
        stream.write(json.dumps({"phase": name, "event": "started", "at": start, "limit": limit}) + "\n")
        stream.flush()
        try:
            yield
        except BaseException:
            stream.write(
                json.dumps({"phase": name, "event": "error", "error": traceback.format_exc()}) + "\n"
            )
            stream.flush()
            raise
        else:
            elapsed = time.perf_counter() - start
            stream.write(json.dumps({"phase": name, "event": "returned", "seconds": elapsed}) + "\n")
            stream.flush()
            if elapsed > limit:
                raise TimeoutError(f"{name} exceeded its registered limit")


def extract(policy, pre, output, counts):
    samples, source_manifest = [], []
    prepared = torch.load(PREPARED_PAIRS, map_location="cpu", weights_only=False)
    for case in prepared:
        spec = case["spec"]
        task, ordinal = spec["task_id"], spec["ordinal"]
        pairs, excluded = case["pairs"], case["excluded"]
        source_manifest.append(
            {
                "task": task,
                "ordinal": ordinal,
                "split": split_for(task),
                "requests": [p["request_id"] for p in pairs],
                "excluded": excluded,
            }
        )
        language = spec["task_name"].replace("_", " ")
        for number, pair in enumerate(pairs):
            cached = pair["cached"]
            inputs = tuple(tensor(v) for v in cached["inputs"])
            with torch.inference_mode():
                for kind in ("current", "future") if number == 0 else ("future",):
                    if counts["encoding_batches"] >= 87:
                        raise RuntimeError("Encoding budget exhausted")
                    with phase(output, f"encode_{task}_{pair['request_id']}_{kind}", 15):
                        counts["encoding_batches"] += 1
                        pre.reset()
                        batch = pre(worker_batch(pair[f"{kind}_observation"], language, "cuda"))
                        images, masks = policy.prepare_images(batch)
                        z, zm = policy.model.encode_image_tokens(images, masks)
                        z = tuple(v.detach().cpu().clone() for v in z)
                        zm = tuple(v.detach().cpu().clone() for v in zm)
                        state = policy.prepare_state(batch).detach().cpu().clone()
                    if kind == "current":
                        comparison = {
                            "task": task,
                            "request_id": pair["request_id"],
                            "camera_max_abs": [
                                (z[c].float() - inputs[c].float()).abs().max().item() for c in range(2)
                            ],
                            "state_exact": torch.equal(state, inputs[6]),
                        }
                        with (output / "current_input_comparisons.jsonl").open("a") as stream:
                            stream.write(json.dumps(comparison) + "\n")
                        for camera in range(2):
                            require_equal(
                                z[camera], inputs[camera], "Current-token extraction differs from archive"
                            )
                            require_equal(zm[camera], inputs[camera + 2], "Current camera mask changed")
                        require_equal(state, inputs[6], "Current model-ready state changed")
                    else:
                        for camera in range(2):
                            require_equal(zm[camera], inputs[camera + 2], "Future camera validity changed")
                        future = z
            if len(samples) >= 77:
                raise RuntimeError("Source sample budget exhausted")
            samples.append(
                {
                    "task": task,
                    "ordinal": ordinal,
                    "split": split_for(task),
                    "request_id": pair["request_id"],
                    "current_index": pair["current_index"],
                    "future_index": pair["future_index"],
                    "delay": pair["delay"],
                    "actions": pair["actions"].clone(),
                    "mask": pair["mask"].clone(),
                    "inputs": tuple(v.clone() for v in inputs),
                    "future": future,
                    "archived_full_chunk": tensor(cached["full_chunk"]).clone(),
                }
            )
    if {s["task"] for s in samples} != set(range(10)):
        raise ValueError("A registered task supplied no aligned pair")
    torch.save(samples, output / "aligned_cache.pt")
    e.write_json(output / "source_manifest.json", source_manifest)
    return samples, source_manifest


def batch_for(samples, indices, device):
    selected = [samples[i] for i in indices]
    return {
        "tokens": tuple(torch.cat([s["inputs"][c] for s in selected]).to(device).float() for c in range(2)),
        "masks": tuple(torch.cat([s["inputs"][c + 2] for s in selected]).to(device) for c in range(2)),
        "actions": torch.cat([s["actions"] for s in selected]).to(device),
        "action_mask": torch.cat([s["mask"] for s in selected]).to(device),
        "state": torch.cat([s["inputs"][6] for s in selected]).to(device),
        "delay": torch.tensor([s["delay"] for s in selected], device=device),
        "future": tuple(torch.cat([s["future"][c] for s in selected]).to(device).float() for c in range(2)),
    }


def prediction(model, batch, arm, quantized=True):
    actions = batch["actions"] if arm == "conditioned" else torch.zeros_like(batch["actions"])
    delta = model(
        batch["tokens"], batch["masks"], actions, batch["action_mask"], batch["state"], batch["delay"]
    )
    future = tuple(z + dz for z, dz in zip(batch["tokens"], delta.delta_tokens, strict=True))
    return tuple(v.to(torch.bfloat16).float() for v in future) if quantized else future


def per_sample_mse(predicted, targets, masks):
    numerator, denominator = 0, 0
    for p, t, mask in zip(predicted, targets, masks, strict=True):
        numerator = numerator + ((p.float() - t.float()).square() * mask.unsqueeze(-1)).sum((1, 2))
        denominator = denominator + mask.sum(1) * p.shape[-1]
    return numerator / denominator.clamp_min(1)


def train(samples, output, counts):
    indices = {
        name: [i for i, s in enumerate(samples) if s["split"] == name]
        for name in ("train", "validation", "test")
    }
    train_batch = batch_for(samples, indices["train"], "cuda")
    valid_batch = batch_for(samples, indices["validation"], "cuda")
    checkpoints, histories = {}, {}
    for arm in ARMS:
        with phase(output, f"train_{arm}", 120):
            torch.manual_seed(SEED)
            model = LightweightFutureLatentPredictor(config()).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            generator = torch.Generator().manual_seed(SEED + 1)
            best, best_step, best_weights = float("inf"), None, None
            history = []
            for step in range(UPDATES + 1):
                if step % 25 == 0:
                    model.eval()
                    with torch.no_grad():
                        loss = (
                            per_sample_mse(
                                prediction(model, valid_batch, arm),
                                valid_batch["future"],
                                valid_batch["masks"],
                            )
                            .mean()
                            .item()
                        )
                    history.append({"step": step, "validation_mse": loss})
                    if loss < best:
                        best, best_step = loss, step
                        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step == UPDATES:
                    break
                model.train()
                chosen = torch.randint(len(indices["train"]), (8,), generator=generator).to("cuda")
                batch = {
                    k: tuple(vv[chosen] for vv in v) if isinstance(v, tuple) else v[chosen]
                    for k, v in train_batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                loss = per_sample_mse(
                    prediction(model, batch, arm, quantized=False), batch["future"], batch["masks"]
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                counts["training_updates"] += 1
            torch.cuda.synchronize()
            checkpoint = {
                "config": asdict(config()),
                "arm": arm,
                "state_dict": best_weights,
                "best_step": best_step,
                "validation_mse": best,
                "seed": SEED,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
            torch.save(checkpoint, output / f"{arm}.pt")
            e.write_json(output / f"{arm}_training.json", history)
            model.load_state_dict(best_weights)
            checkpoints[arm] = model.eval()
            histories[arm] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return checkpoints, histories, indices


def summarize_errors(rows, key):
    values = [r[key] for r in rows]
    tasks = sorted({r["task"] for r in rows})
    means = {str(t): float(np.mean([r[key] for r in rows if r["task"] == t])) for t in tasks}
    return {
        "n": len(values),
        "sample_mean": float(np.mean(values)),
        "task_macro": float(np.mean(list(means.values()))),
        "per_task": means,
    }


def evaluate(samples, models, policy, output, counts, indices):
    selected = indices["test"]
    if not 0 < len(selected) <= 24:
        raise ValueError("Held-out sample count outside the pre-registered bound")
    batch = batch_for(samples, selected, "cuda")
    with torch.no_grad():
        predictions = {a: prediction(m, batch, a) for a, m in models.items()}
        errors = {
            "identity": per_sample_mse(batch["tokens"], batch["future"], batch["masks"]).tolist(),
            **{
                a: per_sample_mse(p, batch["future"], batch["masks"]).tolist() for a, p in predictions.items()
            },
        }
    predictions = {a: tuple(v.detach().cpu() for v in values) for a, values in predictions.items()}
    torch.save(predictions, output / "heldout_prediction_tokens.pt")
    rows, action_arrays = [], []
    with torch.inference_mode(), SmolVLAGraphRuntime(policy.model) as runtime:
        for j, i in enumerate(selected):
            sample = samples[i]
            runtime.begin_episode("graph", f"F-LAT1-task{sample['task']}")
            values = tuple(v.cuda() for v in sample["inputs"])
            contexts = {
                "identity": values[:2],
                **{
                    a: tuple(p[c][j : j + 1].to(device="cuda", dtype=values[c].dtype) for c in range(2))
                    for a, p in predictions.items()
                },
                "oracle": tuple(v.cuda() for v in sample["future"]),
            }
            outputs = {}
            for arm, visual in contexts.items():
                if counts["decoder_calls"] >= 96:
                    raise RuntimeError("Decoder budget exhausted")
                with phase(output, f"decode_{sample['task']}_{sample['request_id']}_{arm}", 15):
                    counts["decoder_calls"] += 1
                    decoded = runtime(
                        None,
                        None,
                        values[4],
                        values[5],
                        values[6],
                        noise=values[7],
                        future_image_tokens=tuple(visual),
                        future_image_token_masks=tuple(values[2:4]),
                    )
                    outputs[arm] = decoded.detach().cpu().clone()
                if arm == "identity":
                    require_equal(
                        outputs[arm],
                        sample["archived_full_chunk"],
                        "Offline identity Graph replay differs from native archive",
                    )
            row = {k: sample[k] for k in ("task", "request_id", "current_index", "future_index", "delay")}
            row.update({f"{a}_latent_mse": errors[a][j] for a in errors})
            for arm in ("identity", *ARMS):
                difference = (outputs[arm][..., :7].float() - outputs["oracle"][..., :7].float()).square()
                row[f"{arm}_row0_oracle_mse"] = difference[:, 0].mean().item()
                row[f"{arm}_chunk_oracle_mse"] = difference.mean().item()
            rows.append(row)
            action_arrays.append(
                {"task": sample["task"], "request_id": sample["request_id"], "outputs": outputs}
            )
        counts["captures"] = len(runtime.captures)
        if counts["captures"] != 2:
            raise ValueError("Expected exactly one Graph capture per held-out task")
        e.write_json(output / "captures.json", runtime.captures)
    torch.save(action_arrays, output / "heldout_action_outputs.pt")
    summary = {k: summarize_errors(rows, k) for k in rows[0] if k.endswith("mse")}
    return rows, summary


def worker(args):
    counts = Counter()
    result = {"status": "technical_failure", "first_failure": None, "experiment": "F-LAT1-r1"}
    try:
        e.require_source(args.execution_head)
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("Registered GPU changed")
        with phase(args.output, "load", 120):
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            result["policy_load"] = report
        with phase(args.output, "extract", 120):
            samples, manifest = extract(policy, pre, args.output, counts)
        models, training, indices = train(samples, args.output, counts)
        e.write_json(args.output / "frozen_checkpoints.json", training)
        with phase(args.output, "heldout_evaluation", 120):
            rows, summary = evaluate(samples, models, policy, args.output, counts, indices)
        result.update(
            status="completed",
            offline_contract_passed=True,
            source_manifest=manifest,
            samples=len(samples),
            split_counts={k: len(v) for k, v in indices.items()},
            training=training,
            heldout_rows=rows,
            metrics=summary,
        )
    except BaseException:
        result.update(first_failure=traceback.format_exc(), offline_contract_passed=False)
    finally:
        result.update(
            execution_head=args.execution_head,
            counts=dict(counts),
            attempts=1,
            retries=0,
            new_env=0,
            new_native=0,
            real_robot=0,
            state_mode="current_model_ready_identity",
            baseline_qualified=False,
            realtime_qualified=False,
            predictor_benefit_tested=False,
            risk_thresholds=None,
            old_confirmation="not_started_untouched",
        )
        e.write_json(args.output / "worker_result.json", result)
    print(
        json.dumps({k: result[k] for k in ("status", "offline_contract_passed", "first_failure")}), flush=True
    )
    return 0 if result["offline_contract_passed"] else 2


def supervise(args):
    e.require_source(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_libero_future_latent_r1_{args.execution_head[:8]}"
    if args.output != expected or args.output.exists() or sys.executable != e.PYTHON:
        raise ValueError("Use the frozen Python and absent exclusive output")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if registration["body"] != (PREPARATION / "registration.md").read_text():
        raise ValueError("Registration differs")
    gates = json.loads((PREPARATION / "gates.json").read_text())
    if not all(gates.values()):
        raise ValueError("Preparation gates did not pass")
    args.output.mkdir()
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--execution-head",
        args.execution_head,
        "--output",
        str(args.output),
        "--worker",
    ]
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    stopped = None
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(command, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT)
        cursor, active = 0, {}
        try:
            while child.poll() is None:
                now = time.perf_counter()
                path = args.output / "events.jsonl"
                if path.exists():
                    with path.open() as stream:
                        stream.seek(cursor)
                        while line := stream.readline():
                            if not line.endswith("\n"):
                                break
                            cursor = stream.tell()
                            event = json.loads(line)
                            if event["event"] == "started":
                                active[event["phase"]] = event
                            else:
                                active.pop(event["phase"], None)
                expired = [v for v in active.values() if now - v["at"] > v["limit"]]
                if expired or now - start >= 600:
                    stopped = {"expired": expired, "outer_limit": now - start >= 600}
                    break
                time.sleep(0.1)
        except BaseException:
            stopped = {"supervisor_error": traceback.format_exc()}
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
        code = child.wait()
    active = {}
    if (args.output / "events.jsonl").exists():
        for line in (args.output / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "started":
                active[event["phase"]] = event
            else:
                active.pop(event["phase"], None)
    result_path = args.output / "worker_result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"status": "technical_failure", "offline_contract_passed": False}
    )
    result["execution"] = {
        "child_pid": child.pid,
        "child_exit_code": code,
        "exit_confirmed": True,
        "started_at_utc": utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - start,
        "stop_reason": stopped,
        "active_at_exit": active,
        "command": command,
    }
    if code != 0 or stopped or active:
        result.update(status="technical_failure", offline_contract_passed=False)
    result["environment_after"] = {
        "python": sys.executable,
        "version": sys.version,
        "packages": sorted([[d.metadata["Name"], d.version] for d in importlib.metadata.distributions()]),
    }
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "offline_contract_passed", "samples", "split_counts", "first_failure")
            }
        ),
        flush=True,
    )
    return 0 if result["offline_contract_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
