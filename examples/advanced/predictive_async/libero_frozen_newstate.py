"""F-LAT2: frozen visual predictors on eight newly collected initial states."""

import argparse
import copy
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_future_latent_pilot as pilot
import libero_graph_natural_native as natural
import torch

e = natural.e
PREPARATION = e.REPO / "outputs/smolvla_frozen_newstate_preparation_8db97488"
CHECKPOINTS = e.REPO / "outputs/smolvla_libero_future_latent_r1_63f1f306"
LIMITS = {
    "episodes": (1, 8),
    "settling": (10, 80),
    "measurement": (280, 2240),
    "model": (160, 1280),
    "capture": (2, 16),
}
ARMS = ("identity", "conditioned", "no_action", "mismatched_action")
METRICS = ("latent_mse", "row0_oracle_mse", "chunk_oracle_mse")


def manifest():
    old = e.fixed_manifest()
    rows = []
    for state in (42, 43, 44, 45):
        for task in (8, 9) if state % 2 == 0 else (9, 8):
            spec = copy.deepcopy(
                next(
                    r
                    for r in old["rows"]
                    if r["task_id"] == task and r["condition"] == "graph_identity_async"
                )
            )
            spec.update(
                ordinal=len(rows),
                pair_index=len(rows),
                initial_state_id=state,
                environment_seed=970000 + 100 * task + state,
                policy_seed=960000 + 100 * task + state,
                recovery_policy=natural.recovery.RECOVERY_POLICY,
                max_recovery_probes_per_episode=50,
            )
            rows.append(spec)
    return {
        "experiment": "F-LAT2-frozen-newstate",
        "rows": rows,
        "source_initial_state": 41,
        "protected_confirmation_states": [21, 40],
        "checkpoints": str(CHECKPOINTS),
        "max_samples_per_episode": 12,
        "training_updates": 0,
        "limits": {k: {"per_episode": a, "total": b} for k, (a, b) in LIMITS.items()},
        **{k: old[k] for k in ("policy_revision", "vlm_revision", "assets_revision")},
    }


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"F-LAT2 {kind} budget exhausted before dispatch")


def donor_indices(samples):
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[(sample["ordinal"], sample["delay"])].append(i)
    donors = {}
    for group in groups.values():
        group.sort(key=lambda i: samples[i]["request_id"])
        if len(group) >= 2:
            donors.update({index: group[(pos + 1) % len(group)] for pos, index in enumerate(group)})
    return donors


def aggregate(rows):
    summaries = {}
    for arm in ARMS:
        for metric in METRICS:
            key = f"{arm}_{metric}"
            eligible = [r for r in rows if key in r]
            grouped = defaultdict(list)
            for row in eligible:
                grouped[row["ordinal"]].append(row[key])
            means = {str(i): sum(v) / len(v) for i, v in grouped.items()}
            summaries[key] = {
                "n": len(eligible),
                "episodes": len(means),
                "sample_mean": sum(r[key] for r in eligible) / len(eligible) if eligible else None,
                "episode_macro": sum(means.values()) / len(means) if means else None,
                "per_episode": means,
            }
    complete = all(summaries[f"{a}_row0_oracle_mse"]["episodes"] == 8 for a in ARMS[:3])
    increment = complete and all(
        summaries[f"conditioned_{metric}"]["episode_macro"] < summaries[f"{other}_{metric}"]["episode_macro"]
        for metric in METRICS[1:]
        for other in ("identity", "no_action")
    )
    return {
        "metrics": summaries,
        "eight_episodes_have_samples": complete,
        "frozen_action_increment_observed": bool(increment),
    }


def load_predictors():
    models, frozen = {}, {}
    for arm, step in (("conditioned", 125), ("no_action", 175)):
        checkpoint = torch.load(CHECKPOINTS / f"{arm}.pt", map_location="cpu", weights_only=False)
        if (
            checkpoint["config"] != asdict(pilot.config())
            or checkpoint["best_step"] != step
            or checkpoint["arm"] != arm
            or checkpoint["seed"] != 20260910
            or checkpoint["parameter_count"] != 69680
        ):
            raise ValueError("The frozen F-LAT1-r1 checkpoint identity differs")
        model = pilot.LightweightFutureLatentPredictor(pilot.config())
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[arm] = model.eval().requires_grad_(False).cuda()
        frozen[arm] = checkpoint
    return models, frozen


def choose_pairs(record, arrays):
    pairs, excluded = pilot.aligned_pairs(record, arrays)
    pairs.sort(key=lambda p: p["request_id"])
    excluded += [
        {"request_id": p["request_id"], "reason": "fixed_first_12_complete_pairs"} for p in pairs[12:]
    ]
    return pairs[:12], excluded


def extract(records, policy, pre, output, counts):
    samples, selections = [], []
    for record in records:
        spec = record["spec"]
        directory = output / f"episode_{spec['ordinal']:03d}"
        arrays = torch.load(directory / "arrays.pt", map_location="cpu", weights_only=False)
        pairs, excluded = choose_pairs(record, arrays)
        selections.append(
            {
                "ordinal": spec["ordinal"],
                "task": spec["task_id"],
                "initial_state_id": spec["initial_state_id"],
                "requests": [p["request_id"] for p in pairs],
                "excluded": excluded,
            }
        )
        for number, pair in enumerate(pairs):
            inputs = tuple(pilot.tensor(v).clone() for v in pair["cached"]["inputs"])
            for kind in ("current", "future") if number == 0 else ("future",):
                if counts["encoding_batches"] >= 104:
                    raise RuntimeError("F-LAT2 encoding budget exhausted")
                with (
                    pilot.phase(output, f"encode_{spec['ordinal']}_{pair['request_id']}_{kind}", 15),
                    torch.inference_mode(),
                ):
                    counts["encoding_batches"] += 1
                    pre.reset()
                    batch = pre(
                        pilot.worker_batch(
                            pair[f"{kind}_observation"], spec["task_name"].replace("_", " "), "cuda"
                        )
                    )
                    images, masks = policy.prepare_images(batch)
                    z, zm = policy.model.encode_image_tokens(images, masks)
                    z = tuple(v.detach().cpu().clone() for v in z)
                    zm = tuple(v.detach().cpu().clone() for v in zm)
                    state = policy.prepare_state(batch).detach().cpu().clone()
                for c in range(2):
                    pilot.require_equal(zm[c], inputs[c + 2], "Camera validity differs")
                    if kind == "current":
                        pilot.require_equal(z[c], inputs[c], "Native current tokens differ")
                if kind == "current":
                    pilot.require_equal(state, inputs[6], "Native current state differs")
                    counts["current_exact_episodes"] += 1
            samples.append(
                {
                    "task": spec["task_id"],
                    "ordinal": spec["ordinal"],
                    "initial_state_id": spec["initial_state_id"],
                    **{
                        k: pair[k]
                        for k in ("request_id", "current_index", "future_index", "delay", "actions", "mask")
                    },
                    "inputs": inputs,
                    "future": z,
                    "archived_full_chunk": pilot.tensor(pair["cached"]["full_chunk"]).clone(),
                }
            )
        del arrays, pairs
    samples.sort(key=lambda s: (s["task"], s["ordinal"], s["request_id"]))
    torch.save(samples, output / "aligned_cache.pt")
    e.write_json(output / "selections.json", selections)
    return samples, selections


def evaluate(samples, models, policy, output, counts):
    if not samples:
        return [], aggregate([])
    donors = donor_indices(samples)
    batch = pilot.batch_for(samples, list(range(len(samples))), "cuda")
    with pilot.phase(output, "frozen_predictors", 30), torch.inference_mode():
        predictions = {"identity": batch["tokens"], "oracle": batch["future"]}
        for arm in pilot.ARMS:
            counts["predictor_batches"] += 1
            predictions[arm] = pilot.prediction(models[arm], batch, arm)
        wrong = None
        if donors:
            wrong_batch = {
                k: tuple(t[list(donors)] for t in v) if isinstance(v, tuple) else v[list(donors)]
                for k, v in batch.items()
            }
            wrong_batch["actions"] = batch["actions"][list(donors.values())].clone()
            counts["predictor_batches"] += 1
            wrong = pilot.prediction(models["conditioned"], wrong_batch, "conditioned")
        errors = {
            arm: pilot.per_sample_mse(values, batch["future"], batch["masks"]).cpu().tolist()
            for arm, values in predictions.items()
            if arm != "oracle"
        }
    predictions = {arm: tuple(v.detach().cpu() for v in values) for arm, values in predictions.items()}
    wrong = None if wrong is None else tuple(v.detach().cpu() for v in wrong)
    wrong_positions = {i: pos for pos, i in enumerate(donors)}
    torch.save(
        {"predictions": predictions, "mismatched_predictions": wrong, "donors": donors},
        output / "predictions.pt",
    )
    rows, outputs_archive = [], []
    with torch.inference_mode(), pilot.SmolVLAGraphRuntime(policy.model) as runtime:
        for i, sample in enumerate(samples):
            runtime.begin_episode("graph", f"F-LAT2-task{sample['task']}")
            inputs = tuple(v.cuda() for v in sample["inputs"])
            contexts = {
                a: tuple(v[i : i + 1].to(device="cuda", dtype=inputs[c].dtype) for c, v in enumerate(z))
                for a, z in predictions.items()
            }
            if i in donors:
                pos = wrong_positions[i]
                contexts["mismatched_action"] = tuple(
                    v[pos : pos + 1].to(device="cuda", dtype=inputs[c].dtype) for c, v in enumerate(wrong)
                )
            decoded = {}
            for arm in ("identity", "conditioned", "no_action", "oracle", "mismatched_action"):
                if arm not in contexts:
                    continue
                if counts["decoder_calls"] >= 480:
                    raise RuntimeError("F-LAT2 decoder budget exhausted")
                with pilot.phase(output, f"decode_{sample['ordinal']}_{sample['request_id']}_{arm}", 15):
                    counts["decoder_calls"] += 1
                    value = runtime(
                        None,
                        None,
                        inputs[4],
                        inputs[5],
                        inputs[6],
                        noise=inputs[7],
                        future_image_tokens=contexts[arm],
                        future_image_token_masks=tuple(inputs[2:4]),
                    )
                    decoded[arm] = value.detach().cpu().clone()
                if arm == "identity":
                    pilot.require_equal(
                        decoded[arm],
                        sample["archived_full_chunk"],
                        "Identity full chunk differs from new native archive",
                    )
                    counts["identity_exact_cases"] += 1
            row = {
                k: sample[k]
                for k in (
                    "task",
                    "ordinal",
                    "initial_state_id",
                    "request_id",
                    "delay",
                    "current_index",
                    "future_index",
                )
            }
            for arm in ARMS:
                if arm not in decoded:
                    continue
                difference = (decoded[arm][..., :7].float() - decoded["oracle"][..., :7].float()).square()
                row[f"{arm}_row0_oracle_mse"] = difference[:, 0].mean().item()
                row[f"{arm}_chunk_oracle_mse"] = difference.mean().item()
                row[f"{arm}_latent_mse"] = (
                    errors[arm][i]
                    if arm != "mismatched_action"
                    else pilot.per_sample_mse(
                        tuple(v.cpu().float() for v in contexts[arm]),
                        sample["future"],
                        tuple(inputs[c].cpu() for c in (2, 3)),
                    ).item()
                )
            if i in donors:
                donor = samples[donors[i]]
                row.update(
                    donor_request_id=donor["request_id"],
                    mismatched_prefix_changed=not torch.equal(sample["actions"], donor["actions"]),
                    conditioned_mismatch_row0_mse=(
                        decoded["conditioned"][:, 0, :7].float()
                        - decoded["mismatched_action"][:, 0, :7].float()
                    )
                    .square()
                    .mean()
                    .item(),
                )
            rows.append(row)
            outputs_archive.append(
                {"ordinal": sample["ordinal"], "request_id": sample["request_id"], "outputs": decoded}
            )
        counts["offline_captures"] = len(runtime.captures)
        if counts["offline_captures"] > 2:
            raise ValueError("More than the two registered offline Graph captures")
        e.write_json(output / "offline_captures.json", runtime.captures)
    torch.save(outputs_archive, output / "action_outputs.pt")
    e.write_json(output / "metric_rows.json", rows)
    return rows, aggregate(rows)


def worker(args):
    import faulthandler

    e.require_source(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls = e.Calls(args.output / "calls.jsonl")
    budget, counts, records = Budget(), Counter(), []
    result = {"status": "technical_failure", "first_failure": None, "experiment": "F-LAT2-frozen-newstate"}
    try:
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("Registered GPU differs")
        with pilot.phase(args.output, "load", 60):
            models, frozen = load_predictors()
            torch.save(frozen, args.output / "frozen_checkpoints_before.pt")
            e.write_json(
                args.output / "frozen_checkpoint_metadata.json",
                {a: {k: v for k, v in ck.items() if k != "state_dict"} for a, ck in frozen.items()},
            )
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            e.write_json(args.output / "policy_load.json", report)
        factory = natural.trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
        with pilot.phase(args.output, "native_collection", 720):
            for spec in manifest()["rows"]:
                print(
                    f"START F-LAT2 {spec['ordinal']} task={spec['task_id']} state={spec['initial_state_id']}",
                    flush=True,
                )
                record, _ = e.run_episode(
                    spec,
                    args.output / f"episode_{spec['ordinal']:03d}",
                    policy,
                    pre,
                    post,
                    factory,
                    budget,
                    calls,
                    engine_class=natural.NaturalEngine,
                )
                records.append(record)
                print(f"END F-LAT2 {spec['ordinal']} {record['status']}", flush=True)
                if record["status"] != "completed":
                    raise RuntimeError(record.get("first_failure") or "Native collection failed")
        with pilot.phase(args.output, "offline_evaluation", 240):
            samples, selections = extract(records, policy, pre, args.output, counts)
            rows, summary = evaluate(samples, models, policy, args.output, counts)
        after = {
            arm: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for arm, model in models.items()
        }
        for arm, state in after.items():
            for key, value in state.items():
                pilot.require_equal(value, frozen[arm]["state_dict"][key], "Frozen predictor weights changed")
        torch.save(after, args.output / "frozen_weights_after.pt")
        result.update(
            status="completed",
            samples=len(samples),
            selections=selections,
            metric_rows=rows,
            frozen_weights_unchanged=True,
            **summary,
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        result.update(
            execution_head=args.execution_head,
            episodes_completed=sum(r["status"] == "completed" for r in records),
            native_budget=dict(budget.total),
            offline_counts=dict(counts),
            training_updates=0,
            real_robot=0,
            state_mode="current_model_ready_identity",
            predictor_controls_environment=False,
            baseline_qualified=False,
            realtime_qualified=False,
            predictor_benefit_tested=False,
            risk_thresholds=None,
            old_confirmation="not_started_untouched",
            attempts=1,
            retries=0,
        )
        calls.close()
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def update_pending(output, cursors, pending, active):
    for name in ("calls.jsonl", "events.jsonl"):
        path = output / name
        if not path.exists():
            continue
        with path.open() as stream:
            stream.seek(cursors.get(name, 0))
            while line := stream.readline():
                if not line.endswith("\n"):
                    break
                cursors[name] = stream.tell()
                event = json.loads(line)
                if name == "calls.jsonl":
                    if event["event"] == "call_intent":
                        pending[event["call_id"]] = event
                    elif event["event"] in ("call_return", "call_error"):
                        pending.pop(event["call_id"], None)
                elif event["event"] == "started":
                    active[event["phase"]] = event
                else:
                    active.pop(event["phase"], None)


def supervise(args):
    e.require_source(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_frozen_newstate_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use fixed interpreter and fresh exclusive output")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("Registration/readback differs")
    gates = json.loads((PREPARATION / "gates.json").read_text())
    if not all(gates.values()):
        raise ValueError("Preparation failed")
    args.output.mkdir()
    e.write_json(args.output / "manifest.json", manifest())
    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--execution-head",
        args.execution_head,
        "--output",
        str(args.output),
        "--worker",
    ]
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    stop_reason, signaled_at, forced = None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=e.REPO,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"F-LAT2 only queue pid={child.pid} head={args.execution_head}", flush=True)
        try:
            while child.poll() is None:
                update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired_calls = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
                expired_phases = [v for v in active.values() if now - v["at"] > v["limit"]]
                if signaled_at is None and (expired_calls or expired_phases or now - start >= 900):
                    stop_reason = {
                        "expired_calls": expired_calls,
                        "expired_phases": expired_phases,
                        "outer_soft_limit": now - start >= 900,
                    }
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    signaled_at = now
                if now - start >= 930 or (signaled_at is not None and now - signaled_at >= 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            stop_reason = stop_reason or {"supervisor_exception": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        code = child.wait()
    update_pending(args.output, cursors, pending, active)
    path = args.output / "worker_result.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "technical_failure", "first_failure": "Worker result absent"}
    )
    result["execution"] = {
        "command": command,
        "child_pid": child.pid,
        "child_exit_code": code,
        "child_exit_confirmed": True,
        "started_at_utc": utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - start,
        "stop_reason": stop_reason,
        "forced_termination": forced,
        "pending_calls_at_exit": list(pending.values()),
        "active_phases_at_exit": active,
    }
    accounting = natural.trace.journal_accounting(natural.trace.read_events(args.output / "calls.jsonl"))
    result["accounting"] = accounting
    if (
        code
        or stop_reason
        or pending
        or active
        or accounting["call_errors"]
        or not accounting["journal_consistent"]
    ):
        result.update(status="technical_failure", frozen_action_increment_observed=False)
        result["first_failure"] = result.get("first_failure") or result["execution"]
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
                for k in (
                    "status",
                    "episodes_completed",
                    "samples",
                    "frozen_action_increment_observed",
                    "first_failure",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "completed" else 2


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
