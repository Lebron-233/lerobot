"""F-ACT1: new-data 2x2 comparison of token and first-action supervision."""

import argparse
import copy
import faulthandler
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

import libero_frozen_newstate as previous
import libero_future_latent_pilot as pilot
import libero_graph_natural_native as natural
import torch

e = natural.e
PREPARATION = e.REPO / "outputs/smolvla_action_objective_preparation_b7f0831f"
ARMS = ("token_conditioned", "token_no_action", "joint_conditioned", "joint_no_action")
SEED, UPDATES = 20260912, 60
LIMITS = {
    "episodes": (1, 12),
    "settling": (10, 120),
    "measurement": (280, 3360),
    "model": (160, 1920),
    "capture": (2, 24),
}


def manifest():
    original = e.fixed_manifest()
    rows = []
    cases = [(t, 46) for t in range(8)] + [(8, 46), (9, 46), (9, 47), (8, 47)]
    for task, state in cases:
        row = copy.deepcopy(
            next(
                r
                for r in original["rows"]
                if r["task_id"] == task and r["condition"] == "graph_identity_async"
            )
        )
        row.update(
            ordinal=len(rows),
            pair_index=len(rows),
            initial_state_id=state,
            environment_seed=980000 + 100 * task + state,
            policy_seed=990000 + 100 * task + state,
            recovery_policy=natural.recovery.RECOVERY_POLICY,
            max_recovery_probes_per_episode=50,
            split="train" if task < 6 else "validation" if task < 8 else "test",
        )
        rows.append(row)
    return {
        "experiment": "F-ACT1",
        "rows": rows,
        "arms": list(ARMS),
        "seed": SEED,
        "updates_per_arm": UPDATES,
        "max_pairs_per_episode": 8,
        "limits": {k: {"per_episode": v[0], "total": v[1]} for k, v in LIMITS.items()},
        **{k: original[k] for k in ("policy_revision", "vlm_revision", "assets_revision")},
    }


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"F-ACT1 {kind} budget exhausted before dispatch")


def chosen_pairs(record, arrays):
    pairs, excluded = pilot.aligned_pairs(record, arrays)
    pairs.sort(key=lambda p: p["request_id"])
    excluded += [{"request_id": p["request_id"], "reason": "fixed_first_8_complete"} for p in pairs[8:]]
    if not pairs:
        raise ValueError("A registered episode has no complete sample")
    return pairs[:8], excluded


def macro(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["ordinal"]].append(float(row[key]))
    per_episode = {str(k): sum(v) / len(v) for k, v in grouped.items()}
    return {
        "n": len(rows),
        "episodes": len(grouped),
        "per_episode": per_episode,
        "sample_mean": sum(float(r[key]) for r in rows) / len(rows) if rows else None,
        "episode_macro": sum(per_episode.values()) / len(per_episode) if per_episode else None,
    }


def aggregate(rows):
    metrics = {
        f"{a}_{m}": macro(rows, f"{a}_{m}")
        for a in ("identity", *ARMS)
        for m in ("latent_mse", "row0_oracle_mse", "chunk_oracle_mse")
    }
    complete = bool(rows) and len({r["ordinal"] for r in rows}) == 4
    key = "joint_conditioned_row0_oracle_mse"

    def better(a):
        return complete and metrics[key]["episode_macro"] < metrics[f"{a}_row0_oracle_mse"]["episode_macro"]

    objective = bool(better("token_conditioned"))
    increment = bool(better("joint_no_action"))
    gate = bool(
        objective
        and increment
        and better("identity")
        and metrics["joint_conditioned_chunk_oracle_mse"]["episode_macro"]
        <= metrics["identity_chunk_oracle_mse"]["episode_macro"]
    )
    return {
        "metrics": metrics,
        "action_objective_row0_improved": objective,
        "joint_action_input_increment_observed": increment,
        "joint_candidate_gate_passed": gate,
    }


def predict(model, sample, arm):
    batch = pilot.batch_for([sample], [0], "cuda" if next(model.parameters()).is_cuda else "cpu")
    context = "conditioned" if arm.endswith("_conditioned") else "no_action"
    return pilot.prediction(model, batch, context), batch


def objective_loss(latent, row0, arm, scales):
    loss = latent / scales["latent"]
    if arm.startswith("joint_"):
        if row0 is None:
            raise ValueError("Joint objective needs the real decoder row0 loss")
        loss = loss + row0 / scales["row0"]
    return loss


def extract(records, policy, pre, output, counts, split_label):
    samples, selections = [], []
    for record in records:
        spec = record["spec"]
        arrays = torch.load(
            output / f"episode_{spec['ordinal']:03d}/arrays.pt", map_location="cpu", weights_only=False
        )
        pairs, excluded = chosen_pairs(record, arrays)
        selections.append(
            {
                "ordinal": spec["ordinal"],
                "split": spec["split"],
                "request_ids": [p["request_id"] for p in pairs],
                "excluded": excluded,
            }
        )
        for number, pair in enumerate(pairs):
            inputs = tuple(pilot.tensor(v).clone() for v in pair["cached"]["inputs"])
            for kind in ("current", "future") if number == 0 else ("future",):
                if counts["encoding_batches"] >= 108:
                    raise RuntimeError("F-ACT1 encoding budget exhausted")
                with (
                    pilot.phase(output, f"encode_{spec['ordinal']}_{pair['request_id']}_{kind}", 30),
                    torch.no_grad(),
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
                    z, zm = (
                        tuple(v.detach().cpu().clone() for v in z),
                        tuple(v.detach().cpu().clone() for v in zm),
                    )
                    state = policy.prepare_state(batch).detach().cpu().clone()
                for c in range(2):
                    pilot.require_equal(zm[c], inputs[c + 2], "Camera validity changed")
                    if kind == "current":
                        pilot.require_equal(z[c], inputs[c], "Current tokens differ from native archive")
                if kind == "current":
                    pilot.require_equal(state, inputs[6], "Current state differs from native archive")
                    counts["current_exact_episodes"] += 1
            samples.append(
                {
                    "task": spec["task_id"],
                    "ordinal": spec["ordinal"],
                    "split": spec["split"],
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
    torch.save(samples, output / f"{split_label}_cache.pt")
    e.write_json(output / f"{split_label}_selection.json", selections)
    return samples


class OfflineRuntime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        if len(self.captures) >= 34:
            raise RuntimeError("F-ACT1 offline capture budget exhausted")
        return super()._capture(inputs)


def decode(policy, runtime, sample, visual, output, counts, label, *, gradients=False):
    if counts["decoder_calls"] >= 640:
        raise RuntimeError("F-ACT1 offline decoder budget exhausted")
    values = tuple(v.to("cuda") for v in sample["inputs"])
    visual = tuple(v.to(device="cuda", dtype=values[c].dtype) for c, v in enumerate(visual))
    with pilot.phase(output, f"decode_{counts['decoder_calls']}_{label}", 30):
        counts["decoder_calls"] += 1
        counts["differentiable_decoder_calls" if gradients else "graph_decoder_calls"] += 1
        if gradients:
            value = policy.model.sample_actions(
                None,
                None,
                values[4],
                values[5],
                values[6],
                noise=values[7].clone(),
                future_image_tokens=visual,
                future_image_token_masks=tuple(values[2:4]),
            )
        else:
            runtime.begin_episode("graph", sample["task"])
            with torch.no_grad():
                value = runtime(
                    None,
                    None,
                    values[4],
                    values[5],
                    values[6],
                    noise=values[7],
                    future_image_tokens=visual,
                    future_image_token_masks=tuple(values[2:4]),
                ).clone()
        if value.shape != (1, 50, 32) or not torch.isfinite(value).all():
            raise ValueError("Invalid full decoder output")
        return value


def annotate(policy, runtime, samples, output, counts):
    with torch.no_grad():
        for sample in samples:
            identity = decode(policy, runtime, sample, sample["inputs"][:2], output, counts, "label_identity")
            pilot.require_equal(
                identity, sample["archived_full_chunk"], "Reference identity differs from native archive"
            )
            oracle = decode(policy, runtime, sample, sample["future"], output, counts, "label_oracle")
            sample["oracle"] = oracle.detach().cpu().clone()
    torch.save(samples, output / "development_labels.pt")


def train_scales(samples):
    selected = [s for s in samples if s["split"] == "train"]
    if not selected:
        raise ValueError("No training samples")
    z_errors, a_errors = [], []
    for s in selected:
        z_errors.append(
            pilot.per_sample_mse(
                tuple(v.float() for v in s["inputs"][:2]), s["future"], s["inputs"][2:4]
            ).item()
        )
        a_errors.append(
            (s["archived_full_chunk"][:, 0, :7].float() - s["oracle"][:, 0, :7].float())
            .square()
            .mean()
            .item()
        )
    return {
        "latent": max(sum(z_errors) / len(z_errors), 1e-12),
        "row0": max(sum(a_errors) / len(a_errors), 1e-12),
        "train_count": len(selected),
    }


def validate(model, arm, samples, policy, runtime, output, counts, step):
    rows, saved = [], []
    with torch.no_grad():
        for sample in samples:
            visual, batch = predict(model, sample, arm)
            value = decode(policy, runtime, sample, visual, output, counts, f"validation_{arm}_{step}")
            error = (
                (value[:, 0, :7].float() - sample["oracle"][:, 0, :7].to("cuda").float())
                .square()
                .mean()
                .item()
            )
            rows.append({"ordinal": sample["ordinal"], "request_id": sample["request_id"], "row0": error})
            saved.append(
                {
                    "ordinal": sample["ordinal"],
                    "request_id": sample["request_id"],
                    "output": value.detach().cpu().clone(),
                    "visual": tuple(v.cpu().clone() for v in visual),
                }
            )
    torch.save(saved, output / f"validation_{arm}_{step}.pt")
    return {"step": step, "row0_macro": macro(rows, "row0")["episode_macro"], "rows": rows}


def train(samples, policy, runtime, output, counts):
    training = [s for s in samples if s["split"] == "train"]
    validation = [s for s in samples if s["split"] == "validation"]
    scales = train_scales(samples)
    e.write_json(output / "train_scales.json", scales)
    models, checkpoints = {}, {}
    for arm in ARMS:
        with pilot.phase(output, f"train_{arm}", 600):
            torch.manual_seed(SEED)
            model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            generator = torch.Generator().manual_seed(SEED + 1)
            best, best_step, best_weights = float("inf"), None, None
            history = []
            for step in range(UPDATES + 1):
                if step % 30 == 0:
                    model.eval()
                    entry = validate(model, arm, validation, policy, runtime, output, counts, step)
                    history.append(entry)
                    if entry["row0_macro"] < best:
                        best, best_step = entry["row0_macro"], step
                        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step == UPDATES:
                    break
                model.train()
                sample_index = int(torch.randint(len(training), (1,), generator=generator))
                sample = training[sample_index]
                with pilot.phase(output, f"update_{arm}_{step}", 30):
                    optimizer.zero_grad(set_to_none=True)
                    visual, batch = predict(model, sample, arm)
                    latent = pilot.per_sample_mse(visual, batch["future"], batch["masks"]).mean()
                    row0, action_gradient = None, None
                    if arm.startswith("joint_"):
                        value = decode(
                            policy,
                            runtime,
                            sample,
                            visual,
                            output,
                            counts,
                            f"train_{arm}_{step}",
                            gradients=True,
                        )
                        row0 = (
                            (value[:, 0, :7].float() - sample["oracle"][:, 0, :7].to("cuda").float())
                            .square()
                            .mean()
                        )
                        if step == 0:
                            pilot.require_equal(
                                value,
                                sample["archived_full_chunk"],
                                "Zero-residual differentiable decoder differs from native archive",
                            )
                            grads = torch.autograd.grad(row0, visual, retain_graph=True)
                            action_gradient = (
                                sum(g.detach().float().square().sum().item() for g in grads) ** 0.5
                            )
                            if not action_gradient > 0:
                                raise ValueError("The action-only derivative to visual tokens is absent")
                            counts["action_only_vjp"] += 1
                    loss = objective_loss(latent, row0, arm, scales)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training objective")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm) or any(p.grad is not None for p in policy.parameters()):
                        raise ValueError("Training did not isolate finite predictor gradients")
                    optimizer.step()
                    counts["training_updates"] += 1
                    if row0 is not None:
                        counts["decoder_backwards"] += 1
                    evidence = {
                        "arm": arm,
                        "step": step + 1,
                        "sample_index": sample_index,
                        "ordinal": sample["ordinal"],
                        "request_id": sample["request_id"],
                        "latent_loss": latent.item(),
                        "row0_loss": None if row0 is None else row0.item(),
                        "objective": loss.item(),
                        "gradient_norm": float(norm),
                        "action_only_token_gradient_norm": action_gradient,
                        "policy_gradients": 0,
                    }
                    with (output / "training_steps.jsonl").open("a") as stream:
                        stream.write(json.dumps(evidence, allow_nan=False) + "\n")
                        stream.flush()
            checkpoint = {
                "arm": arm,
                "seed": SEED,
                "config": asdict(pilot.config()),
                "best_step": best_step,
                "validation_row0": best,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "state_dict": best_weights,
            }
            torch.save(checkpoint, output / f"{arm}.pt")
            e.write_json(output / f"{arm}_validation.json", history)
            model.load_state_dict(best_weights, strict=True)
            models[arm] = model.eval().requires_grad_(False)
            checkpoints[arm] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    e.write_json(output / "frozen_checkpoints.json", checkpoints)
    return models, checkpoints


def evaluate(samples, models, policy, runtime, output, counts):
    rows, saved = [], []
    with torch.no_grad():
        for sample in sorted(samples, key=lambda s: (s["task"], s["ordinal"], s["request_id"])):
            visual = {
                "identity": tuple(v.float().cuda() for v in sample["inputs"][:2]),
                **{arm: predict(model, sample, arm)[0] for arm, model in models.items()},
                "oracle": tuple(v.cuda() for v in sample["future"]),
            }
            values = {
                arm: decode(policy, runtime, sample, z, output, counts, f"test_{arm}").cpu().clone()
                for arm, z in visual.items()
            }
            pilot.require_equal(
                values["identity"], sample["archived_full_chunk"], "Test identity differs from native archive"
            )
            counts["test_identity_exact"] += 1
            row = {k: sample[k] for k in ("task", "ordinal", "request_id", "initial_state_id", "delay")}
            for arm in ("identity", *ARMS):
                z = tuple(v.cpu().float() for v in visual[arm])
                row[f"{arm}_latent_mse"] = pilot.per_sample_mse(
                    z, sample["future"], sample["inputs"][2:4]
                ).item()
                diff = (values[arm][..., :7].float() - values["oracle"][..., :7].float()).square()
                row[f"{arm}_row0_oracle_mse"], row[f"{arm}_chunk_oracle_mse"] = (
                    diff[:, 0].mean().item(),
                    diff.mean().item(),
                )
            rows.append(row)
            saved.append(
                {
                    "ordinal": sample["ordinal"],
                    "request_id": sample["request_id"],
                    "visual": {k: tuple(v.cpu().clone() for v in z) for k, z in visual.items()},
                    "outputs": values,
                }
            )
    torch.save(saved, output / "test_outputs.pt")
    return rows, aggregate(rows)


def worker(args):
    e.require_source(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, budget, records = Counter(), Budget(), []
    calls = e.Calls(args.output / "calls.jsonl")
    result = {"status": "technical_failure", "first_failure": None, "experiment": "F-ACT1"}
    runtime = None
    try:
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("The registered GPU changed")
        with pilot.phase(args.output, "load", 60):
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            e.write_json(args.output / "policy_load.json", report)
        factory = natural.trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
        with pilot.phase(args.output, "native_collection", 960):
            for spec in manifest()["rows"]:
                print(
                    f"START F-ACT1 {spec['ordinal']} task={spec['task_id']} state={spec['initial_state_id']} {spec['split']}",
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
                print(f"END F-ACT1 {spec['ordinal']} {record['status']}", flush=True)
                if record["status"] != "completed":
                    raise RuntimeError(record["first_failure"])
        runtime = OfflineRuntime(policy.model)
        with pilot.phase(args.output, "development", 300):
            dev = extract(records[:8], policy, pre, args.output, counts, "development")
            annotate(policy, runtime, dev, args.output, counts)
        models, checkpoints = train(dev, policy, runtime, args.output, counts)
        print("All four checkpoints frozen; starting new held-out evaluation.", flush=True)
        with pilot.phase(args.output, "test", 300):
            test = extract(records[8:], policy, pre, args.output, counts, "test")
            rows, summary = evaluate(test, models, policy, runtime, args.output, counts)
        result.update(
            status="completed",
            checkpoints=checkpoints,
            metric_rows=rows,
            **summary,
            split_counts=dict(Counter(s["split"] for s in dev + test)),
            vla_parameters_frozen=all(not p.requires_grad and p.grad is None for p in policy.parameters()),
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if runtime is not None:
            e.write_json(args.output / "offline_captures.json", runtime.captures)
            counts["offline_captures"] = len(runtime.captures)
            runtime.release_graph()
        calls.close()
        result.update(
            execution_head=args.execution_head,
            native_budget=dict(budget.total),
            counts=dict(counts),
            episodes_completed=sum(r["status"] == "completed" for r in records),
            predictor_controls_environment=False,
            state_mode="current_model_ready_identity",
            real_robot=0,
            attempts=1,
            retries=0,
            baseline_qualified=False,
            realtime_qualified=False,
            predictor_benefit_tested=False,
            risk_thresholds=None,
            old_confirmation="not_started_untouched",
        )
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    e.require_source(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_action_objective_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use fixed interpreter and exclusive output")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("Registration/readback differs")
    if not all(json.loads((PREPARATION / "gates.json").read_text()).values()):
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
        print(f"F-ACT1 only queue pid={child.pid} head={args.execution_head}", flush=True)
        try:
            while child.poll() is None:
                previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
                phases = [v for v in active.values() if now - v["at"] > v["limit"]]
                if signaled_at is None and (expired or phases or now - start >= 1800):
                    stop_reason = {
                        "expired_calls": expired,
                        "expired_phases": phases,
                        "outer_soft": now - start >= 1800,
                    }
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    signaled_at = now
                if now - start >= 1830 or (signaled_at is not None and now - signaled_at >= 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            stop_reason = stop_reason or {"exception": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        code = child.wait()
    previous.update_pending(args.output, cursors, pending, active)
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
        "exit_confirmed": True,
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
        or accounting["unknown_calls"]
        or not accounting["journal_consistent"]
    ):
        result.update(status="technical_failure", joint_candidate_gate_passed=False)
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
                    "split_counts",
                    "joint_candidate_gate_passed",
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
