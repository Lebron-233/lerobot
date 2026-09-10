"""F-COV1: matched single/multi-state training coverage, development validation only."""

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
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_optimization_probe as opt
import torch

e, pilot, parent = opt.e, opt.pilot, opt.parent
natural = parent.natural
PREP = e.REPO / "outputs/smolvla_coverage_preparation_7b9d527d"
ARMS = ("single_conditioned", "multi_conditioned", "single_no_action", "multi_no_action")
SEED, UPDATES = 20260912, 72
NATIVE_LIMITS = {
    "episodes": (1, 16),
    "settling": (10, 160),
    "measurement": (280, 4480),
    "model": (160, 2560),
    "capture": (2, 32),
}
LIMITS = {"encoding": 80, "decoder": 900, "gradient_decoder": 288, "backward": 288, "updates": 288}


def manifest():
    old = e.fixed_manifest()
    rows = []
    for state, tasks in ((48, range(8)), (49, range(7, -1, -1))):
        for task in tasks:
            row = copy.deepcopy(
                next(
                    r
                    for r in old["rows"]
                    if r["task_id"] == task and r["condition"] == "graph_identity_async"
                )
            )
            row.update(
                ordinal=len(rows),
                pair_index=len(rows),
                initial_state_id=state,
                environment_seed=1000000 + 100 * task + state,
                policy_seed=1010000 + 100 * task + state,
                split="train" if task < 6 else "validation",
                recovery_policy=natural.recovery.RECOVERY_POLICY,
                max_recovery_probes_per_episode=50,
            )
            rows.append(row)
    return {
        "experiment": "F-COV1",
        "rows": rows,
        "arms": list(ARMS),
        "updates_per_arm": UPDATES,
        "seed": SEED,
        "test_reads": 0,
        "max_samples_per_episode": 4,
        "native_limits": NATIVE_LIMITS,
        "offline_limits": LIMITS,
        **{k: old[k] for k in ("policy_revision", "vlm_revision", "assets_revision")},
    }


class Budget(e.Budget):
    def check(self, kind):
        local, total = NATIVE_LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"F-COV1 {kind} budget exhausted")


class Runtime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        if len(self.captures) >= 64:
            raise RuntimeError("F-COV1 offline capture budget exhausted")
        return super()._capture(inputs)


def take(counts, key):
    if counts[key] >= LIMITS[key]:
        raise RuntimeError(f"F-COV1 {key} limit exhausted before dispatch")
    counts[key] += 1


def sample_key(sample):
    return sample["task"], sample["initial_state_id"], sample["request_id"]


def choose_old(samples):
    opt.select_development(samples)
    chosen = []
    for task in range(6):
        group = sorted((s for s in samples if s["task"] == task), key=sample_key)[:4]
        for value in group:
            row = dict(value)
            row["origin"] = "F-ACT1-development"
            chosen.append(row)
    return chosen


def selected_pairs(record, arrays):
    pairs, excluded = pilot.aligned_pairs(record, arrays)
    pairs.sort(key=lambda p: p["request_id"])
    excluded += [{"request_id": p["request_id"], "reason": "fixed_first_four_complete"} for p in pairs[4:]]
    if not pairs:
        raise ValueError("Registered episode has no complete pair; no replacement")
    return pairs[:4], excluded


def pool_for(samples, arm):
    states = (46,) if arm.startswith("single_") else (46, 48, 49)
    pool = sorted(
        (s for s in samples if s["split"] == "train" and s["initial_state_id"] in states), key=sample_key
    )
    if {(s["task"], s["initial_state_id"]) for s in pool} != {(t, st) for t in range(6) for st in states}:
        raise ValueError("Training coverage is incomplete or contains a forbidden task")
    return pool


def schedule(pool, arm):
    states = (46,) if arm.startswith("single_") else (46, 48, 49)
    order = []
    for step in range(UPDATES):
        task, sweep = step % 6, step // 6
        state = states[sweep % len(states)]
        indices = [i for i, s in enumerate(pool) if s["task"] == task and s["initial_state_id"] == state]
        order.append(indices[(sweep // len(states)) % len(indices)])
    return order


def decode(policy, runtime, sample, visual, output, counts, label, gradients=False):
    take(counts, "decoder")
    if gradients:
        take(counts, "gradient_decoder")
    values = tuple(v.cuda() for v in sample["inputs"])
    visual = tuple(v.to(device="cuda", dtype=values[c].dtype) for c, v in enumerate(visual))
    with pilot.phase(output, f"decode_{counts['decoder']}_{label}", 30):
        if gradients:
            decoded = policy.model.sample_actions(
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
                decoded = runtime(
                    None,
                    None,
                    values[4],
                    values[5],
                    values[6],
                    noise=values[7],
                    future_image_tokens=visual,
                    future_image_token_masks=tuple(values[2:4]),
                ).clone()
        if decoded.shape != (1, 50, 32) or not torch.isfinite(decoded).all():
            raise ValueError("Invalid original decoder output")
        return decoded


def extract(records, policy, pre, runtime, output, counts):
    selected, selections = [], []
    for record in sorted(records, key=lambda r: (r["spec"]["task_id"], r["spec"]["initial_state_id"])):
        spec = record["spec"]
        arrays = torch.load(
            output / f"episode_{spec['ordinal']:03d}/arrays.pt", map_location="cpu", weights_only=False
        )
        pairs, excluded = selected_pairs(record, arrays)
        selections.append(
            {"ordinal": spec["ordinal"], "requests": [p["request_id"] for p in pairs], "excluded": excluded}
        )
        for number, pair in enumerate(pairs):
            inputs = tuple(pilot.tensor(v).clone() for v in pair["cached"]["inputs"])
            for kind in ("current", "future") if number == 0 else ("future",):
                take(counts, "encoding")
                with (
                    pilot.phase(output, f"encode_{spec['ordinal']}_{pair['request_id']}_{kind}", 30),
                    torch.no_grad(),
                ):
                    pre.reset()
                    batch = pre(
                        pilot.worker_batch(
                            pair[f"{kind}_observation"], spec["task_name"].replace("_", " "), "cuda"
                        )
                    )
                    images, masks = policy.prepare_images(batch)
                    z, zm = policy.model.encode_image_tokens(images, masks)
                    z, zm = tuple(v.cpu().clone() for v in z), tuple(v.cpu().clone() for v in zm)
                    state = policy.prepare_state(batch).cpu().clone()
                for c in range(2):
                    pilot.require_equal(zm[c], inputs[c + 2], "New camera mask differs")
                    if kind == "current":
                        pilot.require_equal(z[c], inputs[c], "New current tokens differ")
                if kind == "current":
                    pilot.require_equal(state, inputs[6], "New current state differs")
                    counts["current_exact"] += 1
            sample = {
                "origin": "F-COV1",
                "task": spec["task_id"],
                "initial_state_id": spec["initial_state_id"],
                "ordinal": spec["ordinal"],
                "split": spec["split"],
                "inputs": inputs,
                "future": z,
                "archived_full_chunk": pilot.tensor(pair["cached"]["full_chunk"]).clone(),
                **{
                    k: pair[k]
                    for k in ("request_id", "current_index", "future_index", "delay", "actions", "mask")
                },
            }
            with torch.no_grad():
                identity = decode(policy, runtime, sample, inputs[:2], output, counts, "new_identity")
                pilot.require_equal(identity, sample["archived_full_chunk"], "New identity decoder differs")
                sample["oracle"] = (
                    decode(policy, runtime, sample, z, output, counts, "new_oracle").cpu().clone()
                )
            selected.append(sample)
        del arrays, pairs
    torch.save(selected, output / "new_labels.pt")
    e.write_json(output / "selections.json", selections)
    return selected


def metric_values(visual, value, sample):
    latent = pilot.per_sample_mse(
        tuple(v.float() for v in visual),
        tuple(v.to(value.device) for v in sample["future"]),
        tuple(v.to(value.device) for v in sample["inputs"][2:4]),
    ).mean()
    difference = (value[..., :7].float() - sample["oracle"].to(value.device)[..., :7].float()).square()
    return latent, difference[:, 0].mean(), difference.mean()


def summarize_rows(rows):
    result = {}
    for name in ("latent", "row0", "chunk"):
        groups = {}
        for row in rows:
            key = f"{row['task']}/{row['state']}"
            groups.setdefault(key, []).append(row[name])
        means = {k: sum(v) / len(v) for k, v in groups.items()}
        result[name] = {
            "n": len(rows),
            "episodes": len(means),
            "per_episode": means,
            "sample_mean": sum(r[name] for r in rows) / len(rows),
            "episode_macro": sum(means.values()) / len(means),
        }
    return result


def assess(model, arm, samples, policy, runtime, output, counts, label):
    rows, arrays = [], []
    with torch.no_grad():
        for sample in sorted(samples, key=sample_key):
            visual, _ = parent.predict(model, sample, arm)
            value = decode(policy, runtime, sample, visual, output, counts, label)
            values = metric_values(visual, value, sample)
            if label.endswith("_0"):
                pilot.require_equal(
                    value, sample["archived_full_chunk"], "Zero-step output differs from identity"
                )
            row = {
                "task": sample["task"],
                "state": sample["initial_state_id"],
                "request_id": sample["request_id"],
                **{k: float(v) for k, v in zip(("latent", "row0", "chunk"), values, strict=True)},
            }
            rows.append(row)
            arrays.append(
                {**row, "visual": tuple(v.cpu().clone() for v in visual), "output": value.cpu().clone()}
            )
    torch.save(arrays, output / f"assessment_{label}.pt")
    return {"label": label, "rows": rows, "metrics": summarize_rows(rows)}


def gate(checkpoints, histories):
    chosen = {a: next(h for h in histories[a] if h["step"] == checkpoints[a]["best_step"]) for a in ARMS}
    row0 = {a: h["metrics"]["row0"]["episode_macro"] for a, h in chosen.items()}
    baseline = histories[ARMS[0]][0]["metrics"]
    nonzero = (
        checkpoints["multi_conditioned"]["best_step"] > 0
        and checkpoints["multi_conditioned"]["nonzero_residual"]
    )
    return {
        "coverage_row0_improved": row0["multi_conditioned"] < row0["single_conditioned"],
        "action_increment_observed": row0["multi_conditioned"] < row0["multi_no_action"],
        "nonidentity_selected": bool(nonzero),
        "validation_candidate_gate_passed": bool(
            nonzero
            and row0["multi_conditioned"]
            < min(row0["single_conditioned"], row0["multi_no_action"], baseline["row0"]["episode_macro"])
            and chosen["multi_conditioned"]["metrics"]["chunk"]["episode_macro"]
            <= baseline["chunk"]["episode_macro"]
        ),
    }


def train(samples, validation, scales, policy, runtime, output, counts):
    checkpoints, histories, training_results = {}, {}, {}
    for arm in ARMS:
        pool = pool_for(samples, arm)
        order = schedule(pool, arm)
        torch.manual_seed(SEED)
        model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        torch.save(opt.freeze_state(model), output / f"initial_{arm}.pt")
        history, best_weights, best = [], None, float("inf")
        print(f"TRAIN {arm} samples={len(pool)} updates={UPDATES}", flush=True)
        with pilot.phase(output, f"training_{arm}", 600):
            for step in range(UPDATES + 1):
                if step in (0, 36, 72):
                    model.eval()
                    entry = assess(
                        model, arm, validation, policy, runtime, output, counts, f"val_{arm}_{step}"
                    )
                    entry["step"] = step
                    history.append(entry)
                    if entry["metrics"]["row0"]["episode_macro"] < best:
                        best = entry["metrics"]["row0"]["episode_macro"]
                        best_step, best_weights = step, opt.freeze_state(model)
                    e.write_json(output / f"history_{arm}.json", history)
                if step == UPDATES:
                    break
                model.train()
                sample = pool[order[step]]
                with pilot.phase(output, f"update_{arm}_{step}", 30):
                    optimizer.zero_grad(set_to_none=True)
                    visual, _ = parent.predict(model, sample, arm)
                    value = decode(
                        policy, runtime, sample, visual, output, counts, f"train_{arm}", gradients=True
                    )
                    latent, row0, chunk = metric_values(visual, value, sample)
                    loss = parent.objective_loss(latent, row0, "joint_conditioned", scales)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite objective")
                    loss.backward()
                    take(counts, "backward")
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm) or any(p.grad is not None for p in policy.parameters()):
                        raise ValueError("Gradients do not isolate the predictor")
                    optimizer.step()
                    take(counts, "updates")
                    evidence = {
                        "arm": arm,
                        "step": step + 1,
                        "sample": sample_key(sample),
                        "latent": latent.item(),
                        "row0": row0.item(),
                        "chunk": chunk.item(),
                        "objective": loss.item(),
                        "gradient_norm": float(norm),
                        "policy_gradients": 0,
                    }
                    with (output / "training_steps.jsonl").open("a") as stream:
                        stream.write(json.dumps(evidence, allow_nan=False) + "\n")
            model.load_state_dict(best_weights, strict=True)
            model.eval().requires_grad_(False)
            nonzero = any(
                bool(best_weights[k].count_nonzero()) for k in ("up_projection.weight", "up_projection.bias")
            )
            checkpoint = {
                "arm": arm,
                "config": asdict(pilot.config()),
                "seed": SEED,
                "best_step": best_step,
                "validation_row0": best,
                "nonzero_residual": nonzero,
                "state_dict": best_weights,
            }
            torch.save(checkpoint, output / f"{arm}.pt")
            checkpoints[arm] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
            histories[arm] = history
            training_results[arm] = assess(
                model, arm, pool, policy, runtime, output, counts, f"train_selected_{arm}"
            )
        print(f"END {arm} best_step={best_step} validation_row0={best}", flush=True)
    return {
        "checkpoints": checkpoints,
        "histories": histories,
        "training_selected": training_results,
        **gate(checkpoints, histories),
    }


def worker(args):
    opt.source_gate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    result, records, counts, budget = (
        {"status": "technical_failure", "first_failure": None, "experiment": "F-COV1"},
        [],
        Counter(),
        Budget(),
    )
    calls = e.Calls(args.output / "calls.jsonl")
    runtime = None
    try:
        with pilot.phase(args.output, "load", 60):
            old = torch.load(opt.LABELS, map_location="cpu", weights_only=False)
            old_selected, scales = choose_old(old), parent.train_scales(old)
            e.write_json(args.output / "scales.json", scales)
            torch.save(old_selected, args.output / "old_selected.pt")
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Fixed GPU differs")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            e.write_json(args.output / "policy_load.json", report)
        factory = natural.trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
        with pilot.phase(args.output, "native_collection", 1440):
            for spec in manifest()["rows"]:
                print(
                    f"START F-COV1 {spec['ordinal']} task={spec['task_id']} state={spec['initial_state_id']}",
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
                print(f"END F-COV1 {spec['ordinal']} {record['status']}", flush=True)
                if record["status"] != "completed":
                    raise RuntimeError(record["first_failure"])
        runtime = Runtime(policy.model)
        with pilot.phase(args.output, "labels", 300), torch.no_grad():
            for task in range(6):
                sample = next(s for s in old_selected if s["task"] == task)
                for label, z, target in (
                    ("identity", sample["inputs"][:2], sample["archived_full_chunk"]),
                    ("oracle", sample["future"], sample["oracle"]),
                ):
                    value = decode(policy, runtime, sample, z, args.output, counts, f"old_{label}")
                    pilot.require_equal(value, target, "Old training reference no longer reproduces")
                    counts["old_reference_exact"] += 1
            new = extract(records, policy, pre, runtime, args.output, counts)
        training = old_selected + [s for s in new if s["split"] == "train"]
        validation = [s for s in new if s["split"] == "validation"]
        e.write_json(
            args.output / "data_summary.json",
            {
                "single_train": len(old_selected),
                "multi_train": len(training),
                "validation": len(validation),
                "new_samples": len(new),
            },
        )
        result.update(train(training, validation, scales, policy, runtime, args.output, counts))
        result.update(
            status="completed",
            vla_frozen=all(not p.requires_grad and p.grad is None for p in policy.parameters()),
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if runtime is not None:
            e.write_json(args.output / "offline_captures.json", runtime.captures)
            runtime.release_graph()
            counts["offline_captures"] = len(runtime.captures)
        calls.close()
        result.update(
            execution_head=args.execution_head,
            counts=dict(counts),
            native_budget=dict(budget.total),
            episodes_completed=sum(r["status"] == "completed" for r in records),
            test_reads=0,
            predictor_controls_environment=False,
            baseline_qualified=False,
            realtime_qualified=False,
            predictor_benefit_tested=False,
            risk_thresholds=None,
            old_confirmation="not_started_untouched",
            attempts=1,
            retries=0,
        )
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    opt.source_gate(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_coverage_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use fixed interpreter and fresh registered output")
    registration = json.loads((PREP / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREP / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("Registration differs")
    if not all(json.loads((PREP / "gates.json").read_text()).values()):
        raise ValueError("Preparation failed")
    args.output.mkdir()
    e.write_json(args.output / "manifest.json", manifest())
    cmd = [
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
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            cmd, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"F-COV1 worker pid={child.pid}", flush=True)
        try:
            while child.poll() is None:
                parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
                phases = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or phases or now - start > 2400):
                    reason = {
                        "expired_calls": expired,
                        "expired_phases": phases,
                        "outer_soft": now - start > 2400,
                    }
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - start > 2430 or (terminated is not None and now - terminated > 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            reason = {"supervisor_error": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        code = child.wait()
    parent.previous.update_pending(args.output, cursors, pending, active)
    path = args.output / "worker_result.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "technical_failure", "first_failure": "Worker receipt absent"}
    )
    result["execution"] = {
        "command": cmd,
        "child_pid": child.pid,
        "child_exit_code": code,
        "exit_confirmed": True,
        "started_at_utc": utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - start,
        "stop_reason": reason,
        "forced_termination": forced,
        "pending_calls_at_exit": list(pending.values()),
        "active_phases_at_exit": active,
    }
    accounting = natural.trace.journal_accounting(natural.trace.read_events(args.output / "calls.jsonl"))
    result["accounting"] = accounting
    if (
        code
        or reason
        or pending
        or active
        or accounting["unknown_calls"]
        or accounting["call_errors"]
        or not accounting["journal_consistent"]
    ):
        result.update(status="technical_failure", validation_candidate_gate_passed=False)
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
                    "counts",
                    "validation_candidate_gate_passed",
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
