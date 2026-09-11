"""F-ACR1: frozen no-action base plus centered or ordinary action residual."""

import argparse
import faulthandler
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

import libero_prefix_mismatch as pfx
import torch

scale, cov, e, pilot, opt = pfx.scale, pfx.cov, pfx.e, pfx.pilot, pfx.opt
PREP = e.REPO / "outputs/smolvla_action_centered_preparation_11c81ae1"
BASE = e.REPO / "outputs/smolvla_prefix_mismatch_a6a6e997"
ARMS = ("centered", "ordinary")
STEPS = (0, 36, 72)
SEED = 20260912
LIMITS = {"predictor_forwards": 1143, "decoder": 850, "updates": 144, "backward": 144}
PENDING = opt.PENDING_DOCS | {
    "docs/experiments/SMOLVLA_PREFIX_MISMATCH_STATUS.md",
    "docs/experiments/SMOLVLA_PREFIX_MISMATCH_RESULT.md",
    "docs/experiments/SMOLVLA_PREFIX_MISMATCH_NEXT.md",
}


def source_gate(head):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=e.REPO, text=True).strip()
    if actual != head:
        raise ValueError("Execution HEAD changed")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=e.REPO, text=True)
    if any(line[3:] not in PENDING or line[:2] not in (" M", "??") for line in status.splitlines()):
        raise ValueError(f"Unexpected source change: {status}")


def take(counts, name):
    if counts[name] >= LIMITS[name]:
        raise RuntimeError(f"F-ACR1 {name} budget exhausted")
    counts[name] += 1


def compose(base, action_delta, zero_delta, arm):
    if arm not in ARMS:
        raise ValueError("Unknown residual arm")
    # Subtract before addition: base + h(a) - h(0) could round even for a=0.
    return tuple(b + (a - z if arm == "centered" else a)
                 for b, a, z in zip(base, action_delta, zero_delta, strict=True))


def load_data():
    accepted = json.loads((BASE / "independent_audit.json").read_text())
    if accepted["independent_contract_accepted"] is not True:
        raise ValueError("Frozen base experiment is not independently accepted")
    training, validation = pfx.load_data()
    manifest = pfx.mapping_contract(training, validation)
    records = torch.load(BASE / "predictions.pt", map_location="cpu", weights_only=False)
    bases = {tuple(r["key"]): r for r in records if r["context"] == "no_action"}
    if len(bases) != 88 or set(bases) != {pfx.key(s) for s in training + validation}:
        raise ValueError("No-action base coverage changed")
    for r in bases.values():
        for raw, quantized in zip(r["raw_visual"], r["visual"], strict=True):
            pilot.require_equal(raw.to(torch.bfloat16).float(), quantized, "Base precision changed")
    weights = json.loads((pfx.SOURCE / "training_weights.json").read_text())
    if [r["key"] for r in weights["rows"]] != [list(pfx.key(s)) for s in training]:
        raise ValueError("Frozen case weight identities changed")
    expected_hashes = json.loads((pfx.PREP / "preparation.json").read_text())["source_hashes"]
    if {str(p): scale.file_digest(p) for p in (scale.OLD, scale.NEW)} != expected_hashes:
        raise ValueError("Frozen development labels changed")
    return training, validation, manifest, bases, weights


def prepare():
    if PREP.exists():
        raise ValueError("Preparation directory already exists")
    training, validation, manifest, _, weights = load_data()
    PREP.mkdir()
    e.write_json(PREP / "manifest.json", manifest)
    e.write_json(PREP / "weights.json", weights)
    e.write_json(PREP / "preparation.json", {
        "experiment": "F-ACR1", "train": len(training), "validation": len(validation),
        "base_source": str(BASE), "config": asdict(pilot.config()), "seed": SEED,
        "limits": LIMITS, "cuda_initialized": torch.cuda.is_initialized(),
        "new_model_forward": 0, "new_env": 0, "test_reads": 0,
    })
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation initialized CUDA")
    print(json.dumps({"prepared": True, "train": 72, "validation": 16, "limits": LIMITS}), flush=True)


def predict(model, sample, base, arm, actions, counts):
    batch = pilot.batch_for([sample], [0], "cuda")
    arguments = (batch["tokens"], batch["masks"])
    tail = (batch["action_mask"], batch["state"], batch["delay"])
    take(counts, "predictor_forwards")
    delta = model(*arguments, actions.to("cuda"), *tail).delta_tokens
    if arm == "centered":
        take(counts, "predictor_forwards")
        zero = model(*arguments, torch.zeros_like(batch["actions"]), *tail).delta_tokens
    else:
        zero = tuple(torch.zeros_like(z) for z in delta)
    raw = compose(tuple(z.cuda() for z in base["raw_visual"]), delta, zero, arm)
    visual = tuple(z.to(torch.bfloat16).float() for z in raw)
    evidence = {
        "action_delta": tuple(z.detach().cpu().clone() for z in delta),
        "zero_delta": tuple(z.detach().cpu().clone() for z in zero),
        "raw_visual": tuple(z.detach().cpu().clone() for z in raw),
        "visual": tuple(z.detach().cpu().clone() for z in visual),
        "actual_actions": actions.cpu().clone(), "actual_mask": sample["mask"].clone(),
    }
    return visual, evidence


def decoded(policy, runtime, sample, visual, output, counts, label, gradients=False):
    if counts["decoder"] >= LIMITS["decoder"]:
        raise RuntimeError("F-ACR1 decoder budget exhausted")
    return cov.decode(policy, runtime, sample, visual, output, counts, label, gradients)


def row(sample, arm, context, evidence, value, metrics):
    return {"key": list(pfx.key(sample)), "split": sample["split"], "arm": arm, "context": context,
            **evidence, "output": value.detach().cpu().clone(),
            "metrics": {k: float(v) for k, v in zip(("latent", "row0", "chunk"), metrics, strict=True)}}


def assess(model, arm, samples, bases, donors, policy, runtime, output, counts, label, contexts):
    records = []
    with torch.no_grad():
        for sample in sorted(samples.values(), key=pfx.key):
            k, base = pfx.key(sample), bases[pfx.key(sample)]
            for context in contexts:
                if context == "mismatched" and donors[k] is None:
                    continue
                actions = (torch.zeros_like(sample["actions"]) if context == "zero" else
                           samples[donors[k]]["actions"] if context == "mismatched" else sample["actions"])
                with pilot.phase(output, f"predict_{counts['predictor_forwards']}_{label}_{context}", 30):
                    visual, evidence = predict(model, sample, base, arm, actions, counts)
                value = decoded(policy, runtime, sample, visual, output, counts, label)
                record = row(sample, arm, context, evidence, value, cov.metric_values(visual, value, sample))
                records.append(record)
                if context == "zero" and arm == "centered" or label.endswith("_0"):
                    for actual, expected in zip(evidence["raw_visual"], base["raw_visual"], strict=True):
                        pilot.require_equal(actual, expected, "Zero-input or initial base tokens differ")
                    pilot.require_equal(value, base["output"], "Zero-input or initial base output differs")
                    counts["base_invariant_exact"] += 1
    torch.save(records, output / f"assessment_{label}.pt")
    metrics = cov.summarize_rows([
        {"task": r["key"][0], "state": r["key"][1], **r["metrics"]}
        for r in records if r["context"] == "true"
    ])
    return metrics


def worker(args):
    source_gate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime, histories, checkpoints = Counter(), None, {}, {}
    result = {"experiment": "F-ACR1", "status": "technical_failure", "first_failure": None}
    try:
        with pilot.phase(args.output, "load_development", 30):
            training, validation, manifest, bases, weights = load_data()
            if manifest != json.loads((PREP / "manifest.json").read_text()):
                raise ValueError("Registered donor map changed")
            if weights != json.loads((PREP / "weights.json").read_text()):
                raise ValueError("Registered weights changed")
            e.write_json(args.output / "manifest.json", manifest)
            e.write_json(args.output / "training_weights.json", weights)
            samples = {pfx.key(s): s for s in training + validation}
            donors = pfx.donor_map(training + validation)
        with pilot.phase(args.output, "load_model", 90):
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Registered GPU differs")
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts["vla_loads"] += 1
            e.write_json(args.output / "policy_load.json", report)
            runtime = cov.Runtime(policy.model)
        references = []
        with torch.no_grad():
            for sample in sorted(samples.values(), key=pfx.key):
                base = bases[pfx.key(sample)]
                value = decoded(policy, runtime, sample, base["visual"], args.output, counts, "base_reference")
                pilot.require_equal(value, base["output"], "Frozen base replay differs")
                references.append({"key": list(pfx.key(sample)), "output": value.cpu().clone()})
                counts["base_reference_exact"] += 1
        torch.save(references, args.output / "base_references.pt")
        for arm in ARMS:
            torch.manual_seed(SEED)
            model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
            counts["branch_initializations"] += 1
            torch.save(opt.freeze_state(model), args.output / f"initial_{arm}.pt")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            best, best_step, best_weights, history, updates = float("inf"), None, None, [], []
            order = cov.schedule(training, "multi_conditioned")
            print(f"F-ACR1 TRAIN {arm}", flush=True)
            for step in range(73):
                if step in STEPS:
                    model.eval()
                    metrics = assess(model, arm, {pfx.key(s): s for s in validation}, bases, donors,
                                     policy, runtime, args.output, counts, f"val_{arm}_{step}", ("true",))
                    score = metrics["row0"]["episode_macro"]
                    history.append({"step": step, "metrics": metrics})
                    torch.save(opt.freeze_state(model), args.output / f"weights_{arm}_{step}.pt")
                    if score < best:
                        best, best_step, best_weights = score, step, opt.freeze_state(model)
                    e.write_json(args.output / f"history_{arm}.json", history)
                    print(f"F-ACR1 {arm} step={step} validation_row0={score:.12f}", flush=True)
                if step == 72:
                    break
                sample = training[order[step]]
                weight = weights["rows"][order[step]]["case_weight"]
                with pilot.phase(args.output, f"update_{arm}_{step + 1}", 30):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    visual, evidence = predict(model, sample, bases[pfx.key(sample)], arm, sample["actions"], counts)
                    value = decoded(policy, runtime, sample, visual, args.output, counts, f"train_{arm}", True)
                    metrics = cov.metric_values(visual, value, sample)
                    loss = metrics[0] / weights["scales"]["latent"] + weight * metrics[1] / weights["scales"]["row0"]
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite objective")
                    take(counts, "backward")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm) or any(p.grad is not None for p in policy.parameters()):
                        raise ValueError("Nonfinite branch gradients or unfrozen VLA")
                    take(counts, "updates")
                    optimizer.step()
                    evidence_row = row(sample, arm, "true", evidence, value, metrics)
                    evidence_row.update(step=step + 1, weight=weight, objective=float(loss), gradient_norm=float(norm))
                    updates.append(evidence_row)
                    with (args.output / "training_steps.jsonl").open("a") as stream:
                        stream.write(json.dumps({k: evidence_row[k] for k in
                            ("arm", "step", "key", "weight", "objective", "gradient_norm", "metrics")}) + "\n")
            torch.save(updates, args.output / f"training_{arm}.pt")
            model.load_state_dict(best_weights, strict=True)
            model.eval().requires_grad_(False)
            checkpoint = {"arm": arm, "best_step": best_step, "validation_row0": best,
                          "seed": SEED, "config": asdict(pilot.config()), "state_dict": best_weights}
            torch.save(checkpoint, args.output / f"{arm}.pt")
            checkpoints[arm] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
            histories[arm] = history
            assess(model, arm, samples, bases, donors, policy, runtime, args.output, counts,
                   f"selected_{arm}", ("true", "zero", "mismatched"))
            print(f"F-ACR1 END {arm} best_step={best_step}", flush=True)
            del model, optimizer, updates
        if any(counts[k] != v for k, v in LIMITS.items()):
            raise ValueError(f"Formal budget incomplete: {dict(counts)}")
        result.update(status="completed", histories=histories, checkpoints=checkpoints,
                      vla_frozen=all(not p.requires_grad and p.grad is None for p in policy.parameters()))
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if runtime is not None:
            e.write_json(args.output / "offline_captures.json", runtime.captures)
            counts["offline_captures"] = len(runtime.captures)
            runtime.release_graph()
            result["graph_released"] = runtime.graph is None
        result.update(execution_head=args.execution_head, counts=dict(counts), attempts=1, retries=0,
                      new_env=0, new_native=0, visual_encodings=0, test_reads=0, real_robot=0,
                      baseline_qualified=False, realtime_qualified=False, predictor_benefit_tested=False,
                      risk_thresholds=None, old_confirmation="not_started_untouched")
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    source_gate(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_action_centered_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use fixed interpreter and new registered output")
    registration = json.loads((PREP / "registration_readback.json").read_text())
    body = (PREP / "registration.md").read_text()
    if registration["body"] != body or args.execution_head not in body or str(args.output) not in body:
        raise ValueError("Registration exact gate failed")
    args.output.mkdir()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
               "--execution-head", args.execution_head, "--output", str(args.output)]
    started, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(command, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        print(f"F-ACR1 worker={child.pid}", flush=True)
        try:
            while child.poll() is None:
                cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or now - started > 1200):
                    reason = {"expired_phases": expired, "soft_timeout": now - started > 1200}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - started > 1230 or (terminated is not None and now - terminated > 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            reason = {"supervisor_exception": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        code = child.wait()
    cov.parent.previous.update_pending(args.output, cursors, pending, active)
    path = args.output / "worker_result.json"
    result = json.loads(path.read_text()) if path.exists() else {
        "status": "technical_failure", "first_failure": "Worker receipt absent"}
    result["execution"] = {
        "command": command, "child_pid": child.pid, "child_exit_code": code, "exit_confirmed": True,
        "started_at_utc": utc, "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started, "stop_reason": reason,
        "active_at_exit": active, "forced_termination": forced,
    }
    if code or reason or active:
        result["status"] = "technical_failure"
    e.write_json(args.output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in
                     ("status", "first_failure", "counts", "checkpoints", "execution")}), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--execution-head")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    if not args.execution_head or not args.output:
        parser.error("Running requires --execution-head and --output")
    args.output = args.output.resolve()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
