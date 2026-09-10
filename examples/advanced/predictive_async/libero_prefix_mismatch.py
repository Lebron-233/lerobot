"""F-PFX1: a registered, frozen-model committed-prefix intervention."""

import argparse
import faulthandler
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

import libero_case_scale as scale
import torch

cov, e, pilot, opt = scale.cov, scale.e, scale.pilot, scale.opt
PREP = e.REPO / "outputs/smolvla_prefix_mismatch_preparation_ece0d933"
SOURCE = e.REPO / "outputs/smolvla_case_scale_r1_6ddb7505"
ARMS = {"true": "case_conditioned", "no_action": "case_no_action"}


def key(sample):
    return tuple(cov.sample_key(sample))


def donor_map(samples):
    groups = defaultdict(list)
    seen = set()
    for sample in samples:
        if key(sample) in seen:
            raise ValueError("Duplicate sample identity")
        seen.add(key(sample))
        group = (sample["split"], sample["task"], sample["initial_state_id"], sample["delay"])
        groups[group].append(sample)
    result = {}
    for group in groups.values():
        ordered = sorted(group, key=key)
        for i, sample in enumerate(ordered):
            result[key(sample)] = key(ordered[(i + 1) % len(ordered)]) if len(ordered) > 1 else None
    return result


def with_prefix(recipient, donor):
    fields = ("split", "task", "initial_state_id", "delay")
    if key(recipient) == key(donor) or any(recipient[f] != donor[f] for f in fields):
        raise ValueError("Donor must be a different example in the same registered group")
    if not torch.equal(recipient["mask"], donor["mask"]):
        raise ValueError("Do not change the recipient mask")
    return {**recipient, "actions": donor["actions"].clone()}


def load_data():
    old = torch.load(scale.OLD, map_location="cpu", weights_only=False)
    new = torch.load(scale.NEW, map_location="cpu", weights_only=False)
    return scale.select_data(old, new)


def mapping_contract(training, validation):
    samples = training + validation
    donors = donor_map(samples)
    rows = [
        {
            "key": list(key(s)), "split": s["split"], "delay": s["delay"],
            "donor": list(donors[key(s)]) if donors[key(s)] is not None else None,
        }
        for s in sorted(samples, key=key)
    ]
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "validation")}
    eligible = {
        split: sum(r["split"] == split and r["donor"] is not None for r in rows)
        for split in counts
    }
    if counts != {"train": 72, "validation": 16} or eligible != {"train": 69, "validation": 16}:
        raise ValueError("Registered sample or singleton counts differ")
    return {"rows": rows, "counts": counts, "eligible": eligible, "formal_decodes": 261}


def prepare():
    if PREP.exists():
        raise ValueError("Preparation must use its new, unused directory")
    PREP.mkdir()
    training, validation = load_data()
    contract = mapping_contract(training, validation)
    expected = json.loads((SOURCE / "source_hashes.json").read_text())
    actual = {str(p): scale.file_digest(p) for p in (scale.OLD, scale.NEW)}
    if actual != expected:
        raise ValueError("Existing frozen label identities differ")
    checkpoints = {}
    for arm in ARMS.values():
        saved = torch.load(SOURCE / f"{arm}.pt", map_location="cpu", weights_only=False)
        if saved["arm"] != arm or saved["best_step"] != 36 or saved["config"] != asdict(pilot.config()):
            raise ValueError("Wrong frozen checkpoint")
        checkpoints[arm] = {k: saved[k] for k in ("arm", "best_step", "seed", "config")}
    e.write_json(PREP / "manifest.json", contract)
    e.write_json(PREP / "preparation.json", {
        "experiment": "F-PFX1", "source_hashes": actual, "checkpoints": checkpoints,
        "cuda_initialized": torch.cuda.is_initialized(), "model_loads": 0,
        "predictor_forwards": 0, "new_env": 0, "test_reads": 0,
    })
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation unexpectedly initialized CUDA")
    print(json.dumps({k: contract[k] for k in ("counts", "eligible", "formal_decodes")}), flush=True)


def archived_assessments(arm):
    rows = []
    for label in (f"train_selected_{arm}", f"val_{arm}_36"):
        rows.extend(torch.load(SOURCE / f"assessment_{label}.pt", map_location="cpu", weights_only=False))
    mapped = {(r["task"], r["state"], r["request_id"]): r for r in rows}
    if len(mapped) != 88:
        raise ValueError("Frozen assessment identity count differs")
    return mapped


def worker(args):
    opt.source_gate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime, records = Counter(), None, []
    result = {"experiment": "F-PFX1", "status": "technical_failure", "first_failure": None}
    try:
        with pilot.phase(args.output, "load_development", 30):
            training, validation = load_data()
            contract = mapping_contract(training, validation)
            if contract != json.loads((PREP / "manifest.json").read_text()):
                raise ValueError("Registered donor assignment changed")
            prep = json.loads((PREP / "preparation.json").read_text())
            if {str(p): scale.file_digest(p) for p in (scale.OLD, scale.NEW)} != prep["source_hashes"]:
                raise ValueError("Frozen source bytes changed")
            samples = {key(s): s for s in training + validation}
            references = {c: archived_assessments(a) for c, a in ARMS.items()}
        with pilot.phase(args.output, "load_model", 90):
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Registered GPU changed")
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts["vla_loads"] += 1
            e.write_json(args.output / "policy_load.json", report)
            models, initial = {}, {}
            for context, arm in ARMS.items():
                saved = torch.load(SOURCE / f"{arm}.pt", map_location="cpu", weights_only=False)
                if {k: saved[k] for k in prep["checkpoints"][arm]} != prep["checkpoints"][arm]:
                    raise ValueError("Frozen checkpoint metadata changed")
                model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
                model.load_state_dict(saved["state_dict"], strict=True)
                models[context] = model.eval().requires_grad_(False)
                initial[context] = saved["state_dict"]
                counts["predictor_loads"] += 1
            runtime = cov.Runtime(policy.model)
        with torch.no_grad():
            for assignment in contract["rows"]:
                recipient = samples[tuple(assignment["key"])]
                donor = samples[tuple(assignment["donor"])] if assignment["donor"] else None
                for context in ("true", "mismatched", "no_action"):
                    if context == "mismatched" and donor is None:
                        continue
                    if counts["predictor_forwards"] >= 261 or counts["decoder"] >= 261:
                        raise RuntimeError("F-PFX1 formal budget exhausted before dispatch")
                    used = with_prefix(recipient, donor) if context == "mismatched" else recipient
                    batch = pilot.batch_for([used], [0], "cuda")
                    model = models["no_action" if context == "no_action" else "true"]
                    mode = "no_action" if context == "no_action" else "conditioned"
                    with pilot.phase(args.output, f"predict_{counts['predictor_forwards']}_{context}", 30):
                        counts["predictor_forwards"] += 1
                        raw = pilot.prediction(model, batch, mode, quantized=False)
                        visual = tuple(z.to(torch.bfloat16).float() for z in raw)
                    value = cov.decode(policy, runtime, recipient, visual, args.output, counts, context)
                    metric_values = cov.metric_values(visual, value, recipient)
                    row = {
                        **assignment, "context": context,
                        "actual_actions": (torch.zeros_like(used["actions"]) if context == "no_action"
                                           else used["actions"]).clone(),
                        "actual_mask": used["mask"].clone(),
                        "raw_visual": tuple(z.cpu().clone() for z in raw),
                        "visual": tuple(z.cpu().clone() for z in visual),
                        "output": value.cpu().clone(),
                        "metrics": {name: float(v) for name, v in
                                    zip(("latent", "row0", "chunk"), metric_values, strict=True)},
                    }
                    records.append(row)
                    if context in references:
                        expected = references[context][key(recipient)]
                        for actual, original in zip(row["visual"], expected["visual"], strict=True):
                            pilot.require_equal(actual, original, "Selected predictor token replay differs")
                        pilot.require_equal(row["output"], expected["output"], "Selected full-output replay differs")
                        counts["selected_replay_exact"] += 1
                print(f"F-PFX1 sample={assignment['key']} decodes={counts['decoder']}", flush=True)
        for context, model in models.items():
            for name, value in model.state_dict().items():
                pilot.require_equal(value, initial[context][name], "Frozen predictor was modified")
        if not all(not p.requires_grad and p.grad is None for p in policy.parameters()):
            raise ValueError("VLA freeze contract failed")
        if counts["decoder"] != 261 or counts["selected_replay_exact"] != 176:
            raise ValueError("Formal completion counts differ")
        result.update(status="completed", frozen_models_unchanged=True, vla_frozen=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        torch.save(records, args.output / "predictions.pt")
        if runtime is not None:
            e.write_json(args.output / "offline_captures.json", runtime.captures)
            counts["offline_captures"] = len(runtime.captures)
            runtime.release_graph()
            result["graph_released"] = runtime.graph is None
        result.update(
            execution_head=args.execution_head, counts=dict(counts), attempts=1, retries=0,
            python=sys.executable, python_version=sys.version,
            updates=0, backward=0, visual_encodings=0, new_env=0, new_native=0,
            test_reads=0, real_robot=0, baseline_qualified=False, realtime_qualified=False,
            predictor_benefit_tested=False, risk_thresholds=None, old_confirmation="not_started_untouched",
        )
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    opt.source_gate(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_prefix_mismatch_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use the fixed interpreter and new registered output")
    registration = json.loads((PREP / "registration_readback.json").read_text())
    body = (PREP / "registration.md").read_text()
    if registration["body"] != body or args.execution_head not in body or str(args.output) not in body:
        raise ValueError("Exact registration gate failed")
    args.output.mkdir()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
               "--execution-head", args.execution_head, "--output", str(args.output)]
    started, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(command, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        print(f"F-PFX1 worker={child.pid}", flush=True)
        try:
            while child.poll() is None:
                cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or now - started > 600):
                    reason = {"expired_phases": expired, "soft_timeout": now - started > 600}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - started > 630 or (terminated is not None and now - terminated > 5):
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
        "command": command, "child_pid": child.pid, "child_exit_code": code,
        "exit_confirmed": True, "started_at_utc": utc,
        "finished_at_utc": datetime.now(UTC).isoformat(), "wall_seconds": time.perf_counter() - started,
        "stop_reason": reason, "active_at_exit": active, "forced_termination": forced,
    }
    if code or reason or active:
        result["status"] = "technical_failure"
    e.write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
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
