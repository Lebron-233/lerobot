"""F-SCL1-r1: preserve verified mixed native delays in the fixed scale study."""

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import math
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

import libero_coverage_study as cov
import torch

e, pilot, opt = cov.e, cov.pilot, cov.opt
PREP = e.REPO / "outputs/smolvla_case_scale_r1_preparation_eae8c863"
OLD = e.REPO / "outputs/smolvla_action_objective_1666c067/development_labels.pt"
NEW = e.REPO / "outputs/smolvla_coverage_6249b03d/new_labels.pt"
ARMS = ("global_conditioned", "case_conditioned", "global_no_action", "case_no_action")
SEED, UPDATES = 20260912, 72
DELAY_FOUR_KEYS = frozenset(((3, 48, 6), (3, 49, 6), (4, 48, 6)))


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_prefix(sample):
    """Accept the exact source delay, not a blanket relaxation to mixed delays."""
    expected = 4 if cov.sample_key(sample) in DELAY_FOUR_KEYS else 3
    delay = sample["delay"]
    if isinstance(delay, bool) or not isinstance(delay, int) or delay != expected:
        raise ValueError("Native delay differs from the verified sample identity")
    if sample["future_index"] != sample["current_index"] + delay:
        raise ValueError("Future observation index differs from the committed prefix")
    actions, mask = sample["actions"], sample["mask"]
    if actions.shape != (1, 8, 7) or not actions.is_floating_point():
        raise ValueError("Expected the unchanged padded normalized seven-dimensional prefix")
    if mask.dtype != torch.bool or not torch.equal(mask, (torch.arange(8) < delay)[None]):
        raise ValueError("Committed mask must match the verified delay")
    if not torch.isfinite(actions).all() or torch.count_nonzero(actions[:, delay:]):
        raise ValueError("Committed actions must be finite with original zero padding")


def select_data(old, new):
    selected = cov.choose_old(old)
    keys = [cov.sample_key(s) for s in new]
    expected = {(t, st) for t in range(8) for st in (48, 49)}
    if len(new) != 64 or len(set(keys)) != 64:
        raise ValueError("Expected exactly 64 unique F-COV1 development examples")
    if {(s["task"], s["initial_state_id"]) for s in new} != expected:
        raise ValueError("Only the declared training/validation identities may be read")
    for s in new:
        if s["split"] != ("train" if s["task"] < 6 else "validation"):
            raise ValueError("Development split changed")
        validate_prefix(s)
    if {cov.sample_key(s) for s in new if s["delay"] == 4} != DELAY_FOUR_KEYS:
        raise ValueError("The three source-verified four-step examples changed")
    for s in selected:
        validate_prefix(s)
    training = cov.pool_for(selected + [s for s in new if s["split"] == "train"], "multi_conditioned")
    validation = sorted((s for s in new if s["split"] == "validation"), key=cov.sample_key)
    groups = Counter((s["task"], s["initial_state_id"]) for s in training + validation)
    if len(training) != 72 or len(validation) != 16 or set(groups.values()) != {4}:
        raise ValueError("The common 72/16 data contract changed")
    return training, validation


def identity_row0(sample):
    return (
        (sample["archived_full_chunk"][:, 0, :7].double() - sample["oracle"][:, 0, :7].double())
        .square()
        .mean()
        .item()
    )


def normalized_weights(errors, scale):
    if not errors or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Finite positive training scale and nonempty errors required")
    if any(not math.isfinite(v) or v < 0 for v in errors):
        raise ValueError("Identity errors must be finite and nonnegative")
    floor = max(0.1 * scale, 1e-12)
    raw = [scale / max(v, floor) for v in errors]
    mean = math.fsum(raw) / len(raw)
    return [v / mean for v in raw], floor


def weight_table(training, scales):
    if any(s["split"] != "train" or s["task"] not in range(6) for s in training):
        raise ValueError("Weights and low-error threshold use training data only")
    errors = [identity_row0(s) for s in training]
    weights, floor = normalized_weights(errors, scales["row0"])
    threshold = torch.quantile(torch.tensor(errors, dtype=torch.float64), 0.25).item()
    return {
        "scales": scales,
        "floor": floor,
        "low_error_threshold": threshold,
        "weight_mean": math.fsum(weights) / len(weights),
        "rows": [
            {"key": list(cov.sample_key(s)), "identity_row0": err, "case_weight": w}
            for s, err, w in zip(training, errors, weights, strict=True)
        ],
    }


def objective(latent, row0, arm, scales, case_weight):
    if arm not in ARMS:
        raise ValueError("Unknown registered arm")
    weight = case_weight if arm.startswith("case_") else 1.0
    return latent / scales["latent"] + weight * row0 / scales["row0"]


def low_error_summary(rows, samples, threshold):
    by_key = {cov.sample_key(s): s for s in samples}
    eligible = []
    for row in rows:
        key = (row["task"], row["state"], row["request_id"])
        baseline = identity_row0(by_key[key])
        if baseline <= threshold:
            eligible.append({"key": list(key), "identity": baseline, "predicted": row["row0"]})
    return {
        "count": len(eligible),
        "threshold_from_training": threshold,
        "identity_mean": sum(v["identity"] for v in eligible) / len(eligible) if eligible else None,
        "predicted_mean": sum(v["predicted"] for v in eligible) / len(eligible) if eligible else None,
        "harmed_cases": sum(
            v["predicted"] > v["identity"] + 1e-7 + 1e-6 * abs(v["identity"]) for v in eligible
        ),
        "rows": eligible,
    }


def decision(checkpoints, histories):
    chosen = {a: next(h for h in histories[a] if h["step"] == checkpoints[a]["best_step"]) for a in ARMS}
    scores = {a: h["metrics"]["row0"]["episode_macro"] for a, h in chosen.items()}
    baseline = histories[ARMS[0]][0]["metrics"]
    nonzero = (
        checkpoints["case_conditioned"]["best_step"] > 0
        and checkpoints["case_conditioned"]["nonzero_residual"]
    )
    improvement = scores["case_conditioned"] < scores["global_conditioned"]
    action = scores["case_conditioned"] < scores["case_no_action"]
    passed = (
        nonzero
        and improvement
        and action
        and scores["case_conditioned"] < baseline["row0"]["episode_macro"]
        and chosen["case_conditioned"]["metrics"]["chunk"]["episode_macro"]
        <= baseline["chunk"]["episode_macro"]
    )
    return {
        "scale_row0_improved": improvement,
        "action_increment_observed": action,
        "nonidentity_selected": bool(nonzero),
        "development_candidate_gate_passed": bool(passed),
        "all_row0_comparators_better": bool(
            passed and scores["case_conditioned"] < scores["global_no_action"]
        ),
    }


def run_training(training, validation, table, policy, runtime, output, counts):
    order = cov.schedule(training, "multi_conditioned")
    checkpoints, histories, assessments = {}, {}, {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        torch.save(opt.freeze_state(model), output / f"initial_{arm}.pt")
        best, best_weights, best_step, history = float("inf"), None, None, []
        print(f"TRAIN {arm} common_samples=72 updates=72", flush=True)
        with pilot.phase(output, f"training_{arm}", 600):
            for step in range(UPDATES + 1):
                if step in (0, 36, 72):
                    model.eval()
                    entry = cov.assess(
                        model, arm, validation, policy, runtime, output, counts, f"val_{arm}_{step}"
                    )
                    entry["step"] = step
                    entry["low_error"] = low_error_summary(
                        entry["rows"], validation, table["low_error_threshold"]
                    )
                    history.append(entry)
                    if entry["metrics"]["row0"]["episode_macro"] < best:
                        best, best_step = entry["metrics"]["row0"]["episode_macro"], step
                        best_weights = opt.freeze_state(model)
                    e.write_json(output / f"history_{arm}.json", history)
                if step == UPDATES:
                    break
                sample = training[order[step]]
                weight = table["rows"][order[step]]["case_weight"]
                with pilot.phase(output, f"update_{arm}_{step}", 30):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    visual, _ = cov.parent.predict(model, sample, arm)
                    value = cov.decode(
                        policy, runtime, sample, visual, output, counts, f"train_{arm}", gradients=True
                    )
                    latent, row0, chunk = cov.metric_values(visual, value, sample)
                    loss = objective(latent, row0, arm, table["scales"], weight)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training objective")
                    cov.take(counts, "backward")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm) or any(p.grad is not None for p in policy.parameters()):
                        raise ValueError("Predictor gradients must be finite and VLA gradients absent")
                    cov.take(counts, "updates")
                    optimizer.step()
                    evidence = {
                        "arm": arm,
                        "step": step + 1,
                        "sample": list(cov.sample_key(sample)),
                        "weight": weight if arm.startswith("case_") else 1.0,
                        "latent": latent.item(),
                        "row0": row0.item(),
                        "chunk": chunk.item(),
                        "objective": loss.item(),
                        "gradient_norm": float(norm),
                        "policy_gradients": 0,
                    }
                    with (output / "training_steps.jsonl").open("a") as stream:
                        stream.write(json.dumps(evidence, allow_nan=False) + "\n")
                        stream.flush()
            model.load_state_dict(best_weights, strict=True)
            model.eval().requires_grad_(False)
            nonzero = any(
                bool(best_weights[k].count_nonzero()) for k in ("up_projection.weight", "up_projection.bias")
            )
            checkpoint = {
                "arm": arm,
                "seed": SEED,
                "config": asdict(pilot.config()),
                "best_step": best_step,
                "nonzero_residual": nonzero,
                "validation_row0": best,
                "state_dict": best_weights,
            }
            torch.save(checkpoint, output / f"{arm}.pt")
            checkpoints[arm] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
            histories[arm] = history
            assessment = cov.assess(
                model, arm, training, policy, runtime, output, counts, f"train_selected_{arm}"
            )
            assessment["low_error"] = low_error_summary(
                assessment["rows"], training, table["low_error_threshold"]
            )
            assessments[arm] = assessment
        print(f"END {arm} best_step={best_step} row0={best:.12f}", flush=True)
    return {
        "checkpoints": checkpoints,
        "histories": histories,
        "training_selected": assessments,
        **decision(checkpoints, histories),
    }


def worker(args):
    opt.source_gate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime = Counter(), None
    result = {"experiment": "F-SCL1-r1", "status": "technical_failure", "first_failure": None}
    try:
        with pilot.phase(args.output, "load_development", 30):
            hashes = {str(p): file_digest(p) for p in (OLD, NEW)}
            if hashes != json.loads((PREP / "source_hashes.json").read_text()):
                raise ValueError("Frozen development label bytes changed")
            trace = json.loads((PREP / "source_trace.json").read_text())
            if trace["source_trace_passed"] is not True or trace["label_sha256"] != hashes[str(NEW)]:
                raise ValueError("CPU native source trace is missing or belongs to different labels")
            old = torch.load(OLD, map_location="cpu", weights_only=False)
            new = torch.load(NEW, map_location="cpu", weights_only=False)
            training, validation = select_data(old, new)
            summary = {
                "train": dict(Counter(str(s["delay"]) for s in training)),
                "validation": dict(Counter(str(s["delay"]) for s in validation)),
                "four_step_keys": sorted([list(cov.sample_key(s)) for s in training if s["delay"] == 4]),
            }
            if summary != json.loads((PREP / "data_summary.json").read_text()):
                raise ValueError("Actual data differs from the CPU preparation snapshot")
            result["actual_delay_summary"] = summary
            table = weight_table(training, cov.parent.train_scales(old))
            e.write_json(args.output / "training_weights.json", table)
            e.write_json(args.output / "source_hashes.json", hashes)
            e.write_json(
                args.output / "selection.json",
                {
                    "train": [list(cov.sample_key(s)) for s in training],
                    "validation": [list(cov.sample_key(s)) for s in validation],
                },
            )
        with pilot.phase(args.output, "load_model", 60):
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Registered GPU differs")
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            e.write_json(args.output / "policy_load.json", report)
        runtime = cov.Runtime(policy.model)
        with pilot.phase(args.output, "reference_reproduction", 120), torch.no_grad():
            seen = set()
            for sample in sorted(training + validation, key=cov.sample_key):
                key = (sample["task"], sample["initial_state_id"])
                if key in seen:
                    continue
                seen.add(key)
                for label, z, target in (
                    ("identity", sample["inputs"][:2], sample["archived_full_chunk"]),
                    ("oracle", sample["future"], sample["oracle"]),
                ):
                    value = cov.decode(policy, runtime, sample, z, args.output, counts, f"reference_{label}")
                    pilot.require_equal(value, target, "Development reference replay differs")
                    counts["reference_exact"] += 1
        result.update(run_training(training, validation, table, policy, runtime, args.output, counts))
        result.update(
            status="completed",
            source_hashes=hashes,
            data_counts={"train": 72, "validation": 16},
            vla_frozen=all(not p.requires_grad and p.grad is None for p in policy.parameters()),
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if runtime is not None:
            e.write_json(args.output / "offline_captures.json", runtime.captures)
            runtime.release_graph()
            counts["offline_captures"] = len(runtime.captures)
            result["graph_released"] = runtime.graph is None
        result.update(
            execution_head=args.execution_head,
            counts=dict(counts),
            attempts=1,
            retries=0,
            test_reads=0,
            new_env=0,
            new_native=0,
            visual_encodings=0,
            real_robot=0,
            baseline_qualified=False,
            realtime_qualified=False,
            predictor_benefit_tested=False,
            risk_thresholds=None,
            old_confirmation="not_started_untouched",
        )
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    opt.source_gate(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_case_scale_r1_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use the fixed interpreter and unused output")
    registration = json.loads((PREP / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREP / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("Registration gate failed")
    if not all(json.loads((PREP / "gates.json").read_text()).values()):
        raise ValueError("Preparation failed")
    args.output.mkdir()
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
    started, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active, reason, terminated, forced = {}, {}, {}, None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            cmd, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"F-SCL1-r1 only worker pid={child.pid}", flush=True)
        try:
            while child.poll() is None:
                cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or now - started > 1400):
                    reason = {"expired_phases": expired, "outer_soft": now - started > 1400}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - started > 1430 or (terminated is not None and now - terminated > 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            reason = reason or {"supervisor_exception": traceback.format_exc()}
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
        "wall_seconds": time.perf_counter() - started,
        "stop_reason": reason,
        "active_at_exit": active,
        "forced_termination": forced,
    }
    if code or reason or active:
        result.update(status="technical_failure", development_candidate_gate_passed=False)
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
                for k in ("status", "counts", "development_candidate_gate_passed", "first_failure")
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
