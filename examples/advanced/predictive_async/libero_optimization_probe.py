"""F-OPT1: train/validation-only, paired before/after optimization diagnostic."""

import argparse
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

import libero_action_objective as parent
import torch

e, pilot = parent.e, parent.pilot
PREP = e.REPO / "outputs/smolvla_optimization_probe_preparation_d4488fe1"
LABELS = e.REPO / "outputs/smolvla_action_objective_1666c067/development_labels.pt"
SEED, UPDATES = 20260912, 24
ARMS = {
    "joint_original": {"objective": "joint", "lr": 0.001, "actions": True},
    "joint_small": {"objective": "joint", "lr": 0.0001, "actions": True},
    "action_small": {"objective": "action", "lr": 0.0001, "actions": True},
    "joint_small_no_action": {"objective": "joint", "lr": 0.0001, "actions": False},
}
LIMITS = {"decoder": 336, "updates": 96, "backward": 96}
PENDING_DOCS = {
    "docs/experiments/SMOLVLA_ASYNC_NEXT_REVIEW.md",
    "docs/experiments/SMOLVLA_ACTION_OBJECTIVE_RECEIPT.md",
    "docs/experiments/SMOLVLA_ACTION_OBJECTIVE_RECEIPT.json",
    *e.UNTRACKED_DOCS,
}


def source_gate(head):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=e.REPO, text=True).strip()
    if actual != head:
        raise ValueError("Execution HEAD differs")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=e.REPO, text=True)
    for line in status.splitlines():
        if line[3:] not in PENDING_DOCS or line[:2] not in ("??", " M"):
            raise ValueError(f"Unexpected worktree change: {line}")
    return status


def select_development(samples):
    """Select by registered identity only, never by error, success, or gradients."""
    if len(samples) != 59:
        raise ValueError("Expected the 59 F-ACT1 development labels only")
    keys = [(s["ordinal"], s["request_id"]) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate development identity")
    for s in samples:
        expected = "train" if s["task"] < 6 else "validation"
        if s["task"] not in range(8) or s["split"] != expected:
            raise ValueError("Only original train/validation tasks may be read")
        if s["initial_state_id"] != 46 or s["delay"] != 3 or s["ordinal"] != s["task"]:
            raise ValueError("Development identity changed")
    training = sorted((s for s in samples if s["split"] == "train"), key=lambda s: keysort(s))
    validation = sorted((s for s in samples if s["split"] == "validation"), key=lambda s: keysort(s))
    if len(training) != 47 or len(validation) != 12:
        raise ValueError("Original development split changed")
    anchors = [next(s for s in training if s["task"] == task) for task in range(6)]
    return anchors, validation


def keysort(sample):
    return sample["ordinal"], sample["request_id"]


def combined_loss(latent, action, settings, scales):
    loss = action / scales["row0"]
    if settings["objective"] == "joint":
        loss = loss + latent / scales["latent"]
    elif settings["objective"] != "action":
        raise ValueError("Unknown objective")
    return loss


def take(counts, key):
    if counts[key] >= LIMITS[key]:
        raise RuntimeError(f"F-OPT1 {key} budget exhausted before dispatch")
    counts[key] += 1


def freeze_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def predict(model, sample, settings):
    batch = pilot.batch_for([sample], [0], "cuda")
    context = "conditioned" if settings["actions"] else "no_action"
    return pilot.prediction(model, batch, context), batch


def decode(policy, sample, visual, counts):
    take(counts, "decoder")
    values = tuple(v.cuda() for v in sample["inputs"])
    result = policy.model.sample_actions(
        None,
        None,
        values[4],
        values[5],
        values[6],
        noise=values[7].clone(),
        future_image_tokens=tuple(z.to(dtype=values[c].dtype) for c, z in enumerate(visual)),
        future_image_token_masks=tuple(values[2:4]),
    )
    if result.shape != (1, 50, 32) or not torch.isfinite(result).all():
        raise ValueError("Decoder must produce a finite original full chunk")
    return result


def losses(visual, batch, decoded, sample):
    latent = pilot.per_sample_mse(visual, batch["future"], batch["masks"]).mean()
    difference = (decoded[..., :7].float() - sample["oracle"].cuda()[..., :7].float()).square()
    return latent, difference[:, 0].mean(), difference.mean()


def snapshot(visual, decoded, scalars):
    return {
        "visual": tuple(v.detach().cpu().clone() for v in visual),
        "output": decoded.detach().cpu().clone(),
        "latent": float(scalars[0].detach()),
        "row0": float(scalars[1].detach()),
        "chunk": float(scalars[2].detach()),
    }


def assess(model, settings, samples, policy, output, counts, label):
    rows = []
    model.eval()
    for sample in samples:
        with pilot.phase(output, f"assessment_{label}_{sample['ordinal']}_{sample['request_id']}", 30):
            with torch.no_grad():
                visual, batch = predict(model, sample, settings)
                decoded = decode(policy, sample, visual, counts)
                values = snapshot(visual, decoded, losses(visual, batch, decoded, sample))
            if label.endswith("_0"):
                pilot.require_equal(
                    decoded, sample["archived_full_chunk"], "Zero-residual eager output differs"
                )
                counts["initial_identity_exact"] += 1
            rows.append({"ordinal": sample["ordinal"], "request_id": sample["request_id"], **values})
    torch.save(rows, output / f"assessment_{label}.pt")
    return {key: parent.macro(rows, key)["episode_macro"] for key in ("latent", "row0", "chunk")}


def run_arm(arm, settings, samples, anchors, validation, policy, scales, output, counts):
    torch.manual_seed(SEED)
    model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
    initial = freeze_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.0001)
    evaluation = anchors + validation
    before = assess(model, settings, evaluation, policy, output, counts, f"{arm}_0")
    steps = []
    for step in range(UPDATES):
        sample = anchors[step % len(anchors)]
        with pilot.phase(output, f"update_{arm}_{step + 1}", 30):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            visual, batch = predict(model, sample, settings)
            decoded = decode(policy, sample, visual, counts)
            values = losses(visual, batch, decoded, sample)
            loss = combined_loss(values[0], values[1], settings, scales)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            pre = snapshot(visual, decoded, values)
            take(counts, "backward")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm) or any(p.grad is not None for p in policy.parameters()):
                raise ValueError("Predictor gradient or frozen VLA contract failed")
            take(counts, "updates")
            optimizer.step()
            # Measure the SAME input, saved noise, teacher, and current state after this update.
            model.eval()
            with torch.no_grad():
                next_visual, next_batch = predict(model, sample, settings)
                next_decoded = decode(policy, sample, next_visual, counts)
                post = snapshot(
                    next_visual, next_decoded, losses(next_visual, next_batch, next_decoded, sample)
                )
            changed = sum(int((a != b).sum()) for a, b in zip(pre["visual"], post["visual"], strict=True))
            item = {
                "arm": arm,
                "step": step + 1,
                "ordinal": sample["ordinal"],
                "request_id": sample["request_id"],
                "before": pre,
                "after": post,
                "gradient_norm": float(norm),
                "quantized_token_elements_changed": changed,
                "policy_gradients": 0,
            }
            torch.save(item, output / f"update_{arm}_{step + 1:03d}.pt")
            small = {k: v for k, v in item.items() if k not in ("before", "after")}
            small.update(
                before={k: pre[k] for k in ("latent", "row0", "chunk")},
                after={k: post[k] for k in ("latent", "row0", "chunk")},
            )
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(small, allow_nan=False) + "\n")
            steps.append(small)
    after = assess(model, settings, evaluation, policy, output, counts, f"{arm}_{UPDATES}")
    final = freeze_state(model)
    torch.save(
        {
            "arm": arm,
            "settings": settings,
            "seed": SEED,
            "steps": UPDATES,
            "config": asdict(pilot.config()),
            "initial": initial,
            "state_dict": final,
        },
        output / f"{arm}_checkpoint.pt",
    )
    return {
        "settings": settings,
        "updates": UPDATES,
        "same_sample_decreases": sum(s["after"]["row0"] < s["before"]["row0"] for s in steps),
        "same_sample_unchanged": sum(s["after"]["row0"] == s["before"]["row0"] for s in steps),
        "nonzero_residual_weights": bool(
            final["up_projection.weight"].count_nonzero() or final["up_projection.bias"].count_nonzero()
        ),
        "initial_all_development_macro": before,
        "final_all_development_macro": after,
    }


def worker(args):
    source_gate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    counts, result = Counter(), {"experiment": "F-OPT1", "status": "technical_failure", "first_failure": None}
    try:
        torch.set_num_threads(1)
        with pilot.phase(args.output, "development_only_load", 30):
            samples = torch.load(LABELS, map_location="cpu", weights_only=False)
            anchors, validation = select_development(samples)
            scales = parent.train_scales(samples)
            e.write_json(
                args.output / "selection.json",
                {
                    "source": str(LABELS),
                    "anchors": [keysort(s) for s in anchors],
                    "validation": [keysort(s) for s in validation],
                    "scales": scales,
                    "test_reads": 0,
                },
            )
        with pilot.phase(args.output, "model_load", 60):
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Registered GPU differs")
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            e.write_json(args.output / "policy_load.json", report)
        arms = {}
        for arm, settings in ARMS.items():
            print(f"START F-OPT1 {arm}", flush=True)
            with pilot.phase(args.output, f"arm_{arm}", 300):
                arms[arm] = run_arm(
                    arm, settings, samples, anchors, validation, policy, scales, args.output, counts
                )
            e.write_json(args.output / f"{arm}_summary.json", arms[arm])
            print(
                f"END F-OPT1 {arm} same_sample_decreases={arms[arm]['same_sample_decreases']}/{UPDATES}",
                flush=True,
            )
        result.update(
            status="completed",
            arms=arms,
            vla_parameters_frozen=all(not p.requires_grad and p.grad is None for p in policy.parameters()),
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        result.update(
            execution_head=args.execution_head,
            counts=dict(counts),
            test_reads=0,
            new_env=0,
            new_native=0,
            visual_encodings=0,
            graph_captures=0,
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
    source_gate(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_optimization_probe_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use the registered interpreter and unused output")
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
    e.write_json(
        args.output / "protocol.json",
        {"arms": ARMS, "updates": UPDATES, "limits": LIMITS, "seed": SEED, "source": str(LABELS)},
    )
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
        print(f"F-OPT1 worker pid={child.pid}", flush=True)
        try:
            while child.poll() is None:
                parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or now - start >= 1200):
                    reason = {"expired_phases": expired, "outer_limit": now - start >= 1200}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - start >= 1230 or (terminated is not None and now - terminated >= 5):
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
        "active_at_exit": active,
        "forced_termination": forced,
    }
    if code or reason or active:
        result["status"] = "technical_failure"
    result["environment_after"] = {
        "python": sys.executable,
        "version": sys.version,
        "packages": sorted([[d.metadata["Name"], d.version] for d in importlib.metadata.distributions()]),
    }
    e.write_json(args.output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "counts", "first_failure")}), flush=True)
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
