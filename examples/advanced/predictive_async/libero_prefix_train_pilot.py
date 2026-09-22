"""F-PTP1: matched-budget offline projection tuning, not a deployable policy.

One fixed 128-update unconditioned control and one clean-prefix-conditioned arm.
Both receive the same samples, noise, time and supervised suffix coordinates.
Evaluate final checkpoints only; never read sealed images or tune on outcomes.
"""

import argparse
import hashlib
import json
import random
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from smolvla_prefix_training import (
    build_prefix_flow,
    episode_action_window,
    prefix_training_forward,
    suffix_loss,
)
from verify_smolvla_prefix_training import MODULES, e, tensor_sha

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/smolvla_prefix_data_contract_development_20260922"
COHORT = DATA / "cohort"
TIME_AUDIT = ROOT / "outputs/smolvla_demo_time_86c6af43/independent_audit.json"
VERSION = "smolvla-prefix-projection-pilot-v1"
UPDATES = 128
ARMS = ("ordinary", "prefix")
MODES = ("unconditioned", "correct_prefix", "mismatched_prefix")
CODE = ("examples/advanced/predictive_async/libero_prefix_train_pilot.py",
        "examples/advanced/predictive_async/audit_libero_prefix_train_pilot.py",
        "examples/advanced/predictive_async/smolvla_prefix_training.py",
        "tests/test_libero_prefix_train_pilot.py", "docs/experiments/SMOLVLA_PREFIX_TRAIN_PILOT_PLAN.md")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def paths(head):
    return (ROOT / f"outputs/smolvla_prefix_train_pilot_preparation_{head[:8]}",
            ROOT / f"outputs/smolvla_prefix_train_pilot_{head[:8]}")


def corpus():
    manifest = json.loads((COHORT / "manifest.json").read_text())
    packets = json.loads((COHORT / "packets.json").read_text())
    result = {}
    for row in manifest["rows"]:
        if row["split"] == "sealed":
            continue
        packet = next(p for p in packets if p["trajectory_id"] == row["trajectory_id"])
        path = COHORT / packet["path"]
        if sha(path) != packet["sha256"]:
            raise ValueError("Packet changed")
        result[row["trajectory_id"]] = {**row, "packet": packet}
    if len(result) != 16:
        raise ValueError("Expected 12 train and 4 development trajectories")
    return result


def schedule(rows):
    rng = random.Random(20260922)
    train = []
    tasks = (0, 2, 6, 7)
    for step in range(UPDATES):
        task = tasks[step % 4]
        choices = sorted(k for k, v in rows.items() if v["split"] == "train" and v["task"] == task)
        key = choices[rng.randrange(len(choices))]
        row = rows[key]
        anchors = [i for i in row["packet"]["anchors"] if row["rows"] - i > 8]
        train.append({"step": step, "trajectory_id": key, "anchor": rng.choice(anchors),
                      "C": rng.randrange(9), "noise_seed": 20262000 + step,
                      "time": 0.001 + 0.999 * rng.betavariate(1.5, 1.0)})
    dev = sorted((k for k, v in rows.items() if v["split"] == "dev"), key=lambda k: rows[k]["task"])
    tests = []
    for i, key in enumerate(dev):
        anchors = [a for a in rows[key]["packet"]["anchors"] if rows[key]["rows"] - a > 8]
        donor = dev[(i + 1) % len(dev)]
        da = [a for a in rows[donor]["packet"]["anchors"] if rows[donor]["rows"] - a > 8]
        for j in (1, 2):
            anchor = anchors[(j * len(anchors)) // 3]
            for count in (3, 8):
                tests.append({"case": len(tests), "trajectory_id": key, "anchor": anchor, "C": count,
                              "donor": donor, "donor_anchor": da[(j * len(da)) // 3],
                              "noise_seed": 20264000 + len(tests), "time": 0.5})
    if len(tests) != 16:
        raise ValueError("Development population differs")
    return {"train": train, "dev": tests}


def sources():
    rows = corpus()
    files = [*(ROOT / f for f in CODE), COHORT / "manifest.json", COHORT / "packets.json",
             COHORT / "independent_audit.json", TIME_AUDIT, DATA / "upstream_receipt.json"]
    files += [COHORT / r["packet"]["path"] for r in rows.values()]
    lock = json.loads((ROOT / "docs/experiments/SMOLVLA_GRAPH_CANDIDATE_LOCK.json").read_text())
    files += [ROOT / p for p in lock["core_sha256"]]
    for directory in (e.POLICY, e.VLM):
        files += sorted(directory.glob("*.safetensors")) + sorted(directory.glob("*.json"))
    return {str(p): sha(p) for p in files}


def guard(head):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != head:
        raise ValueError("HEAD changed")
    old = json.loads((DATA / "baseline.json").read_text())["pending"]
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    current = {r[3:]: {"status": r[:2], "sha256": sha(ROOT / r[3:])} for r in status.splitlines()}
    if current != old:
        raise ValueError("Original pending changed or uncommitted execution files")


def prepare(head):
    guard(head)
    timing = json.loads(TIME_AUDIT.read_text())
    cohort = json.loads((COHORT / "independent_audit.json").read_text())
    if not timing["independent_contract_accepted"] or not timing["twenty_hz_more_consistent"] or not cohort["independent_cohort_audit_accepted"]:
        raise ValueError("Data prerequisites failed")
    prep, out = paths(head)
    if prep.exists() or out.exists():
        raise ValueError("Unique pilot required")
    prep.mkdir()
    write(prep / "preparation.json", {"head": head, "sources": sources(), "schedule": schedule(corpus()),
          "version": VERSION, "updates_per_arm": UPDATES, "arms": ARMS, "modes": MODES,
          "modules": MODULES, "lr": 2.5e-6, "betas": [0.9, 0.95], "eps": 1e-8,
          "weight_decay": 0.0, "gradient_clip_norm": 1.0, "batch": 1, "max_forwards": 400,
          "new_env": 0, "sealed_evaluations": 0, "projection_only": True,
          "note": "Raw-RLDS pilot; four-task bitwise physics replay and deployment not claimed."})
    print(json.dumps({"prep": str(prep), "out": str(out), "sha256": sha(prep / "preparation.json")}), flush=True)


def objective(model, inputs, truth, noise, scalar_time, counts, valid, mode, donor=None):
    """Equal supervised suffix for both arms; only input conditioning changes."""
    target_flow = build_prefix_flow(truth, noise, scalar_time, counts, valid)
    if mode not in MODES:
        raise ValueError("Unknown context mode")
    supplied = truth
    input_counts = torch.zeros_like(counts) if mode == "unconditioned" else counts
    if mode == "mismatched_prefix":
        if donor is None or donor.shape != truth.shape:
            raise ValueError("Mismatched prefix needs same-shaped donor")
        supplied = torch.where(target_flow.clean_prefix[..., None], donor.detach(), truth)
    result = prefix_training_forward(model, *inputs, supplied, noise, scalar_time, input_counts, valid)
    return suffix_loss(result["velocity"], target_flow), result["velocity"], target_flow


def batch_for(policy, pre, post, meta, packet, anchor):
    k = packet["anchors"].tolist().index(anchor)
    actions, valid = episode_action_window(torch.from_numpy(packet["actions"]), anchor)
    raw = {"observation.state": torch.from_numpy(packet["states"][anchor : anchor + 1]),
           "action": actions[None], "task": [meta["language"]]}
    for i, key in enumerate(e.CAMERA_KEYS):
        raw[key] = torch.from_numpy(packet["pixels"][k, i].copy()).permute(2, 0, 1)[None].float() / 255
    policy.reset()
    pre.reset()
    post.reset()
    batch = pre(raw)
    torch.testing.assert_close(post(batch["action"]).detach().cpu()[0, valid], actions[valid], atol=1e-5, rtol=1e-5)
    images, masks = policy.prepare_images(batch)
    state, actions = policy.prepare_state(batch), policy.prepare_action(batch)
    return (images, masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state), actions, valid[None].to(actions.device)


def execute(head):
    guard(head)
    prep, out = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    if saved["sources"] != sources() or saved["schedule"] != schedule(corpus()):
        raise ValueError("Prepared pilot changed")
    body = (prep / "registration.md").read_text()
    rb = json.loads((prep / "registration_readback.json").read_text())
    if (type(rb.get("id")) is not int or rb.get("body") != body
        or rb.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        or f"F-PTP1-REGISTER:{head}" not in body or sha(prep / "preparation.json") not in body):
        raise ValueError("Actual preregistration missing")
    out.mkdir()
    result = {"experiment": "F-PTP1", "head": head, "status": "technical_failure", "first_failure": None,
              "registration_id": rb["id"], "forwards_started": 0, "forwards_returned": 0,
              "backwards": 0, "updates": dict.fromkeys(ARMS, 0), "model_loads": 0,
              "attempts": 1, "retries": 0, "new_env": 0, "sealed_evaluations": 0,
              "graph_captures": 0, "deployment_qualified": False}
    began = time.perf_counter()
    policy, named, initial, frozen_digest = None, [], {}, {}
    train_rows, evaluation = [], []
    def alarm(_s, _f):
        raise TimeoutError("Pilot budget900s exhausted")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(900)
    try:
        torch.set_num_threads(1)
        rows = corpus()
        packets = {k: dict(np.load(COHORT / row["packet"]["path"], allow_pickle=False)) for k, row in rows.items()}
        policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
        result["model_loads"] = 1
        write(out / "policy_load.json", report)
        named = [(f"{module}.{n}", p) for module in MODULES for n, p in getattr(policy.model, module).named_parameters()]
        params = [p for _, p in named]
        train_names = {"model." + n for n, _ in named}
        initial = {n: p.detach().cpu().clone() for n, p in named}
        frozen_digest = {n: tensor_sha(p) for n, p in policy.named_parameters() if n not in train_names}
        torch.save(initial, out / "initial_projections.pt")
        result["trainable_numel"] = sum(p.numel() for p in params)
        result["trainable_tensors"] = [n for n, _ in named]

        def forward(spec, mode):
            inputs, actions, valid = batch_for(policy, pre, post, rows[spec["trajectory_id"]], packets[spec["trajectory_id"]], spec["anchor"])
            donor = None
            if mode == "mismatched_prefix":
                _, donor, _ = batch_for(policy, pre, post, rows[spec["donor"]], packets[spec["donor"]], spec["donor_anchor"])
            gen = torch.Generator(device=actions.device).manual_seed(spec["noise_seed"])
            noise = torch.randn(actions.shape, generator=gen, device=actions.device)
            count = torch.tensor([spec["C"]], device=actions.device)
            scalar_time = torch.tensor([spec["time"]], dtype=actions.dtype, device=actions.device)
            result["forwards_started"] += 1
            loss, velocity, flow = objective(policy.model, inputs, actions, noise, scalar_time, count, valid, mode, donor)
            result["forwards_returned"] += 1
            return loss, velocity, flow

        def evaluate(label):
            for spec in saved["schedule"]["dev"]:
                for mode in MODES:
                    with torch.no_grad():
                        loss, velocity, flow = forward(spec, mode)
                    record = {"checkpoint": label, "mode": mode, **spec, "loss": float(loss),
                              "coordinates": int(flow.loss_mask.sum()),
                              "velocity": velocity.cpu(), "target": flow.target_velocity.cpu(),
                              "mask": flow.loss_mask.cpu()}
                    evaluation.append(record)
            print(json.dumps({"evaluated": label, "cases": 48}), flush=True)

        evaluate("frozen")
        for arm in ARMS:
            with torch.no_grad():
                for name, parameter in named:
                    parameter.copy_(initial[name].to(parameter.device))
            if any(tensor_sha(p) != tensor_sha(initial[n]) for n, p in named):
                raise ValueError("Arm did not start from same original weights")
            for parameter in params:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(params, lr=2.5e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, foreach=False)
            for spec in saved["schedule"]["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss, velocity, flow = forward(spec, "unconditioned" if arm == "ordinary" else "correct_prefix")
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                result["backwards"] += 1
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise ValueError("Invalid parameter gradient")
                optimizer.step()
                result["updates"][arm] += 1
                row = {"arm": arm, **spec, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                       "coordinates": int(flow.loss_mask.sum()), "target_sha256": tensor_sha(flow.target_velocity),
                       "mask_sha256": tensor_sha(flow.loss_mask)}
                train_rows.append(row)
                with (out / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if (spec["step"] + 1) % 32 == 0:
                    print(json.dumps({"arm": arm, "updates": spec["step"] + 1, "last_loss": row["loss"]}), flush=True)
            checkpoint = {"interface_version": VERSION, "arm": arm, "base_policy": str(e.POLICY),
                          "data_manifest_sha256": sha(COHORT / "manifest.json"), "updates": UPDATES,
                          "deployment_ready": False, "conditioned_sampler_required": arm == "prefix",
                          "projections": {n: p.detach().cpu().clone() for n, p in named},
                          "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, out / f"{arm}_checkpoint.pt")
            for parameter in params:
                parameter.requires_grad_(False)
                parameter.grad = None
            evaluate(arm)
            del optimizer, checkpoint
        after_frozen = {n: tensor_sha(p) for n, p in policy.named_parameters() if n not in train_names}
        if after_frozen != frozen_digest:
            raise ValueError("Non-projection pretrained weights changed")
        if result["forwards_started"] != 400 or result["updates"] != {"ordinary": 128, "prefix": 128}:
            raise ValueError("Formal call budget differs")
        result.update(status="completed", frozen_parameters_unchanged=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if initial:
            with torch.no_grad():
                for name, parameter in named:
                    parameter.copy_(initial[name].to(parameter.device))
                    parameter.requires_grad_(False)
                    parameter.grad = None
            result["base_parameters_restored"] = all(tensor_sha(p) == tensor_sha(initial[n]) for n, p in named)
        signal.alarm(0)
        torch.save(evaluation, out / "evaluation.pt")
        result.update(evaluation_rows=len(evaluation), wall_s=time.perf_counter() - began,
                      sources_unchanged=sources() == saved["sources"],
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        write(out / "result.json", result)
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.execution_head)
        return 0
    return execute(args.execution_head)


if __name__ == "__main__":
    raise SystemExit(main())
