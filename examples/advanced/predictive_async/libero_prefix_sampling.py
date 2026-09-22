"""F-PSD1: fixed offline ten-step action diagnosis; no training or Env dispatch."""

import argparse
import json
import os
import signal
import time
import traceback

import libero_prefix_train_pilot as train
import numpy as np
import torch
from smolvla_prefix_sampling import (
    SAMPLER_VERSION,
    load_projection_checkpoint,
    projection_parameters,
    restore_projections,
    sample_prefix_actions,
)
from verify_smolvla_prefix_training import tensor_sha

ROOT, COHORT, e, sha, write = train.ROOT, train.COHORT, train.e, train.sha, train.write
PRIOR_HEAD = "24c7ed49c6f05299991e605b7b92a21721c69e09"
PRIOR_PREP, PRIOR_OUT = train.paths(PRIOR_HEAD)
LABELS = ("frozen", "ordinary", "prefix")
MODES = ("native_unconditioned", "correct_prefix", "mismatched_prefix")
CHECKPOINTS = {
    "ordinary": "2377b2c3d849c909f9cdf5b320e6d97b6fab07008f96541c5b86f62a41e77ac9",
    "prefix": "0d424426bd2fa9a2f0f4abb69b4e10c3443a2029749366bf16f2cca9821c2bed",
}
CODE = (
    "examples/advanced/predictive_async/smolvla_prefix_sampling.py",
    "examples/advanced/predictive_async/libero_prefix_sampling.py",
    "examples/advanced/predictive_async/audit_libero_prefix_sampling.py",
    "tests/test_smolvla_prefix_sampling.py",
    "docs/experiments/SMOLVLA_PREFIX_SAMPLING_PLAN.md",
)
ENV_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "LIBERO_CONFIG_PATH",
    "LD_PRELOAD",
)


def dev_contract():
    manifest = json.loads((COHORT / "manifest.json").read_text())
    packets = json.loads((COHORT / "packets.json").read_text())
    rows = {}
    for row in manifest["rows"]:
        if row["split"] == "dev":
            packet = next(p for p in packets if p["trajectory_id"] == row["trajectory_id"])
            if sha(COHORT / packet["path"]) != packet["sha256"]:
                raise ValueError("Development packet changed")
            rows[row["trajectory_id"]] = {**row, "packet": packet}
    schedule = json.loads((PRIOR_PREP / "preparation.json").read_text())["schedule"]["dev"]
    if (
        len(rows) != 4
        or len(schedule) != 16
        or {r["task"] for r in rows.values()} != {0, 2, 6, 7}
        or [s["case"] for s in schedule] != list(range(16))
        or any(
            s["trajectory_id"] not in rows or s["donor"] not in rows or s["C"] not in (3, 8) for s in schedule
        )
    ):
        raise ValueError("Original development population changed")
    return rows, schedule


def paths(head):
    return (
        ROOT / f"outputs/smolvla_prefix_sampling_preparation_{head[:8]}",
        ROOT / f"outputs/smolvla_prefix_sampling_{head[:8]}",
    )


def sources():
    values = train.sources()
    files = [
        *(ROOT / p for p in CODE),
        PRIOR_PREP / "preparation.json",
        PRIOR_OUT / "result.json",
        PRIOR_OUT / "independent_audit.json",
        PRIOR_OUT / "initial_projections.pt",
        PRIOR_OUT / "evaluation.pt",
        *(PRIOR_OUT / f"{arm}_checkpoint.pt" for arm in CHECKPOINTS),
        ROOT / "src/lerobot/policies/common/flow_matching.py",
        ROOT / "src/lerobot/policies/common/vla_utils.py",
    ]
    values.update({str(p): sha(p) for p in files})
    return values


def guard(head):
    train.guard(head)
    lock = json.loads((ROOT / "docs/experiments/SMOLVLA_GRAPH_CANDIDATE_LOCK.json").read_text())
    if any(sha(ROOT / p) != h for p, h in lock["core_sha256"].items()):
        raise ValueError("Frozen control core changed")
    for label, expected in CHECKPOINTS.items():
        if sha(PRIOR_OUT / f"{label}_checkpoint.pt") != expected:
            raise ValueError("Prior checkpoint changed")


def prepare(head):
    guard(head)
    prior = json.loads((PRIOR_OUT / "independent_audit.json").read_text())
    if not prior["independent_contract_accepted"] or prior["offline_followup_supported"]:
        raise ValueError("Prerequisite evidence changed; this is diagnosis after a negative pilot")
    rows, schedule = dev_contract()
    prep, out = paths(head)
    if prep.exists() or out.exists() or torch.cuda.is_initialized():
        raise ValueError("Unique CPU preparation required")
    prep.mkdir()
    saved = {
        "head": head,
        "sources": sources(),
        "schedule": schedule,
        "sampler_version": SAMPLER_VERSION,
        "checkpoints": CHECKPOINTS,
        "labels": LABELS,
        "modes": MODES,
        "environment": {k: os.environ.get(k) for k in ENV_KEYS},
        "decodes": 192,
        "denoise_steps": 1920,
        "c0_pairs": 48,
        "scored_outputs": 144,
        "traced_calls": 144,
        "new_env": 0,
        "optimizer_updates": 0,
        "sealed_evaluations": 0,
        "guardrails": "old development; no deployment qualification or promotion of F-PTP1",
    }
    write(prep / "preparation.json", saved)
    print(
        json.dumps(
            {
                "prepared": True,
                "prep": str(prep),
                "out": str(out),
                "dev_trajectories": len(rows),
                "sources": len(saved["sources"]),
                "sha256": sha(prep / "preparation.json"),
            }
        ),
        flush=True,
    )


def validate(head):
    guard(head)
    prep, out = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    if (
        saved["head"] != head
        or saved["sources"] != sources()
        or saved["schedule"] != dev_contract()[1]
        or saved["environment"] != {k: os.environ.get(k) for k in ENV_KEYS}
    ):
        raise ValueError("Prepared source/schedule/environment changed")
    body = (prep / "registration.md").read_text()
    rb = json.loads((prep / "registration_readback.json").read_text())
    if (
        type(rb.get("id")) is not int
        or rb.get("body") != body
        or rb.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        or f"F-PSD1-REGISTER:{head}" not in body
        or str(out) not in body
        or sha(prep / "preparation.json") not in body
    ):
        raise ValueError("Actual registration missing or different")
    return saved, rb["id"]


def score(full, target, valid, count):
    err = (full[..., :7] - target[..., :7]).square()[0]
    live = valid[0]
    after = live & (torch.arange(50, device=live.device) >= count)
    short = after & (torch.arange(50, device=live.device) < count + 10)
    if not after.any() or not live[count]:
        raise ValueError("No valid first unexecuted action")
    return {
        "block": float(err[live].mean()),
        "suffix": float(err[after].mean()),
        "first": float(err[count].mean()),
        "short": float(err[short].mean()),
    }


def execute(head):
    saved, registration = validate(head)
    _, out = paths(head)
    if out.exists():
        raise ValueError("Formal output already consumed")
    out.mkdir()
    result = {
        "experiment": "F-PSD1",
        "head": head,
        "status": "technical_failure",
        "first_failure": None,
        "registration_id": registration,
        "attempts": 1,
        "retries": 0,
        "decodes_started": 0,
        "decodes_returned": 0,
        "vision_encodes": 0,
        "denoise_steps": 0,
        "model_loads": 0,
        "optimizer_updates": 0,
        "backwards": 0,
        "new_env": 0,
        "sealed_evaluations": 0,
        "graph_captures": 0,
        "deployment_qualified": False,
    }
    began, evaluation, references = time.perf_counter(), [], {}
    policy, original_encode, initial, frozen_hashes = None, None, {}, {}
    weights = {}

    def alarm(_signum, _frame):
        raise TimeoutError("F-PSD1 600s runtime budget exhausted")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(600)
    calls = (out / "calls.jsonl").open("x")
    try:
        torch.set_num_threads(1)
        rows, schedule = dev_contract()
        packets = {
            key: dict(np.load(COHORT / meta["packet"]["path"], allow_pickle=False))
            for key, meta in rows.items()
        }
        policy, pre, post, load_report = e.load_runtime(e.POLICY, e.VLM)
        result["model_loads"] = 1
        write(out / "policy_load.json", load_report)
        named = projection_parameters(policy.model)
        initial = {k: p.detach().cpu().clone() for k, p in named.items()}
        original_initial = torch.load(
            PRIOR_OUT / "initial_projections.pt", map_location="cpu", weights_only=True
        )
        if set(initial) != set(original_initial) or any(
            not torch.equal(v, original_initial[k]) for k, v in initial.items()
        ):
            raise ValueError("Loaded base differs from pilot initialization")
        train_names = {"model." + k for k in named}
        frozen_hashes = {n: tensor_sha(p) for n, p in policy.named_parameters() if n not in train_names}
        original_encode = policy.model.encode_image_tokens

        def observed_encode(*args, **kwargs):
            result["vision_encodes"] += 1
            return original_encode(*args, **kwargs)

        policy.model.encode_image_tokens = observed_encode
        manifest_sha = sha(COHORT / "manifest.json")
        for label in LABELS:
            restore_projections(policy.model, initial)
            if label != "frozen":
                load_projection_checkpoint(
                    policy.model,
                    PRIOR_OUT / f"{label}_checkpoint.pt",
                    CHECKPOINTS[label],
                    arm=label,
                    base_policy=e.POLICY,
                    manifest_sha=manifest_sha,
                )
            weights[label] = {n: tensor_sha(p) for n, p in projection_parameters(policy.model).items()}
            for spec in schedule:
                key = spec["trajectory_id"]
                inputs, truth, valid = train.batch_for(
                    policy, pre, post, rows[key], packets[key], spec["anchor"]
                )
                _, donor, _ = train.batch_for(
                    policy, pre, post, rows[spec["donor"]], packets[spec["donor"]], spec["donor_anchor"]
                )
                gen = torch.Generator(device=truth.device).manual_seed(spec["noise_seed"])
                noise = torch.randn(truth.shape, generator=gen, device=truth.device)
                fingerprint = {
                    "state": tensor_sha(inputs[-1]),
                    "language": tensor_sha(inputs[2]),
                    "language_mask": tensor_sha(inputs[3]),
                    "images": [tensor_sha(v) for v in inputs[0]],
                    "image_masks": [tensor_sha(v) for v in inputs[1]],
                }
                ref = {
                    "spec": spec,
                    "truth": truth.detach().cpu().clone(),
                    "valid": valid.cpu(),
                    "noise": noise.cpu(),
                    "donor_prefix": donor[:, : spec["C"], :7].cpu(),
                    "post_truth": post(truth[..., :7]).detach().cpu(),
                    "fingerprint": fingerprint,
                }
                if spec["case"] not in references:
                    references[spec["case"]] = ref
                else:
                    old = references[spec["case"]]
                    if old["fingerprint"] != fingerprint or any(
                        not torch.equal(old[k], ref[k])
                        for k in ("truth", "valid", "noise", "donor_prefix", "post_truth")
                    ):
                        raise ValueError("Input/target differs across checkpoints")
                native = None
                for mode in (*MODES, "calibration_c0"):
                    count = spec["C"] if mode in ("correct_prefix", "mismatched_prefix") else 0
                    prefix = (
                        (truth if mode == "correct_prefix" else donor)[:, :count, :7].clone()
                        if count
                        else None
                    )
                    trace, shapes = [], []
                    cid = result["decodes_started"]
                    if cid >= 192:
                        raise ValueError("Decode budget exhausted")
                    calls.write(
                        json.dumps(
                            {
                                "event": "intent",
                                "call_id": cid,
                                "checkpoint": label,
                                "mode": mode,
                                "case": spec["case"],
                            }
                        )
                        + "\n"
                    )
                    calls.flush()
                    hook = policy.model.action_out_proj.register_forward_hook(
                        lambda _m, _a, value, target=shapes: target.append(list(value.shape))
                    )
                    result["decodes_started"] += 1
                    start = time.perf_counter()
                    try:
                        with torch.no_grad():
                            if mode == "native_unconditioned":
                                full = policy.model.sample_actions(*inputs, noise=noise)
                            else:
                                full = sample_prefix_actions(
                                    policy.model,
                                    *inputs,
                                    noise,
                                    prefix=prefix,
                                    counts=torch.tensor([count], device=noise.device),
                                    trace=trace,
                                )
                            processed = post(full[..., :7]).detach().cpu().clone()
                            full_cpu = full.detach().cpu().clone()
                            metrics = score(full, truth, valid, spec["C"])
                            torch.cuda.synchronize()
                    finally:
                        hook.remove()
                    duration = time.perf_counter() - start
                    if shapes != [[1, 50, 32]] * 10 or duration > 30:
                        raise ValueError("Decode shape/step or 30s call budget violation")
                    result["denoise_steps"] += len(shapes)
                    result["decodes_returned"] += 1
                    calls.write(json.dumps({"event": "return", "call_id": cid, "seconds": duration}) + "\n")
                    calls.flush()
                    if mode == "native_unconditioned":
                        native = (full_cpu, processed)
                    elif mode == "calibration_c0" and (
                        not torch.equal(native[0], full_cpu) or not torch.equal(native[1], processed)
                    ):
                        raise ValueError("C0 full native output differs")
                    evaluation.append(
                        {
                            "checkpoint": label,
                            "mode": mode,
                            **spec,
                            "call_id": cid,
                            "input_fingerprint": fingerprint,
                            "condition_count": count,
                            "condition_prefix": None if prefix is None else prefix.cpu(),
                            "full": full_cpu,
                            "processed": processed,
                            "metrics": metrics,
                            "trace": trace,
                            "seconds_including_trace": duration,
                        }
                    )
            if weights[label] != {n: tensor_sha(p) for n, p in projection_parameters(policy.model).items()}:
                raise ValueError("Projection changed during sampling")
            print(
                json.dumps(
                    {
                        "checkpoint": label,
                        "completed_decodes": result["decodes_returned"],
                        "c0_pairs_exact": 16,
                    }
                ),
                flush=True,
            )
        if (
            result["decodes_returned"] != 192
            or result["vision_encodes"] != 192
            or result["denoise_steps"] != 1920
        ):
            raise ValueError("Final call/vision/step population mismatch")
        if any(p.requires_grad or p.grad is not None for p in policy.parameters()):
            raise ValueError("Unexpected gradients")
        if frozen_hashes != {n: tensor_sha(p) for n, p in policy.named_parameters() if n not in train_names}:
            raise ValueError("Non-projection weights changed")
        result.update(status="completed", c0_pairs_exact=48, nonprojection_weights_unchanged=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        calls.close()
        if policy is not None and original_encode is not None:
            policy.model.encode_image_tokens = original_encode
        if initial:
            restore_projections(policy.model, initial)
            result["base_projections_restored"] = all(
                torch.equal(p.detach().cpu(), initial[n])
                for n, p in projection_parameters(policy.model).items()
            )
        signal.alarm(0)
        torch.save(
            {"references": references, "rows": evaluation, "projection_hashes": weights},
            out / "evaluation.pt",
        )
        result.update(
            saved_rows=len(evaluation),
            sources_unchanged=saved["sources"] == sources(),
            wall_s=time.perf_counter() - began,
            evidence_sha256=sha(out / "evaluation.pt"),
        )
        write(out / "result.json", result)
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution-head", required=True)
    p.add_argument("--prepare", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare(args.execution_head)
        return 0
    return execute(args.execution_head)


if __name__ == "__main__":
    raise SystemExit(main())
