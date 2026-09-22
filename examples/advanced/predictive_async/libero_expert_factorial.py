"""F-EAT1: fixed 2x2 learning-scope experiment, no online or sealed evaluation."""

import argparse
import json
import os
import signal
import time
import traceback

import libero_expert_feasibility as feasibility
import libero_prefix_sampling as sampling
import libero_prefix_train_pilot as train
import numpy as np
import torch
from smolvla_expert_lora import ADAPTER_VERSION, ALPHA, RANK, SEED, attach_expert_adapters
from smolvla_prefix_sampling import projection_parameters, restore_projections, sample_prefix_actions
from verify_smolvla_prefix_training import tensor_sha

ROOT, COHORT, e, sha, write = train.ROOT, train.COHORT, train.e, train.sha, train.write
FEASIBILITY_HEAD = "985ee5c057860ad88df543c96d50f1592f8dd24a"
FEASIBILITY_OUT = feasibility.paths(FEASIBILITY_HEAD)[1]
ARMS = ("projection_ordinary", "projection_prefix", "expert_ordinary", "expert_prefix")
LABELS = ("frozen", *ARMS)
MODES = sampling.MODES
VERSION = "smolvla-expert-factorial-v1"
CODE = (
    "examples/advanced/predictive_async/libero_expert_factorial.py",
    "examples/advanced/predictive_async/audit_libero_expert_factorial.py",
    "tests/test_libero_expert_factorial.py",
)


def paths(head):
    return (
        ROOT / f"outputs/smolvla_expert_factorial_preparation_{head[:8]}",
        ROOT / f"outputs/smolvla_expert_factorial_{head[:8]}",
    )


def sources():
    values = feasibility.sources()
    files = [
        *(ROOT / p for p in CODE),
        *(FEASIBILITY_OUT / p for p in ("result.json", "independent_audit.json", "evidence.pt")),
        ROOT / "outputs/smolvla_prefix_sampling_97a02e91/independent_audit.json",
    ]
    values.update({str(p): sha(p) for p in files})
    return values


def specification():
    return {
        "version": VERSION,
        "arms": list(ARMS),
        "labels": list(LABELS),
        "modes": list(MODES),
        "schedule": train.schedule(train.corpus()),
        "rank": RANK,
        "alpha": ALPHA,
        "seed": SEED,
        "updates_per_arm": 128,
        "train_forwards": 512,
        "decodes": 320,
        "denoise_steps": 3200,
        "c0_pairs": 80,
        "scored_outputs": 240,
        "batch": 1,
        "lr": 2.5e-6,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0,
        "clip_norm": 1,
        "no_middle_evaluation": True,
        "new_env": 0,
        "graph_captures": 0,
        "sealed_evaluations": 0,
        "note": "Same update/data budget, NOT matched FLOPs or matched parameter count.",
    }


def prepare(head):
    sampling.guard(head)
    audit = json.loads((FEASIBILITY_OUT / "independent_audit.json").read_text())
    if not audit["independent_contract_accepted"] or not audit["technical_followup_supported"]:
        raise ValueError("Expert feasibility did not pass")
    prep, out = paths(head)
    if prep.exists() or out.exists() or torch.cuda.is_initialized():
        raise ValueError("Unique CPU preparation required")
    prep.mkdir()
    saved = {
        "head": head,
        "sources": sources(),
        "specification": specification(),
        "environment": {k: os.environ.get(k) for k in sampling.ENV_KEYS},
    }
    write(prep / "preparation.json", saved)
    print(
        json.dumps(
            {
                "prep": str(prep),
                "out": str(out),
                "sources": len(saved["sources"]),
                "sha256": sha(prep / "preparation.json"),
            }
        ),
        flush=True,
    )


def validate(head):
    sampling.guard(head)
    prep, out = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    if (
        saved["sources"] != sources()
        or saved["specification"] != specification()
        or saved["environment"] != {k: os.environ.get(k) for k in sampling.ENV_KEYS}
    ):
        raise ValueError("Prepared source or experiment changed")
    body = (prep / "registration.md").read_text()
    rb = json.loads((prep / "registration_readback.json").read_text())
    if (
        type(rb.get("id")) is not int
        or rb.get("body") != body
        or rb.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        or f"F-EAT1-REGISTER:{head}" not in body
        or str(out) not in body
        or sha(prep / "preparation.json") not in body
    ):
        raise ValueError("Missing verified actual registration")
    return saved, rb["id"]


def checkpoint_state(label, model, handle, initial, optimizer):
    return {
        "version": VERSION,
        "arm": label,
        "base_policy": str(e.POLICY),
        "data_manifest_sha256": sha(COHORT / "manifest.json"),
        "updates": 128,
        "deployment_ready": False,
        "adapter_version": ADAPTER_VERSION,
        "adapter_metadata": handle.metadata() if handle else [],
        "projections": {n: p.detach().cpu().clone() for n, p in projection_parameters(model).items()},
        "adapters": handle.snapshot() if handle else {},
        "initial_adapters": initial,
        "optimizer": optimizer.state_dict(),
    }


def load_checkpoint(path, expected_sha, label, model, handle):
    if sha(path) != expected_sha:
        raise ValueError("New checkpoint hash changed")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (
        record["version"] != VERSION
        or record["arm"] != label
        or record["updates"] != 128
        or record["base_policy"] != str(e.POLICY)
        or record["data_manifest_sha256"] != sha(COHORT / "manifest.json")
        or record["deployment_ready"]
        or record["adapter_version"] != ADAPTER_VERSION
        or bool(record["adapters"]) != (handle is not None)
    ):
        raise ValueError("New checkpoint provenance mismatch")
    restore_projections(model, record["projections"])
    if handle:
        if record["adapter_metadata"] != handle.metadata():
            raise ValueError("Target mapping changed at reload")
        handle.restore(record["adapters"])
    return record


def execute(head):
    saved, registration = validate(head)
    _, out = paths(head)
    out.mkdir()
    result = {
        "experiment": "F-EAT1",
        "head": head,
        "registration_id": registration,
        "status": "technical_failure",
        "first_failure": None,
        "model_loads": 0,
        "training_forwards": 0,
        "backwards": 0,
        "updates": dict.fromkeys(ARMS, 0),
        "decodes": 0,
        "denoise_steps": 0,
        "vision_encodes": 0,
        "new_env": 0,
        "sealed_evaluations": 0,
        "graph_captures": 0,
        "attempts": 1,
        "retries": 0,
        "deployment_qualified": False,
    }
    policy, handle, original_encode = None, None, None
    initial, original, initial_hashes, checkpoints = {}, {}, {}, {}
    evaluation, references, first_training, arm_resources = [], {}, {}, {}
    started = time.perf_counter()

    def alarm(_s, _f):
        raise TimeoutError("F-EAT1 fixed1500s budget exhausted")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1500)
    calls = (out / "calls.jsonl").open("x")
    train_log = (out / "training.jsonl").open("x")
    call_count = 0

    def intent(kind, label, index, mode):
        nonlocal call_count
        if call_count >= 832:
            raise ValueError("Call budget exhausted")
        cid = call_count
        call_count += 1
        calls.write(
            json.dumps(
                {
                    "event": "intent",
                    "call_id": cid,
                    "kind": kind,
                    "label": label,
                    "index": index,
                    "mode": mode,
                }
            )
            + "\n"
        )
        calls.flush()
        return cid, time.perf_counter()

    def returned(cid, start):
        duration = time.perf_counter() - start
        if duration > 30:
            raise ValueError("Per-call30s budget exhausted")
        calls.write(json.dumps({"event": "return", "call_id": cid, "seconds": duration}) + "\n")
        calls.flush()

    try:
        torch.set_num_threads(1)
        rows = train.corpus()
        packets = {
            k: dict(np.load(COHORT / v["packet"]["path"], allow_pickle=False)) for k, v in rows.items()
        }
        policy, pre, post, load = e.load_runtime(e.POLICY, e.VLM)
        result["model_loads"] = 1
        write(out / "policy_load.json", load)
        original = dict(policy.model.named_parameters())
        initial_hashes = {n: tensor_sha(p) for n, p in original.items()}
        projection = projection_parameters(policy.model)
        initial = {n: p.detach().cpu().clone() for n, p in projection.items()}
        torch.save(initial, out / "initial_projections.pt")
        original_encode = policy.model.encode_image_tokens

        def observed_encode(*args, **kwargs):
            result["vision_encodes"] += 1
            return original_encode(*args, **kwargs)

        policy.model.encode_image_tokens = observed_encode
        schedule = saved["specification"]["schedule"]
        for arm in ARMS:
            restore_projections(policy.model, initial)
            handle = attach_expert_adapters(policy.model) if arm.startswith("expert_") else None
            selected = dict(projection)
            adapter_initial = handle.snapshot() if handle else {}
            if handle:
                selected.update(handle.parameters_by_name())
            for p in selected.values():
                p.requires_grad_(True)
            if any(p.requires_grad for n, p in original.items() if n not in projection):
                raise ValueError("Unexpected original expert/VLM training")
            optimizer = torch.optim.AdamW(
                list(selected.values()), lr=2.5e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False
            )
            torch.cuda.reset_peak_memory_stats()
            arm_start = time.perf_counter()
            forward_times, backward_times, update_times = [], [], []
            for spec in schedule["train"]:
                meta = rows[spec["trajectory_id"]]
                if meta["split"] != "train":
                    raise ValueError("Nontrain optimization sample")
                inputs, truth, valid = train.batch_for(
                    policy, pre, post, meta, packets[spec["trajectory_id"]], spec["anchor"]
                )
                generator = torch.Generator(device=truth.device).manual_seed(spec["noise_seed"])
                noise = torch.randn(truth.shape, generator=generator, device=truth.device)
                counts = torch.tensor([spec["C"]], device=truth.device)
                times = torch.tensor([spec["time"]], dtype=truth.dtype, device=truth.device)
                mode = "correct_prefix" if arm.endswith("_prefix") else "unconditioned"
                optimizer.zero_grad(set_to_none=True)
                cid, call_start = intent("train", arm, spec["step"], mode)
                torch.cuda.synchronize()
                timer = time.perf_counter()
                loss, velocity, flow = train.objective(
                    policy.model, inputs, truth, noise, times, counts, valid, mode
                )
                result["training_forwards"] += 1
                torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - timer)
                timer = time.perf_counter()
                loss.backward()
                result["backwards"] += 1
                torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - timer)
                if not torch.isfinite(loss) or any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in selected.values()
                ):
                    raise ValueError("Invalid training gradients")
                if spec["step"] == 0:
                    first_training[arm] = {
                        "velocity": velocity.detach().cpu().clone(),
                        "loss": float(loss.detach()),
                        "projection_gradients": {
                            n: p.grad.detach().cpu().clone() for n, p in projection.items()
                        },
                    }
                norms = (
                    {n: float(p.grad.float().norm()) for n, p in selected.items()}
                    if spec["step"] in (0, 127)
                    else {}
                )
                norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0, error_if_nonfinite=True)
                timer = time.perf_counter()
                optimizer.step()
                result["updates"][arm] += 1
                torch.cuda.synchronize()
                update_times.append(time.perf_counter() - timer)
                record = {
                    "arm": arm,
                    **spec,
                    "call_id": cid,
                    "loss": float(loss.detach()),
                    "coordinates": int(flow.loss_mask.sum()),
                    "target_sha256": tensor_sha(flow.target_velocity),
                    "mask_sha256": tensor_sha(flow.loss_mask),
                    "gradient_norm_before_clip": float(norm),
                    "gradient_norms": norms,
                    "forward_s": forward_times[-1],
                    "backward_s": backward_times[-1],
                    "update_s": update_times[-1],
                }
                train_log.write(json.dumps(record) + "\n")
                train_log.flush()
                returned(cid, call_start)
                if (spec["step"] + 1) % 32 == 0:
                    print(json.dumps({"arm": arm, "updates": spec["step"] + 1}), flush=True)
            resource = {
                "trainable_parameters": sum(p.numel() for p in selected.values()),
                "trainable_tensors": len(selected),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "training_wall_s": time.perf_counter() - arm_start,
                "forward_mean_s": sum(forward_times) / 128,
                "backward_mean_s": sum(backward_times) / 128,
                "update_mean_s": sum(update_times) / 128,
            }
            arm_resources[arm] = resource
            if resource["peak_allocated"] > 8 * 2**30:
                raise ValueError("8GiB training bound exceeded")
            state = checkpoint_state(arm, policy.model, handle, adapter_initial, optimizer)
            torch.save(state, out / f"{arm}_checkpoint.pt")
            checkpoints[arm] = sha(out / f"{arm}_checkpoint.pt")
            for p in selected.values():
                p.requires_grad_(False)
                p.grad = None
            if handle:
                handle.detach()
                handle = None
            del optimizer, state, selected
        for ordinary, prefix in [
            ("projection_ordinary", "expert_ordinary"),
            ("projection_prefix", "expert_prefix"),
        ]:
            a, b = first_training[ordinary], first_training[prefix]
            if (
                not torch.equal(a["velocity"], b["velocity"])
                or a["loss"] != b["loss"]
                or any(
                    not torch.equal(v, b["projection_gradients"][n])
                    for n, v in a["projection_gradients"].items()
                )
            ):
                raise ValueError("First zero-adapter training equivalence failed")
        parameter_hashes = {}
        for label in LABELS:
            restore_projections(policy.model, initial)
            handle = attach_expert_adapters(policy.model) if label.startswith("expert_") else None
            if label != "frozen":
                load_checkpoint(
                    out / f"{label}_checkpoint.pt", checkpoints[label], label, policy.model, handle
                )
            selected = dict(projection)
            if handle:
                selected.update(handle.parameters_by_name())
            for p in selected.values():
                p.requires_grad_(False)
                p.grad = None
            parameter_hashes[label] = {n: tensor_sha(p) for n, p in selected.items()}
            for spec in schedule["dev"]:
                key = spec["trajectory_id"]
                if rows[key]["split"] != "dev":
                    raise ValueError("Wrong evaluation split")
                inputs, truth, valid = train.batch_for(
                    policy, pre, post, rows[key], packets[key], spec["anchor"]
                )
                _, donor, _ = train.batch_for(
                    policy, pre, post, rows[spec["donor"]], packets[spec["donor"]], spec["donor_anchor"]
                )
                generator = torch.Generator(device=truth.device).manual_seed(spec["noise_seed"])
                noise = torch.randn(truth.shape, generator=generator, device=truth.device)
                fingerprint = {
                    "state": tensor_sha(inputs[-1]),
                    "language": tensor_sha(inputs[2]),
                    "language_mask": tensor_sha(inputs[3]),
                    "images": [tensor_sha(v) for v in inputs[0]],
                    "image_masks": [tensor_sha(v) for v in inputs[1]],
                }
                ref = {
                    "spec": spec,
                    "truth": truth.cpu(),
                    "valid": valid.cpu(),
                    "noise": noise.cpu(),
                    "donor_prefix": donor[:, : spec["C"], :7].cpu(),
                    "post_truth": post(truth[..., :7]).detach().cpu(),
                    "fingerprint": fingerprint,
                }
                if spec["case"] not in references:
                    references[spec["case"]] = ref
                else:
                    previous = references[spec["case"]]
                    if previous["fingerprint"] != fingerprint or any(
                        not torch.equal(ref[k], previous[k])
                        for k in ("truth", "valid", "noise", "donor_prefix", "post_truth")
                    ):
                        raise ValueError("Evaluation input differs across labels")
                native = None
                for mode in (*MODES, "calibration_c0"):
                    c = spec["C"] if mode in ("correct_prefix", "mismatched_prefix") else 0
                    prefix = (truth if mode == "correct_prefix" else donor)[:, :c, :7].clone() if c else None
                    trace, shapes = [], []
                    cid, call_start = intent("decode", label, spec["case"], mode)
                    hook = policy.model.action_out_proj.register_forward_hook(
                        lambda _m, _a, v, target=shapes: target.append(list(v.shape))
                    )
                    try:
                        with torch.no_grad():
                            full = (
                                policy.model.sample_actions(*inputs, noise=noise)
                                if mode == "native_unconditioned"
                                else sample_prefix_actions(
                                    policy.model,
                                    *inputs,
                                    noise,
                                    prefix=prefix,
                                    counts=torch.tensor([c], device=noise.device),
                                    trace=trace,
                                )
                            )
                            processed = post(full[..., :7]).detach().cpu().clone()
                            full_cpu = full.detach().cpu().clone()
                            metrics = sampling.score(full, truth, valid, spec["C"])
                        torch.cuda.synchronize()
                    finally:
                        hook.remove()
                    if shapes != [[1, 50, 32]] * 10:
                        raise ValueError("Wrong denoising count/shape")
                    result["decodes"] += 1
                    result["denoise_steps"] += 10
                    returned(cid, call_start)
                    if mode == "native_unconditioned":
                        native = (full_cpu, processed)
                    elif mode == "calibration_c0" and (
                        not torch.equal(native[0], full_cpu) or not torch.equal(native[1], processed)
                    ):
                        raise ValueError("New checkpoint C0 native mismatch")
                    evaluation.append(
                        {
                            "checkpoint": label,
                            "mode": mode,
                            **spec,
                            "call_id": cid,
                            "input_fingerprint": fingerprint,
                            "condition_count": c,
                            "condition_prefix": None if prefix is None else prefix.cpu(),
                            "full": full_cpu,
                            "processed": processed,
                            "metrics": metrics,
                            "trace": trace,
                        }
                    )
            if parameter_hashes[label] != {n: tensor_sha(p) for n, p in selected.items()}:
                raise ValueError("Evaluation changed parameters")
            if handle:
                handle.detach()
                handle = None
            print(json.dumps({"evaluated": label, "decodes_completed": result["decodes"]}), flush=True)
        restore_projections(policy.model, initial)
        if any(tensor_sha(p) != initial_hashes[n] for n, p in original.items()):
            raise ValueError("Original model not restored")
        if (
            result["updates"] != dict.fromkeys(ARMS, 128)
            or result["training_forwards"] != 512
            or result["decodes"] != 320
            or result["vision_encodes"] != 832
            or result["denoise_steps"] != 3200
        ):
            raise ValueError("Final population mismatch")
        result.update(
            status="completed",
            base_restored=True,
            frozen_parameters_unchanged=True,
            c0_pairs_exact=80,
            zero_adapter_start_pairs_exact=2,
            checkpoints=checkpoints,
            resources=arm_resources,
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
        parameter_hashes = locals().get("parameter_hashes", {})
    finally:
        if handle:
            handle.detach()
        if initial:
            restore_projections(policy.model, initial)
        for p in original.values():
            p.requires_grad_(False)
            p.grad = None
        if policy is not None and original_encode is not None:
            policy.model.encode_image_tokens = original_encode
        signal.alarm(0)
        calls.close()
        train_log.close()
        torch.save(
            {
                "references": references,
                "rows": evaluation,
                "first_training": first_training,
                "parameter_hashes": parameter_hashes,
            },
            out / "evidence.pt",
        )
        result.update(
            sources_unchanged=sources() == saved["sources"],
            saved_rows=len(evaluation),
            calls=call_count,
            wall_s=time.perf_counter() - started,
            evidence_sha256=sha(out / "evidence.pt"),
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
