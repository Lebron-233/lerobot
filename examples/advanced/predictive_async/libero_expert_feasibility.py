"""F-EAF1: six matched technical optimizer probes; no scientific promotion."""

import argparse
import json
import os
import signal
import time
import traceback

import libero_prefix_sampling as prior
import libero_prefix_train_pilot as train
import numpy as np
import torch
from smolvla_expert_lora import ADAPTER_VERSION, ALPHA, RANK, SEED, attach_expert_adapters
from smolvla_prefix_sampling import projection_parameters, restore_projections
from verify_smolvla_prefix_training import tensor_sha

ROOT, COHORT, e, sha, write = train.ROOT, train.COHORT, train.e, train.sha, train.write
CODE = (
    "examples/advanced/predictive_async/smolvla_expert_lora.py",
    "examples/advanced/predictive_async/libero_expert_feasibility.py",
    "examples/advanced/predictive_async/audit_libero_expert_feasibility.py",
    "tests/test_smolvla_expert_lora.py",
    "docs/experiments/SMOLVLA_EXPERT_ADAPTATION_PLAN.md",
)
STAGES = ("projection", "expert")


def paths(head):
    return (
        ROOT / f"outputs/smolvla_expert_feasibility_preparation_{head[:8]}",
        ROOT / f"outputs/smolvla_expert_feasibility_{head[:8]}",
    )


def sources():
    values = prior.sources()
    values.update(
        {str(ROOT / p): sha(ROOT / p) for p in (*CODE, "src/lerobot/policies/smolvla/smolvlm_with_expert.py")}
    )
    return values


def spec():
    value = dict(train.schedule(train.corpus())["train"][0])
    value.update(C=3, time=0.5, noise_seed=2026092202)
    return value


def prepare(head):
    prior.guard(head)
    prep, out = paths(head)
    if prep.exists() or out.exists() or torch.cuda.is_initialized():
        raise ValueError("Unique CPU preparation required")
    prep.mkdir()
    write(
        prep / "preparation.json",
        {
            "head": head,
            "sources": sources(),
            "sample": spec(),
            "adapter_version": ADAPTER_VERSION,
            "rank": RANK,
            "alpha": ALPHA,
            "seed": SEED,
            "stages": STAGES,
            "updates_per_stage": 3,
            "environment": {k: os.environ.get(k) for k in prior.ENV_KEYS},
        },
    )
    print(
        json.dumps({"prep": str(prep), "out": str(out), "sha256": sha(prep / "preparation.json")}), flush=True
    )


def validate(head):
    prior.guard(head)
    prep, out = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    if (
        saved["sources"] != sources()
        or saved["sample"] != spec()
        or saved["environment"] != {k: os.environ.get(k) for k in prior.ENV_KEYS}
    ):
        raise ValueError("Prepared sources or settings changed")
    body = (prep / "registration.md").read_text()
    rb = json.loads((prep / "registration_readback.json").read_text())
    if (
        type(rb.get("id")) is not int
        or rb.get("body") != body
        or rb.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        or f"F-EAF1-REGISTER:{head}" not in body
        or str(out) not in body
        or sha(prep / "preparation.json") not in body
    ):
        raise ValueError("Missing actual registration")
    return saved, rb["id"]


def execute(head):
    saved, registration = validate(head)
    _, out = paths(head)
    out.mkdir()
    result = {
        "experiment": "F-EAF1",
        "head": head,
        "registration_id": registration,
        "status": "technical_failure",
        "first_failure": None,
        "model_loads": 0,
        "forwards": 0,
        "backwards": 0,
        "optimizer_updates": 0,
        "new_env": 0,
        "sealed_evaluations": 0,
        "graph_captures": 0,
        "attempts": 1,
        "retries": 0,
    }
    policy, handle, initial, original, evidence = None, None, {}, {}, {}
    began = time.perf_counter()

    def timeout(_s, _f):
        raise TimeoutError("F-EAF1 300s budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        torch.set_num_threads(1)
        rows = train.corpus()
        row = rows[saved["sample"]["trajectory_id"]]
        if row["split"] != "train":
            raise ValueError("Preflight uses train only")
        packet = dict(np.load(COHORT / row["packet"]["path"], allow_pickle=False))
        policy, pre, post, load = e.load_runtime(e.POLICY, e.VLM)
        result["model_loads"] = 1
        write(out / "policy_load.json", load)
        original = dict(policy.model.named_parameters())
        original_hashes = {n: tensor_sha(p) for n, p in original.items()}
        projection = projection_parameters(policy.model)
        initial = {n: p.detach().cpu().clone() for n, p in projection.items()}
        inputs, truth, valid = train.batch_for(policy, pre, post, row, packet, saved["sample"]["anchor"])
        generator = torch.Generator(device=truth.device).manual_seed(saved["sample"]["noise_seed"])
        noise = torch.randn(truth.shape, generator=generator, device=truth.device)
        times, counts = torch.tensor([0.5], device=truth.device), torch.tensor([3], device=truth.device)
        for label in STAGES:
            restore_projections(policy.model, initial)
            selected = dict(projection)
            handles, hits = [], {}
            if label == "expert":
                handle = attach_expert_adapters(policy.model)
                metadata = handle.metadata()
                if [v["global_layer"] for v in metadata] != [i for i in range(0, 32, 2) for _ in range(2)]:
                    raise ValueError("Unexpected actual target layers")
                selected.update(handle.parameters_by_name())
                result["targets"] = metadata
                for path, _parent, _name, _base, adapter, _index in handle.slots:
                    hits[path] = 0

                    def observe(_m, _a, _v, key=path, target=hits):
                        target[key] += 1

                    handles.append(adapter.register_forward_hook(observe))
            for p in selected.values():
                p.requires_grad_(True)
            initial_selected = {n: p.detach().cpu().clone() for n, p in selected.items()}
            optimizer = torch.optim.AdamW(
                list(selected.values()), lr=2.5e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False
            )
            torch.cuda.reset_peak_memory_stats()
            steps = []
            for i in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss, velocity, flow = train.objective(
                    policy.model, inputs, truth, noise, times, counts, valid, "correct_prefix"
                )
                result["forwards"] += 1
                torch.cuda.synchronize()
                fwd = time.perf_counter() - start
                start_backward = time.perf_counter()
                loss.backward()
                result["backwards"] += 1
                torch.cuda.synchronize()
                bwd = time.perf_counter() - start_backward
                if not torch.isfinite(loss) or any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in selected.values()
                ):
                    raise ValueError("Nonfinite or disconnected gradients")
                gradient = {n: p.grad.detach().cpu().clone() for n, p in selected.items()}
                norms = {n: float(g.float().norm()) for n, g in gradient.items()}
                norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0, error_if_nonfinite=True)
                before_update = time.perf_counter()
                optimizer.step()
                result["optimizer_updates"] += 1
                torch.cuda.synchronize()
                upd = time.perf_counter() - before_update
                steps.append(
                    {
                        "step": i,
                        "loss": float(loss.detach()),
                        "velocity": velocity.detach().cpu(),
                        "target": flow.target_velocity.cpu(),
                        "mask": flow.loss_mask.cpu(),
                        "projection_gradients": {n: gradient[n] for n in projection} if i == 0 else {},
                        "gradient_norms": norms,
                        "clip_norm": float(norm),
                        "forward_s": fwd,
                        "backward_s": bwd,
                        "update_s": upd,
                    }
                )
                if fwd + bwd + upd > 30:
                    raise ValueError("Per-step technical budget exceeded")
            for h in handles:
                h.remove()
            final_selected = {n: p.detach().cpu().clone() for n, p in selected.items()}
            evidence[label] = {
                "initial": initial_selected,
                "final": final_selected,
                "steps": steps,
                "optimizer": optimizer.state_dict(),
                "hits": hits,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "trainable_parameters": sum(p.numel() for p in selected.values()),
            }
            if evidence[label]["peak_allocated"] > 8 * 2**30:
                raise ValueError("8GiB preflight memory gate exceeded")
            for p in selected.values():
                p.requires_grad_(False)
                p.grad = None
            if handle:
                handle.detach()
                handle = None
            del optimizer
        a, b = evidence["projection"]["steps"][0], evidence["expert"]["steps"][0]
        if (
            not torch.equal(a["velocity"], b["velocity"])
            or a["loss"] != b["loss"]
            or any(
                not torch.equal(a["projection_gradients"][n], b["projection_gradients"][n])
                for n in projection
            )
        ):
            raise ValueError("Zero-LoRA output or gradient identity failed")
        if any(v != 3 for v in evidence["expert"]["hits"].values()):
            raise ValueError("An adapter did not execute once per forward")
        for suffix, step, positive in [("lora_A", 0, False), ("lora_B", 0, True), ("lora_A", 1, True)]:
            values = [
                v
                for n, v in evidence["expert"]["steps"][step]["gradient_norms"].items()
                if n.endswith(suffix)
            ]
            if not values or (positive and not all(v > 0 for v in values)) or (not positive and any(values)):
                raise ValueError("LoRA initialization/gradient connectivity gate failed")
        restore_projections(policy.model, initial)
        if any(tensor_sha(p) != original_hashes[n] for n, p in original.items()):
            raise ValueError("Original pretrained parameters changed")
        result.update(
            status="completed",
            zero_adapter_exact=True,
            base_restored=True,
            technical_followup_supported=True,
            stages={
                k: {n: v[n] for n in ("peak_allocated", "peak_reserved", "trainable_parameters")}
                for k, v in evidence.items()
            },
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if handle:
            handle.detach()
        if initial:
            restore_projections(policy.model, initial)
        for p in original.values():
            p.requires_grad_(False)
            p.grad = None
        signal.alarm(0)
        torch.save(evidence, out / "evidence.pt")
        result.update(
            sources_unchanged=sources() == saved["sources"],
            wall_s=time.perf_counter() - began,
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
