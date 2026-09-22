"""F-PTI1: five fixed real-checkpoint training forwards, no optimizer or rollout."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from libero_rtc_graph import BASELINE, LOCK, REPO, e
from smolvla_prefix_training import build_prefix_flow, episode_action_window, prefix_training_forward

DATA = REPO / "outputs/smolvla_prefix_training_interface_20260922"
CASES = (
    ("native_c0", 0, 0),
    ("interface_c0", 0, 0),
    ("interface_c3", 3, 0),
    ("interface_c8", 8, 0),
    ("terminal_c1", 1, 1),
)
MODULES = ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj")
CODE = (
    "examples/advanced/predictive_async/audit_smolvla_prefix_training.py",
    "examples/advanced/predictive_async/smolvla_prefix_training.py",
    "examples/advanced/predictive_async/verify_smolvla_prefix_training.py",
    "examples/advanced/predictive_async/prepare_prefix_training_demo.py",
    "tests/test_smolvla_prefix_training.py",
    "docs/experiments/SMOLVLA_PREFIX_TRAINING_INTERFACE_PLAN.md",
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def pending():
    text = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    return {s[3:]: {"status": s[:2], "sha256": sha(REPO / s[3:])} for s in text.splitlines()}


def sources():
    lock = json.loads(LOCK.read_text())
    paths = [REPO / p for p in (*CODE, *lock["core_sha256"])]
    paths += [DATA / "demo_contract.json", DATA / "demo_packet.npz"]
    for directory in (e.POLICY, e.VLM):
        paths += sorted(directory.glob("*.safetensors")) + sorted(directory.glob("*.json"))
    return {str(p): sha(p) for p in paths}


def paths(head):
    return DATA / f"preparation_{head[:8]}.json", REPO / f"outputs/smolvla_prefix_training_real_{head[:8]}"


def guard(head):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != head:
        raise ValueError("HEAD changed")
    if pending() != json.loads(BASELINE.read_text())["pending"]:
        raise ValueError("Old pending changed or uncommitted execution files")
    lock = json.loads(LOCK.read_text())
    if any(sha(REPO / p) != value for p, value in lock["core_sha256"].items()):
        raise ValueError("Frozen execution core changed")


def prepare(head):
    guard(head)
    prep, out = paths(head)
    if prep.exists() or out.exists():
        raise ValueError("Unique preparation required")
    contract = json.loads((DATA / "demo_contract.json").read_text())
    if contract["episode"] != 814 or contract["packet_sha256"] != sha(DATA / "demo_packet.npz"):
        raise ValueError("Wrong demo packet")
    if any(sha(p) != h for p, h in contract["files"].items()):
        raise ValueError("Demo source changed")
    if torch.cuda.is_initialized():
        raise ValueError("Preparation must remain CPU-only")
    write(
        prep,
        {
            "head": head,
            "out": str(out),
            "sources": sources(),
            "cases": CASES,
            "model_loads": 1,
            "forwards": 5,
            "backwards": 5,
            "optimizer_updates": 0,
            "env": 0,
            "graph_captures": 0,
        },
    )
    print(json.dumps({"prepared": True, "prep": str(prep), "out": str(out), "sha256": sha(prep)}), flush=True)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def execute(head):
    guard(head)
    prep, out = paths(head)
    saved = json.loads(prep.read_text())
    if saved["sources"] != sources() or saved["cases"] != [list(c) for c in CASES] or out.exists():
        raise ValueError("Prepared sources/contract changed or output consumed")
    body = (DATA / "registration.md").read_text()
    rb = json.loads((DATA / "registration_readback.json").read_text())
    if (
        type(rb.get("id")) is not int
        or rb.get("body") != body
        or rb.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        or f"F-PTI1-REGISTER:{head}" not in body
        or sha(prep) not in body
    ):
        raise ValueError("Actual registration not verified")
    out.mkdir()
    result = {
        "experiment": "F-PTI1",
        "head": head,
        "status": "technical_failure",
        "first_failure": None,
        "registration_id": rb["id"],
        "forwards_started": 0,
        "forwards_returned": 0,
        "backwards": 0,
        "model_loads": 0,
        "optimizer_updates": 0,
        "new_env": 0,
        "graph_captures": 0,
        "attempts": 1,
        "retries": 0,
        "cases": [],
    }
    policy, params, evidence = None, [], {}
    started = time.perf_counter()

    def timeout(_signum, _frame):
        raise TimeoutError("F-PTI1 total time budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        torch.set_num_threads(1)
        policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
        result["model_loads"] = 1
        write(out / "policy_load.json", report)
        if time.perf_counter() - started > 90:
            raise TimeoutError("Model load exceeded 90 seconds")
        named = [
            (f"{module}.{name}", p)
            for module in MODULES
            for name, p in getattr(policy.model, module).named_parameters()
        ]
        params = [p for _, p in named]
        before = {name: tensor_sha(p) for name, p in named}
        for p in params:
            p.requires_grad_(True)
        packet = np.load(DATA / "demo_packet.npz", allow_pickle=False)
        contract = json.loads((DATA / "demo_contract.json").read_text())
        action_values = torch.from_numpy(packet["actions"])
        for label, count, anchor_select in CASES:
            result["forwards_started"] += 1
            began = time.perf_counter()
            index = int(packet["anchors"][anchor_select])
            raw_actions, valid = episode_action_window(action_values, index)
            raw = {
                "observation.state": torch.from_numpy(packet["states"][index : index + 1]),
                "action": raw_actions[None],
                "task": [contract["language"]],
            }
            for k, key in enumerate(e.CAMERA_KEYS):
                raw[key] = (
                    torch.from_numpy(packet["pixels"][anchor_select, k].copy()).permute(2, 0, 1)[None].float()
                    / 255
                )
            policy.reset()
            pre.reset()
            post.reset()
            batch = pre(raw)
            images, image_masks = policy.prepare_images(batch)
            state, actions = policy.prepare_state(batch), policy.prepare_action(batch)
            normal_roundtrip = post(batch["action"]).detach().cpu()
            torch.testing.assert_close(normal_roundtrip[0, valid], raw_actions[valid], atol=1e-5, rtol=1e-5)
            language_tokens = batch["observation.language.tokens"]
            language_masks = batch["observation.language.attention_mask"]
            generator = torch.Generator(device=actions.device).manual_seed(20260922)
            noise = torch.randn(actions.shape, generator=generator, device=actions.device)
            scalar_time = torch.full((1,), 0.5, device=actions.device)
            counts = torch.tensor([count], device=actions.device)
            valid = valid[None].to(actions.device)
            flow = build_prefix_flow(actions, noise, scalar_time, counts, valid)
            if label == "native_c0":
                captured = {}
                hook = policy.model.action_out_proj.register_forward_hook(
                    lambda _m, _a, v, storage=captured: storage.update(velocity=v)
                )
                try:
                    elementwise = policy.model(
                        images,
                        image_masks,
                        language_tokens,
                        language_masks,
                        state,
                        actions,
                        noise,
                        scalar_time,
                    )
                finally:
                    hook.remove()
                velocity = captured["velocity"]
                loss = elementwise[..., :7].mean()
            else:
                value = prefix_training_forward(
                    policy.model,
                    images,
                    image_masks,
                    language_tokens,
                    language_masks,
                    state,
                    actions,
                    noise,
                    scalar_time,
                    counts,
                    valid,
                )
                loss, velocity, elementwise, flow = (
                    value["loss"],
                    value["velocity"],
                    value["elementwise"],
                    value["flow"],
                )
            grads = torch.autograd.grad(loss, [velocity, *params])
            result["backwards"] += 1
            if any(not torch.isfinite(g).all() for g in grads) or any(g.abs().sum() == 0 for g in grads[1:]):
                raise ValueError("Nonfinite/zero projection gradient")
            if torch.count_nonzero(grads[0][~flow.loss_mask]):
                raise ValueError("Excluded prediction coordinates carry direct loss gradients")
            item = {
                "elementwise": elementwise.detach().cpu(),
                "velocity": velocity.detach().cpu(),
                "loss": loss.detach().cpu(),
                "prediction_gradient": grads[0].cpu(),
                "parameter_gradients": {n: g.cpu() for (n, _), g in zip(named, grads[1:], strict=True)},
                "loss_mask": flow.loss_mask.cpu(),
                "valid_steps": valid.cpu(),
                "counts": counts.cpu(),
                "flow_times": flow.flow_times.cpu(),
                "target": flow.target_velocity.cpu(),
                "noisy_actions": flow.noisy_actions.cpu(),
                "actions": actions.detach().cpu(),
                "noise": noise.cpu(),
                "state": state.detach().cpu(),
                "source_row": index,
            }
            evidence[label] = item
            if label == "interface_c0":
                reference = evidence["native_c0"]
                for key in ("elementwise", "velocity", "loss", "prediction_gradient"):
                    if not torch.equal(reference[key], item[key]):
                        raise ValueError("C0 native equivalence differs: " + key)
                for name in item["parameter_gradients"]:
                    if not torch.equal(
                        reference["parameter_gradients"][name], item["parameter_gradients"][name]
                    ):
                        raise ValueError("C0 parameter gradient differs: " + name)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            if elapsed > 60:
                raise TimeoutError("Case exceeded 60 seconds")
            result["forwards_returned"] += 1
            result["cases"].append(
                {
                    "label": label,
                    "C": count,
                    "source_row": index,
                    "valid_rows": int(valid.sum()),
                    "coordinates": int(flow.loss_mask.sum()),
                    "loss": float(loss.detach()),
                    "seconds": elapsed,
                }
            )
            print(json.dumps(result["cases"][-1]), flush=True)
            del velocity, elementwise, loss, grads
        result["projection_parameters_unchanged"] = before == {name: tensor_sha(p) for name, p in named}
        if not result["projection_parameters_unchanged"] or saved["sources"] != sources():
            raise ValueError("Source/parameter mutation")
        result.update(
            status="completed",
            c0_exact=True,
            projection_gradient_tensors=len(params),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_sources_unchanged=True,
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        for p in params:
            p.requires_grad_(False)
        result["requires_grad_restored"] = policy is None or all(
            not p.requires_grad and p.grad is None for p in policy.parameters()
        )
        result["wall_s"] = time.perf_counter() - started
        torch.save(evidence, out / "evidence.pt")
        result["evidence_sha256"] = sha(out / "evidence.pt")
        write(out / "result.json", result)
        signal.alarm(0)
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
