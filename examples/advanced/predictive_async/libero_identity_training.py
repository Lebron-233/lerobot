"""F-ITC1: matched identity-centered fine-tuning, with/without soft error penalties.

OLD development only. Fixed warm start and final step; never selects on validation.
"""

import argparse
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import libero_identity_anchor_probe as p
import torch

q, e, pilot = p.q, p.e, p.pilot
REPO = p.REPO
IAR = REPO / "outputs/smolvla_identity_anchor_1f74c04d"
WEIGHTS = q.pfx.SOURCE / "training_weights.json"
ARMS = ("plain", "guarded")
STEPS, SEED, LR, WD = 72, 20260912, 0.0001, 0.0001
LIMITS = {"decoder": 930, "predictor": 1508, "backward": 144, "updates": 144}
CAPTURES, SOFT, HARD = 24, 900, 930
require, write, digest = p.require, p.write, p.digest


def paths(head):
    return (REPO / f"outputs/smolvla_identity_training_preparation_{head[:8]}",
            REPO / f"outputs/smolvla_identity_training_{head[:8]}")


def specification():
    return {"experiment": "F-ITC1", "arms": list(ARMS), "updates_per_arm": STEPS,
            "seed": SEED, "lr": LR, "weight_decay": WD, "betas": [0.9, 0.999], "eps": 1e-8,
            "foreach": False, "gradient_clip": 1.0, "warm_start_step": 72,
            "checkpoint_selection": "fixed_final_72_no_validation_selection",
            "guard_coefficients": {"chunk": 1.0, "row0_excess": 1.0, "chunk_excess": 1.0},
            "limits": LIMITS, "captures": CAPTURES, "soft": SOFT, "hard": HARD,
            "qualification_reads": 0, "new_env": 0}


def source_hashes():
    files = [*p.source_paths(), WEIGHTS, IAR / "independent_audit.json",
             *sorted((IAR / "predictions").glob("*.pt"))]
    require(len(list((IAR / "predictions").glob("*.pt"))) == 88, "IAR coverage differs")
    return {str(f.relative_to(REPO)): digest(f) for f in files}


def load_data():
    require(json.loads((IAR / "independent_audit.json").read_text())[
        "independent_contract_accepted"] is True, "IAR source not accepted")
    samples, bases, residuals, donors, contract = p.load_sources()
    weights = json.loads(WEIGHTS.read_text())
    train = [s for s in samples if s["split"] == "train"]
    require([r["key"] for r in weights["rows"]] == [list(q.pfx.key(s)) for s in train], "Weight identities differ")
    scales = {k: weights["scales"][k] for k in ("latent", "row0")}
    scales["chunk"] = math.fsum(p.metrics(s["inputs"][:2], s["archived_full_chunk"], s)["chunk"]
                                for s in train) / len(train)
    require(all(math.isfinite(v) and v > 0 for v in scales.values()), "Invalid training-only scales")
    refs = {q.pfx.key(s): torch.load(IAR / "predictions" / ("_".join(map(str, q.pfx.key(s))) + ".pt"),
                                   map_location="cpu", weights_only=False) for s in samples}
    return samples, bases, residuals, donors, contract, weights, scales, refs


def objective(metrics, identity, scales, weight, arm):
    require(arm in ARMS, "Unknown objective arm")
    latent, row0, chunk = metrics
    loss = latent / scales["latent"] + weight * row0 / scales["row0"]
    if arm == "guarded":
        loss = loss + chunk / scales["chunk"]
        loss = loss + torch.relu(row0 - identity["row0"]) / scales["row0"]
        loss = loss + torch.relu(chunk - identity["chunk"]) / scales["chunk"]
    return loss


def take(counts, name):
    require(counts[name] < LIMITS[name], f"{name} exhausted before dispatch")
    counts[name] += 1


def freeze(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def prediction(model, sample, actions, counts):
    batch = pilot.batch_for([sample], [0], "cuda")
    args = (batch["tokens"], batch["masks"])
    tail = (batch["action_mask"], batch["state"], batch["delay"])
    take(counts, "predictor")
    a = model(*args, actions.to("cuda"), *tail).delta_tokens
    take(counts, "predictor")
    z = model(*args, torch.zeros_like(actions, device="cuda"), *tail).delta_tokens
    raw = p.transplant(tuple(v.cuda() for v in sample["inputs"][:2]), a, z)
    visual = tuple(v.to(torch.bfloat16).float() for v in raw)
    evidence = {"action_delta": q.cpu(a), "zero_delta": q.cpu(z), "raw": q.cpu(raw),
                "visual": q.cpu(visual), "actions": actions.cpu().clone(), "mask": sample["mask"].clone()}
    return visual, evidence


class Runtime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        require(len(self.captures) < CAPTURES, "Capture budget exhausted")
        return super()._capture(inputs)


def decode(policy, runtime, sample, visual, output, counts, label, gradients=False):
    take(counts, "decoder")
    values = tuple(v.cuda() for v in sample["inputs"])
    visual = tuple(v.to(device="cuda", dtype=values[i].dtype) for i, v in enumerate(visual))
    with pilot.phase(output, f"decode_{counts['decoder']}_{label}", 30):
        if gradients:
            counts["gradient_decoder"] += 1
            value = policy.model.sample_actions(None, None, values[4], values[5], values[6],
                noise=values[7].clone(), future_image_tokens=visual, future_image_token_masks=tuple(values[2:4]))
        else:
            runtime.begin_episode("graph", sample["task"])
            with torch.no_grad():
                value = runtime(None, None, values[4], values[5], values[6], noise=values[7],
                    future_image_tokens=visual, future_image_token_masks=tuple(values[2:4])).detach().cpu().clone()
        require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), "Invalid decoder output")
    return value


def statistics(rows):
    summaries = {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        names = sorted({a for r in selected for a in r["metrics"]})
        arms, contrasts = {}, {}
        for name in names:
            groups = defaultdict(list)
            for r in selected:
                if name in r["metrics"]:
                    groups["/".join(map(str, r["key"][:2]))].append(r["metrics"][name])
            eps = {k: {m: math.fsum(x[m] for x in v) / len(v) for m in ("row0", "chunk", "latent")}
                   for k, v in groups.items()}
            arms[name] = {"samples": sum(map(len, groups.values())), "episodes": eps,
                          "macro": {m: math.fsum(v[m] for v in eps.values()) / len(eps)
                                    for m in ("row0", "chunk", "latent")}}
        comparisons = [(f"{a}_true", c) for a in ARMS for c in ("identity", "frozen", f"{a}_mismatched")]
        comparisons.append(("guarded_true", "plain_true"))
        for treatment, control in comparisons:
            subset = [r for r in selected if control in r["metrics"] and treatment in r["metrics"]]
            groups, details = defaultdict(list), []
            for r in subset:
                a, b = r["metrics"][treatment]["row0"], r["metrics"][control]["row0"]
                groups["/".join(map(str, r["key"][:2]))].append((a, b))
                details.append({"key": r["key"], "benefit": b-a, "direction":
                    "improved" if b-a > p.tolerance(b) else "worsened" if a-b > p.tolerance(b) else "tied"})
            eps = {k: {"treatment": math.fsum(a for a, _ in v)/len(v),
                       "control": math.fsum(b for _, b in v)/len(v)} for k, v in groups.items()}
            gains = {k: v["control"]-v["treatment"] for k, v in eps.items()}
            contrasts[f"{treatment}_vs_{control}"] = {"samples": len(subset), "episodes": eps,
                "macro_benefit": math.fsum(gains.values())/len(gains),
                "sample_directions": dict(Counter(r["direction"] for r in details)),
                "episode_improved": sum(gains[k] > p.tolerance(v["control"]) for k, v in eps.items()),
                "leave_one_out": {k: math.fsum(v for j, v in gains.items() if j != k)/(len(gains)-1) for k in gains},
                "per_sample": details}
        harms = {a: max([0.0] + [r["metrics"][a]["row0"]-r["metrics"]["identity"]["row0"]
                                  for r in selected]) for a in ("frozen", "plain_true", "guarded_true")}
        summaries[split] = {"metrics": arms, "contrasts": contrasts, "worst_row0_excess_vs_identity": harms}
    v, t = summaries["validation"], summaries["train"]
    g = v["metrics"]["guarded_true"]["macro"]
    checks = {}
    for name in ("identity", "guarded_mismatched"):
        c = v["contrasts"][f"guarded_true_vs_{name}"]
        control = math.fsum(x["control"] for x in c["episodes"].values()) / len(c["episodes"])
        checks[f"row0_better_{name}"] = c["macro_benefit"] > p.tolerance(control)
    frozen = v["metrics"]["frozen"]["macro"]
    checks["row0_not_worse_frozen"] = g["row0"] <= frozen["row0"] + p.tolerance(frozen["row0"])
    for name in ("frozen", "plain_true"):
        c = v["metrics"][name]["macro"]["chunk"]
        checks[f"chunk_better_{name}"] = g["chunk"] < c-p.tolerance(c)
        h = t["worst_row0_excess_vs_identity"]
        checks[f"train_worst_excess_better_{name}"] = h["guarded_true"] < h[name]-p.tolerance(h[name])
    c = v["contrasts"]["guarded_true_vs_identity"]
    checks["three_of_four_episodes"] = c["episode_improved"] >= 3
    checks["strict_sample_majority"] = c["sample_directions"].get("improved", 0) > 8
    h = v["worst_row0_excess_vs_identity"]
    checks["validation_worst_excess_not_worse_frozen"] = h["guarded_true"] <= h["frozen"]+p.tolerance(h["frozen"])
    return {"splits": summaries, "development_checks": checks,
            "development_followup_supported": all(checks.values()), "independent_qualification_claimed": False}


def prepare(head):
    p.worktree_gate(head)
    environment = q.runtime_environment()
    prep, output = paths(head)
    require(not prep.exists() and not output.exists(), "Preparation/output exists; do not overwrite")
    samples, _, _, _, manifest, weights, scales, _ = load_data()
    saved = torch.load(p.ARCHIVE / "centered.pt", map_location="cpu", weights_only=False)
    q.checkpoint_valid(saved, "centered")
    require(not torch.cuda.is_initialized(), "Preparation initialized CUDA")
    prep.mkdir()
    write(prep / "preparation.json", {"execution_head": head, "specification": specification(),
          "source_hashes": source_hashes(), "manifest": manifest, "runtime_environment": environment,
          "scales": scales, "weights": weights, "samples": len(samples), "model_forwards": 0,
          "new_env": 0, "qualification_reads": 0})
    print(json.dumps({"prepared": True, "prep": str(prep), "output": str(output),
                      "sha256": digest(prep / "preparation.json"), "scales": scales}), flush=True)


def validate(head):
    p.worktree_gate(head)
    prep, output = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    require(saved["execution_head"] == head and saved["specification"] == specification(), "Contract changed")
    require(saved["source_hashes"] == source_hashes(), "Sources changed")
    require(saved["runtime_environment"] == q.runtime_environment(), "Runtime environment changed")
    body = (prep / "registration.md").read_text()
    post = json.loads((prep / "registration_post.json").read_text())
    got = json.loads((prep / "registration_readback.json").read_text())
    require(type(got.get("id")) is int and got["id"] == post.get("id") and got.get("body") == post.get("body") == body
            and got.get("issue_url") == "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
            and all(x in body for x in (f"F-ITC1-REGISTER:{head}", str(output), digest(prep / "preparation.json"))),
            "Actual-ID registration exact gate failed")
    return saved


def worker(args):
    prepared = validate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime, rows = Counter(), None, []
    result = {"experiment": "F-ITC1", "execution_head": args.execution_head, "status": "technical_failure", "first_failure": None}
    try:
        with pilot.phase(args.output, "load_development", 30):
            samples, _, residuals, donors, manifest, weights, scales, refs = load_data()
            require(manifest == prepared["manifest"] and weights == prepared["weights"] and scales == prepared["scales"],
                    "Prepared development metadata changed")
            train = [s for s in samples if s["split"] == "train"]
            order = q.cov.schedule(train, "multi_conditioned")
            require(len(order) == STEPS and len(set(order)) == 72, "Training schedule differs")
            by_key = {q.pfx.key(s): s for s in samples}
        with pilot.phase(args.output, "load_models", 90):
            require(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU differs")
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts["vla_loads"] += 1
            write(args.output / "policy_load.json", report)
            runtime = Runtime(policy.model)
            checkpoint = torch.load(p.ARCHIVE / "centered.pt", map_location="cpu", weights_only=False)
            q.checkpoint_valid(checkpoint, "centered")
            models = {}
            for arm in ARMS:
                torch.manual_seed(SEED)
                model = pilot.LightweightFutureLatentPredictor(pilot.config()).cuda()
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                require(sum(v.numel() for v in model.parameters()) == 69680, "Predictor size changed")
                models[arm] = model.eval()
                counts["predictor_loads"] += 1
                torch.save(freeze(model), args.output / f"initial_{arm}.pt")
        for directory in ("controls", "training", "predictions"):
            (args.output / directory).mkdir()
        controls = {}
        with torch.no_grad(), pilot.phase(args.output, "initial_replay", 180):
            for sample in samples:
                key, ref = q.pfx.key(sample), refs[q.pfx.key(sample)]
                visual, evidence = prediction(models["plain"], sample, sample["actions"], counts)
                for field, old_field in (("action_delta", "action_delta"), ("zero_delta", "zero_delta")):
                    for a, b in zip(evidence[field], residuals[key, "true"][old_field], strict=True):
                        pilot.require_equal(a, b, "Warm-start branch differs from F-ACR1")
                for a, b in zip(evidence["raw"], ref["raw_new"]["identity_true"], strict=True):
                    pilot.require_equal(a, b, "Warm-start identity raw differs")
                outputs, scores = {}, {}
                for name, old_name, v in (("identity", "identity", sample["inputs"][:2]),
                    ("old_centered", "centered", ref["visual"]["centered"]), ("frozen", "identity_true", visual)):
                    value = decode(policy, runtime, sample, v, args.output, counts, f"initial_{name}")
                    pilot.require_equal(value, ref["outputs"][old_name], f"Initial {name} differs")
                    outputs[name], scores[name] = value, p.metrics(v, value, sample)
                    counts[f"{name}_exact"] += 1
                controls[key] = {"key": list(key), "outputs": outputs, "metrics": scores, "evidence": evidence}
                torch.save(controls[key], args.output / "controls" / ("_".join(map(str, key))+".pt"))
        runtime.release_graph()
        assessed = {}
        for arm in ARMS:
            model = models[arm]
            model.train().requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999), eps=1e-8, foreach=False)
            with pilot.phase(args.output, f"train_{arm}", 180):
                for step, index in enumerate(order, 1):
                    sample = train[index]
                    key = q.pfx.key(sample)
                    with pilot.phase(args.output, f"update_{arm}_{step}", 30):
                        optimizer.zero_grad(set_to_none=True)
                        visual, evidence = prediction(model, sample, sample["actions"], counts)
                        value = decode(policy, runtime, sample, visual, args.output, counts, arm, True)
                        metrics = q.cov.metric_values(visual, value, sample)
                        weight = weights["rows"][index]["case_weight"]
                        loss = objective(metrics, controls[key]["metrics"]["identity"], scales, weight, arm)
                        require(torch.isfinite(loss), "Nonfinite loss")
                        take(counts, "backward")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        require(torch.isfinite(norm) and all(v.grad is None for v in policy.parameters()), "Gradient or VLA freeze failed")
                        gradients = {n: v.grad.detach().cpu().clone() for n, v in model.named_parameters() if v.grad is not None}
                        require(gradients and all(torch.isfinite(v).all() for v in gradients.values()), "Invalid predictor gradients")
                        take(counts, "updates")
                        optimizer.step()
                        record = {"arm": arm, "step": step, "key": list(key), "weight": weight, "scales": scales,
                                  "metrics": dict(zip(("latent", "row0", "chunk"), map(float, metrics), strict=True)),
                                  "objective": float(loss), "gradient_norm_before_clip": float(norm),
                                  "evidence": evidence, "output": value.detach().cpu().clone(),
                                  "clipped_gradients": gradients, "weights_after": freeze(model)}
                        torch.save(record, args.output / "training" / f"{arm}_{step:03d}.pt")
                        with (args.output / "training.jsonl").open("a") as stream:
                            stream.write(json.dumps({k: record[k] for k in ("arm", "step", "key", "metrics", "objective", "gradient_norm_before_clip")})+"\n")
            model.eval().requires_grad_(False)
            torch.save({"arm": arm, "step": STEPS, "source_step": 72, "state_dict": freeze(model),
                        "specification": specification()}, args.output / f"{arm}.pt")
            with torch.no_grad(), pilot.phase(args.output, f"evaluate_{arm}", 180):
                for sample in samples:
                    key = q.pfx.key(sample)
                    evidence, outputs, scores = {}, {}, {}
                    for context in ("true", "zero", "mismatched"):
                        if context == "mismatched" and donors[key] is None:
                            continue
                        actions = (torch.zeros_like(sample["actions"]) if context == "zero" else
                                   by_key[donors[key]]["actions"] if context == "mismatched" else sample["actions"])
                        visual, ev = prediction(model, sample, actions, counts)
                        value = decode(policy, runtime, sample, visual, args.output, counts, f"{arm}_{context}")
                        if context == "zero":
                            for a, b in zip(ev["raw"], sample["inputs"][:2], strict=True):
                                pilot.require_equal(a, b.float(), "Zero raw invariant failed")
                            pilot.require_equal(value, controls[key]["outputs"]["identity"], "Zero output invariant failed")
                            counts["zero_exact"] += 1
                        evidence[context], outputs[context], scores[context] = ev, value, p.metrics(visual, value, sample)
                    assessed[arm, key] = scores
                    torch.save({"key": list(key), "arm": arm, "donor": list(donors[key]) if donors[key] else None,
                                "evidence": evidence, "outputs": outputs, "metrics": scores},
                               args.output / "predictions" / (arm+"_"+"_".join(map(str, key))+".pt"))
            runtime.release_graph()
            del optimizer
        for s in samples:
            key = q.pfx.key(s)
            scores = dict(controls[key]["metrics"])
            for arm in ARMS:
                scores.update({f"{arm}_{c}": v for c, v in assessed[arm, key].items()})
            rows.append({"key": list(key), "split": s["split"], "delay": s["delay"],
                         "donor": list(donors[key]) if donors[key] else None, "metrics": scores})
        with (args.output / "rows.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        require(all(counts[k] == v for k, v in LIMITS.items()), "Formal budget incomplete")
        require(counts["zero_exact"] == 176 and counts["gradient_decoder"] == 144, "Evaluation/gradient counts differ")
        require(all(not v.requires_grad and v.grad is None for v in policy.parameters()), "VLA not frozen")
        require(source_hashes() == prepared["source_hashes"], "Source files changed")
        result.update(status="completed", vla_frozen=True, samples=88, **statistics(rows))
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if runtime is not None:
            write(args.output / "captures.json", runtime.captures)
            counts["captures"] = len(runtime.captures)
            runtime.release_graph()
            result["graph_released"] = runtime.graph is None
        result.update(counts=dict(counts), specification=specification(), attempts=1, retries=0,
                      new_env=0, image_encodings=0, qualification_reads=0, real_robot=0,
                      baseline_qualified=False, realtime_qualified=False, predictor_benefit_tested=False,
                      risk_thresholds=None, old_confirmation="untouched")
        write(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    validate(args.execution_head)
    require(args.output == paths(args.execution_head)[1] and not args.output.exists(), "Use unique registered output")
    args.output.mkdir()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--execution-head", args.execution_head,
               "--output", str(args.output), "--worker"]
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"F-ITC1 worker={child.pid}", flush=True)
        try:
            while child.poll() is None:
                q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now-v["at"] > v["limit"]]
                if terminated is None and (expired or now-start >= SOFT):
                    reason = {"expired": expired, "soft_timeout": now-start >= SOFT}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now-start >= HARD or (terminated is not None and now-terminated >= 5):
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
    q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
    file = args.output / "worker_result.json"
    result = json.loads(file.read_text()) if file.exists() else {"status": "technical_failure",
             "first_failure": "Worker receipt absent", "execution_head": args.execution_head}
    result["execution"] = {"child_pid": child.pid, "child_exit_code": code, "exit_confirmed": True,
        "started_at_utc": utc, "finished_at_utc": datetime.now(UTC).isoformat(), "wall_seconds": time.perf_counter()-start,
        "stop_reason": reason, "forced_termination": forced, "pending": pending, "active": active}
    if code or reason or pending or active:
        result["status"] = "technical_failure"
    write(args.output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "first_failure", "development_followup_supported")}), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        require(not args.worker and args.output is None, "Prepare cannot start worker")
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
