"""F-IAR1: one frozen residual transplant on OLD development data only.

No training, predictor forward, image encoding, native environment, or ACQ data.
The residual is replayed from independently accepted F-ACR1 saved branch arrays.
"""

import argparse
import faulthandler
import hashlib
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

import libero_action_qualification as q
import torch

REPO, e, pilot = q.e.REPO, q.e, q.pilot
ARCHIVE = REPO / "outputs/smolvla_action_centered_2926678f"
PRESERVED = REPO / "outputs/smolvla_identity_anchor_development_20260917/preserved_worktree.json"
ARMS = ("identity", "base", "centered", "identity_true", "identity_zero", "identity_mismatched")
CONTRASTS = (("base", "identity"), ("centered", "base"), ("identity_true", "identity"),
             ("identity_true", "centered"), ("identity_true", "identity_mismatched"))
DECODERS, CAPTURES = 525, 8
SOFT, HARD = 300, 330


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paths(head):
    return (REPO / f"outputs/smolvla_identity_anchor_preparation_{head[:8]}",
            REPO / f"outputs/smolvla_identity_anchor_{head[:8]}")


def source_paths():
    return [q.pfx.scale.OLD, q.pfx.scale.NEW, q.acr.BASE / "predictions.pt",
            q.acr.BASE / "independent_audit.json", ARCHIVE / "assessment_selected_centered.pt",
            ARCHIVE / "independent_audit.json", ARCHIVE / "centered.pt", PRESERVED]


def source_hashes():
    return {str(p.relative_to(REPO)): digest(p) for p in source_paths()}


def worktree_gate(head):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    require(actual == head, "Execution HEAD changed")
    before = json.loads(PRESERVED.read_text())
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    current = {line[3:]: {"status": line[:2], "sha256": digest(REPO / line[3:])}
               for line in status.splitlines()}
    require(current == before, "Only byte-identical original pending documents may remain")


def load_sources():
    for directory in (q.acr.BASE, ARCHIVE):
        require(json.loads((directory / "independent_audit.json").read_text())[
            "independent_contract_accepted"] is True, "Source audit not accepted")
    train, validation = q.pfx.load_data()
    samples = sorted(train + validation, key=q.pfx.key)
    contract = q.pfx.mapping_contract(train, validation)
    bases = {tuple(r["key"]): r for r in torch.load(q.acr.BASE / "predictions.pt",
             map_location="cpu", weights_only=False) if r["context"] == "no_action"}
    residuals = {(tuple(r["key"]), r["context"]): r for r in torch.load(
        ARCHIVE / "assessment_selected_centered.pt", map_location="cpu", weights_only=False)}
    require(len(bases) == 88 and len(residuals) == 261, "Archive coverage differs")
    require(set(bases) == {q.pfx.key(s) for s in samples}, "Base identities differ")
    donors = q.pfx.donor_map(samples)
    by_key = {q.pfx.key(s): s for s in samples}
    for sample in samples:
        key = q.pfx.key(sample)
        require(sample["split"] in ("train", "validation") and sample["task"] < 8
                and sample["initial_state_id"] in (46, 48, 49), "Nondevelopment sample rejected")
        for context in ("true", "zero", "mismatched"):
            if context == "mismatched" and donors[key] is None:
                require((key, context) not in residuals, "Unexpected singleton donor")
                continue
            r = residuals[key, context]
            expected = (torch.zeros_like(sample["actions"]) if context == "zero" else
                        by_key[donors[key]]["actions"] if context == "mismatched" else sample["actions"])
            require(r["arm"] == "centered" and r["split"] == sample["split"], "Residual identity differs")
            pilot.require_equal(r["actual_actions"], expected, "Saved actual actions differ")
            pilot.require_equal(r["actual_mask"], sample["mask"], "Receiver mask differs")
            for b, a, z, raw, visual in zip(bases[key]["raw_visual"], r["action_delta"],
                                          r["zero_delta"], r["raw_visual"], r["visual"], strict=True):
                pilot.require_equal(b + (a - z), raw, "Archived subtract-before-add differs")
                pilot.require_equal(raw.to(torch.bfloat16).float(), visual, "Archived precision differs")
    return samples, bases, residuals, donors, contract


def transplant(identity, action, zero):
    require(len(identity) == len(action) == len(zero), "Camera count differs")
    return tuple(i.float() + (a.float() - z.float()) for i, a, z in zip(identity, action, zero, strict=True))


def variants(sample, base, residuals, donors):
    key = q.pfx.key(sample)
    identity = tuple(v.float() for v in sample["inputs"][:2])
    visual = {"identity": identity, "base": base["visual"], "centered": residuals[key, "true"]["visual"]}
    raw = {}
    for context in ("true", "zero", "mismatched"):
        if context == "mismatched" and donors[key] is None:
            continue
        r = residuals[key, context]
        name = f"identity_{context}"
        raw[name] = transplant(identity, r["action_delta"], r["zero_delta"])
        visual[name] = tuple(v.to(torch.bfloat16).float() for v in raw[name])
    for actual, expected in zip(raw["identity_zero"], identity, strict=True):
        pilot.require_equal(actual, expected, "Identity-centered zero raw invariant failed")
    return raw, visual


def metrics(visual, output, sample):
    return q.metrics(visual, output, sample, sample["oracle"])


def tolerance(x):
    return 1e-7 + 1e-6 * abs(x)


def summarize(rows):
    result = {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        arms, comparisons = {}, {}
        for arm in ARMS:
            groups = defaultdict(list)
            for r in selected:
                if arm in r["metrics"]:
                    groups["/".join(map(str, r["key"][:2]))].append(r["metrics"][arm])
            episodes = {k: {m: math.fsum(x[m] for x in v) / len(v)
                           for m in ("row0", "chunk", "latent")} for k, v in groups.items()}
            arms[arm] = {"samples": sum(len(v) for v in groups.values()), "episodes": episodes,
                         "macro": {m: math.fsum(v[m] for v in episodes.values()) / len(episodes)
                                   for m in ("row0", "chunk", "latent")}}
        for treatment, control in CONTRASTS:
            paired = [r for r in selected if treatment in r["metrics"] and control in r["metrics"]]
            groups = defaultdict(list)
            per_sample = []
            for r in paired:
                a, b = r["metrics"][treatment]["row0"], r["metrics"][control]["row0"]
                groups[tuple(r["key"][:2])].append((a, b))
                per_sample.append({"key": r["key"], "benefit": b - a,
                                   "direction": "improved" if a < b - tolerance(b) else
                                   "worsened" if a > b + tolerance(b) else "tied"})
            ep = {"/".join(map(str, k)): {"treatment": math.fsum(a for a, _ in v) / len(v),
                  "control": math.fsum(b for _, b in v) / len(v)} for k, v in groups.items()}
            benefits = {k: v["control"] - v["treatment"] for k, v in ep.items()}
            comparisons[f"{treatment}_vs_{control}"] = {
                "paired_samples": len(paired), "paired_episodes": len(ep), "episodes": ep,
                "macro_benefit": math.fsum(benefits.values()) / len(ep),
                "sample_directions": dict(Counter(r["direction"] for r in per_sample)),
                "episode_improved": sum(benefits[k] > tolerance(v["control"]) for k, v in ep.items()),
                "leave_one_out": {k: math.fsum(x for j, x in benefits.items() if j != k) / (len(ep) - 1)
                                  for k in ep}, "per_sample": per_sample}
        result[split] = {"metrics": arms, "contrasts": comparisons}
    v = result["validation"]
    checks = {}
    for control in ("identity", "centered", "identity_mismatched"):
        c = v["contrasts"][f"identity_true_vs_{control}"]
        b = math.fsum(x["control"] for x in c["episodes"].values()) / c["paired_episodes"]
        checks[f"better_{control}"] = c["macro_benefit"] > tolerance(b)
    c = v["contrasts"]["identity_true_vs_identity"]
    checks["three_of_four_episodes"] = c["episode_improved"] >= 3
    checks["strict_sample_majority"] = c["sample_directions"].get("improved", 0) > 8
    a, b = v["metrics"]["identity_true"]["macro"], v["metrics"]["identity"]["macro"]
    checks["chunk_not_worse_identity"] = a["chunk"] <= b["chunk"] + tolerance(b["chunk"])
    return {"splits": result, "development_checks": checks,
            "development_followup_supported": all(checks.values()),
            "independent_qualification_claimed": False}


class Runtime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        require(len(self.captures) < CAPTURES, "Capture budget exhausted before dispatch")
        return super()._capture(inputs)


def decode(runtime, sample, visual, output, counts, name):
    require(counts["decoder"] < DECODERS, "Decoder budget exhausted before dispatch")
    counts["decoder"] += 1
    with pilot.phase(output, f"decode_{counts['decoder']}_{name}", 30):
        values = tuple(v.cuda() for v in sample["inputs"])
        runtime.begin_episode("graph", sample["task"])
        value = runtime(None, None, values[4], values[5], values[6], noise=values[7],
                        future_image_tokens=tuple(v.to(device="cuda", dtype=values[i].dtype)
                                                  for i, v in enumerate(visual)),
                        future_image_token_masks=tuple(values[2:4])).detach().cpu().clone()
        require(value.shape == (1, 50, 32) and torch.isfinite(value).all(), "Invalid decoder output")
    return value


def prepare(head):
    worktree_gate(head)
    environment = q.runtime_environment()
    prep, output = paths(head)
    require(not prep.exists() and not output.exists(), "Preparation/output already exists")
    samples, _, _, _, contract = load_sources()
    require(not torch.cuda.is_initialized(), "CPU prepare initialized CUDA")
    prep.mkdir()
    write(prep / "preparation.json", {"experiment": "F-IAR1", "execution_head": head,
          "source_hashes": source_hashes(), "manifest": contract, "runtime_environment": environment,
          "samples": len(samples), "model_forwards": 0, "new_env": 0, "qualification_reads": 0})
    print(json.dumps({"prepared": True, "sha256": digest(prep / "preparation.json"),
                      "prep": str(prep), "output": str(output), "samples": len(samples)}), flush=True)


def validate(head, registration=True):
    worktree_gate(head)
    prep, output = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    require(saved["execution_head"] == head and saved["source_hashes"] == source_hashes(), "Prepared source changed")
    require(saved["runtime_environment"] == q.runtime_environment(), "Runtime environment changed")
    if registration:
        body = (prep / "registration.md").read_text()
        r = json.loads((prep / "registration_readback.json").read_text())
        p = json.loads((prep / "registration_post.json").read_text())
        require(type(r.get("id")) is int and r["id"] == p.get("id") and r.get("body") == p.get("body") == body
                and r.get("issue_url") == "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
                and all(x in body for x in (f"F-IAR1-REGISTER:{head}", str(output), digest(prep / "preparation.json"))),
                "Registration exact/actual-ID gate failed")
    return saved


def worker(args):
    validate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, runtime, rows = Counter(), None, []
    result = {"experiment": "F-IAR1", "execution_head": args.execution_head,
              "status": "technical_failure", "first_failure": None}
    try:
        with pilot.phase(args.output, "load_development", 30):
            samples, bases, residuals, donors, contract = load_sources()
            require(contract == json.loads((paths(args.execution_head)[0] / "preparation.json").read_text())["manifest"],
                    "Development manifest changed")
        with pilot.phase(args.output, "load_vla", 90):
            require(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, _, _, report = e.load_runtime(e.POLICY, e.VLM)
            counts["vla_loads"] += 1
            write(args.output / "policy_load.json", report)
            runtime = Runtime(policy.model)
        (args.output / "predictions").mkdir()
        with torch.no_grad(), pilot.phase(args.output, "paired_development_probe", 180):
            for sample in samples:
                key = q.pfx.key(sample)
                with pilot.phase(args.output, f"sample_{key}", 60):
                    raw, visual = variants(sample, bases[key], residuals, donors)
                    outputs = {}
                    for name, reference in (("identity", sample["archived_full_chunk"]),
                                            ("base", bases[key]["output"]),
                                            ("centered", residuals[key, "true"]["output"])):
                        outputs[name] = decode(runtime, sample, visual[name], args.output, counts, name)
                        pilot.require_equal(outputs[name], reference, f"Frozen {name} replay differs")
                        counts[f"{name}_exact"] += 1
                    for name in ("identity_true", "identity_zero", "identity_mismatched"):
                        if name in visual:
                            outputs[name] = decode(runtime, sample, visual[name], args.output, counts, name)
                    pilot.require_equal(outputs["identity_zero"], outputs["identity"], "Zero full output differs")
                    counts["zero_exact"] += 1
                    row = {"key": list(key), "split": sample["split"], "delay": sample["delay"],
                           "donor": list(donors[key]) if donors[key] else None,
                           "metrics": {name: metrics(visual[name], value, sample) for name, value in outputs.items()}}
                    torch.save({**row, "raw_new": raw, "visual": visual, "outputs": outputs},
                               args.output / "predictions" / ("_".join(map(str, key)) + ".pt"))
                    rows.append(row)
                    with (args.output / "rows.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
        require(counts == Counter(decoder=525, vla_loads=1, identity_exact=88, base_exact=88,
                                  centered_exact=88, zero_exact=88), "Formal count differs")
        require(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "VLA not frozen")
        require(source_hashes() == json.loads((paths(args.execution_head)[0] / "preparation.json").read_text())[
            "source_hashes"], "Source bytes changed during execution")
        result.update(status="completed", samples=88, vla_frozen=True, **summarize(rows))
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if runtime is not None:
            write(args.output / "captures.json", runtime.captures)
            counts["captures"] = len(runtime.captures)
            runtime.release_graph()
            result["graph_released"] = runtime.graph is None
        result.update(counts=dict(counts), attempts=1, retries=0, predictor_forwards=0, training_updates=0,
                      backward=0, new_env=0, image_encodings=0, qualification_reads=0, real_robot=0,
                      predictor_benefit_tested=False, realtime_qualified=False, risk_thresholds=None)
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
        print(f"F-IAR1 worker={child.pid}", flush=True)
        try:
            while child.poll() is None:
                q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or now - start >= SOFT):
                    reason = {"expired": expired, "soft_timeout": now - start >= SOFT}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - start >= HARD or (terminated is not None and now - terminated >= 5):
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
    result = json.loads(file.read_text()) if file.exists() else {
        "status": "technical_failure", "first_failure": "Worker receipt absent", "execution_head": args.execution_head}
    result["execution"] = {"child_pid": child.pid, "child_exit_code": code, "exit_confirmed": True,
                           "started_at_utc": utc, "finished_at_utc": datetime.now(UTC).isoformat(),
                           "wall_seconds": time.perf_counter() - start, "stop_reason": reason,
                           "forced_termination": forced, "pending": pending, "active": active}
    if code or reason or active or pending:
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
        require(not args.worker and args.output is None, "Prepare cannot run a worker")
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
