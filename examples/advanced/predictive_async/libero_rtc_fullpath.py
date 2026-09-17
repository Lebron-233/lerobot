"""Stage 2: 192 fixed complete eager requests; no Env, training or oracle selection."""

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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import libero_action_qualification as q
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

REPO, e, pilot = q.e.REPO, q.e, q.pilot
DEVELOPMENT = REPO / "outputs/smolvla_coverage_6249b03d"
BASELINE = REPO / "outputs/smolvla_rtc_stage1_development_20260917/baseline.json"
EXPECTED_KEYS = [(t, s, r) for t in (6, 7) for s in (48, 49) for r in (3, 4, 5, 6)]
SOFT, HARD = 600, 630


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paths(head):
    return (REPO / f"outputs/smolvla_rtc_fullpath_preparation_{head[:8]}",
            REPO / f"outputs/smolvla_rtc_fullpath_{head[:8]}")


def specification():
    return {"experiment": "RTC-FULLPATH1", "keys": [list(k) for k in EXPECTED_KEYS],
            "arms": ["base", "rtc"], "repetitions": 6, "cold_no_prefix_per_arm": 16,
            "steady_per_arm": 80, "requests": 192, "chunk_size": 50, "num_steps": 10,
            "rtc": {"mode": "guided", "execution_horizon": 10,
                    "max_guidance_weight": 10.0, "schedule": "EXP"},
            "fps": 20, "max_delay": 8, "margin_steps": 1, "queue_remaining": 30,
            "latency_gate": "ceil(20*p99_complete_seconds)+1 <= 8 for BOTH arms",
            "soft_seconds": SOFT, "hard_seconds": HARD, "graph_capture": 0,
            "qualification_reads": 0, "training": 0, "new_env": 0}


def source_files():
    files = [BASELINE, q.pfx.scale.NEW]
    for ordinal in (6, 7, 8, 9):
        folder = DEVELOPMENT / f"episode_{ordinal:03d}"
        files += [folder / "arrays.pt", folder / "result.json"]
    files += sorted(e.POLICY.glob("*.safetensors")) + sorted(e.VLM.glob("*.safetensors"))
    return files


def hashes():
    code = subprocess.check_output(["git", "ls-files", "src/lerobot/policies", "src/lerobot/processor",
                                    "examples/advanced/predictive_async"], text=True, cwd=REPO).splitlines()
    return {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): digest(p)
            for p in [*source_files(), *(REPO / name for name in code)]}


def tree_gate(head):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip() == head,
            "Execution HEAD changed")
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=REPO)
    actual = {s[3:]: {"status": s[:2], "sha256": digest(REPO / s[3:])} for s in status.splitlines()}
    require(actual == json.loads(BASELINE.read_text())["pending"], "Original pending state changed")


def collect_data():
    samples = sorted((s for s in load(q.pfx.scale.NEW) if s["split"] == "validation"), key=q.pfx.key)
    require([q.pfx.key(s) for s in samples] == EXPECTED_KEYS, "Fixed development keys changed")
    collected = {}
    for ordinal in (6, 7, 8, 9):
        folder = DEVELOPMENT / f"episode_{ordinal:03d}"
        arrays, record = load(folder / "arrays.pt"), json.loads((folder / "result.json").read_text())
        observations = {o["index"]: o for o in arrays["observations"]}
        dispatches = {d["action_index"]: d for d in arrays["control"]["dispatches"]}
        for s in (v for v in samples if v["ordinal"] == ordinal):
            k, start = q.pfx.key(s), s["current_index"]
            d = dispatches[start]
            previous = arrays["requests"][f"request_{d['source_request_id']}"]
            tail = previous["policy_chunk"][d["source_row_offset"]:].clone()
            require(tail.shape == (30, 7) and s["delay"] == 3, "Original prefix coverage changed")
            pilot.require_equal(tail[:s["delay"]], s["actions"][0, :s["delay"]], "Prefix provenance differs")
            collected[k] = {"key": list(k), "observation": observations[start], "inputs": s["inputs"],
                            "archived_full_chunk": s["archived_full_chunk"], "prefix": tail,
                            "delay": s["delay"], "language": record["spec"]["task_name"].replace("_", " "),
                            "source_request": d["source_request_id"], "source_row": d["source_row_offset"]}
        del arrays
    return [collected[k] for k in EXPECTED_KEYS]


def prepare(head):
    tree_gate(head)
    environment = q.runtime_environment()
    prep, out = paths(head)
    require(not prep.exists() and not out.exists(), "Unique preparation/output required")
    data = collect_data()
    require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    prep.mkdir()
    torch.save(data, prep / "data.pt")
    write(prep / "preparation.json", {"head": head, "specification": specification(),
        "sources": hashes(), "data_sha256": digest(prep / "data.pt"), "environment": environment})
    print(json.dumps({"prepared": True, "prep": str(prep), "output": str(out),
                      "sha256": digest(prep / "preparation.json"), "samples": len(data)}), flush=True)


def validate(head):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    require(saved["head"] == head and saved["specification"] == specification(), "Specification changed")
    require(saved["sources"] == hashes() and saved["data_sha256"] == digest(prep / "data.pt"), "Source changed")
    require(saved["environment"] == q.runtime_environment(), "Environment changed")
    body = (prep / "registration.md").read_text()
    got = json.loads((prep / "registration_readback.json").read_text())
    require(type(got.get("id")) is int and got.get("body") == body and got.get("issue_url") ==
            "https://api.github.com/repos/Lebron-233/lerobot/issues/1" and
            all(x in body for x in (f"RTC-FULLPATH1-REGISTER:{head}", str(out), digest(prep / "preparation.json"))),
            "Actual registration readback differs")
    return saved


def nearest(values, probability):
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)]


def summary(rows):
    result = {}
    for arm in ("base", "rtc"):
        selected = [r for r in rows if r["arm"] == arm and r["repeat"] > 0]
        latency = [r["complete_s"] for r in selected]
        p99 = nearest(latency, .99)
        result[arm] = {"n": len(selected), "p50_s": nearest(latency, .5),
            "p95_s": nearest(latency, .95), "p99_s": p99, "max_s": max(latency),
            "mean_s": math.fsum(latency) / len(latency), "required_delay_steps": math.ceil(p99 * 20) + 1,
            "component_mean_s": {k: math.fsum(r["components"][k] for r in selected) / len(selected)
                for k in selected[0]["components"]}}
    return {"timing": result, "latency_gate_passed": all(v["required_delay_steps"] <= 8 for v in result.values())}


def worker(args):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, rows = Counter(), []
    result = {"experiment": "RTC-FULLPATH1", "execution_head": args.execution_head,
              "status": "technical_failure", "first_failure": None}
    try:
        prep = validate(args.execution_head)
        data = load(paths(args.execution_head)[0] / "data.pt")
        with pilot.phase(args.output, "load_model", 90):
            require(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            require(not policy.config.compile_model, "Both arms must use the same eager sampler")
            counts["vla_loads"] += 1
            write(args.output / "policy_load.json", report)
            policy.config.rtc_config = RTCConfig(enabled=True, mode="guided", execution_horizon=10,
                max_guidance_weight=10.0, prefix_attention_schedule=RTCAttentionSchedule.EXP)
            policy.init_rtc_processor()
        capture = {}
        original_profiled = policy.model.sample_actions_profiled
        original_encode = policy.model.encode_image_tokens
        original_rtc = policy.rtc_processor.denoise_step

        def encoded(*a, **kw):
            z, mask = original_encode(*a, **kw)
            counts["vision_encodes"] += 1
            capture["tokens"], capture["masks"] = q.cpu(z), q.cpu(mask)
            return z, mask

        def profiled(*a, **kw):
            counts["chunks_started"] += 1
            capture["state"] = a[4].detach().cpu().clone()
            capture["language"] = a[2].detach().cpu().clone()
            value = original_profiled(*a, **kw)
            capture["full"] = value.detach().cpu().clone()
            counts["chunks_returned"] += 1
            return value

        def rtc_step(*a, **kw):
            value = original_rtc(*a, **kw)
            counts["rtc_steps"] += 1
            if kw.get("prev_chunk_left_over") is not None:
                counts["guided_vjps_returned"] += 1
            return value

        policy.model.encode_image_tokens = encoded
        policy.model.sample_actions_profiled = profiled
        policy.rtc_processor.denoise_step = rtc_step
        for index, sample in enumerate(data):
            cold = {}
            for repeat in range(6):
                for arm in (("base", "rtc") if (index + repeat) % 2 == 0 else ("rtc", "base")):
                    number = len(rows)
                    require(number < 192, "Request budget exhausted")
                    with pilot.phase(args.output, f"request_{number}", 30), torch.no_grad():
                        policy.config.rtc_config.enabled = arm == "rtc"
                        capture.clear()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        policy.reset()
                        pre.reset()
                        post.reset()
                        batch = pre(pilot.worker_batch(sample["observation"], sample["language"], "cuda"))
                        prefix = sample["prefix"].cuda() if arm == "rtc" and repeat > 0 else None
                        noise = sample["inputs"][7].cuda().clone()
                        torch.cuda.synchronize()
                        ready = time.perf_counter()
                        parts = {}
                        normalized = policy.predict_action_chunk(batch, noise=noise, timings=parts,
                            prev_chunk_left_over=prefix, inference_delay=sample["delay"], execution_horizon=10)
                        torch.cuda.synchronize()
                        inferred = time.perf_counter()
                        processed = post(normalized).detach().cpu().clone()
                        normalized = normalized.detach().cpu().clone()
                        torch.cuda.synchronize()
                        ended = time.perf_counter()
                        require(normalized.shape == processed.shape == (1, 50, 7), "Public chunk shape changed")
                        require(torch.isfinite(normalized).all() and torch.isfinite(processed).all(), "Nonfinite output")
                        for i in range(2):
                            pilot.require_equal(capture["tokens"][i], sample["inputs"][i], "Current pixels/tokens differ")
                            pilot.require_equal(capture["masks"][i], sample["inputs"][i+2], "Camera masks differ")
                        pilot.require_equal(capture["state"], sample["inputs"][6], "Current normalized state differs")
                        pilot.require_equal(capture["language"], sample["inputs"][4], "Language differs")
                        pilot.require_equal(normalized, capture["full"][..., :7], "Unpadding changed")
                        counts["input_exact"] += 1
                        if repeat == 0:
                            cold[arm] = normalized.clone()
                        if arm == "base" or repeat == 0:
                            pilot.require_equal(capture["full"], sample["archived_full_chunk"], "No-guidance anchor changed")
                            counts["no_guidance_exact"] += 1
                        components = {"preprocess_transfer_s": ready-started, **parts,
                                      "public_predict_total_s": inferred-ready, "postprocess_transfer_s": ended-inferred}
                        row = {"index": number, "key": sample["key"], "arm": arm, "repeat": repeat,
                               "complete_s": ended-started, "components": components,
                               "prefix_present": prefix is not None, "prefix_shape": list(sample["prefix"].shape),
                               "delay": sample["delay"], "output": normalized, "processed": processed,
                               "full_output": capture["full"].clone(), "queue_wait_s": None}
                        rows.append(row)
                        torch.save(row, args.output / f"request_{number:03d}.pt")
                        with (args.output / "timing.jsonl").open("a") as f:
                            f.write(json.dumps({k: v for k, v in row.items() if not isinstance(v, torch.Tensor)})+"\n")
            pilot.require_equal(cold["base"], cold["rtc"], "No-prefix RTC differs from baseline")
            counts["no_prefix_pair_exact"] += 1
            print(f"RTC-FULLPATH1 completed sample {index+1}/16; requests={len(rows)}", flush=True)
        require(dict(counts) == {"vla_loads": 1, "chunks_started": 192, "vision_encodes": 192,
            "chunks_returned": 192, "input_exact": 192, "no_guidance_exact": 112,
            "rtc_steps": 960, "guided_vjps_returned": 800, "no_prefix_pair_exact": 16}, "Call accounting differs")
        require(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "Policy not frozen")
        policy.config.rtc_config = None
        policy.init_rtc_processor()
        require(hashes() == prep["sources"], "Source mutation")
        result.update(status="completed", vla_frozen=True, **summary(rows))
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    result.update(counts=dict(counts), completed_requests=len(rows), attempts=1, retries=0,
                  training_updates=0, new_env=0, qualification_reads=0, graph_captures=0,
                  realtime_qualified=False, risk_thresholds=None)
    write(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    validate(args.execution_head)
    require(args.output == paths(args.execution_head)[1] and not args.output.exists(), "Unique output required")
    args.output.mkdir()
    start = time.perf_counter()
    utc = datetime.now(UTC).isoformat()
    command = [sys.executable, "-u", __file__, "--execution-head", args.execution_head,
               "--output", str(args.output), "--worker"]
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in active.values() if now-v["at"] > v["limit"]]
                if terminated is None and (expired or now-start > SOFT):
                    reason = {"expired": expired, "soft_timeout": now-start > SOFT}
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now-start > HARD or (terminated is not None and now-terminated > 5):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(.1)
        except BaseException:
            reason = reason or {"supervisor_exception": traceback.format_exc()}
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
            code = child.wait()
    q.cov.parent.previous.update_pending(args.output, cursors, pending, active)
    receipt = args.output / "worker_result.json"
    result = json.loads(receipt.read_text()) if receipt.exists() else {
        "experiment": "RTC-FULLPATH1", "execution_head": args.execution_head,
        "status": "technical_failure", "first_failure": "Worker receipt absent"}
    result["execution"] = {"child_pid": child.pid, "exit_code": code, "exit_confirmed": True,
        "started_utc": utc, "finished_utc": datetime.now(UTC).isoformat(), "wall_s": time.perf_counter()-start,
        "stop_reason": reason, "forced": forced, "active": active, "pending": pending}
    if code or reason or active or pending:
        result["status"] = "technical_failure"
    write(args.output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in ("status", "completed_requests", "latency_gate_passed", "first_failure")}), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        require(not args.worker and args.output is None, "Preparation cannot run a worker")
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
