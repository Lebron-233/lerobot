"""E-RCC1: one declared-prefix candidate and three concurrent development controls."""

import argparse
import faulthandler
import json
import math
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import libero_rtc_dynamic as prior
import torch
from rtc_commitment_owner import CommittedPrefixOwner
from rtc_dynamic_runtime import DynamicRTCPredictor

old, q, e, pilot, REPO = prior.old, prior.q, prior.e, prior.pilot, prior.REPO
check, write, digest, load = prior.check, prior.write, prior.digest, prior.load
BASE_HEAD = "993442a39fef11e92c622e8ef3798af233e4d356"
VARIANTS = ("serialized", "aligned_async", "rtc_async", "committed_rtc")
LIMITS = {"episodes": (1, 40), "settling": (10, 400), "measurement": (280, 11200), "model": (160, 6400)}
ADDITIONS = (
    "examples/advanced/predictive_async/rtc_commitment_owner.py",
    "examples/advanced/predictive_async/libero_rtc_commitment.py",
    "examples/advanced/predictive_async/audit_libero_rtc_commitment.py",
)
PLAN = "docs/experiments/SMOLVLA_RTC_COMMITMENT_PLAN.md"
TEST = "tests/test_rtc_commitment_owner.py"
PREREQUISITE = REPO / "outputs/smolvla_rtc_dynamic_feedback_fa80d0b5/independent_audit.json"


def paths(head):
    return (REPO / f"outputs/smolvla_rtc_commitment_preparation_{head[:8]}",
            REPO / f"outputs/smolvla_rtc_commitment_{head[:8]}")


def manifest():
    template = prior.manifest("feedback")["rows"][0]
    rows = []
    for pair, (task, state) in enumerate(prior.PAIRS):
        shift = pair % 4
        for variant in VARIANTS[shift:] + VARIANTS[:shift]:
            rows.append({**template, "ordinal": len(rows), "pair_index": pair,
                "task_id": task, "task_name": e.reference.TASK_NAMES[task], "initial_state_id": state,
                "environment_seed": 1160000 + 100 * task + state,
                "policy_seed": 1170000 + 100 * task + state,
                "arm": "rtc_async" if variant == "committed_rtc" else variant,
                "variant": variant, "condition": "serialized" if variant == "serialized" else "async",
                "cohort": "pilot_development" if pair < 8 else "known_diagnostic",
                "limits": {k: v[0] for k, v in LIMITS.items()}})
    return {"experiment": "E-RCC1", "rows": rows, "episodes": 40, "new_qualification": False,
        "commitment": "C=original estimated coverage; install only after C old actions, wait if late",
        "early_stop": "cancel outstanding commitment without further environment steps",
        "trim": "unchanged original actual-consumed-once RTCExecutionQueue",
        "config": {"chunk": 50, "steps": 10, "fps": 20, "cap": 8, "margin": 1,
                   "schedule": "EXP", "horizon": 10, "guidance": 10.0},
        "captures": 80, "extra_sampler_calls": 400,
        "global_limits": {k: v[1] for k, v in LIMITS.items()}, "attempts": 1, "retries": 0}


def tree_gate(head):
    check(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == head,
          "Execution HEAD changed")
    check(old.pending_state() == json.loads(prior.previous.BASELINE.read_text())["pending"], "Old pending changed")
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE_HEAD, "--", "src/lerobot",
        "examples/advanced/predictive_async"], cwd=REPO, text=True).splitlines()
    check(set(changed) <= set(ADDITIONS), f"Frozen source modified: {changed}")
    lock = json.loads(prior.previous.LOCK.read_text())
    check(all(digest(REPO / p) == h for p, h in lock["core_sha256"].items()), "Frozen core changed")


def sources():
    values = prior.sources()
    for p in (PREREQUISITE, REPO / PLAN, REPO / TEST, *(REPO / a for a in ADDITIONS)):
        values[str(p)] = digest(p)
    return values


def prepare(head):
    tree_gate(head)
    previous = json.loads(PREREQUISITE.read_text())
    check(previous["independent_contract_accepted"] and not previous["development_followup_supported"],
          "Prior result changed")
    prep, out = paths(head)
    check(not prep.exists() and not out.exists(), "Unique preparation/output required")
    saved = {"head": head, "manifest": manifest(), "sources": sources(),
             "environment": q.runtime_environment(), "history": q.history_inventory(REPO / "outputs")}
    check(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    prep.mkdir()
    write(prep / "preparation.json", saved)
    print(json.dumps({"prepared": True, "head": head, "prep": str(prep), "out": str(out),
        "sha256": digest(prep / "preparation.json"), "sources": len(saved["sources"])}), flush=True)


def validate(head, after=False):
    tree_gate(head)
    prep, out = paths(head)
    saved = json.loads((prep / "preparation.json").read_text())
    check(saved["head"] == head and saved["manifest"] == manifest(), "Contract changed")
    check(saved["sources"] == sources() and saved["environment"] == q.runtime_environment(), "Source/environment changed")
    history = q.history_inventory(REPO / "outputs")
    if after:
        history = [v for v in history if not v["path"].startswith(out.name + "/")]
    check(saved["history"] == history, "Other collection/history change")
    body = (prep / "registration.md").read_text()
    readback = json.loads((prep / "registration_readback.json").read_text())
    check(type(readback.get("id")) is int and readback.get("body") == body and
        readback.get("issue_url") == "https://api.github.com/repos/Lebron-233/lerobot/issues/1" and
        all(v in body for v in (f"E-RCC1-REGISTER:{head}", str(out), digest(prep / "preparation.json"))),
        "Actual registration missing or different")
    return saved


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, f"E-RCC1 {kind} exhausted")


def episode(spec, output, policy, pre, post, factory, budget, calls, paired=None):
    folder = output / f"episode_{spec['ordinal']:03d}"
    folder.mkdir()
    write(folder / "started.json", {"spec": spec, "started_at_utc": datetime.now(UTC).isoformat()})
    budget.begin_episode()
    native = e.NativeSession(spec, budget, calls, factory)
    owner, predict, control, initial = None, None, None, None
    observations = []
    result = {"spec": spec, "status": "technical_failure", "first_failure": None,
        "worker_joined": False, "environment_closed": False, "initial_pair_exact": None,
        "controller_thread": threading.get_ident()}
    try:
        def observe(raw, index, returned):
            _, _, row = e.observation(raw, spec["task_name"].replace("_", " "), index, returned)
            observations.append(row)
            return row
        initial = observe(native.create_reset(), 0, time.perf_counter())
        torch.save(initial, folder / "initial_checkpoint.pt")
        if paired is not None:
            check(e.initial_difference(paired, initial) is None, "Initial quartet differs")
            result["initial_pair_exact"] = True
        predict = DynamicRTCPredictor(policy, pre, post, spec)
        owner = CommittedPrefixOwner(predict, calls, spec, budget, on_owner_close=predict.close,
                                     request_kind="dynamic_graph_request")
        start = time.perf_counter()
        owner.submit(initial, 0)
        owner.receive(block=True)
        result["startup_s"] = time.perf_counter() - start
        control = old.control_loop(owner, native, spec, initial, observe)
        result.update(status="completed", success=control["success"], terminal_reason=control["terminal_reason"])
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if owner is not None:
            control = getattr(owner, "control_result", control)
            try:
                owner.close()
                result["worker_joined"] = True
            except BaseException:
                result["first_failure"] = result["first_failure"] or traceback.format_exc()
                result["status"] = "technical_failure"
        if owner is None or result["worker_joined"]:
            try:
                result["environment_closed"] = native.close()
            except BaseException:
                result["first_failure"] = result["first_failure"] or traceback.format_exc()
                result["status"] = "technical_failure"
            if owner is not None:
                result.update(requests=owner.rows, owner_cleanup=owner.cleanup_evidence,
                              predictor_cleanup=predict.receipt)
                torch.save({"observations": observations, "control": control, "outputs": owner.outputs}, folder / "arrays.pt")
        if not result["worker_joined"] or not result["environment_closed"]:
            result["status"] = "technical_failure"
        result.update(budget=dict(budget.episode), native_steps=native.native_steps, native_returned=dict(native.returned))
        if control is not None:
            wall = control["ended_at"] - control["t0"]
            slots = min(spec["ready_wall_slots"], max(1, math.ceil(wall * 20)))
            result.update(wall_s=wall, measured_actions=len(control["dispatches"]), wall_slots=slots,
                no_action_slots=slots-len(control["dispatches"]),
                underflows=sum(v["outcome"] == "underflow" for v in control["gets"]))
        write(folder / "result.json", result)
    return result, initial


def worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output / "calls.jsonl"), Budget(), []
    result = {"experiment": "E-RCC1", "execution_head": args.execution_head,
              "status": "technical_failure", "first_failure": None}
    try:
        saved = validate(args.execution_head)
        with pilot.phase(args.output, "environment_preflight", 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), "Preflight initialized CUDA")
        with pilot.phase(args.output, "load_model", 90):
            check(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, "Model config changed")
            write(args.output / "policy_load.json", report)
            result["vla_loads"] = 1
        initials, boots = {}, {}
        for spec in manifest()["rows"]:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-RCC1 START {spec['ordinal']}/40 {spec['variant']} {spec['task_id']}/{spec['initial_state_id']}", flush=True)
                row, initial = episode(spec, args.output, policy, pre, post, factory, budget, calls,
                                       initials.get(spec["pair_index"]))
                records.append(row)
                check(row["status"] == "completed", row["first_failure"] or "Incomplete episode")
                arrays = load(args.output / f"episode_{spec['ordinal']:03d}/arrays.pt")
                boot, pair = arrays["outputs"][1], spec["pair_index"]
                if pair in boots:
                    for field in ("full", "noise"):
                        pilot.require_equal(boot[field], boots[pair][field], "Bootstrap differs:"+field)
                else:
                    initials[pair], boots[pair] = initial, {k: boot[k] for k in ("full", "noise")}
                del arrays
                print(f"E-RCC1 END {spec['ordinal']} success={row['success']} actions={row['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "Model unfrozen")
        tree_gate(args.execution_head)
        check(saved["sources"] == sources(), "Sources changed during run")
        result.update(status="completed", vla_frozen=True, frozen_baseline_unchanged=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        calls.close()
        caps = [c for row in records for c in (row.get("predictor_cleanup") or {}).get("captures", [])]
        result.update(episodes_completed=sum(v["status"] == "completed" for v in records),
            native_budget=dict(budget.total), graph_captures=len(caps),
            capture_internal={k: sum(c[k] for c in caps) for k in ("setup", "warmup", "capture")},
            outcomes=[{k: v.get(k) for k in ("spec", "success", "measured_actions", "terminal_reason")} for v in records],
            attempts=1, retries=0, training_updates=0, qualification_reads=0, real_robot=0,
            new_qualification_claimed=False, deployment_qualified=False)
        write(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution-head", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        check(not args.worker and args.output is None, "Prepare cannot dispatch")
        prepare(args.execution_head)
        return 0
    args.output = (args.output or paths(args.execution_head)[1]).resolve()
    return worker(args) if args.worker else old.supervise(args, validate_run=validate, paths_for=paths,
        manifest_for=manifest, worker_script=str(Path(__file__).resolve()), experiment="E-RCC1")


if __name__ == "__main__":
    raise SystemExit(main())
