"""Separately registered E-RPI1 input gate and E-RPF1 five-arm feedback experiment."""

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

import libero_rtc_commitment as commitment
import torch
from rtc_prefix_scope_runtime import (
    VARIANTS,
    PrefixOnlyGraphRuntime,
    PrefixScopeOwner,
    PrefixScopePredictor,
    prefix_config,
)

prior = commitment.prior
old, q, e, pilot, REPO = prior.old, prior.q, prior.e, prior.pilot, prior.REPO
check, write, digest, load = prior.check, prior.write, prior.digest, prior.load
BASE_HEAD = "65ec2f8d33da961504a3808272cf11b5b4ee4530"
LIMITS = {"episodes": (1, 50), "settling": (10, 500), "measurement": (280, 14000), "model": (160, 8000)}
ADDITIONS = (
    "examples/advanced/predictive_async/rtc_prefix_scope_runtime.py",
    "examples/advanced/predictive_async/libero_rtc_prefix_scope.py",
    "examples/advanced/predictive_async/libero_rtc_prefix_scope_feedback.py",
    "examples/advanced/predictive_async/audit_libero_rtc_prefix_scope.py",
)
PLAN = "docs/experiments/SMOLVLA_RTC_PREFIX_SCOPE_PLAN.md"
TEST = "tests/test_rtc_prefix_scope.py"
PREREQUISITE = REPO / "outputs/smolvla_rtc_commitment_586dab66/independent_audit.json"


def paths(head, stage):
    check(stage in ("inputs", "feedback"), "Unknown stage")
    stem = f"smolvla_rtc_prefix_scope_{stage}"
    return REPO / f"outputs/{stem}_preparation_{head[:8]}", REPO / f"outputs/{stem}_{head[:8]}"


def experiment(stage):
    check(stage in ("inputs", "feedback"), "Unknown stage")
    return "E-RPI1" if stage == "inputs" else "E-RPF1"


def manifest(stage):
    common = {"experiment": experiment(stage), "stage": stage, "chunk": 50, "steps": 10,
        "fps": 20, "cap": 8, "margin": 1, "horizon": 10, "guidance": 10.0,
        "candidate_weights": "1(j<C), else 0; native ZEROS; complete VJP",
        "origin": "observation_index", "trim": "actual_consumed_once",
        "attempts": 1, "retries": 0, "new_qualification": False, "latency_limit_s": .35}
    if stage == "inputs":
        return {**common, "rows": prior.input_schedule(), "requests": 482, "pairs": 241,
            "guided_pairs": 225, "captures": 2, "new_env": 0,
            "reference": "same new ZEROS native_eager, NOT old EXP"}
    rows = []
    template = prior.manifest("feedback")["rows"][0]
    for pair, (task, state) in enumerate(prior.PAIRS):
        shift = pair % 5
        for variant in VARIANTS[shift:] + VARIANTS[:shift]:
            arm = "rtc_async" if variant in ("committed_exp", "committed_prefix") else (
                "serialized" if variant == "serialized" else "aligned_async")
            rows.append({**template, "ordinal": len(rows), "pair_index": pair,
                "task_id": task, "task_name": e.reference.TASK_NAMES[task], "initial_state_id": state,
                "environment_seed": 1160000 + 100*task + state,
                "policy_seed": 1170000 + 100*task + state,
                "variant": variant, "arm": arm, "condition": "serialized" if arm == "serialized" else "async",
                "cohort": "pilot_development" if pair < 8 else "known_diagnostic",
                "constraint_schedule": "ZEROS" if variant == "committed_prefix" else "EXP",
                "limits": {k: v[0] for k, v in LIMITS.items()}})
    return {**common, "rows": rows, "episodes": 50, "captures": 100, "extra_sampler_calls": 500,
        "global_limits": {k: v[1] for k, v in LIMITS.items()},
        "commitment": "same original coverage estimate; old actions until C then receive, wait if late",
        "controls": list(VARIANTS[:-1]), "candidate": VARIANTS[-1]}


def tree_gate(head):
    check(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == head,
          "Execution HEAD changed")
    check(old.pending_state() == json.loads(prior.previous.BASELINE.read_text())["pending"], "Old pending changed")
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE_HEAD, "--", "src/lerobot",
        "examples/advanced/predictive_async"], cwd=REPO, text=True).splitlines()
    check(set(changed) <= set(ADDITIONS), f"Frozen source modified: {changed}")
    lock = json.loads(prior.previous.LOCK.read_text())
    check(all(digest(REPO / p) == h for p, h in lock["core_sha256"].items()), "Frozen core changed")


def sources(head, stage):
    values = commitment.sources()
    extra = [PREREQUISITE, REPO / PLAN, REPO / TEST, *(REPO / p for p in ADDITIONS)]
    if stage == "feedback":
        folder = paths(head, "inputs")[1]
        extra += [folder / "result.json", folder / "independent_audit.json"]
    values.update({str(p): digest(p) for p in extra})
    return values


def prerequisite(head, stage):
    old_result = json.loads(PREREQUISITE.read_text())
    check(old_result["independent_contract_accepted"] and not old_result["development_followup_supported"],
          "Old commitment result changed")
    if stage == "feedback":
        accepted = json.loads((paths(head, "inputs")[1] / "independent_audit.json").read_text())
        check(accepted["independent_contract_accepted"] and accepted["prefix_scope_equivalence_passed"] and
              accepted["graph_budget_passed"], "New formula input prerequisite not met")


def prepare(head, stage):
    tree_gate(head)
    prerequisite(head, stage)
    prep, out = paths(head, stage)
    check(not prep.exists() and not out.exists(), "Unique preparation/output required")
    data = load(prior.previous.DATA)
    check([s["key"] for s in data] == [list(k) for k in prior.previous.prior.EXPECTED_KEYS], "Data identities changed")
    saved = {"head": head, "manifest": manifest(stage), "sources": sources(head, stage),
        "environment": q.runtime_environment(), "history": q.history_inventory(REPO / "outputs")}
    if stage == "feedback":
        check(set(prior.PAIRS) <= {(v["task"], v["state"]) for v in saved["history"]}, "Known identity missing")
    check(not torch.cuda.is_initialized(), "CPU prepare initialized CUDA")
    prep.mkdir()
    write(prep / "preparation.json", saved)
    print(json.dumps({"prepared": True, "stage": stage, "prep": str(prep), "out": str(out),
        "sha256": digest(prep / "preparation.json"), "sources": len(saved["sources"])}), flush=True)


def validate(head, stage, after=False):
    tree_gate(head)
    prerequisite(head, stage)
    prep, out = paths(head, stage)
    saved = json.loads((prep / "preparation.json").read_text())
    check(saved["head"] == head and saved["manifest"] == manifest(stage), "Contract changed")
    check(saved["sources"] == sources(head, stage) and saved["environment"] == q.runtime_environment(),
          "Source/environment changed")
    history = q.history_inventory(REPO / "outputs")
    if after:
        history = [v for v in history if not v["path"].startswith(out.name + "/")]
    check(saved["history"] == history, "Other collection/history changed")
    body = (prep / "registration.md").read_text()
    got = json.loads((prep / "registration_readback.json").read_text())
    check(type(got.get("id")) is int and got.get("body") == body and got.get("issue_url") ==
        "https://api.github.com/repos/Lebron-233/lerobot/issues/1" and all(v in body for v in
        (f"{experiment(stage)}-REGISTER:{head}", str(out), digest(prep / "preparation.json"))),
        "Actual registration missing or changed")
    return saved


def input_worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, rows, rt, policy = e.Calls(args.output / "calls.jsonl"), [], None, None
    result = {"experiment": "E-RPI1", "stage": "inputs", "execution_head": args.execution_head,
              "status": "technical_failure", "first_failure": None}
    try:
        saved = validate(args.execution_head, "inputs")
        data = load(prior.previous.DATA)
        with pilot.phase(args.output, "load_model", 90):
            check(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, "Model config changed")
            write(args.output / "policy_load.json", report)
            policy.config.rtc_config = prefix_config()
            policy.init_rtc_processor()
            result["vla_loads"] = 1
        rt = PrefixOnlyGraphRuntime(policy.model)
        with rt:
            pairs = {}
            for group in range(10):
                with pilot.phase(args.output, f"input_group_{group}", 90):
                    for spec in (s for s in prior.input_schedule() if s["group"] == group):
                        check(spec["ordinal"] == len(rows) and len(rows) < 482, "Input order/budget")
                        sample = data[spec["sample"]]
                        row = prior.input_request(spec, sample, rt, policy, pre, post, calls)
                        row["constraint_schedule"] = "ZEROS"
                        rows.append(row)
                        torch.save(row, args.output / f"request_{spec['ordinal']:03d}.pt")
                        for j in range(8):
                            pilot.require_equal(row["inputs"][j], sample["inputs"][j], "Current input differs")
                        if spec["length"]:
                            pilot.require_equal(row["inputs"][8], sample["prefix"][:spec["length"]], "Wrong prefix")
                        else:
                            pilot.require_equal(row["full"], sample["archived_full_chunk"], "No-prefix anchor differs")
                        if spec["pair"] in pairs:
                            for field in ("full", "normalized", "processed"):
                                pilot.require_equal(row[field], pairs[spec["pair"]][field], "New formula differs:"+field)
                            del pairs[spec["pair"]]
                        else:
                            pairs[spec["pair"]] = row
                print(f"E-RPI1 group{group} complete: {len(rows)}/482", flush=True)
            check(not pairs and len(rows) == 482, "Input pair coverage")
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "Model unfrozen")
        tree_gate(args.execution_head)
        check(saved["sources"] == sources(args.execution_head, "inputs"), "Sources changed")
        result.update(status="completed", vla_frozen=True, prefix_scope_equivalence_passed=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if rt is not None:
            result["runtime_receipt"] = rt.receipt
        if policy is not None:
            policy.config.rtc_config = None
            policy.init_rtc_processor()
        calls.close()
        result.update(completed_requests=len(rows), attempts=1, retries=0, training_updates=0,
                      new_env=0, qualification_reads=0, deployment_qualified=False)
        write(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, f"E-RPF1 {kind} exhausted")


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
            check(e.initial_difference(paired, initial) is None, "Paired initial differs")
            result["initial_pair_exact"] = True
        predict = PrefixScopePredictor(policy, pre, post, spec)
        owner = PrefixScopeOwner(predict, calls, spec, budget, on_owner_close=predict.close,
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
            count = min(spec["ready_wall_slots"], max(1, math.ceil(wall*20)))
            result.update(wall_s=wall, measured_actions=len(control["dispatches"]), wall_slots=count,
                no_action_slots=count-len(control["dispatches"]),
                underflows=sum(v["outcome"] == "underflow" for v in control["gets"]))
        write(folder / "result.json", result)
    return result, initial


def feedback_worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output / "calls.jsonl"), Budget(), []
    result = {"experiment": "E-RPF1", "stage": "feedback", "execution_head": args.execution_head,
              "status": "technical_failure", "first_failure": None}
    try:
        saved = validate(args.execution_head, "feedback")
        with pilot.phase(args.output, "environment_preflight", 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), "Factory initialized CUDA")
        with pilot.phase(args.output, "load_model", 90):
            check(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(not policy.config.compile_model and policy.config.rtc_config is None, "Model config changed")
            write(args.output / "policy_load.json", report)
            result["vla_loads"] = 1
        initials, boots = {}, {}
        for spec in manifest("feedback")["rows"]:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(f"E-RPF1 START {spec['ordinal']}/50 {spec['variant']} {spec['task_id']}/{spec['initial_state_id']}", flush=True)
                row, initial = episode(spec, args.output, policy, pre, post, factory, budget, calls,
                                       initials.get(spec["pair_index"]))
                records.append(row)
                check(row["status"] == "completed", row["first_failure"] or "Episode incomplete")
                arrays = load(args.output / f"episode_{spec['ordinal']:03d}/arrays.pt")
                boot, pair = arrays["outputs"][1], spec["pair_index"]
                if pair in boots:
                    for field in ("full", "noise"):
                        pilot.require_equal(boot[field], boots[pair][field], "Bootstrap differs:"+field)
                else:
                    initials[pair], boots[pair] = initial, {k: boot[k] for k in ("full", "noise")}
                del arrays
                print(f"E-RPF1 END {spec['ordinal']} success={row['success']} actions={row['measured_actions']}", flush=True)
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "Model unfrozen")
        tree_gate(args.execution_head)
        check(saved["sources"] == sources(args.execution_head, "feedback"), "Sources changed")
        result.update(status="completed", vla_frozen=True, frozen_baseline_unchanged=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        calls.close()
        caps = [c for row in records for c in (row.get("predictor_cleanup") or {}).get("captures", [])]
        result.update(episodes_completed=sum(v["status"] == "completed" for v in records),
            native_budget=dict(budget.total), graph_captures=len(caps),
            capture_internal={k: sum(c[k] for c in caps) for k in ("setup", "warmup", "capture")},
            outcomes=[{k: row.get(k) for k in ("spec", "success", "measured_actions", "terminal_reason")} for row in records],
            attempts=1, retries=0, training_updates=0, qualification_reads=0, real_robot=0,
            new_qualification_claimed=False, deployment_qualified=False)
        write(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def main(stage="inputs"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution-head", required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    if args.prepare:
        check(not args.worker and args.output is None, "Prepare cannot dispatch")
        prepare(args.execution_head, stage)
        return 0
    args.output = (args.output or paths(args.execution_head, stage)[1]).resolve()
    if args.worker:
        return input_worker(args) if stage == "inputs" else feedback_worker(args)
    script = __file__ if stage == "inputs" else str(Path(__file__).with_name("libero_rtc_prefix_scope_feedback.py"))
    return old.supervise(args, validate_run=lambda h: validate(h, stage), paths_for=lambda h: paths(h, stage),
        manifest_for=lambda: manifest(stage), worker_script=script, experiment=experiment(stage))


if __name__ == "__main__":
    raise SystemExit(main())
