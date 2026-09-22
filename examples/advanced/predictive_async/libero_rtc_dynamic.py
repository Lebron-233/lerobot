"""E-RDC1 / E-RDF1: dynamic RTC equivalence, then separately registered live feedback."""

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

import libero_rtc_graph as previous
import torch
from rtc_dynamic_runtime import (
    DynamicRTCGraphRuntime,
    DynamicRTCPredictor,
    RTCPrefixOwner,
    config,
    cpu_optional,
    g,
)

old, q, e, pilot, REPO = g.old, g.q, g.e, g.pilot, g.REPO
check, write, digest = g.check, g.write, g.digest
BASE_HEAD = "cd63157eae17721582c8f24b7e9f3081eaeaf214"
PAIRS = ((0, 10), (2, 10), (6, 10), (7, 10), (7, 11), (6, 11), (2, 11), (0, 11), (7, 18), (7, 19))
ARMS = ("serialized", "aligned_async", "rtc_async")
LIMITS = {"episodes": (1, 30), "settling": (10, 300), "measurement": (280, 8400), "model": (160, 4800)}
ADDITIONS = (
    "examples/advanced/predictive_async/rtc_dynamic_runtime.py",
    "examples/advanced/predictive_async/libero_rtc_dynamic.py",
    "examples/advanced/predictive_async/libero_rtc_dynamic_feedback.py",
    "examples/advanced/predictive_async/audit_libero_rtc_dynamic.py",
)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def paths(head, stage):
    check(stage in ("inputs", "feedback"), "Unknown stage")
    stem = f"smolvla_rtc_dynamic_{stage}"
    return REPO / f"outputs/{stem}_preparation_{head[:8]}", REPO / f"outputs/{stem}_{head[:8]}"


def experiment(stage):
    return "E-RDC1" if stage == "inputs" else "E-RDF1"


def input_schedule():
    groups = [(0, i, 0, 0) for i in range(16)]
    number = 0
    for delay in range(9):
        for length in range(delay + 2, 31):
            groups.append((delay + 1, number % 16, delay, length))
            number += 1
    rows = []
    for pair, (group, sample, delay, length) in enumerate(groups):
        modes = ("native_eager", "graph") if pair % 2 == 0 else ("graph", "native_eager")
        for mode in modes:
            rows.append(
                {
                    "ordinal": len(rows),
                    "pair": pair,
                    "group": group,
                    "sample": sample,
                    "delay": delay,
                    "length": length,
                    "arm": mode,
                }
            )
    return rows


def manifest(stage):
    base = {
        "experiment": experiment(stage),
        "stage": stage,
        "chunk": 50,
        "steps": 10,
        "schedule": "EXP",
        "horizon": 10,
        "guidance": 10.0,
        "fps": 20,
        "cap": 8,
        "margin": 1,
        "latency_limit_s": 0.35,
        "training": 0,
        "new_qualification": False,
        "origin": "observation_index",
        "trim": "actual_consumed_once",
        "attempts": 1,
        "retries": 0,
    }
    if stage == "inputs":
        return {
            **base,
            "requests": 482,
            "pairs": 241,
            "guided_pairs": 225,
            "rows": input_schedule(),
            "scope": "all submit-feasible d0..8, L=d+2..30; 16 cyclic old inputs",
            "prefix_source": "exact deterministic leading L rows of archived normalized tail; interface probe",
            "captures": 2,
            "new_env": 0,
        }
    template = g.manifest()["rows"][0]
    rows = []
    for pair, (task, state) in enumerate(PAIRS):
        shift = pair % 3
        for arm in ARMS[shift:] + ARMS[:shift]:
            rows.append(
                {
                    **template,
                    "ordinal": len(rows),
                    "pair_index": pair,
                    "arm": arm,
                    "task_id": task,
                    "task_name": e.reference.TASK_NAMES[task],
                    "initial_state_id": state,
                    "condition": "serialized" if arm == "serialized" else "async",
                    "environment_seed": 1160000 + 100 * task + state,
                    "policy_seed": 1170000 + 100 * task + state,
                    "cohort": "pilot_development" if pair < 8 else "known_diagnostic",
                    "limits": {k: v[0] for k, v in LIMITS.items()},
                }
            )
    return {
        **base,
        "rows": rows,
        "captures": 60,
        "episodes": 30,
        "global_limits": {k: v[1] for k, v in LIMITS.items()},
        "forced_wait_in_async": False,
        "control_loop": "unchanged",
        "queue": "unchanged RTCExecutionQueue",
        "gate": "no lost successes vs both controls, all request budgets, real overlap, less waiting, no queue faults",
    }


def sources():
    values = g.source_hashes()
    files = [
        previous.DATA,
        previous.LOCK,
        previous.BASELINE,
        REPO / "outputs/smolvla_rtc_graph_a43a31d8/independent_audit.json",
        REPO / "docs/experiments/SMOLVLA_RTC_DYNAMIC_PLAN.md",
        REPO / "tests/test_rtc_dynamic_runtime.py",
        *(REPO / p for p in ADDITIONS),
    ]
    values.update({str(p): digest(p) for p in files})
    return values


def tree_gate(head):
    check(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == head,
        "Execution HEAD changed",
    )
    check(old.pending_state() == json.loads(previous.BASELINE.read_text())["pending"], "Old pending changed")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_HEAD, "--", "src/lerobot", "examples/advanced/predictive_async"],
        cwd=REPO,
        text=True,
    ).splitlines()
    check(set(changed) <= set(ADDITIONS), f"Frozen source modified: {changed}")
    lock = json.loads(previous.LOCK.read_text())
    check(all(digest(REPO / p) == h for p, h in lock["core_sha256"].items()), "Frozen candidate changed")


def prepare(head, stage):
    tree_gate(head)
    if stage == "feedback":
        accepted = json.loads((paths(head, "inputs")[1] / "independent_audit.json").read_text())
        check(
            accepted["independent_contract_accepted"]
            and accepted["dynamic_equivalence_passed"]
            and accepted["graph_budget_passed"],
            "Dynamic input prerequisite not met",
        )
    prior = json.loads((REPO / "outputs/smolvla_rtc_graph_a43a31d8/independent_audit.json").read_text())
    check(prior["full_rtc_equivalence_passed"], "Fixed RTC prerequisite missing")
    prep, out = paths(head, stage)
    check(not prep.exists() and not out.exists(), "Unique preparation/output required")
    data = load(previous.DATA)
    check(
        [s["key"] for s in data] == [list(k) for k in previous.prior.EXPECTED_KEYS],
        "Input identities changed",
    )
    env, history = q.runtime_environment(), q.history_inventory(REPO / "outputs")
    if stage == "feedback":
        check(set(PAIRS) <= {(v["task"], v["state"]) for v in history}, "Development identities missing")
    check(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    prep.mkdir()
    write(
        prep / "preparation.json",
        {
            "head": head,
            "manifest": manifest(stage),
            "sources": sources(),
            "environment": env,
            "history": history,
        },
    )
    print(
        json.dumps(
            {
                "prepared": True,
                "stage": stage,
                "prep": str(prep),
                "out": str(out),
                "sha256": digest(prep / "preparation.json"),
            }
        ),
        flush=True,
    )


def validate(head, stage, after=False):
    tree_gate(head)
    prep, out = paths(head, stage)
    saved = json.loads((prep / "preparation.json").read_text())
    check(saved["head"] == head and saved["manifest"] == manifest(stage), "Contract changed")
    check(
        saved["sources"] == sources() and saved["environment"] == q.runtime_environment(),
        "Source/environment changed",
    )
    history = q.history_inventory(REPO / "outputs")
    if after:
        history = [v for v in history if not v["path"].startswith(out.name + "/")]
    check(saved["history"] == history, "History or concurrent collection changed")
    body = (prep / "registration.md").read_text()
    got = json.loads((prep / "registration_readback.json").read_text())
    check(
        type(got.get("id")) is int
        and got.get("body") == body
        and got.get("issue_url") == "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        and all(
            s in body
            for s in (f"{experiment(stage)}-REGISTER:{head}", str(out), digest(prep / "preparation.json"))
        ),
        "Actual registration missing or changed",
    )
    if stage == "feedback":
        prerequisite = json.loads((paths(head, "inputs")[1] / "independent_audit.json").read_text())
        check(
            prerequisite["dynamic_equivalence_passed"] and prerequisite["graph_budget_passed"],
            "Input gate not passed",
        )
    return saved


def input_request(spec, sample, rt, policy, pre, post, calls):
    cid = calls.start("dynamic_request", spec["group"], limit=30, request_id=spec["ordinal"])
    entered = time.perf_counter()
    try:
        rt.mode = spec["arm"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            policy.reset()
            pre.reset()
            post.reset()
            batch = pre(pilot.worker_batch(sample["observation"], sample["language"], "cuda"))
            noise = sample["inputs"][7].cuda().clone()
            prefix = sample["prefix"][: spec["length"]].cuda().clone() if spec["length"] else None
            normal = policy.predict_action_chunk(
                batch,
                noise=noise,
                prev_chunk_left_over=prefix,
                inference_delay=spec["delay"],
                execution_horizon=10,
            )
            processed = post(normal).detach().cpu().clone()
            normal = normal.detach().cpu().clone()
            full, inputs = rt.latest.detach().cpu().clone(), q.cpu(rt.latest_inputs)
            weights, padded = cpu_optional(rt.latest_weights), cpu_optional(rt.latest_padded_prefix)
            torch.cuda.synchronize()
            end = time.perf_counter()
        row = {
            **spec,
            "complete_s": end - start,
            "full": full,
            "normalized": normal,
            "processed": processed,
            "inputs": inputs,
            "weights": weights,
            "padded": padded,
            "metadata": dict(rt.metadata),
        }
        calls.emit("call_return", call_id=cid, elapsed=time.perf_counter() - entered)
        return row
    except BaseException:
        calls.emit("call_error", call_id=cid, exception=traceback.format_exc())
        raise


def input_worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls = e.Calls(args.output / "calls.jsonl")
    result = {
        "experiment": "E-RDC1",
        "stage": "inputs",
        "execution_head": args.execution_head,
        "status": "technical_failure",
        "first_failure": None,
    }
    rt, policy, rows = None, None, []
    try:
        prepared = validate(args.execution_head, "inputs")
        data = load(previous.DATA)
        with pilot.phase(args.output, "load_model", 90):
            check(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(
                not policy.config.compile_model and policy.config.rtc_config is None,
                "Model configuration changed",
            )
            write(args.output / "policy_load.json", report)
            policy.config.rtc_config = config()
            policy.init_rtc_processor()
        rt = DynamicRTCGraphRuntime(policy.model)
        with rt:
            paired = {}
            for group in range(10):
                with pilot.phase(args.output, f"input_group_{group}", 90):
                    for spec in [s for s in input_schedule() if s["group"] == group]:
                        check(spec["ordinal"] == len(rows) and len(rows) < 482, "Input budget/order")
                        sample = data[spec["sample"]]
                        row = input_request(spec, sample, rt, policy, pre, post, calls)
                        rows.append(row)
                        torch.save(row, args.output / f"request_{spec['ordinal']:03d}.pt")
                        for j in range(8):
                            pilot.require_equal(
                                row["inputs"][j], sample["inputs"][j], f"Dynamic input{j} differs"
                            )
                        if spec["length"]:
                            pilot.require_equal(
                                row["inputs"][8], sample["prefix"][: spec["length"]], "Wrong prefix"
                            )
                        else:
                            pilot.require_equal(
                                row["full"], sample["archived_full_chunk"], "No-prefix anchor differs"
                            )
                        if spec["pair"] in paired:
                            for field in ("full", "normalized", "processed"):
                                pilot.require_equal(
                                    row[field], paired[spec["pair"]][field], f"Dynamic {field} differs"
                                )
                            del paired[spec["pair"]]
                        else:
                            paired[spec["pair"]] = row
                print(f"E-RDC1 group{group} complete: {len(rows)}/482 requests", flush=True)
            check(not paired and len(rows) == 482, "Pair coverage")
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "Model unfrozen")
        check(prepared["sources"] == sources(), "Source mutation")
        result.update(status="completed", vla_frozen=True, dynamic_equivalence_passed=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if rt is not None:
            result["runtime_receipt"] = rt.receipt
        if policy is not None:
            policy.config.rtc_config = None
            policy.init_rtc_processor()
        calls.close()
        result.update(
            completed_requests=len(rows),
            attempts=1,
            retries=0,
            new_env=0,
            training_updates=0,
            qualification_reads=0,
            realtime_qualified=False,
        )
        write(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


class Budget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        check(self.episode[kind] < local and self.total[kind] < total, f"E-RDF1 {kind} budget exhausted")


def episode(spec, output, policy, pre, post, factory, budget, calls, paired=None):
    folder = output / f"episode_{spec['ordinal']:03d}"
    folder.mkdir()
    write(folder / "started.json", {"spec": spec, "started_at_utc": datetime.now(UTC).isoformat()})
    budget.begin_episode()
    native = e.NativeSession(spec, budget, calls, factory)
    owner, predict, control, initial = None, None, None, None
    observations = []
    result = {
        "spec": spec,
        "status": "technical_failure",
        "first_failure": None,
        "worker_joined": False,
        "environment_closed": False,
        "initial_pair_exact": None,
        "controller_thread": threading.get_ident(),
    }
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
        predict = DynamicRTCPredictor(policy, pre, post, spec)
        owner = RTCPrefixOwner(
            predict, calls, spec, budget, on_owner_close=predict.close, request_kind="dynamic_graph_request"
        )
        start = time.perf_counter()
        owner.submit(initial, 0)
        owner.receive(block=True)
        result["startup_s"] = time.perf_counter() - start
        control = old.control_loop(owner, native, spec, initial, observe)
        result.update(
            status="completed", success=control["success"], terminal_reason=control["terminal_reason"]
        )
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
                result.update(
                    requests=owner.rows,
                    owner_cleanup=owner.cleanup_evidence,
                    predictor_cleanup=predict.receipt,
                )
                torch.save(
                    {"observations": observations, "control": control, "outputs": owner.outputs},
                    folder / "arrays.pt",
                )
        if not result["worker_joined"] or not result["environment_closed"]:
            result["status"] = "technical_failure"
        result.update(
            budget=dict(budget.episode),
            native_steps=native.native_steps,
            native_returned=dict(native.returned),
        )
        if control is not None:
            wall = control["ended_at"] - control["t0"]
            count = min(spec["ready_wall_slots"], max(1, math.ceil(wall * 20)))
            result.update(
                wall_s=wall,
                measured_actions=len(control["dispatches"]),
                wall_slots=count,
                no_action_slots=count - len(control["dispatches"]),
                underflows=sum(x["outcome"] == "underflow" for x in control["gets"]),
            )
        write(folder / "result.json", result)
    return result, initial


def feedback_worker(args):
    torch.set_num_threads(1)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    calls, budget, records = e.Calls(args.output / "calls.jsonl"), Budget(), []
    result = {
        "experiment": "E-RDF1",
        "stage": "feedback",
        "execution_head": args.execution_head,
        "status": "technical_failure",
        "first_failure": None,
    }
    try:
        prepared = validate(args.execution_head, "feedback")
        with pilot.phase(args.output, "environment_preflight", 30):
            factory = e.reference.make_native_env_factory()
            check(not torch.cuda.is_initialized(), "Preflight initialized CUDA")
        with pilot.phase(args.output, "load_model", 90):
            check(torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4070 Ti SUPER", "GPU changed")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            check(
                not policy.config.compile_model and policy.config.rtc_config is None,
                "Unexpected model config",
            )
            write(args.output / "policy_load.json", report)
        initials, boots = {}, {}
        for spec in manifest("feedback")["rows"]:
            with pilot.phase(args.output, f"episode_{spec['ordinal']}", 100):
                print(
                    f"E-RDF1 START {spec['ordinal']}/30 {spec['arm']} {spec['task_id']}/{spec['initial_state_id']}",
                    flush=True,
                )
                rec, initial = episode(
                    spec,
                    args.output,
                    policy,
                    pre,
                    post,
                    factory,
                    budget,
                    calls,
                    initials.get(spec["pair_index"]),
                )
                records.append(rec)
                check(rec["status"] == "completed", rec["first_failure"] or "Incomplete episode")
                arrays = load(args.output / f"episode_{spec['ordinal']:03d}/arrays.pt")
                boot, pair = arrays["outputs"][1], spec["pair_index"]
                if pair in boots:
                    for field in ("full", "noise"):
                        pilot.require_equal(
                            boot[field], boots[pair][field], "Bootstrap " + field + " differs"
                        )
                else:
                    initials[pair], boots[pair] = initial, {k: boot[k] for k in ("full", "noise")}
                del arrays
                print(
                    f"E-RDF1 END {spec['ordinal']} success={rec['success']} actions={rec['measured_actions']}",
                    flush=True,
                )
        check(all(not p.requires_grad and p.grad is None for p in policy.parameters()), "Model unfrozen")
        tree_gate(args.execution_head)
        check(sources() == prepared["sources"], "Source mutation")
        result.update(status="completed", vla_frozen=True, frozen_baseline_unchanged=True)
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        calls.close()
        caps = [c for v in records for c in (v.get("predictor_cleanup") or {}).get("captures", [])]
        result.update(
            episodes_completed=sum(v["status"] == "completed" for v in records),
            native_budget=dict(budget.total),
            graph_captures=len(caps),
            capture_internal={k: sum(c[k] for c in caps) for k in ("setup", "warmup", "capture")},
            outcomes=[
                {k: v.get(k) for k in ("spec", "success", "measured_actions", "terminal_reason")}
                for v in records
            ],
            attempts=1,
            retries=0,
            training_updates=0,
            qualification_reads=0,
            real_robot=0,
            realtime_qualified=False,
        )
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
    script = (
        __file__ if stage == "inputs" else str(Path(__file__).with_name("libero_rtc_dynamic_feedback.py"))
    )
    return old.supervise(
        args,
        validate_run=lambda h: validate(h, stage),
        paths_for=lambda h: paths(h, stage),
        manifest_for=lambda: manifest(stage),
        worker_script=script,
        experiment=experiment(stage),
    )


if __name__ == "__main__":
    raise SystemExit(main())
