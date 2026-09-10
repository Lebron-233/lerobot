"""E-NAT1: one fixed 20-episode scheduling comparison without injected delays.

The accepted recovery policy is enabled in both scheduling arms. This is a new
development experiment, not a continuation of E or a predictor benefit test.
"""

import argparse
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_cap_recovery_native as recovery
import libero_graph_cap_trace_native as trace
import torch

e = recovery.e
REPO = e.REPO
PREPARATION = REPO / "outputs/smolvla_graph_natural_preparation_4ab3a03e"
EXPERIMENT = "E-NAT1"


def manifest():
    declared = e.fixed_manifest()
    declared.update(
        experiment=EXPERIMENT,
        predecessor_execution_head="563a8077b69785e90763be98da4dbce1bea7a996",
        predecessor_result_head="83017092832a53b870aa3491ac4bd1835a10481d",
        replaces_or_completes_predecessor=False,
        intervention="none",
        recovery_policy=recovery.RECOVERY_POLICY,
        recovery_probe_limit={"per_episode": 50, "total": 1000, "included_in_main_calls": True},
        production_default="disabled",
        attempts=1,
        retries=0,
        predictor_calls=0,
        training_calls=0,
        real_robot_calls=0,
    )
    for row in declared["rows"]:
        row.update(recovery_policy=recovery.RECOVERY_POLICY, max_recovery_probes_per_episode=50)
    return declared


class NaturalEngine(trace.TraceEngine):
    """Retain the accepted phase/terminal journal, but never inject a pause."""

    def _prepare_queue_actions(self, actions, metrics):
        with self.traced("independent_cpu_chunks_and_original_completion_barrier"):
            return recovery.RecoveryEngine._prepare_queue_actions(self, actions, metrics)


def summarize(records, declared):
    result = e.summarize(records, declared)
    for key in e.E_FIELDS:
        result[f"nat_{key}"] = result.pop(key)
    analyses = [
        {"ordinal": row["spec"]["ordinal"], **recovery.recovery_evidence(row)}
        for row in records
        if row.get("requests")
    ]
    result.update(
        experiment=EXPERIMENT,
        recovery_episodes=analyses,
        nat_recovery_probe_observed=any(row["probe_count"] for row in analyses),
        nat_recovery_row0_observed=any(row["native_recovery_chains"] for row in analyses),
        nat_no_injected_pause=not any(
            request.get("host_intervention") for row in records for request in row.get("requests", [])
        ),
        production_default_unchanged=True,
        predecessor_results_unchanged=True,
        independent_cpu_audit_completed=False,
    )
    return result


def finalize(result, execution, accounting, worker_result):
    result.update(execution=execution, **accounting)
    result["budget"] = worker_result.get("budget")
    result["budget_complete"] = result["budget"] is not None
    result["first_failure"] = result.get("first_failure") or worker_result.get("first_failure")
    failed = bool(
        result["status"] != "completed"
        or execution["stop_reason"]
        or execution["child_exit_code"] != 0
        or accounting["unknown_calls"]
        or accounting["call_errors"]
        or not accounting["journal_consistent"]
        or not result["budget_complete"]
        or not result["nat_no_injected_pause"]
    )
    if failed:
        result.update(
            status="technical_failure",
            nat_native_closed_loop_contract_passed=False,
            nat_paired_scheduling_comparison_complete=False,
        )
        result["first_failure"] = result["first_failure"] or {
            "source": "supervisor_and_flushed_journal",
            "stop_reason": execution["stop_reason"],
            "child_exit_code": execution["child_exit_code"],
            "unknown_calls": accounting["unknown_calls"],
            "worker_final_receipt_present": bool(worker_result),
        }
    return result


def worker(args):
    e.require_source(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    declared, budget = manifest(), e.Budget()
    calls = e.Calls(args.output / "calls.jsonl")
    records, result = [], {}
    try:
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("The registered GPU changed")
        calls.emit("policy_load_started")
        start = time.perf_counter()
        policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
        load = {"model_load_seconds": time.perf_counter() - start, "policy_load": report}
        e.write_json(args.output / "policy_load.json", load)
        calls.emit("policy_load_returned", model_load_seconds=load["model_load_seconds"])
        result.update(load)
        factory = trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
        paired_initial = None
        for spec in declared["rows"]:
            print(
                f"START {EXPERIMENT} {spec['ordinal']} task={spec['task_id']} {spec['condition']}",
                flush=True,
            )
            record, initial = e.run_episode(
                spec,
                args.output / f"episode_{spec['ordinal']:03d}",
                policy,
                pre,
                post,
                factory,
                budget,
                calls,
                paired_initial,
                engine_class=NaturalEngine,
            )
            records.append(record)
            print(f"END {EXPERIMENT} {spec['ordinal']} {record['status']}", flush=True)
            if record["status"] != "completed":
                break
            paired_initial = initial if spec["ordinal"] % 2 == 0 else None
        result.update(summarize(records, declared))
    except BaseException:
        result.update(status="technical_failure", first_failure=traceback.format_exc())
    finally:
        result.update(budget=dict(budget.total), calls_logging_seconds=calls.logging_seconds)
        calls.close()
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result.get("nat_native_closed_loop_contract_passed") else 2


def supervise(args):
    e.require_source(args.execution_head)
    expected = REPO / "outputs" / f"smolvla_graph_natural_{args.execution_head[:8]}"
    if Path(sys.executable) != Path(e.PYTHON) or args.output != expected or args.output.exists():
        raise ValueError("Use the fixed Python and exclusive E-NAT1 output")
    gates = json.loads((PREPARATION / "preparation_gates.json").read_text())
    if not all(
        gates.get(key) is True
        for key in ("cpu_contract_passed", "model_entry_passed", "lint_format_passed", "environment_exact")
    ):
        raise ValueError("E-NAT1 preparation gates failed")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("E-NAT1 registration/readback differs")
    args.output.mkdir()
    for path in PREPARATION.iterdir():
        if path.is_file():
            shutil.copyfile(path, args.output / path.name)
    declared = manifest()
    e.write_json(args.output / "manifest.json", declared)
    command = [
        e.PYTHON,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--execution-head",
        args.execution_head,
        "--output",
        str(args.output),
        "--worker",
    ]
    start, started_at = time.perf_counter(), datetime.now(UTC).isoformat()
    cursor, pending, stop_reason, signaled_at, forced = 0, {}, None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"{EXPERIMENT} only queue pid={child.pid} head={args.execution_head}", flush=True)
        try:
            while child.poll() is None:
                now = time.perf_counter()
                path = args.output / "calls.jsonl"
                if path.exists():
                    with path.open() as stream:
                        stream.seek(cursor)
                        while line := stream.readline():
                            if not line.endswith("\n"):
                                break
                            cursor = stream.tell()
                            event = json.loads(line)
                            if event["event"] == "call_intent":
                                pending[event["call_id"]] = event
                            elif event["event"] in ("call_return", "call_error"):
                                pending.pop(event["call_id"], None)
                expired = [value for value in pending.values() if now - value["timestamp"] > value["limit"]]
                if signaled_at is None and (expired or now - start >= declared["soft_stop_seconds"]):
                    stop_reason = (
                        {"expired_calls": expired}
                        if expired
                        else {"outer_soft_limit": declared["soft_stop_seconds"]}
                    )
                    os.kill(child.pid, signal.SIGUSR1)
                    stop_reason["stack_dump_requested"] = True
                    os.killpg(child.pid, signal.SIGTERM)
                    signaled_at = now
                if now - start >= declared["hard_stop_seconds"] or (
                    signaled_at is not None and stop_reason.get("expired_calls") and now - signaled_at >= 5
                ):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            stop_reason = stop_reason or {"supervisor_exception": traceback.format_exc()}
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
        exit_code = child.wait()
    execution = {
        "execution_head": args.execution_head,
        "command": command,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "child_pid": child.pid,
        "child_exit_code": exit_code,
        "child_exit_confirmed": True,
        "wall_seconds": time.perf_counter() - start,
        "stop_reason": stop_reason,
        "forced_termination": forced,
        "attempts": 1,
        "retries": 0,
    }
    e.write_json(args.output / "execution.json", execution)
    records = []
    for spec in declared["rows"]:
        directory = args.output / f"episode_{spec['ordinal']:03d}"
        path = directory / "result.json"
        records.append(
            json.loads(path.read_text())
            if path.exists()
            else {
                "spec": spec,
                "status": "started_return_unknown" if (directory / "started.json").exists() else "not_run",
            }
        )
    path = args.output / "worker_result.json"
    final_worker = json.loads(path.read_text()) if path.exists() else {}
    result = finalize(
        summarize(records, declared),
        execution,
        trace.journal_accounting(trace.read_events(args.output / "calls.jsonl")),
        final_worker,
    )
    load_path = args.output / "policy_load.json"
    result.update(
        json.loads(load_path.read_text())
        if load_path.exists()
        else {"policy_load": None, "model_load_seconds": None}
    )
    result.update(
        execution_head=args.execution_head,
        preparation_gates=gates,
        episode_statuses=[{"ordinal": row["spec"]["ordinal"], "status": row["status"]} for row in records],
    )
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps({key: result[key] for key in ("status", "episodes_completed", "first_failure")}),
        flush=True,
    )
    return 0 if result["nat_native_closed_loop_contract_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    signal.signal(signal.SIGTERM, e.request_shutdown)
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
