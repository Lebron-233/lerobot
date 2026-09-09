"""E-RCV3: one independently registered, phase-journaled native cap-recovery diagnostic.

This does not resume E-RCV2 or change its results. The policy, controller, four
cases, limits and 600 ms intervention are inherited unchanged. New evidence is
flushed before/after model phases and initial observations survive worker loss.
"""

import argparse
import copy
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_cap_recovery_native as recovery
import libero_graph_cap_stress_native as stress
import torch

e = stress.e
REPO = e.REPO
PREPARATION = REPO / "outputs/smolvla_graph_cap_trace_preparation_4d62316f"
EXPERIMENT = "E-RCV3-trace"
pause = time.sleep


def manifest():
    value = copy.deepcopy(stress.read_manifest())
    value.update(
        experiment=EXPERIMENT,
        predecessor_execution_head="9b7aa8685ae1978c553da6e6d7ac3d4b38e3b9e6",
        predecessor_result_head="f132f6cec76bf952bbbfa014318e6aefb815e650",
        predecessor_is_terminal=True,
        replaces_or_completes_predecessor=False,
        instrumentation=[
            "flushed_model_phase_boundaries_no_new_cuda_calls",
            "initial_cpu_checkpoint_before_first_model_request",
            "policy_load_receipt_before_first_environment",
            "request_terminal_snapshots_and_cleanup_boundaries",
            "all_thread_stack_dump_on_original_watchdog_expiry",
            "supervisor_failure_and_partial_journal_accounting",
        ],
    )
    return value


class TraceEngine(recovery.RecoveryEngine):
    """Original graph worker, with CPU-only observability and the original pause."""

    def __init__(self, *args, **kwargs):
        self.pause_count = 0
        self.trace_request = None
        super().__init__(*args, **kwargs)

    def trace(self, phase, boundary, **values):
        request = self.trace_request
        self.calls.emit(
            "model_phase",
            ordinal=self.spec["ordinal"],
            request_id=None if request is None else request.request_id,
            reset_epoch=None if request is None else request.reset_epoch,
            task_epoch=None if request is None else request.task_epoch,
            phase=phase,
            boundary=boundary,
            **values,
        )

    @contextmanager
    def traced(self, phase):
        self.trace(phase, "started")
        try:
            yield
        except BaseException:
            self.trace(phase, "error", exception=traceback.format_exc())
            raise
        else:
            self.trace(phase, "returned")

    def _run_request(self, request):
        self.trace_request = request
        with self.traced("whole_request"):
            return super()._run_request(request)

    @contextmanager
    def _record_metrics_phase(self, phase, metrics, cuda_events):
        with self.traced(phase), super()._record_metrics_phase(phase, metrics, cuda_events):
            yield

    def _prepare_queue_actions(self, actions, metrics):
        with self.traced("independent_cpu_chunks_and_original_completion_barrier"):
            chunks = super()._prepare_queue_actions(actions, metrics)
        if self.current["kind"] == "planned" and self.pause_count == 0:
            self.pause_count += 1
            self.trace("host_pause", "started")
            start = time.perf_counter()
            pause(self.spec["host_pause_seconds"])
            end = time.perf_counter()
            self.current["host_intervention"] = {
                "kind": "post_cpu_completion_pre_admission_pause",
                "requested_seconds": self.spec["host_pause_seconds"],
                "started_at": start,
                "returned_at": end,
                "elapsed_seconds": end - start,
                "ordinal_within_engine": self.pause_count,
            }
            self.trace("host_pause", "returned", intervention=self.current["host_intervention"])
        return chunks

    def _request_finished(self, request):
        # A snapshot is evidence, not an independent estimator sample or action.
        try:
            super()._request_finished(request)
        finally:
            terminal = next(
                (
                    v
                    for v in reversed(self.metrics.events)
                    if v.get("request_id") == request.request_id
                    and v["event"] in ("chunk_request", "request_error")
                ),
                None,
            )
            self.calls.emit(
                "request_snapshot",
                ordinal=self.spec["ordinal"],
                request_id=request.request_id,
                row=copy.deepcopy(self.current),
                terminal=terminal,
                cumulative_api_counts=dict(self.audit.counts),
            )

    def stop(self):
        with self.traced("engine_stop"):
            super().stop()
        self.calls.emit(
            "engine_cleanup",
            ordinal=self.spec["ordinal"],
            worker_joined=self.worker_joined,
            graph_released=self.graph_released,
            original_sampler_restored=self.original_sampler_restored,
            metrics_closed=self.metrics.closed,
        )


def checkpoint_factory(factory, calls):
    """Checkpoint the actual reset return; do not create or replay another Env."""

    def create(spec):
        env = factory(spec)
        original_reset = env.reset
        saved = False

        def reset(*args, **kwargs):
            nonlocal saved
            if saved:
                raise RuntimeError("E-RCV3 initial checkpoint cannot be replaced")
            result = original_reset(*args, **kwargs)
            _, _, initial = e.observation(
                result[0], spec["task_name"].replace("_", " "), 0, time.perf_counter()
            )
            path = Path(calls.file.name).parent / f"episode_{spec['ordinal']:03d}" / "initial_checkpoint.pt"
            with path.open("xb") as stream:
                torch.save(initial, stream)
                stream.flush()
            saved = True
            calls.emit("initial_checkpoint", ordinal=spec["ordinal"], bytes=path.stat().st_size)
            return result

        env.reset = reset
        return env

    return create


def journal_accounting(events):
    """Separate errors from unknown returns and never substitute missing with zero."""
    intents = {v["call_id"]: v for v in events if v["event"] == "call_intent"}
    returns = {v["call_id"] for v in events if v["event"] == "call_return"}
    errors = {v["call_id"] for v in events if v["event"] == "call_error"}
    terminal_ids = returns | errors
    intent_count = Counter(v["call_id"] for v in events if v["event"] == "call_intent")
    terminal_count = Counter(v["call_id"] for v in events if v["event"] in ("call_return", "call_error"))
    return {
        "call_intents": dict(Counter(v["kind"] for v in intents.values())),
        "call_returns": dict(Counter(v["kind"] for k, v in intents.items() if k in returns)),
        "call_errors": dict(Counter(v["kind"] for k, v in intents.items() if k in errors)),
        "unknown_calls": [v for k, v in intents.items() if k not in terminal_ids],
        "journal_consistent": not (returns & errors)
        and not (terminal_ids - intents.keys())
        and all(n == 1 for n in intent_count.values())
        and all(n == 1 for n in terminal_count.values()),
        "measured_native_returns": sum(
            v["kind"] == "native_step" and v.get("segment") == "measurement" and k in returns
            for k, v in intents.items()
        ),
        "settling_native_returns": sum(
            v["kind"] == "native_step" and v.get("segment") == "settling" and k in returns
            for k, v in intents.items()
        ),
        "request_snapshots": sum(v["event"] == "request_snapshot" for v in events),
        "last_model_phase": next((v for v in reversed(events) if v["event"] == "model_phase"), None),
        "accounting_basis": "flushed_call_intent_return_error_journal",
    }


def summarize(records, declared):
    original = stress.summarize(records, declared)
    result = {key.replace("stress_", "trace_"): value for key, value in original.items()}
    result["experiment"] = EXPERIMENT
    result["predecessor_results_unchanged"] = True
    return result


def finalize(result, execution, accounting, worker_result):
    result.update(execution=execution, **accounting)
    result["budget"] = worker_result.get("budget")
    result["budget_complete"] = result["budget"] is not None
    result["first_failure"] = result.get("first_failure") or worker_result.get("first_failure")
    technical_failure = bool(
        result.get("status") == "technical_failure"
        or execution["stop_reason"]
        or execution["child_exit_code"] != 0
        or accounting["unknown_calls"]
        or accounting["call_errors"]
        or not accounting["journal_consistent"]
        or not result["budget_complete"]
    )
    if technical_failure:
        result.update(status="technical_failure", trace_four_episode_contract_passed=False)
        result["first_failure"] = result["first_failure"] or {
            "source": "supervisor_and_flushed_journal",
            "stop_reason": execution["stop_reason"],
            "child_exit_code": execution["child_exit_code"],
            "unknown_calls": accounting["unknown_calls"],
            "worker_final_receipt_present": bool(worker_result),
        }
    result["trace_mechanism_contrast_passed"] = bool(
        result["trace_four_episode_contract_passed"]
        and result["trace_disabled_latch_both_observed"]
        and result["trace_candidate_recovery_both_observed"]
    )
    return result


def worker(args):
    e.require_source(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    declared = manifest()
    budget = recovery.RecoveryBudget()
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
        factory = checkpoint_factory(e.reference.make_native_env_factory(), calls)
        paired_initial = None
        for spec in declared["rows"]:
            print(
                f"START {EXPERIMENT} {spec['ordinal']} task={spec['task_id']} arm={spec['arm']}", flush=True
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
                engine_class=TraceEngine,
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
    return 0 if result.get("trace_four_episode_contract_passed") else 2


def read_events(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith("\n")]


def supervise(args):
    e.require_source(args.execution_head)
    expected = REPO / "outputs" / f"smolvla_graph_cap_trace_{args.execution_head[:8]}"
    if Path(sys.executable) != Path(e.PYTHON) or args.output != expected or args.output.exists():
        raise ValueError("Use the fixed Python and exclusive E-RCV3 output")
    gates = json.loads((PREPARATION / "preparation_gates.json").read_text())
    if not all(
        gates.get(k) is True for k in ("cpu_contract_passed", "model_entry_passed", "lint_format_passed")
    ):
        raise ValueError("E-RCV3 preparation gates failed")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("E-RCV3 registration/readback differs")
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
                expired = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
                if signaled_at is None and (expired or now - start >= 870):
                    stop_reason = {"expired_calls": expired} if expired else {"outer_soft_limit": 870}
                    # The handler is registered before CUDA/model loading. No retry,
                    # timeout extension, extra model call or other process is involved.
                    os.kill(child.pid, signal.SIGUSR1)
                    stop_reason["stack_dump_requested"] = True
                    os.killpg(child.pid, signal.SIGTERM)
                    signaled_at = now
                if now - start >= 900 or (
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
        journal_accounting(read_events(args.output / "calls.jsonl")),
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
        episode_statuses=[{"ordinal": r["spec"]["ordinal"], "status": r["status"]} for r in records],
    )
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "episodes_completed", "trace_mechanism_contrast_passed", "first_failure")
            }
        ),
        flush=True,
    )
    return 0 if result["trace_four_episode_contract_passed"] else 2


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
