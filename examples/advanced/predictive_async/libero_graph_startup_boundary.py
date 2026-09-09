"""E-S1: one fresh native startup of the old E case, with cap 8 and no measured actions."""

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_identity_native as e
import torch

REPO = e.REPO
PREPARATION = REPO / "outputs/smolvla_graph_startup_boundary_preparation_9038ba38"
MANIFEST = REPO / "docs/experiments/SMOLVLA_GRAPH_STARTUP_BOUNDARY_MANIFEST.json"
OLD_REPORT = REPO / "docs/experiments/SMOLVLA_GRAPH_IDENTITY_NATIVE_RESULT.json"
OLD_EXECUTION = "e00b8731b44b80a9e80dc0af91e4434a9af3d7dd"
LIMITS = {"episodes": 1, "settling": 10, "measurement": 0, "model": 3, "capture": 2}


def fixed_manifest():
    report = json.loads(OLD_REPORT.read_text())
    old = copy.deepcopy(e.fixed_manifest()["rows"][7])
    if report["execution_head"] != OLD_EXECUTION or report["first_failure"]["spec"] != old:
        raise ValueError("Old E execution or the original failed case differs")
    spec = {**copy.deepcopy(old), "ordinal": 0, "source_ordinal": 7, "limits": dict(LIMITS)}
    spec.pop("ready_wall_slots")
    return {
        "experiment": "E-S1",
        "old_execution_head": OLD_EXECUTION,
        "old_case": old,
        "source_initial": next(
            str(Path(record["directory"]) / "arrays.pt")
            for record in report["archives"]
            if record["ordinal"] == 7
        ),
        "cases": [spec],
        "policy_revision": e.reference.POLICY_REVISION,
        "vlm_revision": e.reference.VLM_REVISION,
        "assets_revision": e.reference.ASSETS_REVISION,
        "chunk_size": 50,
        "n_action_steps": 1,
        "num_steps": 10,
        "main_calls_if_ready": 3,
        "limits": dict(LIMITS),
        "startup_seconds": 30,
        "request_seconds": 15,
        "native_call_seconds": 30,
        "soft_stop_seconds": 270,
        "hard_stop_seconds": 300,
        "reference_calls": 0,
        "training_calls": 0,
        "real_robot_calls": 0,
        "frozen_max_prediction_delay": 8,
        "measured_native_dispatches": 0,
        "source_contract_defect_identified": False,
        "source_contract_defect_fixed": False,
        "native_closed_loop_contract_passed": False,
        "paired_scheduling_comparison_complete": False,
        **e.FLAGS,
    }


def read_manifest():
    value = json.loads(MANIFEST.read_text())
    if value != fixed_manifest():
        raise ValueError("The one-case E-S1 manifest differs from its frozen contract")
    return value


class StartupBudget(e.Budget):
    def check(self, kind):
        if self.episode[kind] >= LIMITS[kind] or self.total[kind] >= LIMITS[kind]:
            raise RuntimeError(f"E-S1 {kind} budget exhausted before dispatch")


class BoundaryEngine(e.NativeEngine):
    """CPU host snapshots around the unchanged original startup and completion notification."""

    def __init__(self, *args, **kwargs):
        self.boundary_events = []
        self.audit_host_seconds = 0.0
        super().__init__(*args, **kwargs)

    def record_boundary(self, kind, **values):
        started = time.perf_counter()
        self.boundary_events.append(
            {
                "event": kind,
                "host_monotonic_seconds": started,
                "owner_thread": threading.get_ident(),
                "reset_epoch": self._reset_epoch,
                "startup_phase": self._startup_phase,
                "tracker_samples_seconds": list(self._latency_tracker._values),
                **values,
            }
        )
        self.audit_host_seconds += time.perf_counter() - started

    def _make_graph_runtime(self):
        runtime = super()._make_graph_runtime()
        self.record_boundary("original_owner_seed_returned", seed=self.spec["policy_seed"])
        return runtime

    def _run_request(self, request):
        self.record_boundary(
            "before_original_request",
            request_id=request.request_id,
            requested_at_s=request.requested_at,
            observation_id=request.observation["e_observation_index"],
            task_epoch=request.task_epoch,
        )
        return super()._run_request(request)

    def _validate_startup_probe(self, metrics, *, latency_s, actions_finite):
        self.record_boundary(
            "before_original_gate",
            request_id=metrics["request_id"],
            latency_argument_seconds=latency_s,
            actions_finite=actions_finite,
        )
        passed = False
        try:
            super()._validate_startup_probe(metrics, latency_s=latency_s, actions_finite=actions_finite)
            passed = True
        finally:
            self.record_boundary(
                "after_original_gate",
                request_id=metrics["request_id"],
                original_gate_returned=passed,
                required_steps_unclamped=metrics["startup_gate_raw_required_delay_steps"],
            )

    def _request_finished(self, request):
        # Snapshot before waking the controller; no next request or stop can precede this record.
        self.record_boundary(
            "before_completion_notification", request_id=request.request_id, ready=self.ready
        )
        super()._request_finished(request)


def run_startup_only(
    policy,
    pre,
    post,
    factory,
    spec,
    directory,
    calls,
    old_initial,
    *,
    device="cuda",
    engine_class=BoundaryEngine,
    clock=None,
):
    clock = clock or e.Clock()
    directory.mkdir()
    budget = StartupBudget()
    budget.begin_episode()
    native = e.NativeSession(spec, budget, calls, factory)
    engine = initial = None
    started = clock.now()
    result = {
        "spec": spec,
        "status": "technical_failure",
        "native_diagnostic_started": False,
        "startup_gate_passed_this_run": None,
        "startup_ready_this_run": False,
        "required_steps_unclamped": None,
        "initial_exact": None,
        "first_failure": None,
        "cleanup_errors": [],
        "cleanup_confirmed": None,
        "measured_native_dispatches": 0,
        "frozen_max_prediction_delay": 8,
        "source_contract_defect_identified": False,
        "source_contract_defect_fixed": False,
        "native_closed_loop_contract_passed": False,
        "paired_scheduling_comparison_complete": False,
        **e.FLAGS,
    }
    e.write_json(directory / "started.json", {"spec": spec, "host_started_at": started})
    try:
        result["native_diagnostic_started"] = True
        raw = native.create_reset()
        obs, features, initial = e.observation(raw, spec["task_name"].replace("_", " "), 0, clock.now())
        difference = e.initial_difference(old_initial, initial)
        result.update(initial_exact=difference is None, initial_difference=difference)
        if difference is not None:
            raise ValueError(f"Original E initial observation differs: {difference}")
        engine = engine_class(policy, pre, post, features, spec, budget, calls, device=device)
        startup_started = clock.now()
        try:
            result["startup_ready_seconds"] = e.startup(engine, obs, clock)
        finally:
            result["startup_attempt_seconds"] = clock.now() - startup_started
        result.update(status="startup_passed", startup_ready_this_run=engine.ready)
        # This diagnostic ends here, including when the original gate passed. No get or env.step.
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        cleanup_started = clock.now()
        if engine is not None:
            try:
                engine.stop()
            except BaseException:
                result["cleanup_errors"].append(traceback.format_exc())
            for key in ("worker_joined", "graph_released", "original_sampler_restored"):
                result[key] = getattr(engine, key)
            result.update(metrics_closed=engine.metrics.closed, worker_failed=engine.failed)
        if engine is None or engine.worker_joined:
            try:
                result["environment_closed"] = native.close()
            except BaseException:
                result["cleanup_errors"].append(traceback.format_exc())
        result["cleanup_seconds"] = clock.now() - cleanup_started
        cleanup_fields = (
            ["environment_closed"]
            if engine is None
            else [
                "environment_closed",
                "worker_joined",
                "graph_released",
                "original_sampler_restored",
                "metrics_closed",
            ]
        )
        result["cleanup_confirmed"] = not result["cleanup_errors"] and all(
            result.get(k) is True for k in cleanup_fields
        )
        if engine is not None and engine.worker_joined:
            result.update(
                requests=engine.rows,
                metrics=engine.metrics.events,
                stats=asdict(engine.stats),
                captures=engine.runtime.captures if engine.runtime else [],
                boundary_events=engine.boundary_events,
                audit_host_seconds=engine.audit_host_seconds,
                api_counts=dict(engine.audit.counts),
                api_events=engine.audit.events,
                owner_lifecycle=engine.lifecycle,
                owner_thread=engine.owner_thread,
                controller_thread=threading.get_ident(),
                peak_model_inflight=engine.peak_active_calls,
                queue_get_count=sum(event["event"] == "queue_get" for event in engine.metrics.events),
            )
            probe = next(
                (event for event in engine.metrics.events if event.get("request_kind") == "startup_probe"),
                None,
            )
            if probe is not None:
                result["required_steps_unclamped"] = probe["startup_gate_raw_required_delay_steps"]
                result["startup_gate_passed_this_run"] = probe["startup_gate_outcome"] == "passed"
                if probe["startup_gate_outcome"] == "cap_exceeded":
                    result["status"] = "gate_rejected"
        if not result["cleanup_confirmed"]:
            result["status"] = "cleanup_failure"
        if result["cleanup_confirmed"]:
            serialization_started = clock.now()
            torch.save(
                {"initial": initial, "requests": {} if engine is None else engine.arrays},
                directory / "arrays.pt",
            )
            result["serialization_seconds"] = clock.now() - serialization_started
        else:
            result["arrays_not_serialized"] = "Owner/join/Graph/Env cleanup is not confirmed"
        result.update(
            native_steps=native.native_steps,
            native_returned=dict(native.returned),
            budget=dict(budget.total),
            logging_seconds=calls.logging_seconds,
            episode_wall_seconds=clock.now() - started,
        )
        e.write_json(directory / "result.json", result)
    return result


def worker(args):
    e.require_source(args.execution_head)
    manifest = read_manifest()
    calls = e.Calls(args.output / "calls.jsonl")
    result = {
        "status": "technical_failure",
        "native_diagnostic_started": False,
        "first_failure": None,
        "startup_gate_passed_this_run": None,
        "required_steps_unclamped": None,
        "cleanup_confirmed": None,
    }
    try:
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("The registered GPU changed")
        old_initial = torch.load(
            manifest["source_initial"], weights_only=True, map_location="cpu", mmap=True
        )["observations"][0]
        started = time.perf_counter()
        try:
            policy, pre, post, policy_load = e.load_runtime(e.POLICY, e.VLM)
        finally:
            result["model_load_attempt_seconds"] = time.perf_counter() - started
        model_load_seconds = result["model_load_attempt_seconds"]
        result = run_startup_only(
            policy,
            pre,
            post,
            e.reference.make_native_env_factory(),
            manifest["cases"][0],
            args.output / "case",
            calls,
            old_initial,
        )
        result.update(model_load_seconds=model_load_seconds, policy_load=policy_load)
    except BaseException:
        result["first_failure"] = result["first_failure"] or traceback.format_exc()
    finally:
        calls.close()
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "startup_passed" else 2


def supervise(args):
    e.require_source(args.execution_head)
    manifest = read_manifest()
    expected = REPO / "outputs" / f"smolvla_graph_startup_boundary_{args.execution_head[:8]}"
    if Path(sys.executable) != Path(e.PYTHON) or args.output != expected or args.output.exists():
        raise ValueError("Use the unchanged model Python and new exclusive E-S1 output")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("E-S1 registration and exact readback differ")
    args.output.mkdir()
    for path in PREPARATION.iterdir():
        if path.is_file():
            shutil.copyfile(path, args.output / path.name)
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
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    cursor, pending, signaled_at, stop_reason, forced = 0, {}, None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"E-S1 only queue pid={child.pid} head={args.execution_head} measured=0 cap=8", flush=True)
        try:
            while child.poll() is None:
                now = time.perf_counter()
                journal_path = args.output / "calls.jsonl"
                if journal_path.exists():
                    with journal_path.open() as journal:
                        journal.seek(cursor)
                        while line := journal.readline():
                            if not line.endswith("\n"):
                                break
                            cursor = journal.tell()
                            event = json.loads(line)
                            if event["event"] == "call_intent":
                                pending[event["call_id"]] = event
                            elif event["event"] in ("call_return", "call_error"):
                                pending.pop(event["call_id"], None)
                expired = [event for event in pending.values() if now - event["timestamp"] > event["limit"]]
                if signaled_at is None and (expired or now - started >= 270):
                    stop_reason = {"expired_calls": expired} if expired else {"outer_soft_limit": 270}
                    os.killpg(child.pid, signal.SIGTERM)
                    signaled_at = now
                if now - started >= 300 or (
                    signaled_at is not None and stop_reason.get("expired_calls") and now - signaled_at >= 5
                ):
                    os.killpg(child.pid, signal.SIGKILL)
                    forced = True
                    break
                time.sleep(0.1)
        except BaseException:
            stop_reason = {"supervisor_exception": traceback.format_exc()}
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
        "wall_seconds": time.perf_counter() - started,
        "stop_reason": stop_reason,
        "forced_termination": forced,
        "attempts": 1,
        "retries": 0,
    }
    e.write_json(args.output / "execution.json", execution)
    path = args.output / "worker_result.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "status": "technical_failure",
            "first_failure": "Worker result unavailable; see model.log and intent journal",
        }
    )
    events = e.read_call_journal(args.output / "calls.jsonl")
    returned = {v["call_id"] for v in events if v["event"] == "call_return"}
    intents = [v for v in events if v["event"] == "call_intent"]
    unknown = [v for v in intents if v["call_id"] not in returned]
    measured = [v for v in intents if v["kind"] == "native_step" and v.get("segment") == "measurement"]
    result.update(
        execution=execution,
        execution_head=args.execution_head,
        manifest=manifest,
        old_failure_recomputed=json.loads((PREPARATION / "old_audit.json").read_text())[
            "old_failure_recomputed"
        ],
        native_diagnostic_started=any(v["kind"] == "environment_factory" for v in intents),
        unknown_calls=unknown,
        unknown_native_calls=sum(v["kind"] != "model_request" for v in unknown),
        measured_native_dispatches=len(measured),
        call_intents=dict(Counter(v["kind"] for v in intents)),
        call_returns=dict(Counter(v["kind"] for v in intents if v["call_id"] in returned)),
        source_contract_defect_identified=False,
        source_contract_defect_fixed=False,
        native_closed_loop_contract_passed=False,
        paired_scheduling_comparison_complete=False,
        **e.FLAGS,
    )
    if stop_reason or unknown or measured or exit_code not in (0, 2):
        result["status"] = "technical_failure"
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "startup_gate_passed_this_run",
                    "required_steps_unclamped",
                    "measured_native_dispatches",
                    "unknown_native_calls",
                    "cleanup_confirmed",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "startup_passed" else 2


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
