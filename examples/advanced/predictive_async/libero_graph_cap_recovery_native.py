"""E-RCV1: one four-episode native diagnostic of same_path_discard_probe_v1."""

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import libero_graph_identity_native as e
import torch

from lerobot.rollout.inference.latency_replay import latency_to_steps

REPO = e.REPO
PREPARATION = REPO / "outputs/smolvla_graph_cap_recovery_preparation_223156f3"
MANIFEST = REPO / "docs/experiments/SMOLVLA_GRAPH_CAP_RECOVERY_MANIFEST.json"
RECOVERY_POLICY = "same_path_discard_probe_v1"
LIMITS = {
    "episodes": (1, 4),
    "settling": (10, 40),
    "measurement": (280, 1120),
    "model": (160, 640),
    "capture": (2, 8),
}
FIELDS = (
    "recovery_native_diagnostic_started",
    "recovery_probe_observed_native",
    "recovery_estimator_returned_within_cap_native",
    "planned_takeover_after_recovery_observed_native",
    "recovery_native_four_episode_contract_passed",
    "recovery_native_two_pair_comparison_complete",
)


def fixed_manifest():
    original = e.fixed_manifest()
    rows = []
    for ordinal, old_ordinal in enumerate((0, 1, 5, 4)):
        row = copy.deepcopy(original["rows"][old_ordinal])
        row.update(
            ordinal=ordinal,
            source_ordinal=old_ordinal,
            pair_index=ordinal // 2,
            recovery_policy=RECOVERY_POLICY,
            max_recovery_probes_per_episode=50,
        )
        rows.append(row)
    return {
        "experiment": "E-RCV1",
        "old_execution_head": "e00b8731b44b80a9e80dc0af91e4434a9af3d7dd",
        "handoff_head": "223156f3d41dd37bc982bd691123f8490ed93445",
        "rows": rows,
        "recovery_policy": RECOVERY_POLICY,
        "production_default": "disabled",
        "limits": {key: {"per_episode": local, "total": total} for key, (local, total) in LIMITS.items()},
        "recovery_probe_limit": {"per_episode": 50, "total": 200, "included_in_main_calls": True},
        "policy_revision": original["policy_revision"],
        "vlm_revision": original["vlm_revision"],
        "assets_revision": original["assets_revision"],
        "chunk_size": 50,
        "n_action_steps": 1,
        "num_steps": 10,
        "soft_stop_seconds": 870,
        "hard_stop_seconds": 900,
        "request_seconds": 15,
        "native_call_seconds": 30,
        "reference_calls": 0,
        "predictor_calls": 0,
        "training_calls": 0,
        "real_robot_calls": 0,
        "attempts": 1,
        "retries": 0,
        "native_closed_loop_contract_passed": False,
        "paired_scheduling_comparison_complete": False,
        **e.FLAGS,
    }


def read_manifest():
    manifest = json.loads(MANIFEST.read_text())
    if manifest != fixed_manifest():
        raise ValueError("The fixed four-case recovery manifest differs")
    return manifest


class RecoveryBudget(e.Budget):
    def check(self, kind):
        local, total = LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"E-RCV1 {kind} budget exhausted before dispatch")


class RecoveryEngine(e.NativeEngine):
    """The original controller notifications plus CPU history evidence; same original worker."""

    def __init__(self, policy, pre, post, features, spec, budget, calls, *, device="cuda"):
        self.history_before = None
        self.history_audit_seconds = 0.0
        super().__init__(
            policy,
            pre,
            post,
            features,
            spec,
            budget,
            calls,
            device=device,
            recovery_policy=spec.get("recovery_policy", "disabled"),
            max_recovery_probes_per_episode=spec.get("max_recovery_probes_per_episode", 50),
        )

    def history(self):
        samples = list(self._latency_tracker._values)
        estimate = self._latency_tracker.percentile(0.9) if samples else None
        return {
            "samples_seconds": samples,
            "count": len(samples),
            "p90_seconds": estimate,
            "raw_required_steps": None if estimate is None else latency_to_steps(estimate, 20) + 1,
        }

    def _run_request(self, request):
        started = time.perf_counter()
        self.history_before = self.history()
        self.history_audit_seconds += time.perf_counter() - started
        return super()._run_request(request)

    def _request_finished(self, request):
        started = time.perf_counter()
        self.current["recovery"] = {
            "policy": self._recovery_policy,
            "probe_requests_dispatched": self.stats.recovery_probe_requests,
            "history_before": self.history_before,
            "history_after": self.history(),
        }
        self.history_audit_seconds += time.perf_counter() - started
        self.current["history_audit_seconds_cumulative"] = self.history_audit_seconds
        super()._request_finished(request)


def recovery_evidence(record):
    rows = record.get("requests", [])
    terminals = {
        v["request_id"]: v
        for v in record.get("metrics", [])
        if v["event"] in ("chunk_request", "request_error")
    }
    probes = []
    chains = []
    for row in rows:
        if row["kind"] != "recovery_probe":
            continue
        terminal = terminals.get(row["request_id"], {})
        trace = row.get("recovery", {})
        before, after = trace.get("history_before", {}), trace.get("history_after", {})
        valid = (
            terminal.get("outcome") == "discarded_recovery_probe"
            and terminal.get("latency_tracker_admitted") is True
        )
        within = valid and before.get("raw_required_steps", 0) > 8 and after.get("raw_required_steps", 9) <= 8
        probe = {
            "request_id": row["request_id"],
            "reset_epoch": row["reset_epoch"],
            "task_epoch": row["task_epoch"],
            "trigger_raw": terminal.get("raw_required_delay_steps"),
            "completed_seconds": terminal.get("total_chunk_s"),
            "admitted": terminal.get("latency_tracker_admitted", False),
            "outcome": terminal.get("outcome"),
            "history_before": before,
            "history_after": after,
            "estimator_returned_within_cap": within,
        }
        probes.append(probe)
        if not within or not record.get("audit", {}).get("source_audit_passed"):
            continue
        for planned in rows:
            t = terminals.get(planned["request_id"], {})
            if (
                planned["kind"] != "planned"
                or planned["requested_at"] < row["finished_at"]
                or (planned["reset_epoch"], planned["task_epoch"]) != (row["reset_epoch"], row["task_epoch"])
                or t.get("outcome") not in ("staged_early", "staged_on_time")
                or t.get("raw_required_delay_steps", 9) > 8
            ):
                continue
            index = t["takeover_index"]
            get = next(
                (
                    g
                    for g in record["metrics"]
                    if g["event"] == "queue_get" and g["outcome"] == "takeover" and g["action_index"] == index
                ),
                None,
            )
            step = next(
                (
                    s
                    for s in record.get("native_steps", [])
                    if s["segment"] == "measurement" and s["action_index"] == index
                ),
                None,
            )
            if get is not None and step is not None:
                chains.append(
                    {
                        "probe_request_id": row["request_id"],
                        "planned_request_id": planned["request_id"],
                        "reset_epoch": row["reset_epoch"],
                        "task_epoch": row["task_epoch"],
                        "takeover_action_index": index,
                        "source_row_offset": 0,
                        "native_step_number": step["number"],
                        "native_returned_at": step["returned_at"],
                        "source_audit_passed": True,
                    }
                )
                break
    metrics = record.get("metrics", [])
    counts = Counter(v.get("decision") for v in metrics if v["event"] == "planner_decision")
    return {
        "probes": probes,
        "native_recovery_chains": chains,
        "probe_count": len(probes),
        "admitted_probe_count": sum(p["admitted"] for p in probes),
        "estimator_returned_within_cap": any(p["estimator_returned_within_cap"] for p in probes),
        "planned_takeover_after_recovery": bool(chains),
        "planner_decisions": dict(counts),
        "steady_bootstraps": sum(r["kind"] == "bootstrap" and r["startup_phase"] is None for r in rows),
        "history_audit_seconds": rows[-1].get("history_audit_seconds_cumulative", 0) if rows else 0,
    }


def summarize(records, manifest):
    old_summary = e.summarize(records, manifest)
    completed = [r for r in records if r["status"] == "completed"]
    analyses = [
        {"ordinal": r["spec"]["ordinal"], **recovery_evidence(r)} for r in records if r.get("requests")
    ]
    complete = (
        len(completed) == 4
        and len(old_summary["pairs"]) == 2
        and all(p["initial_exact"] for p in old_summary["pairs"])
    )
    return {
        "status": "completed" if complete else "technical_failure",
        "experiment": "E-RCV1",
        "manifest": manifest,
        "episodes_scheduled": 4,
        "episodes_completed": len(completed),
        "pairs": old_summary["pairs"],
        "conditions": old_summary["conditions"],
        "recovery_episodes": analyses,
        "first_failure": old_summary["first_failure"],
        "original_cap_latch_reproduced_cpu": True,
        "same_path_recovery_passed_cpu": True,
        "production_default_unchanged": True,
        "recovery_native_diagnostic_started": any(r["status"] != "not_run" for r in records),
        "recovery_probe_observed_native": any(r["probe_count"] for r in analyses),
        "recovery_estimator_returned_within_cap_native": any(
            r["estimator_returned_within_cap"] for r in analyses
        ),
        "planned_takeover_after_recovery_observed_native": any(
            r["planned_takeover_after_recovery"] for r in analyses
        ),
        "recovery_native_four_episode_contract_passed": complete,
        "recovery_native_two_pair_comparison_complete": complete,
        "native_closed_loop_contract_passed": False,
        "paired_scheduling_comparison_complete": False,
        **e.FLAGS,
    }


def worker(args):
    e.require_source(args.execution_head)
    manifest = read_manifest()
    budget = RecoveryBudget()
    calls = e.Calls(args.output / "calls.jsonl")
    records = []
    outcome = {"execution_head": args.execution_head, **dict.fromkeys(FIELDS, False)}
    try:
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("The registered GPU changed")
        started = time.perf_counter()
        policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
        outcome.update(model_load_seconds=time.perf_counter() - started, policy_load=report)
        factory = e.reference.make_native_env_factory()
        paired_initial = None
        for spec in manifest["rows"]:
            print(f"START E-RCV1 {spec['ordinal']} task={spec['task_id']} {spec['condition']}", flush=True)
            result, initial = e.run_episode(
                spec,
                args.output / f"episode_{spec['ordinal']:03d}",
                policy,
                pre,
                post,
                factory,
                budget,
                calls,
                paired_initial,
                engine_class=RecoveryEngine,
            )
            records.append(result)
            print(
                f"END E-RCV1 {spec['ordinal']} {result['status']} measured={result['native_returned'].get('measurement', 0)} probes={result.get('stats', {}).get('recovery_probe_requests', 0)}",
                flush=True,
            )
            if result["status"] != "completed":
                break
            paired_initial = initial if spec["ordinal"] % 2 == 0 else None
        outcome.update(summarize(records, manifest))
    except BaseException:
        outcome.update(status="technical_failure", first_failure=traceback.format_exc())
    finally:
        outcome.update(budget=dict(budget.total), calls_logging_seconds=calls.logging_seconds)
        calls.close()
        e.write_json(args.output / "worker_result.json", outcome)
    return 0 if outcome.get("recovery_native_four_episode_contract_passed") else 2


def supervise(args):
    e.require_source(args.execution_head)
    manifest = read_manifest()
    expected = REPO / "outputs" / f"smolvla_graph_cap_recovery_{args.execution_head[:8]}"
    if Path(sys.executable) != Path(e.PYTHON) or args.output != expected or args.output.exists():
        raise ValueError("Use the fixed model Python and new exclusive E-RCV1 output")
    gates = json.loads((PREPARATION / "preparation_gates.json").read_text())
    if not all(
        gates[k] is True
        for k in (
            "original_cap_latch_reproduced_cpu",
            "same_path_recovery_passed_cpu",
            "model_entry_passed",
            "lint_format_passed",
        )
    ):
        raise ValueError("E-RCV1 preparation gates did not all pass")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("E-RCV1 registration/readback differs")
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
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
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
        print(
            f"E-RCV1 only four-case queue pid={child.pid} head={args.execution_head} hard_limit=900s",
            flush=True,
        )
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
                if signaled_at is None and (expired or now - start >= 870):
                    stop_reason = {"expired_calls": expired} if expired else {"outer_soft_limit": 870}
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
        "wall_seconds": time.perf_counter() - start,
        "stop_reason": stop_reason,
        "forced_termination": forced,
        "attempts": 1,
        "retries": 0,
    }
    e.write_json(args.output / "execution.json", execution)
    records = []
    for spec in manifest["rows"]:
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
    result = summarize(records, manifest)
    worker_path = args.output / "worker_result.json"
    worker_result = json.loads(worker_path.read_text()) if worker_path.exists() else {}
    events = e.read_call_journal(args.output / "calls.jsonl")
    returned = {v["call_id"] for v in events if v["event"] == "call_return"}
    intents = [v for v in events if v["event"] == "call_intent"]
    unknown = [v for v in intents if v["call_id"] not in returned]
    result.update(
        execution_head=args.execution_head,
        execution=execution,
        preparation_gates=gates,
        first_failure=result["first_failure"] or worker_result.get("first_failure"),
        policy_load=worker_result.get("policy_load"),
        model_load_seconds=worker_result.get("model_load_seconds"),
        recovery_native_diagnostic_started=any(v["kind"] == "environment_factory" for v in intents),
        episode_statuses=[{"ordinal": r["spec"]["ordinal"], "status": r["status"]} for r in records],
        unknown_calls=unknown,
        unknown_native_calls=sum(v["kind"] != "model_request" for v in unknown),
        call_intents=dict(Counter(v["kind"] for v in intents)),
        call_returns=dict(Counter(v["kind"] for v in intents if v["call_id"] in returned)),
        measured_native_dispatches=sum(
            v["kind"] == "native_step" and v.get("segment") == "measurement" for v in intents
        ),
        settling_native_dispatches=sum(
            v["kind"] == "native_step" and v.get("segment") == "settling" for v in intents
        ),
        budget=worker_result.get("budget"),
    )
    if stop_reason or unknown or exit_code != 0:
        result.update(
            status="technical_failure",
            recovery_native_four_episode_contract_passed=False,
            recovery_native_two_pair_comparison_complete=False,
        )
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "episodes_completed", *FIELDS, "unknown_native_calls")}
        ),
        flush=True,
    )
    return 0 if result["recovery_native_four_episode_contract_passed"] else 2


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
