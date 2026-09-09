"""E-RCV2: fixed native disabled/probe comparison with one declared host pause."""

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

import libero_graph_cap_recovery_native as recovery
import libero_graph_identity_native as e
import torch

REPO = e.REPO
PREPARATION = REPO / "outputs/smolvla_graph_cap_stress_preparation_5e2679a9"
MANIFEST = REPO / "docs/experiments/SMOLVLA_GRAPH_CAP_STRESS_MANIFEST.json"
pause = time.sleep
CLEANUP = (
    "worker_joined",
    "graph_released",
    "original_sampler_restored",
    "metrics_closed",
    "environment_closed",
)


def declaration():
    return {
        "experiment": "E-RCV2",
        "handover_head": "5e2679a95adefa0de57e10b05506d20f35c7cac2",
        "source_ordinals": [1, 1, 5, 5],
        "arms": ["disabled", "candidate", "candidate", "disabled"],
        "condition": "graph_identity_async",
        "intervention": "one_host_pause_after_first_planned_cpu_completion_before_admission",
        "host_pause_seconds": 0.6,
        "max_recovery_probes_per_episode": 50,
        "max_recovery_probes_total": 100,
        "attempts": 1,
        "retries": 0,
    }


def read_manifest():
    declared = json.loads(MANIFEST.read_text())
    if declared != declaration():
        raise ValueError("The E-RCV2 fixed intervention/arms changed")
    original = e.fixed_manifest()
    rows = []
    for ordinal, (source, arm) in enumerate(zip(declared["source_ordinals"], declared["arms"], strict=True)):
        row = copy.deepcopy(original["rows"][source])
        row.update(
            ordinal=ordinal,
            source_ordinal=source,
            pair_index=ordinal // 2,
            arm=arm,
            recovery_policy=recovery.RECOVERY_POLICY if arm == "candidate" else "disabled",
            max_recovery_probes_per_episode=50,
            host_pause_seconds=0.6,
        )
        rows.append(row)
    return {
        **declared,
        "rows": rows,
        "policy_revision": original["policy_revision"],
        "vlm_revision": original["vlm_revision"],
        "assets_revision": original["assets_revision"],
        "chunk_size": 50,
        "n_action_steps": 1,
        "num_steps": 10,
        "limits": {k: {"per_episode": a, "total": b} for k, (a, b) in recovery.LIMITS.items()},
        "soft_stop_seconds": 870,
        "hard_stop_seconds": 900,
        "reference_calls": 0,
        "predictor_calls": 0,
        "training_calls": 0,
        "real_robot_calls": 0,
    }


class StressEngine(recovery.RecoveryEngine):
    """A diagnostic-only host scheduling pause; no synthetic tracker samples or GPU work."""

    def __init__(self, *args, **kwargs):
        self.pause_count = 0
        super().__init__(*args, **kwargs)

    def _prepare_queue_actions(self, actions, metrics):
        chunks = super()._prepare_queue_actions(actions, metrics)
        if self.current["kind"] == "planned" and self.pause_count == 0:
            self.pause_count += 1
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
        return chunks


def evidence(record):
    result = recovery.recovery_evidence(record)
    requests = record.get("requests", [])
    injected = [r for r in requests if "host_intervention" in r]
    first = injected[0] if len(injected) == 1 else None
    after = [] if first is None else [r for r in requests if r["request_id"] > first["request_id"]]
    history = {} if first is None else first.get("recovery", {}).get("history_after", {})
    terminal = {} if first is None else first.get("terminal", {})
    valid_pause = bool(
        first is not None
        and first["kind"] == "planned"
        and first["startup_phase"] is None
        and first["model_cpu_completed_at"] <= first["host_intervention"]["started_at"]
        and first["host_intervention"]["elapsed_seconds"] >= 0.6 - 1e-9
        and terminal.get("latency_tracker_admitted") is True
        and history.get("raw_required_steps", 0) > 8
    )
    fast_boots = [
        r
        for r in after
        if r["kind"] == "bootstrap"
        and r.get("terminal", {}).get("outcome") == "installed"
        and r["terminal"].get("latency_tracker_admitted") is False
        and recovery.latency_to_steps(r["terminal"]["total_chunk_s"], 20) + 1 <= 8
    ]
    frozen = bool(after) and all(
        r.get("recovery", {}).get("history_after", {}).get("samples_seconds")
        == history.get("samples_seconds")
        for r in after
    )
    by_id = {r["request_id"]: r for r in requests}
    chains = []
    for chain in result["native_recovery_chains"]:
        if not valid_pause or (chain["reset_epoch"], chain["task_epoch"]) != (
            first["reset_epoch"],
            first["task_epoch"],
        ):
            continue
        probe = by_id[chain["probe_request_id"]]
        planned = by_id[chain["planned_request_id"]]
        if probe["requested_at"] < first["finished_at"]:
            continue
        step = next(
            s
            for s in record["native_steps"]
            if s["segment"] == "measurement" and s["number"] == chain["native_step_number"]
        )
        # The original source audit already matched this takeover's command to
        # this uniquely staged request. Preserve that identity explicitly.
        chains.append(
            {
                **chain,
                "paused_request_id": first["request_id"],
                "paused_history_before": first["recovery"]["history_before"],
                "paused_history_after": history,
                "source_request_id": planned["request_id"],
                "source_row_offset": step["action_index"] - planned["terminal"]["takeover_index"],
                "native_started_at": step["started_at"],
            }
        )
    return {
        **result,
        "native_recovery_chains": chains,
        "planned_takeover_after_recovery": bool(chains),
        "ordinal": record["spec"]["ordinal"],
        "arm": record["spec"]["arm"],
        "valid_host_pause_with_cap_exceedance": valid_pause,
        "injected_request_id": None if first is None else first["request_id"],
        "host_intervention": None if first is None else first["host_intervention"],
        "post_intervention_history": history,
        "post_intervention_planned_requests": sum(r["kind"] == "planned" for r in after),
        "fast_excluded_bootstraps_after_intervention": len(fast_boots),
        "fast_excluded_bootstrap_evidence": [
            {
                "request_id": r["request_id"],
                "reset_epoch": r["reset_epoch"],
                "task_epoch": r["task_epoch"],
                "total_chunk_s": r["terminal"]["total_chunk_s"],
                "raw_required_steps": recovery.latency_to_steps(r["terminal"]["total_chunk_s"], 20) + 1,
                "latency_tracker_admitted": r["terminal"]["latency_tracker_admitted"],
                "history_after": r["recovery"]["history_after"],
            }
            for r in fast_boots
        ],
        "disabled_latch_observed": bool(
            record["spec"]["arm"] == "disabled"
            and valid_pause
            and frozen
            and fast_boots
            and not any(r["kind"] in ("planned", "recovery_probe") for r in after)
        ),
    }


def summarize(records, manifest):
    completed = [r for r in records if r["status"] == "completed" and all(r.get(k) is True for k in CLEANUP)]
    analyses = [evidence(r) for r in records if r.get("requests")]
    pairs = []
    for pair in range(2):
        rows = [r for r in completed if r["spec"]["pair_index"] == pair]
        if len(rows) != 2:
            continue
        arms = {
            r["spec"]["arm"]: {
                "ordinal": r["spec"]["ordinal"],
                "success": r["success"],
                "terminal_reason": r["terminal_reason"],
                "measured_actions": r["native_returned"]["measurement"],
                "wall_seconds": r["wall"]["measured_wall_seconds"],
                "planned_takeovers": r["audit"]["planned_takeovers"],
                "main_calls": r["budget"]["model"],
                "no_action_slots": r["wall"]["no_action_slots"],
            }
            for r in rows
        }
        if set(arms) != {"disabled", "candidate"}:
            raise ValueError("A stress pair must contain exactly one of each arm")
        pairs.append(
            {
                "pair_index": pair,
                "task_id": rows[0]["spec"]["task_id"],
                "initial_exact": any(r["initial_pair_exact"] is True for r in rows),
                "arms": arms,
            }
        )
    complete = len(completed) == 4 and len(pairs) == 2 and all(p["initial_exact"] for p in pairs)
    disabled = [r for r in analyses if r["arm"] == "disabled"]
    candidates = [r for r in analyses if r["arm"] == "candidate"]
    return {
        "experiment": "E-RCV2",
        "status": "completed" if complete else "technical_failure",
        "manifest": manifest,
        "episodes_completed": len(completed),
        "pairs": pairs,
        "stress_episodes": analyses,
        "first_failure": next((r["first_failure"] for r in records if r.get("first_failure")), None),
        "stress_four_episode_contract_passed": complete,
        "stress_disabled_latch_both_observed": len(disabled) == 2
        and all(r["disabled_latch_observed"] for r in disabled),
        "stress_candidate_recovery_both_observed": len(candidates) == 2
        and all(
            r["valid_host_pause_with_cap_exceedance"] and r["planned_takeover_after_recovery"]
            for r in candidates
        ),
        "natural_latency_recovery_demonstrated": False,
        "production_default_unchanged": True,
        "native_closed_loop_contract_passed": False,
        "paired_scheduling_comparison_complete": False,
        **e.FLAGS,
    }


def worker(args):
    e.require_source(args.execution_head)
    manifest = read_manifest()
    budget = recovery.RecoveryBudget()
    calls = e.Calls(args.output / "calls.jsonl")
    records = []
    result = {}
    try:
        torch.set_num_threads(1)
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
            raise ValueError("The registered GPU changed")
        started = time.perf_counter()
        policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
        result.update(model_load_seconds=time.perf_counter() - started, policy_load=report)
        factory = e.reference.make_native_env_factory()
        paired_initial = None
        for spec in manifest["rows"]:
            print(f"START E-RCV2 {spec['ordinal']} task={spec['task_id']} arm={spec['arm']}", flush=True)
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
                engine_class=StressEngine,
            )
            records.append(record)
            print(
                f"END E-RCV2 {spec['ordinal']} {record['status']} {json.dumps(evidence(record))}", flush=True
            )
            if record["status"] != "completed":
                break
            paired_initial = initial if spec["ordinal"] % 2 == 0 else None
        result.update(summarize(records, manifest))
    except BaseException:
        result.update(status="technical_failure", first_failure=traceback.format_exc())
    finally:
        result.update(budget=dict(budget.total), calls_logging_seconds=calls.logging_seconds)
        calls.close()
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result.get("stress_four_episode_contract_passed") else 2


def supervise(args):
    e.require_source(args.execution_head)
    manifest = read_manifest()
    expected = REPO / "outputs" / f"smolvla_graph_cap_stress_{args.execution_head[:8]}"
    if Path(sys.executable) != Path(e.PYTHON) or args.output != expected or args.output.exists():
        raise ValueError("Use the fixed model Python and exclusive E-RCV2 output")
    gates = json.loads((PREPARATION / "preparation_gates.json").read_text())
    if not all(
        gates.get(k) is True for k in ("cpu_contract_passed", "model_entry_passed", "lint_format_passed")
    ):
        raise ValueError("E-RCV2 preparation gates failed")
    registration = json.loads((PREPARATION / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREPARATION / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("E-RCV2 registration/readback differs")
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
        print(f"E-RCV2 only queue pid={child.pid} head={args.execution_head}", flush=True)
        try:
            while child.poll() is None:
                now = time.perf_counter()
                path = args.output / "calls.jsonl"
                if path.exists():
                    with path.open() as journal:
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
                expired = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
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
    path = args.output / "worker_result.json"
    worker_result = json.loads(path.read_text()) if path.exists() else {}
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
        episode_statuses=[{"ordinal": r["spec"]["ordinal"], "status": r["status"]} for r in records],
        unknown_calls=unknown,
        budget=worker_result.get("budget"),
        call_intents=dict(Counter(v["kind"] for v in intents)),
        call_returns=dict(Counter(v["kind"] for v in intents if v["call_id"] in returned)),
        measured_native_dispatches=sum(
            v["kind"] == "native_step" and v.get("segment") == "measurement" for v in intents
        ),
        settling_native_dispatches=sum(
            v["kind"] == "native_step" and v.get("segment") == "settling" for v in intents
        ),
    )
    if stop_reason or unknown or exit_code != 0:
        result.update(status="technical_failure", stress_four_episode_contract_passed=False)
    result["stress_mechanism_contrast_passed"] = bool(
        result["stress_four_episode_contract_passed"]
        and result["stress_disabled_latch_both_observed"]
        and result["stress_candidate_recovery_both_observed"]
    )
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "episodes_completed", "stress_mechanism_contrast_passed", "first_failure")
            }
        ),
        flush=True,
    )
    return 0 if result["stress_four_episode_contract_passed"] else 2


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
