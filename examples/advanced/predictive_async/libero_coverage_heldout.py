"""F-COV2: one independent evaluation of all four frozen F-COV1 predictors."""

import argparse
import copy
import faulthandler
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import libero_coverage_study as cov
import torch

e, pilot, natural, opt = cov.e, cov.pilot, cov.natural, cov.opt
PREP = e.REPO / "outputs/smolvla_coverage_heldout_preparation_23d3932a"
CHECKPOINTS = e.REPO / "outputs/smolvla_coverage_6249b03d"
ARMS = cov.ARMS
NATIVE_LIMITS = {
    "episodes": (1, 4),
    "settling": (10, 40),
    "measurement": (280, 1120),
    "model": (160, 640),
    "capture": (2, 8),
}
OFFLINE_LIMITS = {"encoding": 20, "decoder": 96, "predictor": 64}


def manifest():
    original = e.fixed_manifest()
    rows = []
    for task, state in ((8, 48), (9, 48), (9, 49), (8, 49)):
        row = copy.deepcopy(next(r for r in original["rows"] if r["task_id"] == task))
        row.update(
            ordinal=len(rows),
            pair_index=len(rows),
            initial_state_id=state,
            environment_seed=1020000 + 100 * task + state,
            policy_seed=1030000 + 100 * task + state,
            condition="graph_identity_async",
            split="heldout",
            recovery_policy=natural.recovery.RECOVERY_POLICY,
            max_recovery_probes_per_episode=50,
        )
        rows.append(row)
    return {
        "experiment": "F-COV2",
        "rows": rows,
        "arms": list(ARMS),
        "checkpoint_source": str(CHECKPOINTS),
        "best_step": 72,
        "max_pairs_per_episode": 4,
        "native_limits": NATIVE_LIMITS,
        "offline_limits": OFFLINE_LIMITS,
        "training_updates": 0,
        **{k: original[k] for k in ("policy_revision", "vlm_revision", "assets_revision")},
    }


class Budget(e.Budget):
    def check(self, kind):
        local, total = NATIVE_LIMITS[kind]
        if self.episode[kind] >= local or self.total[kind] >= total:
            raise RuntimeError(f"F-COV2 {kind} limit exhausted before dispatch")


class Runtime(pilot.SmolVLAGraphRuntime):
    def _capture(self, inputs):
        if len(self.captures) >= 2:
            raise RuntimeError("F-COV2 allows at most two offline Graph captures")
        return super()._capture(inputs)


def take(counts, key):
    if counts[key] >= OFFLINE_LIMITS[key]:
        raise RuntimeError(f"F-COV2 {key} limit exhausted before dispatch")
    counts[key] += 1


def validate_checkpoint(checkpoint, arm):
    if (
        checkpoint["arm"] != arm
        or checkpoint["best_step"] != 72
        or checkpoint["seed"] != 20260912
        or checkpoint["config"] != asdict(pilot.config())
        or checkpoint["nonzero_residual"] is not True
        or not any(
            checkpoint["state_dict"][k].count_nonzero()
            for k in ("up_projection.weight", "up_projection.bias")
        )
    ):
        raise ValueError("Frozen F-COV1 checkpoint identity differs")


def load_predictors():
    models, frozen = {}, {}
    for arm in ARMS:
        checkpoint = torch.load(CHECKPOINTS / f"{arm}.pt", map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, arm)
        model = pilot.LightweightFutureLatentPredictor(pilot.config())
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if sum(p.numel() for p in model.parameters()) != 69680:
            raise ValueError("Frozen predictor parameter count differs")
        models[arm] = model.eval().requires_grad_(False).cuda()
        frozen[arm] = checkpoint
    return models, frozen


def aggregate(rows):
    complete = {r["ordinal"] for r in rows} == set(range(4))
    metrics = {}
    for arm in ("identity", *ARMS):
        values = [{"task": r["task"], "state": r["state"], **r[arm]} for r in rows]
        metrics[arm] = cov.summarize_rows(values) if values else {}

    def better(other, metric, strict=True):
        if not complete:
            return False
        left = metrics["multi_conditioned"][metric]["episode_macro"]
        right = metrics[other][metric]["episode_macro"]
        return left < right if strict else left <= right

    return {
        "metrics": metrics,
        "four_episodes_have_samples": complete,
        "heldout_primary_gate_passed": all(
            better(a, "row0") for a in ("identity", "single_conditioned", "multi_no_action")
        )
        and better("identity", "chunk", strict=False),
        "strongest_no_action_row0_better": better("single_no_action", "row0"),
        "strongest_no_action_chunk_better": better("single_no_action", "chunk"),
        "heldout_all_comparators_better": all(
            better(a, m)
            for a in ("identity", "single_conditioned", "multi_no_action", "single_no_action")
            for m in ("row0", "chunk")
        ),
    }


def extract(records, policy, pre, output, counts):
    samples, selections = [], []
    for record in records:
        spec = record["spec"]
        arrays = torch.load(
            output / f"episode_{spec['ordinal']:03d}/arrays.pt", map_location="cpu", weights_only=False
        )
        pairs, excluded = cov.selected_pairs(record, arrays)
        selections.append(
            {"ordinal": spec["ordinal"], "requests": [p["request_id"] for p in pairs], "excluded": excluded}
        )
        for number, pair in enumerate(pairs):
            inputs = tuple(v.detach().cpu().clone() for v in pair["cached"]["inputs"])
            for kind in ("current", "future") if number == 0 else ("future",):
                take(counts, "encoding")
                with (
                    pilot.phase(output, f"encode_{spec['ordinal']}_{pair['request_id']}_{kind}", 15),
                    torch.no_grad(),
                ):
                    pre.reset()
                    batch = pre(
                        pilot.worker_batch(
                            pair[f"{kind}_observation"], spec["task_name"].replace("_", " "), "cuda"
                        )
                    )
                    images, masks = policy.prepare_images(batch)
                    z, zm = policy.model.encode_image_tokens(images, masks)
                    z = tuple(v.detach().cpu().clone() for v in z)
                    state = policy.prepare_state(batch).detach().cpu().clone()
                    for c in range(2):
                        pilot.require_equal(zm[c], inputs[c + 2], "Camera mask differs")
                        if kind == "current":
                            pilot.require_equal(z[c], inputs[c], "Current token differs from native archive")
                    if kind == "current":
                        pilot.require_equal(state, inputs[6], "Current state differs")
                        counts["current_exact"] += 1
            samples.append(
                {
                    "task": spec["task_id"],
                    "ordinal": spec["ordinal"],
                    "initial_state_id": spec["initial_state_id"],
                    **{
                        k: pair[k]
                        for k in ("request_id", "current_index", "future_index", "delay", "actions", "mask")
                    },
                    "inputs": inputs,
                    "future": z,
                    "archived_full_chunk": pair["cached"]["full_chunk"].clone(),
                }
            )
        del arrays, pairs
    samples.sort(key=lambda s: (s["task"], s["ordinal"], s["request_id"]))
    torch.save(samples, output / "aligned_cache.pt")
    e.write_json(output / "selections.json", selections)
    return samples


def evaluate(samples, models, policy, output, counts):
    rows, archives = [], []
    with torch.no_grad(), Runtime(policy.model) as runtime:
        for sample in samples:
            visual = {"identity": sample["inputs"][:2]}
            for arm in ARMS:
                take(counts, "predictor")
                visual[arm] = cov.parent.predict(models[arm], sample, arm)[0]
            visual["oracle"] = sample["future"]
            values = tuple(v.cuda() for v in sample["inputs"])
            runtime.begin_episode("graph", sample["task"])
            outputs = {}
            for arm, z in visual.items():
                take(counts, "decoder")
                with pilot.phase(output, f"decode_{sample['ordinal']}_{sample['request_id']}_{arm}", 15):
                    decoded = runtime(
                        None,
                        None,
                        values[4],
                        values[5],
                        values[6],
                        noise=values[7],
                        future_image_tokens=tuple(
                            v.to(device="cuda", dtype=values[c].dtype) for c, v in enumerate(z)
                        ),
                        future_image_token_masks=tuple(values[2:4]),
                    )
                    outputs[arm] = decoded.detach().cpu().clone()
                if arm == "identity":
                    pilot.require_equal(
                        outputs[arm], sample["archived_full_chunk"], "Heldout identity full chunk differs"
                    )
                    counts["identity_exact"] += 1
            row = {
                "ordinal": sample["ordinal"],
                "task": sample["task"],
                "state": sample["initial_state_id"],
                "request_id": sample["request_id"],
            }
            visual_cpu = {a: tuple(v.detach().cpu().float().clone() for v in z) for a, z in visual.items()}
            for arm in ("identity", *ARMS):
                diff = (outputs[arm][..., :7].float() - outputs["oracle"][..., :7].float()).square()
                row[arm] = {
                    "latent": pilot.per_sample_mse(
                        visual_cpu[arm], sample["future"], sample["inputs"][2:4]
                    ).item(),
                    "row0": diff[:, 0].mean().item(),
                    "chunk": diff.mean().item(),
                }
            rows.append(row)
            archives.append(
                {
                    "ordinal": sample["ordinal"],
                    "request_id": sample["request_id"],
                    "visual": visual_cpu,
                    "outputs": outputs,
                }
            )
        e.write_json(output / "offline_captures.json", runtime.captures)
        counts["offline_captures"] = len(runtime.captures)
    torch.save(archives, output / "evaluation_arrays.pt")
    return rows, aggregate(rows)


def worker(args):
    opt.source_gate(args.execution_head)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    torch.set_num_threads(1)
    counts, budget, records = Counter(), Budget(), []
    result = {"experiment": "F-COV2", "status": "technical_failure", "first_failure": None}
    calls = e.Calls(args.output / "calls.jsonl")
    try:
        with pilot.phase(args.output, "load", 60):
            if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4070 Ti SUPER":
                raise ValueError("Registered GPU differs")
            models, frozen = load_predictors()
            torch.save(frozen, args.output / "frozen_checkpoints_before.pt")
            policy, pre, post, report = e.load_runtime(e.POLICY, e.VLM)
            e.write_json(args.output / "policy_load.json", report)
        factory = natural.trace.checkpoint_factory(e.reference.make_native_env_factory(), calls)
        with pilot.phase(args.output, "native_collection", 480):
            for spec in manifest()["rows"]:
                print(
                    f"START F-COV2 {spec['ordinal']} task={spec['task_id']} state={spec['initial_state_id']}",
                    flush=True,
                )
                record, _ = e.run_episode(
                    spec,
                    args.output / f"episode_{spec['ordinal']:03d}",
                    policy,
                    pre,
                    post,
                    factory,
                    budget,
                    calls,
                    engine_class=natural.NaturalEngine,
                )
                records.append(record)
                print(f"END F-COV2 {spec['ordinal']} {record['status']}", flush=True)
                if record["status"] != "completed":
                    raise RuntimeError(record["first_failure"] or "Native collection did not complete")
        with pilot.phase(args.output, "offline_evaluation", 180):
            samples = extract(records, policy, pre, args.output, counts)
            rows, summary = evaluate(samples, models, policy, args.output, counts)
        after = {a: opt.freeze_state(m) for a, m in models.items()}
        for arm in ARMS:
            for key, value in after[arm].items():
                pilot.require_equal(value, frozen[arm]["state_dict"][key], "Frozen predictor changed")
        torch.save(after, args.output / "frozen_weights_after.pt")
        result.update(
            status="completed",
            samples=len(samples),
            metric_rows=rows,
            frozen_weights_unchanged=True,
            **summary,
        )
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        calls.close()
        result.update(
            execution_head=args.execution_head,
            counts=dict(counts),
            native_budget=dict(budget.total),
            episodes_completed=sum(r["status"] == "completed" for r in records),
            predictor_controls_environment=False,
            training_updates=0,
            real_robot=0,
            baseline_qualified=False,
            realtime_qualified=False,
            predictor_benefit_tested=False,
            risk_thresholds=None,
            old_confirmation="not_started_untouched",
            attempts=1,
            retries=0,
        )
        e.write_json(args.output / "worker_result.json", result)
    return 0 if result["status"] == "completed" else 2


def supervise(args):
    opt.source_gate(args.execution_head)
    expected = e.REPO / "outputs" / f"smolvla_coverage_heldout_{args.execution_head[:8]}"
    if str(Path(sys.executable)) != e.PYTHON or args.output != expected or args.output.exists():
        raise ValueError("Use registered Python and unused output")
    registration = json.loads((PREP / "registration_readback.json").read_text())
    if (
        registration["body"] != (PREP / "registration.md").read_text()
        or args.execution_head not in registration["body"]
        or str(args.output) not in registration["body"]
    ):
        raise ValueError("Registration readback differs")
    if not all(json.loads((PREP / "gates.json").read_text()).values()):
        raise ValueError("Preparation failed")
    args.output.mkdir()
    e.write_json(args.output / "manifest.json", manifest())
    command = [
        sys.executable,
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
    start, utc = time.perf_counter(), datetime.now(UTC).isoformat()
    cursors, pending, active = {}, {}, {}
    reason, terminated, forced = None, None, False
    with (args.output / "model.log").open("x") as log:
        child = subprocess.Popen(
            command, cwd=e.REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"F-COV2 worker pid={child.pid}", flush=True)
        try:
            while child.poll() is None:
                cov.parent.previous.update_pending(args.output, cursors, pending, active)
                now = time.perf_counter()
                expired = [v for v in pending.values() if now - v["timestamp"] > v["limit"]]
                phases = [v for v in active.values() if now - v["at"] > v["limit"]]
                if terminated is None and (expired or phases or now - start >= 870):
                    reason = {
                        "expired_calls": expired,
                        "expired_phases": phases,
                        "outer_soft": now - start >= 870,
                    }
                    os.kill(child.pid, signal.SIGUSR1)
                    os.killpg(child.pid, signal.SIGTERM)
                    terminated = now
                if now - start >= 900 or (terminated is not None and now - terminated >= 5):
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
    cov.parent.previous.update_pending(args.output, cursors, pending, active)
    path = args.output / "worker_result.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "technical_failure", "first_failure": "Worker receipt absent"}
    )
    result["execution"] = {
        "command": command,
        "child_pid": child.pid,
        "child_exit_code": code,
        "exit_confirmed": True,
        "started_at_utc": utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - start,
        "stop_reason": reason,
        "forced_termination": forced,
        "pending_calls_at_exit": list(pending.values()),
        "active_phases_at_exit": active,
    }
    accounting = natural.trace.journal_accounting(natural.trace.read_events(args.output / "calls.jsonl"))
    result["accounting"] = accounting
    if (
        code
        or reason
        or pending
        or active
        or accounting["call_errors"]
        or accounting["unknown_calls"]
        or not accounting["journal_consistent"]
    ):
        result.update(status="technical_failure", heldout_primary_gate_passed=False)
        result["first_failure"] = result.get("first_failure") or result["execution"]
    result["environment_after"] = {
        "python": sys.executable,
        "version": sys.version,
        "packages": sorted([[d.metadata["Name"], d.version] for d in importlib.metadata.distributions()]),
    }
    e.write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "episodes_completed",
                    "samples",
                    "heldout_primary_gate_passed",
                    "first_failure",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "completed" else 2


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
