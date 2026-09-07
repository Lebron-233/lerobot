"""L14 physical simulation with exact seven-step age, NOT wall-clock asynchronous inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch
from collect_so101_predictor_pilot import prepared_observation
from eval_leisaac_so101 import EnvClient, decode_observation
from leisaac_so101_contract import SCALAR_KEYS, action_to_radians, validate_step
from leisaac_so101_matched import TASK, candidate_manifest, load_matched_runtime
from leisaac_so101_predicted import load_so101_l6_predictor
from so101_feasible_actions import FeasibleActionProjector
from so101_joint_runtime import JointContextPredictor, load_future_state
from so101_startup_preparation import warm_environment
from so101_task_evidence import PlacementTracker

from lerobot.policies.rtc.scheduled_action_queue import (
    GetOutcome,
    InstallOutcome,
    PlanOutcome,
    ScheduledActionQueue,
    StageOutcome,
)
from lerobot.utils.random_utils import set_seed


def create_plan(queue: ScheduledActionQueue, request_id: int):
    result = queue.create_takeover_plan(
        request_id=request_id,
        planned_delay_steps=7,
        max_prediction_delay=8,
        committed_guard_steps=2,
        reset_epoch=0,
        task_epoch=0,
        task=TASK,
    )
    if result.outcome != PlanOutcome.CREATED:
        raise RuntimeError(f"Controlled delay could not commit its prefix: {result.outcome}")
    return result.plan


def install_or_stage(queue, normalized, physical, plan, request_id):
    if plan is None:
        result = queue.install_active_chunk(normalized, physical, task=TASK, reset_epoch=0, task_epoch=0)
        if result.outcome != InstallOutcome.INSTALLED:
            raise RuntimeError(f"Unexpected initial/synchronous chunk install: {result.outcome}")
    else:
        result = queue.stage_chunk(
            normalized, physical, request_id=request_id, reset_epoch=0, task_epoch=0, task=TASK
        )
        if result.outcome != StageOutcome.STAGED_EARLY:
            raise RuntimeError(f"The simulator advanced during controlled inference: {result.outcome}")


def run_episode(
    client,
    packet,
    policy,
    pre,
    post,
    visual,
    state_model,
    projector,
    arm,
    seed,
    max_steps,
    ticks,
    events,
    frames,
    stop_subgoal,
    recorder=None,
):
    queue = ScheduledActionQueue(reset_epoch=0, task_epoch=0)
    context = (
        JointContextPredictor(visual, state_model, arm).eval() if arm in ("joint", "state_only") else None
    )
    tracker = PlacementTracker()
    request_id = 0
    for step in range(max_steps):
        started = time.perf_counter()
        if recorder is not None:
            recorder.observe(packet, step, policy, pre)
        raw = decode_observation(packet)
        if step % 150 == 0:
            frames.append((step, {name: raw[name] for name in ("top", "wrist")}))
        row = {
            "tick": step,
            "episode_id": packet["episode_id"],
            "sim_step": packet["step"],
            "sim_time_s": packet["sim_time_s"],
            "started_at_s": started,
            "state": [raw[k] for k in SCALAR_KEYS],
            "camera_frames": packet["camera_frames"],
            "task_diagnostics_before_action": packet.get("task_diagnostics"),
            "dispatch": "not_sent",
            "mode": arm,
        }
        if "camera_world_poses_opengl" in packet:
            row["camera_world_poses_opengl"] = packet["camera_world_poses_opengl"]
        ticks.append(row)
        need_chunk = queue.qsize() == 0 or (
            arm != "sync" and queue.qsize() <= 30 and queue.plan_snapshot() is None
        )
        if need_chunk:
            plan = None if queue.qsize() == 0 else create_plan(queue, request_id)
            event = {
                "event": "controlled_chunk_request",
                "request_id": request_id,
                "request_step": step,
                "takeover_index": None if plan is None else plan.takeover_index,
                "simulated_delay_steps": 0 if plan is None else 7,
                "noise_seed": seed * 100000 + step,
                "state_predictor_calls": 0,
                "visual_predictor_calls": 0,
                "outcome": "started",
                "physics_paused_during_inference": True,
            }
            events.append(event)
            if plan is not None:
                event["committed_normalized_prefix"] = plan.committed_policy_actions[:7].tolist()
            batch, tokens, masks, state = prepared_observation(packet, policy, pre)
            kwargs = {"future_image_tokens": tokens, "future_image_token_masks": masks}
            with torch.inference_mode():
                if plan is not None and arm in ("visual_only", "state_only", "joint"):
                    prefix = plan.committed_policy_actions[None].to("cuda:0")
                    valid = plan.committed_mask[None].to("cuda:0")
                    delay = torch.tensor([7], device="cuda:0")
                    predictor = context if context is not None else visual
                    prediction = predictor(tokens, masks, prefix, valid, state, delay)
                    kwargs["future_image_tokens"] = tuple(
                        (z.float() + delta.float()).to(z.dtype)
                        for z, delta in zip(tokens, prediction.delta_tokens, strict=True)
                    )
                    event["state_predictor_calls"] = int(context is not None)
                    event["visual_predictor_calls"] = int(arm != "state_only")
                    if context is not None:
                        kwargs["future_state"] = prediction.future_state
                generator = torch.Generator(device="cuda:0").manual_seed(event["noise_seed"])
                noise = torch.randn((1, 50, 32), generator=generator, device="cuda:0")
                actions = projector(policy.predict_action_chunk(batch, noise=noise, **kwargs))
                normalized = actions.squeeze(0).detach().cpu()
                physical = post(actions).squeeze(0).detach().cpu()
            if recorder is not None and plan is not None:
                recorder.request(step, plan, batch, tokens, masks, state, kwargs.get("future_state"))
            install_or_stage(queue, normalized, physical, plan, request_id)
            event.update(
                outcome="installed" if plan is None else "staged_for_future_index",
                wall_s=time.perf_counter() - started,
            )
            request_id += 1
        get = queue.get_with_task()
        if get.outcome == GetOutcome.UNDERFLOW or get.action_index != step:
            raise RuntimeError("Controlled-delay queue underflowed or skipped an action")
        row.update(
            action=get.post_policy_action.tolist(),
            normalized_action=get.policy_action.tolist(),
            queue_outcome=get.outcome.value,
            queue_action_index=get.action_index,
        )
        action_to_radians(row["action"])
        row["dispatch"] = "sent_result_unknown"
        response = client.step(packet, row["action"])
        row.update(
            dispatch="completed",
            terminated=bool(response["terminated"]),
            truncated=bool(response["truncated"]),
            task_transition_after_action=response.get("task_transition"),
        )
        terminal = row["terminated"] or row["truncated"]
        if bool(response["post_reset_observation"]) != terminal:
            raise RuntimeError("Automatic-reset provenance changed")
        validate_step(packet, response["observation"], terminal=terminal)
        witness = row["task_transition_after_action"]
        if witness is None or witness["native_success"] != row["terminated"]:
            raise RuntimeError("The pre-reset task witness is missing or inconsistent")
        tracker.update(witness, step)
        row["finished_at_s"] = time.perf_counter()
        row["work_s"] = row["finished_at_s"] - started
        if terminal:
            return {
                "status": "terminal",
                "steps": step + 1,
                "success": row["terminated"],
                "timeout": row["truncated"],
            }
        if stop_subgoal and tracker.first_settled_step is not None:
            return {"status": "task_subgoal_reached", "steps": step + 1, "success": None, "timeout": False}
        packet = response["observation"]
    return {"status": "censored_step_limit", "steps": max_steps, "success": None, "timeout": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("sync", "identity", "visual_only", "state_only", "joint"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke and (
        args.arm != "joint" or args.seed != 20270501 or args.policy_seed != 3401 or args.max_steps != 100
    ):
        parser.error("Only the registered hundred-step joint wiring smoke is supported")
    if not args.smoke and (
        args.max_steps != 3600
        or not 20270510 <= args.seed <= 20270517
        or args.policy_seed != args.seed - 20267100
    ):
        parser.error("Use the frozen L14 task cohort and full simulated-time bound")
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the controlled-delay protocol before execution")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "execution": "L14_controlled_simulated_delay_NOT_realtime",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "candidate": candidate_manifest(snapshot),
        "visual_predictor": "L6_epoch3",
        "state_predictor": "L12_epoch29",
        "realtime_required": False,
        "simulated_delay_steps": 0 if args.arm == "sync" else 7,
        "risk_thresholds": None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    client = EnvClient(
        base / "leisaac-sim-venv/bin/python",
        base
        / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets",
        base / "leisaac-source",
        "cpu",
    )
    client.camera_backend, client.episode_seconds, client.task_evidence = "standard", 120, True
    ticks, events, setup, frames = [], [], [], []
    result = {}
    started = time.perf_counter()
    try:
        policy, pre, post = load_matched_runtime(snapshot, device="cuda:0")
        visual = load_so101_l6_predictor(
            base / "artifacts/m54l6_predictor_training_v1/best.pt", device="cuda:0"
        )
        state_model = load_future_state(
            base / "artifacts/m54l12_state_development_v1/state_best.pt", "cuda:0"
        )
        projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
        client.start()
        packet = client.reset(args.seed)
        warm_environment(client, packet, setup)
        packet = client.reset(args.seed)
        policy.reset()
        pre.reset()
        post.reset()
        set_seed(args.policy_seed)
        result = run_episode(
            client,
            packet,
            policy,
            pre,
            post,
            visual,
            state_model,
            projector,
            args.arm,
            args.policy_seed,
            args.max_steps,
            ticks,
            events,
            frames,
            not args.smoke,
        )
        result["projection_records"] = projector.records
    except Exception:
        result = {"status": "technical_failure", "success": None, "error": traceback.format_exc()}
    finally:
        client.close()
        if client.cleanup_error:
            result.update(status="technical_failure", cleanup_error=client.cleanup_error)
        result.update(
            source_commit=source,
            mode=args.arm,
            execution=manifest["execution"],
            realtime_required=False,
            ticks=len(ticks),
            environment=client.metadata,
            subprocess_returncode=None if client.process is None else client.process.returncode,
            metrics_closed=True,
            setup_actions=sum(r["dispatch"] == "completed" for r in setup),
            wall_s=time.perf_counter() - started,
            task_contract="pickorange_first_settled_v1",
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        for name, rows in (("ticks", ticks), ("events", events), ("setup_ticks", setup)):
            (args.output / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
        if frames:
            from PIL import Image

            folder = args.output / "observations"
            folder.mkdir()
            for index, cameras in frames:
                for name, image in cameras.items():
                    Image.fromarray(image).save(folder / f"step_{index:05d}_{name}.png")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("projection_records", "environment")}, indent=2
        )
    )
    return int(result["status"] == "technical_failure")


if __name__ == "__main__":
    raise SystemExit(main())
