"""L16: explicit target-time information references, never realtime deployment."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from collect_so101_predictor_pilot import prepared_observation
from eval_leisaac_so101 import EnvClient, decode_observation
from eval_so101_controlled_delay import create_plan
from leisaac_so101_contract import SCALAR_KEYS, action_to_radians, validate_step
from leisaac_so101_matched import TASK, load_matched_runtime
from so101_feasible_actions import FeasibleActionProjector
from so101_joint_runtime import load_future_state
from so101_startup_preparation import warm_environment
from so101_task_evidence import PlacementTracker
from train_so101_action_distillation import forecast

from lerobot.policies.rtc.scheduled_action_queue import (
    GetOutcome,
    InstallOutcome,
    ScheduledActionQueue,
    StageOutcome,
)
from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor
from lerobot.utils.random_utils import set_seed


def geometry(packet: dict) -> dict:
    raw = decode_observation(packet)
    return {
        "state": [raw[key] for key in SCALAR_KEYS],
        "objects": copy.deepcopy(packet["task_diagnostics"]["object_positions_world"]),
        "cameras": copy.deepcopy(packet["camera_world_poses_opengl"]),
    }


def stage_at_target(queue, normalized, physical, plan):
    if queue.next_action_index != plan.takeover_index:
        raise ValueError("Privileged reference must generate at the exact target index, not a later tick")
    staged = queue.stage_chunk(
        normalized, physical, request_id=plan.request_id, reset_epoch=0, task_epoch=0, task=TASK
    )
    if staged.outcome != StageOutcome.STAGED_ON_TIME:
        raise ValueError("The target-time reference was not staged on time")


def decoder_context(arm: str, request: dict, future_tokens=None, future_masks=None, future_state=None):
    if arm in ("state_only", "student"):
        if future_tokens is not None or future_state is not None:
            raise ValueError("A causal controller cannot consume the actual successor")
        return request["tokens"], request["masks"], request["predicted_state"]
    if arm not in ("oracle_visual", "oracle_joint") or future_tokens is None or future_state is None:
        raise ValueError("The privileged reference requires an explicitly observed successor")
    state = future_state if arm == "oracle_joint" else request["predicted_state"]
    return future_tokens, future_masks, state


@torch.no_grad()
def generate(policy, post, projector, batch, tokens, masks, state, noise_seed, *, explicit_noise=None):
    if explicit_noise is None:
        generator = torch.Generator(device="cuda:0").manual_seed(noise_seed)
        noise = torch.randn((1, 50, 32), generator=generator, device="cuda:0")
    else:
        if explicit_noise.shape != (1, 50, 32) or not bool(torch.isfinite(explicit_noise).all()):
            raise ValueError("Explicit flow noise requires a finite 1x50x32 action chunk")
        noise = explicit_noise.clone()
    actions = projector(
        policy.predict_action_chunk(
            batch,
            noise=noise,
            future_image_tokens=tokens,
            future_image_token_masks=masks,
            future_state=state,
        )
    )
    return actions[0].detach().cpu(), post(actions)[0].detach().cpu()


def drive(
    client,
    packet,
    policy,
    pre,
    post,
    state_model,
    student,
    projector,
    arm,
    policy_seed,
    bootstrap,
    limit,
    stop_subgoal,
    ticks,
    events,
    *,
    noise_provider=None,
):
    queue = ScheduledActionQueue(reset_epoch=0, task_epoch=0)
    installed = queue.install_active_chunk(
        bootstrap["normalized"], bootstrap["physical"], task=TASK, reset_epoch=0, task_epoch=0
    )
    if installed.outcome != InstallOutcome.INSTALLED:
        raise ValueError("Shared bootstrap did not enter the original queue")
    tracker, pending, request_id = PlacementTracker(), None, 1
    for step in range(limit):
        raw = decode_observation(packet)
        if pending is not None and step == pending["plan"].takeover_index:
            _, tokens, masks, actual_state = prepared_observation(packet, policy, pre)
            z, m, s = decoder_context(arm, pending, tokens, masks, actual_state)
            normal, physical = generate(
                policy,
                post,
                projector,
                pending["batch"],
                z,
                m,
                s,
                pending["event"]["noise_seed"],
                explicit_noise=pending["noise"],
            )
            stage_at_target(queue, normal, physical, pending["plan"])
            pending["event"].update(
                outcome="staged_on_time",
                generated_step=step,
                visual_observation_step=step,
                state_observation_step=step if arm == "oracle_joint" else pending["event"]["request_step"],
            )
            pending = None
        if queue.qsize() <= 30 and queue.plan_snapshot() is None:
            plan = create_plan(queue, request_id)
            batch, tokens, masks, state = prepared_observation(packet, policy, pre)
            actions, valid = (
                plan.committed_policy_actions[None].to("cuda:0"),
                plan.committed_mask[None].to("cuda:0"),
            )
            delay = torch.tensor([7], device="cuda:0")
            with torch.no_grad():
                predicted_state = state_model(state, actions, valid, delay)
                if arm == "student":
                    query = {
                        "z": torch.stack(tokens, dim=1),
                        "masks": torch.stack(masks, dim=1),
                        "state": state,
                        "actions": actions,
                        "valid": valid,
                        "delay": delay,
                    }
                    predicted_tokens = forecast(student, query)
                    tokens = tuple(predicted_tokens[:, c] for c in range(2))
            event = {
                "request_id": request_id,
                "request_step": step,
                "takeover_index": step + 7,
                "noise_seed": policy_seed * 100000 + step,
                "committed_prefix": plan.committed_policy_actions[:7].tolist(),
                "arm": arm,
                "outcome": "pending",
                "generated_step": None,
                "visual_observation_step": None,
                "state_observation_step": step,
                "state_forecaster_calls": 1,
                "visual_forecaster_calls": int(arm == "student"),
                "privileged": arm.startswith("oracle"),
            }
            events.append(event)
            explicit_noise = None if noise_provider is None else noise_provider(step, plan.takeover_index)
            if noise_provider is not None:
                event.update(
                    noise_mode="absolute_action_index",
                    noise_window_start=plan.takeover_index,
                    noise_first_row=explicit_noise[0, 0].detach().cpu().tolist(),
                    noise_last_row=explicit_noise[0, -1].detach().cpu().tolist(),
                )
            # The noise is known at t, including in privileged visual arms.
            request = {
                "plan": plan,
                "batch": batch,
                "tokens": tokens,
                "masks": masks,
                "predicted_state": predicted_state,
                "event": event,
                "noise": explicit_noise,
            }
            if arm.startswith("oracle"):
                pending = request
            else:
                z, m, s = decoder_context(arm, request)
                normal, physical = generate(
                    policy,
                    post,
                    projector,
                    batch,
                    z,
                    m,
                    s,
                    event["noise_seed"],
                    explicit_noise=explicit_noise,
                )
                staged = queue.stage_chunk(
                    normal, physical, request_id=request_id, reset_epoch=0, task_epoch=0, task=TASK
                )
                if staged.outcome != StageOutcome.STAGED_EARLY:
                    raise ValueError("Causal controller advanced physics during its inference")
                event.update(outcome="staged_early", generated_step=step, visual_observation_step=step)
            request_id += 1
        get = queue.get_with_task()
        if get.outcome == GetOutcome.UNDERFLOW or get.action_index != step:
            raise ValueError("Privileged study cannot skip queue indices or hide underflow")
        row = {
            "tick": step,
            "sim_step": packet["step"],
            "state": [raw[k] for k in SCALAR_KEYS],
            "objects_before": packet["task_diagnostics"]["object_positions_world"],
            "camera_frames": packet["camera_frames"],
            "action": get.post_policy_action.tolist(),
            "normalized_action": get.policy_action.tolist(),
            "queue_outcome": get.outcome.value,
            "dispatch": "not_sent",
        }
        ticks.append(row)
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
            raise ValueError("Automatic reset provenance changed")
        validate_step(packet, response["observation"], terminal=terminal)
        witness = row["task_transition_after_action"]
        if witness is None or witness["native_success"] != row["terminated"]:
            raise ValueError("Missing native task evidence before automatic reset")
        tracker.update(witness, step)
        if terminal or (stop_subgoal and tracker.first_settled_step is not None):
            break
        packet = response["observation"]
    if pending is not None:
        pending["event"]["outcome"] = "censored_before_future_observation"
    final = ticks[-1]
    terminal = final["terminated"] or final["truncated"]
    return {
        "status": "terminal"
        if terminal
        else (
            "task_subgoal_reached"
            if stop_subgoal and tracker.first_settled_step is not None
            else "censored_step_limit"
        ),
        "native_success": final["terminated"] if terminal else None,
        "first_settled_step": tracker.first_settled_step,
        "steps": len(ticks),
    }


def paired_summary(rows: list[dict], arms: list[str]) -> dict:
    aggregates = {}
    for arm in arms:
        group = [row for row in rows if row["arm"] == arm]
        aggregates[arm] = {
            "trials": len(group),
            "successes": sum(r["success"] for r in group),
            "technical_failures": sum(not r["technical_valid"] for r in group),
            "mean_restricted_time_s": float(np.mean([r["restricted_time_s"] for r in group]))
            if group
            else None,
        }
    contrasts = {}
    for arm in arms:
        if arm == "state_only":
            continue
        differences, matched = [], []
        for block in range(8):
            pair = {r["arm"]: r for r in rows if r["block"] == block}
            if arm not in pair or "state_only" not in pair:
                continue
            a, b = pair[arm], pair["state_only"]
            differences.append(a["restricted_time_s"] - b["restricted_time_s"])
            matched.append(
                a["initial_geometry_match"]
                and b["initial_geometry_match"]
                and a["bootstrap_matches"]
                and b["bootstrap_matches"]
            )
        contrast = {"differences_s": differences, "matched": matched, "stable_gate": False}
        if len(differences) == 8:
            values = np.asarray(differences)
            rng = np.random.default_rng(3601)
            interval = np.percentile(values[rng.integers(0, 8, (50000, 8))].mean(1), [2.5, 97.5]).tolist()
            contrast.update(
                mean_difference_s=float(values.mean()),
                bootstrap_95_s=interval,
                first_half_s=float(values[:4].mean()),
                second_half_s=float(values[4:].mean()),
            )
            contrast["stable_gate"] = bool(
                all(matched)
                and interval[1] < 0
                and values[:4].mean() < 0
                and values[4:].mean() < 0
                and aggregates[arm]["successes"] >= aggregates["state_only"]["successes"]
                and aggregates[arm]["technical_failures"] <= aggregates["state_only"]["technical_failures"]
            )
        contrasts[arm] = contrast
    reference = aggregates["state_only"]
    reference_ok = (
        reference["trials"] == 8 and reference["successes"] >= 7 and reference["technical_failures"] == 0
    )
    return {
        "aggregates": aggregates,
        "contrasts": contrasts,
        "reliable_state_only_reference": reference_ok,
        "privileged_visual_reference_positive": bool(
            reference_ok and contrasts.get("oracle_visual", {}).get("stable_gate")
        ),
        "learned_task_benefit": bool(reference_ok and contrasts.get("student", {}).get("stable_gate")),
        "realtime_qualified": False,
    }


def prefix_geometry_differences(root: Path, rows: list[dict]) -> list[dict]:
    """Describe physical divergence despite identical commands; not a fitted gate."""
    result = []
    for block in sorted({row["block"] for row in rows}):
        group = [row for row in rows if row["block"] == block]
        if not any(row["arm"] == "state_only" for row in group):
            continue

        def prefix(arm, block=block):
            path = root / f"block{block:02d}_{arm}" / "ticks.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()][:27]

        reference = prefix("state_only")
        for row in group:
            if row["arm"] == "state_only":
                continue
            current = prefix(row["arm"])
            count = min(len(reference), len(current))
            state_difference, object_difference = None, None
            if count:
                state_difference = (
                    np.abs(
                        np.array([r["state"] for r in current[:count]])
                        - np.array([r["state"] for r in reference[:count]])
                    )
                    .max(axis=0)
                    .tolist()
                )
                object_difference = max(
                    abs(a["objects_before"][name][axis] - b["objects_before"][name][axis])
                    for a, b in zip(current[:count], reference[:count], strict=True)
                    for name in a["objects_before"]
                    for axis in range(3)
                )
            result.append(
                {
                    "block": block,
                    "arm": row["arm"],
                    "compared_pre_takeover_states": count,
                    "state_transport_abs_max_by_coordinate": state_difference,
                    "object_position_abs_max_m": object_difference,
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "study"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--student-folder", type=Path)
    parser.add_argument("--student-test", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        parser.error("Commit the exact information-reference protocol before execution")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    smoke = args.phase == "smoke"
    if not smoke:
        if args.smoke is None:
            parser.error("The single oracle wiring smoke must finish before the study")
        ready = json.loads((args.smoke / "summary.json").read_text())
        if not ready["smoke_pass"]:
            parser.error("The fixed oracle smoke failed; no repeated smoke is authorized")
    arms = ["state_only", "oracle_visual", "oracle_joint"]
    selection = None
    if args.student_folder is not None or args.student_test is not None:
        if args.student_folder is None or args.student_test is None:
            parser.error("Student requires both its frozen selection and independent test")
        selection = json.loads((args.student_folder / "selection.json").read_text())
        test = json.loads(args.student_test.read_text())
        if (
            not selection["qualified"]
            or not test["stable_incremental_action_gate"]
            or test["selected_update"] != selection["selected_update"]
        ):
            parser.error("Do not deploy a student that failed its independent gate")
        arms.append("student")
    if smoke:
        orders, first_seed, first_policy = [("oracle_visual",)], 20270701, 3601
    else:
        if len(arms) == 3:
            orders = list(itertools.permutations(arms))
            orders += orders[:2]
        else:
            orders = [tuple(arms[i:] + arms[:i]) for i in range(4)]
            reverse = list(reversed(arms))
            orders += [tuple(reverse[i:] + reverse[:i]) for i in range(4)]
        random.Random(3600).shuffle(orders)
        first_seed, first_policy = 20270710, 3610
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "phase": args.phase,
        "orders": orders,
        "arms": arms,
        "first_seed": first_seed,
        "first_policy_seed": first_policy,
        "simulated_delay": 7,
        "shared_bootstrap": "real_current_observation_policy_chunk_before_any_arm",
        "student_selection": str(args.student_folder) if selection else None,
        "privileged_references_are_not_deployable": True,
        "realtime_required": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    base = root.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    client = EnvClient(
        base / "leisaac-sim-venv/bin/python",
        base
        / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets",
        base / "leisaac-source",
        "cpu",
    )
    client.camera_backend, client.episode_seconds, client.task_evidence = "standard", 120, True
    rows, global_error, total_setup = [], None, 0
    try:
        policy, pre, post = load_matched_runtime(snapshot, device="cuda:0")
        state_model = load_future_state(
            base / "artifacts/m54l12_state_development_v1/state_best.pt", "cuda:0"
        )
        student = None
        if selection is not None:
            saved = torch.load(args.student_folder / "best.pt", map_location="cpu", weights_only=True)
            if (
                saved["kind"] != "l15_state_conditional_visual_v1"
                or saved["selected_update"] != selection["selected_update"]
            ):
                raise ValueError("The frozen student identity changed")
            student = LightweightFutureLatentPredictor(FutureLatentConfig(**saved["config"]))
            student.load_state_dict(saved["state_dict"], strict=True)
            student.to("cuda:0").eval().requires_grad_(False)
        client.start()
        for block, order in enumerate(orders):
            seed, policy_seed = first_seed + block, first_policy + block
            folder = args.output / f"block{block:02d}_bootstrap"
            folder.mkdir()
            setup = []
            try:
                packet = client.reset(seed)
                warm_environment(client, packet, setup)
                packet = client.reset(seed)
                policy.reset()
                pre.reset()
                post.reset()
                set_seed(policy_seed)
                batch, tokens, masks, state = prepared_observation(packet, policy, pre)
                projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
                normal, physical = generate(
                    policy, post, projector, batch, tokens, masks, state, policy_seed * 100000
                )
                bootstrap = {
                    "normalized": normal,
                    "physical": physical,
                    "geometry": geometry(packet),
                    "tokens": torch.stack(tokens, dim=1).cpu(),
                    "state": state.cpu(),
                    "noise_seed": policy_seed * 100000,
                }
                torch.save(bootstrap, folder / "bootstrap.pt")
            finally:
                (folder / "setup_ticks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in setup))
                total_setup += sum(r["dispatch"] == "completed" for r in setup)
            for arm in order:
                folder = args.output / f"block{block:02d}_{arm}"
                folder.mkdir()
                ticks, events, setup = [], [], []
                result, initial = {}, None
                try:
                    packet = client.reset(seed)
                    warm_environment(client, packet, setup)
                    packet = client.reset(seed)
                    initial = geometry(packet)
                    if initial != bootstrap["geometry"]:
                        raise ValueError("Reset geometry differs from the fixed bootstrap; do not resample")
                    policy.reset()
                    pre.reset()
                    post.reset()
                    set_seed(policy_seed)
                    projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
                    started = time.perf_counter()
                    result = drive(
                        client,
                        packet,
                        policy,
                        pre,
                        post,
                        state_model,
                        student,
                        projector,
                        arm,
                        policy_seed,
                        bootstrap,
                        60 if smoke else 3600,
                        not smoke,
                        ticks,
                        events,
                    )
                    result["wall_s"] = time.perf_counter() - started
                except Exception:
                    result = {
                        "status": "technical_failure",
                        "error": traceback.format_exc(),
                        "first_settled_step": None,
                    }
                setup_count = sum(r["dispatch"] == "completed" for r in setup)
                total_setup += setup_count
                completed = sum(t["dispatch"] == "completed" for t in ticks)
                prefix = [t["normalized_action"] for t in ticks[:27] if t["dispatch"] == "completed"]
                matched = bool(prefix) and prefix == bootstrap["normalized"][: len(prefix)].tolist()
                valid = result["status"] != "technical_failure" and completed == len(ticks) and bool(ticks)
                success = valid and result["first_settled_step"] is not None
                row = {
                    "block": block,
                    "arm": arm,
                    "seed": seed,
                    "policy_seed": policy_seed,
                    **result,
                    "completed_actions": completed,
                    "setup_actions": setup_count,
                    "initial_geometry_match": initial == bootstrap["geometry"],
                    "bootstrap_matches": matched,
                    "verified_bootstrap_prefix_actions": len(prefix),
                    "technical_valid": valid,
                    "success": bool(success),
                    "restricted_time_s": (result["first_settled_step"] + 1) / 30 if success else 120.0,
                    "planned_requests": len(events),
                    "takeovers": sum(t.get("queue_outcome") == "takeover" for t in ticks),
                    "privileged_generations": sum(
                        e.get("generated_step") is not None and e["privileged"] for e in events
                    ),
                }
                rows.append(row)
                for name, values in (("ticks", ticks), ("events", events), ("setup_ticks", setup)):
                    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
                (folder / "control_result.json").write_text(json.dumps(row, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "block",
                                "arm",
                                "status",
                                "completed_actions",
                                "success",
                                "restricted_time_s",
                            )
                        }
                    ),
                    flush=True,
                )
    except Exception:
        global_error = traceback.format_exc()
    finally:
        client.close()
        cleanup = (
            client.cleanup_error is None and client.process is not None and client.process.returncode == 0
        )
        if not cleanup:
            for row in rows:
                row.update(technical_valid=False, success=False, restricted_time_s=120.0)
        report = {
            "source_commit": source,
            "phase": args.phase,
            "rows": rows,
            "complete": global_error is None and len(rows) == sum(len(order) for order in orders),
            "shared_simulator_cleanup_complete": cleanup,
            "shared_simulator_returncode": None if client.process is None else client.process.returncode,
            "cleanup_error": client.cleanup_error,
            "global_error": global_error,
            "environment": client.metadata,
            "setup_actions": total_setup,
            "completed_actions": sum(row["completed_actions"] for row in rows),
            "privileged_generations": sum(row["privileged_generations"] for row in rows),
            "realtime_qualified": False,
        }
        if smoke:
            report["smoke_pass"] = bool(
                cleanup
                and len(rows) == 1
                and rows[0]["technical_valid"]
                and rows[0]["completed_actions"] == 60
                and rows[0]["privileged_generations"] == 2
                and rows[0]["takeovers"] == 2
                and rows[0]["bootstrap_matches"]
            )
        else:
            report.update(paired_summary(rows, arms))
            report["pre_takeover_geometry_differences"] = prefix_geometry_differences(args.output, rows)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("rows", "environment")}, indent=2),
        flush=True,
    )
    return int(not report["complete"] or not cleanup or (smoke and not report["smoke_pass"]))


if __name__ == "__main__":
    raise SystemExit(main())
