"""Bounded SO101 transfer smoke/capability/async pilot, using one production queue.

See docs/experiments/LEISAAC_SO101_MINIMAL_EXPERIMENT.md for the fixed run order.
Run this script with the existing model interpreter, not the simulator interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
from leisaac_so101_contract import (
    FPS,
    IMAGE_SHAPE,
    PROFILE,
    SCALAR_KEYS,
    TASK,
    ContractError,
    action_to_radians,
    hardware_features,
    validate_observation,
    validate_step,
)


class EnvClient:
    """An inherited local socket: primitive dictionaries/bytes, no ndarray pickle."""

    # First RTX shader/material compilation can exceed the original 180 s.
    # This budget is outside the measured control clock; step IPC remains 30 s.
    STARTUP_TIMEOUT_S = 600

    def __init__(self, python: Path, assets: Path, source: Path, device: str) -> None:
        # Keep the venv executable path (do not resolve its interpreter symlink).
        self.python = python.absolute()
        self.assets, self.source, self.device = assets.resolve(), source.resolve(), device
        self.process = None
        self.connection = None
        self.log_reader = None
        self.logs: list[bytes] = []
        self.metadata: dict[str, Any] = {}
        self.cleanup_error: str | None = None
        self.episode_seconds = 25
        self.control_fps = 30
        self.initial_pose = "zero"
        self.camera_backend = "tiled"
        self.task_evidence = False
        self.gc_diagnostics = False
        self.profile_steps = False
        self.step_profile: dict | None = None

    def start(self) -> None:
        parent, child = socket.socketpair()
        self.connection = Connection(parent.detach())
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["HF_HUB_OFFLINE"] = "1"
        command = [
            str(self.python),
            str(Path(__file__).with_name("leisaac_so101_env_server.py")),
            "--fd",
            str(child.fileno()),
            "--assets-root",
            str(self.assets),
            "--leisaac-root",
            str(self.source),
            "--device",
            self.device,
            "--episode-seconds",
            str(self.episode_seconds),
            "--control-fps",
            str(self.control_fps),
            "--initial-pose",
            self.initial_pose,
            "--camera-backend",
            self.camera_backend,
        ]
        if self.profile_steps:
            command.append("--profile-steps")
        if self.task_evidence:
            command.append("--task-evidence")
        if self.gc_diagnostics:
            command.append("--gc-diagnostics")
        try:
            self.process = subprocess.Popen(
                command,
                pass_fds=(child.fileno(),),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        finally:
            child.close()

        def drain() -> None:
            while chunk := self.process.stdout.read(65536):
                self.logs.append(chunk)

        self.log_reader = Thread(target=drain, name="LeIsaacLogReader")
        self.log_reader.start()
        self.metadata = self.receive(self.STARTUP_TIMEOUT_S)["metadata"]
        if self.metadata["profile"] != PROFILE:
            raise ContractError("Unexpected environment profile")

    def receive(self, timeout: float = 30) -> dict:
        if not self.connection.poll(timeout):
            raise TimeoutError(f"LeIsaac did not respond within {timeout} seconds")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def reset(self, seed: int) -> dict:
        self.connection.send({"op": "reset", "seed": seed})
        packet = self.receive()["observation"]
        packet["received_at_s"] = time.perf_counter()
        return packet

    def step(self, packet: dict, action: list[float]) -> dict:
        self.connection.send(
            {"op": "step", "episode_id": packet["episode_id"], "step": packet["step"], "action": action}
        )
        result = self.receive()
        result["observation"]["received_at_s"] = time.perf_counter()
        return result

    def close(self) -> None:
        if self.process is None:
            if self.connection is not None:
                self.connection.close()
            return
        try:
            if self.process.poll() is None:
                try:
                    self.connection.send({"op": "close"})
                    self.step_profile = self.receive().get("step_profile")
                except (EOFError, BrokenPipeError, OSError, RuntimeError, TimeoutError) as error:
                    self.cleanup_error = str(error)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.cleanup_error = "Simulator required termination after close timeout"
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.connection.close()
            self.log_reader.join(timeout=5)
            self.process.stdout.close()


class MemoryMetrics:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.closed = False

    def emit(self, event: dict) -> None:
        if self.closed:
            raise RuntimeError("Metrics emitted after sink closure")
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def decode_observation(packet: dict) -> dict:
    values = validate_observation(packet)
    raw = dict(zip(SCALAR_KEYS, values, strict=True))
    for alias, camera in packet["camera_sources"].items():
        raw[alias] = np.frombuffer(packet["images"][camera], dtype=np.uint8).reshape(IMAGE_SHAPE).copy()
    return raw


class SnapshotRobot:
    robot_type = "so101_follower"
    name = PROFILE
    is_connected = True

    def __init__(self, observation: dict) -> None:
        self.observation = observation

    def get_observation(self) -> dict:
        return dict(self.observation)


def prepare_engine(engine: Any, observation: dict, *, timeout: float = 180) -> None:
    """Startup off the measured clock, then quiesce before resetting processors."""
    engine.start()
    engine.resume()
    deadline = time.perf_counter() + timeout
    while not engine.ready:
        if engine.failed:
            raise RuntimeError(engine.failure_traceback)
        if time.perf_counter() >= deadline:
            raise TimeoutError("Production engine startup timed out")
        engine.notify_observation(observation)
        time.sleep(0.005)
    if engine.failed:
        raise RuntimeError(engine.failure_traceback)
    engine.pause()
    # Existing reset() resets model/processors on its caller. Do not race it
    # with an in-flight inference. No production queue/reset code is changed.
    while True:
        with engine._request_lock:
            busy = engine._request_in_flight or engine._pending_request is not None
        if not busy:
            break
        if time.perf_counter() >= deadline:
            raise TimeoutError("Production engine did not quiesce before episode reset")
        time.sleep(0.005)
    engine.reset()


def prime_episode_engine(engine: Any, observation: dict, *, timeout: float = 180) -> None:
    """Compute a fresh initial chunk at reset state before the control clock starts.

    No environment step, action get, manual queue insertion or future observation
    occurs here. The normal production worker owns this bootstrap and its metrics.
    """
    if engine.queue.qsize():
        raise RuntimeError("Fresh episode priming requires an empty reset queue")
    engine.resume()
    engine.notify_observation(observation)
    deadline = time.perf_counter() + timeout
    while True:
        if engine.failed:
            raise RuntimeError(engine.failure_traceback)
        with engine._request_lock:
            busy = engine._request_in_flight or engine._pending_request is not None
        if not busy and engine.queue.qsize():
            return
        if time.perf_counter() >= deadline:
            raise TimeoutError("Fresh episode bootstrap did not finish before control start")
        time.sleep(0.005)


def stop_engine(engine: Any) -> None:
    engine.stop()
    worker = engine._worker
    if worker is not None:
        worker.join(timeout=30)
        if worker.is_alive():
            raise RuntimeError("Policy worker is still alive; do not serialize concurrent telemetry")
    engine.reset()


def drive_episode(
    client: Any,
    packet: dict,
    *,
    max_steps: int,
    ticks: list[dict],
    engine: Any = None,
    sync_action: Any = None,
    realtime: bool = True,
    frames: list | None = None,
    stop_after_placement: bool = False,
) -> dict:
    """One notify/get/real step per tick; underflow holds the previous sent target."""
    from lerobot.utils.cycle_timer import CycleTimer

    if packet.get("control_fps", 30) != 30 and (realtime or engine is not None):
        raise ContractError("Native60 is a synchronous diagnostic, not the qualified async time base")

    last_action = validate_observation(packet)
    timer = CycleTimer(FPS, records_data=False)
    origin = time.perf_counter()
    total_reward = 0.0
    placement_tracker = None
    if stop_after_placement:
        from so101_task_evidence import PlacementTracker

        placement_tracker = PlacementTracker()
    for index in range(max_steps):
        started = time.perf_counter()
        # Pace and judge the same fixed slots; neither action indices nor
        # simulation time are fast-forwarded after bounded late work.
        if realtime and started - (origin + index / FPS) >= 1 / FPS:
            raise ContractError("lost_control_slot_before_tick")
        timer.tick()
        raw = decode_observation(packet)
        if frames is not None and index % 150 == 0:
            frames.append((index, {name: raw[name] for name in ("top", "wrist")}))
        row = {
            "tick": index,
            "episode_id": packet["episode_id"],
            "sim_step": packet["step"],
            "control_fps": packet.get("control_fps", 30),
            "sim_time_s": packet["sim_time_s"],
            "started_at_s": started,
            "scheduled_start_s": origin + index / FPS,
            "start_lateness_s": started - (origin + index / FPS),
            "snapshot_ready_at_s": packet["snapshot_ready_at_s"],
            "received_at_s": packet.get("received_at_s"),
            "state": [raw[k] for k in SCALAR_KEYS],
            "camera_frames": packet["camera_frames"],
            "dispatch": "not_sent",
        }
        if "task_diagnostics" in packet:
            row["task_diagnostics_before_action"] = packet["task_diagnostics"]
        if "camera_world_poses_opengl" in packet:
            row["camera_world_poses_opengl"] = packet["camera_world_poses_opengl"]
        ticks.append(row)
        hold = True
        if engine is not None:
            if engine.failed:
                raise RuntimeError(engine.failure_traceback)
            engine.notify_observation(raw)
            action = engine.get_action(None)
            if action is not None:
                last_action = action.detach().cpu().reshape(-1).tolist()
                hold = False
        elif sync_action is not None:
            last_action = sync_action(raw)
            hold = False
        row.update(action=list(last_action), underflow_hold=hold and engine is not None)
        action_to_radians(last_action)  # Fail before dispatch; never silently clip.
        row["dispatch"] = "sent_result_unknown"
        step_started = time.perf_counter()
        response = client.step(packet, last_action)
        row["ipc_roundtrip_s"] = time.perf_counter() - step_started
        row["controller_before_step_s"] = step_started - started
        row["server_timing_s"] = response.get("server_timing_s", {})
        if "gc_collections" in response:
            row["gc_collections"] = response["gc_collections"]
        if "task_transition" in response:
            row["task_transition_after_action"] = response["task_transition"]
        row.update(
            dispatch="completed",
            reward=float(response["reward"]),
            terminated=bool(response["terminated"]),
            truncated=bool(response["truncated"]),
        )
        terminal = row["terminated"] or row["truncated"]
        if bool(response["post_reset_observation"]) != terminal:
            raise ContractError("Automatic reset provenance does not match returned termination flags")
        successor = response["observation"]
        validate_step(packet, successor, terminal=terminal)
        subgoal_reached = False
        if placement_tracker is not None:
            witness = row.get("task_transition_after_action")
            if witness is None:
                raise ContractError("First-placement protocol requires native pre-reset evidence")
            subgoal_reached = placement_tracker.update(witness, index) >= 1
        row["finished_at_s"] = time.perf_counter()
        row["work_s"] = row["finished_at_s"] - started
        total_reward += row["reward"]
        if realtime and row["finished_at_s"] - (origin + index / FPS) >= 2 / FPS:
            raise ContractError("lost_control_slot_during_tick")
        if terminal:
            # Do not decode/notify with Isaac's already-reset observation.
            return {
                "status": "terminal",
                "success": row["terminated"],
                "timeout": row["truncated"],
                "steps": index + 1,
                "reward": total_reward,
            }
        packet = successor
        if subgoal_reached:
            return {
                "status": "task_subgoal_reached",
                "success": None,
                "subtask_success": True,
                "timeout": False,
                "steps": index + 1,
                "reward": total_reward,
            }
        if realtime:
            timer.wait(deadline=origin + (index + 1) / FPS)
    return {
        "status": "censored_step_limit",
        "success": None,
        "timeout": False,
        "steps": max_steps,
        "reward": total_reward,
    }


def load_runtime(
    mode: str,
    device: str,
    predictor_path: Path | None,
    seed: int,
    sink: MemoryMetrics,
    raw: dict,
    *,
    matched_snapshot: Path | None = None,
    sync_execution_steps: int = 50,
    action_contract: str = "strict",
    minimum_delay: int = 1,
    startup_profile: str = "native",
    context_variant: str = "visual_only",
    future_state_path: Path | None = None,
    sync_task_text: str | None = None,
):
    if sync_task_text is not None and (mode != "sync" or matched_snapshot is None):
        raise ValueError("Task-text diagnostics require the matched synchronous candidate")
    import torch
    from huggingface_hub import snapshot_download

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.future_latent_checkpoint import (
        POLICY_REPO_ID,
        POLICY_REVISION,
        load_frozen_future_latent_predictor,
    )
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.rollout.context import _load_frozen_future_latent_runtime
    from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    task = TASK
    if matched_snapshot is None:
        snapshot = snapshot_download(POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True)
        config = PreTrainedConfig.from_pretrained(snapshot, local_files_only=True)
        policy, preprocessor, postprocessor = _load_frozen_future_latent_runtime(config, device=device)
    else:
        from leisaac_so101_matched import TASK as MATCHED_TASK, load_matched_runtime

        if mode not in ("sync", "identity", "predicted"):
            raise ValueError("Unsupported task-matched inference mode")
        task = MATCHED_TASK
        if sync_task_text is not None:
            task = sync_task_text
        policy, preprocessor, postprocessor = load_matched_runtime(
            matched_snapshot, device=device, execution_steps=sync_execution_steps
        )
    policy.to(device).eval()
    projector = None
    if action_contract == "feasible_v1":
        from so101_feasible_actions import FeasibleActionProjector

        projector = FeasibleActionProjector.from_snapshot(matched_snapshot, device)
    robot = ThreadSafeRobot(SnapshotRobot(raw))
    features = hardware_features()
    if mode == "sync":

        def action(observation: dict) -> list[float]:
            batch = build_dataset_frame(features, observation, prefix="observation")
            batch = prepare_observation_for_inference(batch, torch.device(device), task, robot.robot_type)
            batch["task"] = [task]
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=torch.device(device).type,
                    enabled=matched_snapshot is not None and policy.config.use_amp,
                ),
            ):
                selected = policy.select_action(preprocessor(batch))
                result = postprocessor(projector(selected) if projector is not None else selected)
            return result.detach().cpu().reshape(-1).tolist()

        action.action_projector = projector
        return None, action
    engine_class = PredictiveAsyncInferenceEngine
    predictor = None
    if matched_snapshot is not None and mode == "predicted":
        from leisaac_so101_predicted import SO101PredictiveAsyncInferenceEngine, load_so101_l6_predictor

        predictor = load_so101_l6_predictor(predictor_path, device=device)
        engine_class = SO101PredictiveAsyncInferenceEngine
    elif mode == "predicted":
        predictor = load_frozen_future_latent_predictor(predictor_path, device=device)
    if context_variant != "visual_only":
        from so101_joint_runtime import JointContextPredictor, load_future_state

        if mode != "predicted" or matched_snapshot is None or projector is None or future_state_path is None:
            raise ValueError("Joint context is restricted to the explicit matched feasible simulation")
        predictor = (
            JointContextPredictor(predictor, load_future_state(future_state_path, device), context_variant)
            .eval()
            .requires_grad_(False)
        )
    predictor_warmup = None
    if startup_profile == "warmed_v2" and predictor is not None:
        from so101_startup_preparation import warm_predictor_once

        predictor_warmup = warm_predictor_once(policy, preprocessor, predictor, raw, task, device)
    projection_arguments = {}
    if projector is not None:
        from so101_feasible_actions import SO101FeasibleAsyncEngine

        engine_class = SO101FeasibleAsyncEngine
        projection_arguments["action_projector"] = projector
    if context_variant != "visual_only":
        from so101_joint_runtime import SO101JointAsyncEngine

        engine_class = SO101JointAsyncEngine
    engine = engine_class(
        **projection_arguments,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot,
        hw_features=features,
        task=task,
        fps=FPS,
        device=device,
        queue_threshold=30,
        latency_quantile=0.9,
        latency_window=50,
        delay_safety_margin_steps=1,
        min_prediction_delay=minimum_delay,
        max_prediction_delay=8,
        committed_guard_steps=2,
        max_late_steps=2,
        context_mode=mode,
        future_latent_predictor=predictor,
        fallback_mode="identity",
        metrics_sink=sink,
    )
    engine.predictor_warmup = predictor_warmup
    return engine, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smoke", "env-profile", "sync", "identity", "predicted"), required=True
    )
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=1701)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-seconds", type=int, choices=(25, 60, 120), default=25)
    parser.add_argument("--sync-execution-steps", type=int, choices=(25, 50), default=50)
    parser.add_argument(
        "--sync-task-text",
        choices=("Grab orange and place into plate", "Pick up the orange and put it in the plate"),
        help="Explicit task-text diagnostic for the matched synchronous candidate only",
    )
    parser.add_argument("--control-fps", type=int, choices=(30, 60), default=30)
    parser.add_argument("--initial-pose", choices=("zero", "rest", "training_medoid_v1"), default="zero")
    parser.add_argument("--camera-backend", choices=("tiled", "standard"), default="tiled")
    parser.add_argument("--task-evidence", action="store_true")
    parser.add_argument("--gc-diagnostics", action="store_true")
    parser.add_argument("--minimum-delay", type=int, choices=(1, 7), default=1)
    parser.add_argument("--profile-native-steps", action="store_true")
    parser.add_argument("--action-contract", choices=("strict", "feasible_v1"), default="strict")
    parser.add_argument("--stop-after-first-placement", action="store_true")
    parser.add_argument("--startup-profile", choices=("native", "warmed_v2"), default="native")
    parser.add_argument(
        "--context-variant", choices=("visual_only", "state_only", "joint"), default="visual_only"
    )
    parser.add_argument("--future-state-checkpoint", type=Path)
    parser.add_argument(
        "--matched-snapshot", type=Path, help="Exact independent PickOrange candidate snapshot"
    )
    parser.add_argument(
        "--sim-device", choices=("cpu", "cuda:0"), help="Simulation compute device; model device is unchanged"
    )
    args = parser.parse_args()
    if args.sync_task_text is not None and (args.mode != "sync" or args.matched_snapshot is None):
        parser.error("Task-text diagnostics require matched synchronous inference")
    if args.context_variant != "visual_only":
        if (
            args.mode != "predicted"
            or args.future_state_checkpoint is None
            or args.action_contract != "feasible_v1"
            or args.startup_profile != "warmed_v2"
            or args.minimum_delay != 7
        ):
            parser.error("L13 joint/state-only requires its causal checkpoint and calibrated warmed profile")
    elif args.future_state_checkpoint is not None:
        parser.error("A future-state checkpoint requires an explicit joint/state-only variant")
    if args.minimum_delay != 1 and (
        args.matched_snapshot is None or args.mode not in ("identity", "predicted")
    ):
        parser.error("The calibrated delay floor is for matched async execution only")
    if args.profile_native_steps and (args.mode != "sync" or args.max_steps > 600):
        parser.error("Native per-step profiling is a bounded synchronous diagnostic only")
    if args.startup_profile == "warmed_v2" and args.action_contract != "feasible_v1":
        parser.error("Warmed-v2 is exclusive to the independently registered feasible-action protocol")
    if args.stop_after_first_placement and (
        not args.task_evidence or args.action_contract != "feasible_v1" or args.control_fps != 30
    ):
        parser.error("The first-placement task requires task evidence and the 30Hz feasible-action contract")
    if args.action_contract != "strict":
        from leisaac_so101_matched import WSAGI_REVISION

        if args.matched_snapshot is None or args.matched_snapshot.name != WSAGI_REVISION:
            parser.error("Feasible-action execution is an independent WSAGI-only contract")
    environment_only = args.mode in ("smoke", "env-profile")
    realtime = args.mode not in ("sync", "env-profile")
    if args.episode_seconds != 25 and args.matched_snapshot is None:
        parser.error("Extended development protocols are exclusive to independent matched candidates")
    if args.control_fps != 30 and (args.matched_snapshot is None or args.mode != "sync"):
        parser.error("Native60 time-base diagnostic is restricted to matched synchronous evaluation")
    if args.initial_pose != "zero" and args.matched_snapshot is None:
        parser.error("Rest-pose preparation is a separately recorded matched-candidate protocol")
    if args.initial_pose == "training_medoid_v1":
        from leisaac_so101_matched import WSAGI_REVISION

        if (
            args.mode != "sync"
            or args.matched_snapshot.name != WSAGI_REVISION
            or args.action_contract != "feasible_v1"
            or args.startup_profile != "warmed_v2"
            or args.control_fps != 30
            or args.sim_device != "cpu"
            or args.camera_backend != "standard"
        ):
            parser.error("Training-medoid preparation is a registered WSAGI synchronous diagnostic only")
    if not 1 <= args.max_steps <= (30 if environment_only else args.episode_seconds * args.control_fps):
        parser.error("Step bound exceeds the selected environment-only / episode protocol")
    if args.mode == "predicted" and args.predictor is None:
        parser.error("predicted requires the frozen portable --predictor")
    if args.matched_snapshot is not None and args.mode not in ("sync", "identity", "predicted"):
        parser.error("Task-matched candidate supports sync/identity/predicted only")
    if args.matched_snapshot is not None and args.mode == "predicted":
        from leisaac_so101_matched import WSAGI_REVISION

        if (
            args.matched_snapshot.name != WSAGI_REVISION
            or args.sim_device != "cpu"
            or args.camera_backend != "standard"
            or args.initial_pose != "zero"
            or args.control_fps != 30
        ):
            parser.error("L7 predicted requires WSAGI, CPU PhysX, standard cameras, zero start and 30 Hz")
    if args.sync_execution_steps != 50 and (args.matched_snapshot is None or args.mode != "sync"):
        parser.error("Shortened synchronous execution is a matched-candidate development protocol only")
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        parser.error("Commit the implementation before a source-bound experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "profile": PROFILE,
        "source_commit": commit,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "risk_thresholds": None,
        "simulator_startup_timeout_s": EnvClient.STARTUP_TIMEOUT_S,
        "ipc_timeout_s": 30,
        "operator_eula_acceptance_env": os.environ.get("OMNI_KIT_ACCEPT_EULA"),
        "realtime_required": realtime,
    }
    if args.matched_snapshot is not None:
        from leisaac_so101_matched import candidate_manifest

        manifest["candidate"] = candidate_manifest(
            args.matched_snapshot, execution_steps=args.sync_execution_steps
        )
        if args.sync_task_text is not None:
            manifest["candidate"]["task"] = args.sync_task_text
            manifest["candidate"]["task_text_diagnostic"] = True
        if args.mode == "predicted":
            from leisaac_so101_predicted import PREDICTOR_ID, SELECTED_EPOCH, TRAINING_SOURCE

            manifest["candidate"]["predictor"] = {
                "id": PREDICTOR_ID,
                "path": str(args.predictor),
                "training_source": TRAINING_SOURCE,
                "selected_epoch": SELECTED_EPOCH,
            }
            if args.context_variant != "visual_only":
                from so101_joint_runtime import STATE_EPOCH, STATE_SOURCE

                visual_parent = manifest["candidate"]["predictor"]
                manifest["candidate"]["predictor"] = {
                    "id": "so101_l13_" + args.context_variant,
                    "visual_parent": visual_parent,
                    "visual_forecasting_active": args.context_variant == "joint",
                    "state_path": str(args.future_state_checkpoint),
                    "state_source": STATE_SOURCE,
                    "state_epoch": STATE_EPOCH,
                }
    manifest["pacing"] = "absolute_CycleTimer_deadline_same_origin_as_lost_slot_checks"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    client = EnvClient(args.sim_python, args.assets_root, args.leisaac_root, args.sim_device or args.device)
    client.profile_steps = args.mode == "env-profile" or args.profile_native_steps
    client.episode_seconds = args.episode_seconds
    client.control_fps = args.control_fps
    client.initial_pose = args.initial_pose
    client.camera_backend = args.camera_backend
    client.task_evidence = args.task_evidence
    client.gc_diagnostics = args.gc_diagnostics
    sink, ticks, engine, result = MemoryMetrics(), [], None, {}
    fresh_bootstrap_wall_s = None
    sync_action = None
    setup_rows = []
    frames = [] if args.matched_snapshot is not None else None
    try:
        client.start()
        packet = client.reset(args.seed)
        if not environment_only:
            engine, sync_action = load_runtime(
                args.mode,
                args.device,
                args.predictor,
                args.policy_seed,
                sink,
                decode_observation(packet),
                matched_snapshot=args.matched_snapshot,
                sync_execution_steps=args.sync_execution_steps,
                action_contract=args.action_contract,
                minimum_delay=args.minimum_delay,
                startup_profile=args.startup_profile,
                context_variant=args.context_variant,
                future_state_path=args.future_state_checkpoint,
                sync_task_text=args.sync_task_text,
            )
        else:
            sync_action = None
        if engine is not None:
            prepare_engine(engine, decode_observation(packet))
        if args.startup_profile == "warmed_v2":
            from so101_startup_preparation import warm_environment

            warm_environment(client, packet, setup_rows)
        if engine is not None or args.startup_profile == "warmed_v2":
            packet = client.reset(args.seed)
        if not environment_only:
            from lerobot.utils.random_utils import set_seed

            # Predictor construction and startup consume RNG differently by
            # mode. Seed the measured policy stream only after both are over.
            set_seed(args.policy_seed)
        if engine is not None:
            priming_started = time.perf_counter()
            prime_episode_engine(engine, decode_observation(packet))
            fresh_bootstrap_wall_s = time.perf_counter() - priming_started
        result = drive_episode(
            client,
            packet,
            max_steps=args.max_steps,
            ticks=ticks,
            engine=engine,
            sync_action=sync_action,
            realtime=realtime,
            frames=frames,
            stop_after_placement=args.stop_after_first_placement,
        )
    except Exception:
        result = {"status": "technical_failure", "error": traceback.format_exc(), "success": None}
    finally:
        try:
            if engine is not None:
                stop_engine(engine)
        finally:
            client.close()
        if engine is None:
            sink.close()
        if client.cleanup_error:
            result.update(status="technical_failure", cleanup_error=client.cleanup_error)
        result.update(
            environment=client.metadata,
            ticks=len(ticks),
            source_commit=commit,
            profile=PROFILE,
            mode=args.mode,
            realtime_required=realtime,
            metrics_closed=sink.closed,
            subprocess_returncode=None if client.process is None else client.process.returncode,
            fresh_episode_bootstrap_wall_s=fresh_bootstrap_wall_s,
        )
        if "candidate" in manifest:
            result["candidate"] = manifest["candidate"]
        result["action_execution_contract"] = args.action_contract
        result["startup_profile"] = args.startup_profile
        result["context_variant"] = args.context_variant if args.mode == "predicted" else args.mode
        result["setup_physics_steps"] = sum(row["dispatch"] == "completed" for row in setup_rows)
        result["predictor_kernel_warmup"] = getattr(engine, "predictor_warmup", None)
        result["task_contract"] = (
            "pickorange_first_settled_v1" if args.stop_after_first_placement else "native_pickorange"
        )
        projector = getattr(engine if engine is not None else sync_action, "action_projector", None)
        if projector is not None:
            result["projection_records"] = projector.records
        if engine is not None:
            result["engine_stats"] = asdict(engine.stats)
            if args.matched_snapshot is not None and args.mode == "predicted":
                predictor = engine._future_latent_predictor
                result["loaded_predictor"] = (
                    predictor._so101_joint_binding
                    if args.context_variant != "visual_only"
                    else predictor._so101_l6_binding
                )
        if client.profile_steps:
            result["step_profile"] = client.step_profile
        # Control stopped, policy worker joined, sink closed, simulator exited.
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        for name, rows in (("ticks.jsonl", ticks), ("events.jsonl", sink.events)):
            (args.output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
        if setup_rows:
            (args.output / "setup_ticks.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in setup_rows)
            )
        if frames:
            from PIL import Image

            folder = args.output / "observations"
            folder.mkdir()
            for index, cameras in frames:
                for name, image in cameras.items():
                    Image.fromarray(image).save(folder / f"step_{index:05d}_{name}.png")
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
