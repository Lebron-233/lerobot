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
        ]
        try:
            self.process = subprocess.Popen(
                command,
                pass_fds=(child.fileno(),),
                env=environment,
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
        self.metadata = self.receive(180)["metadata"]
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
                    self.receive()
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
) -> dict:
    """One notify/get/real step per tick; underflow holds the previous sent target."""
    from lerobot.utils.cycle_timer import CycleTimer

    last_action = validate_observation(packet)
    timer = CycleTimer(FPS, records_data=False)
    origin = time.perf_counter()
    total_reward = 0.0
    for index in range(max_steps):
        started = time.perf_counter()
        # This observes missed slots against a fixed origin; pacing itself uses
        # CycleTimer. Neither get indices nor simulation time are fast-forwarded.
        if realtime and started - (origin + index / FPS) >= 1 / FPS:
            raise ContractError("lost_control_slot_before_tick")
        timer.tick()
        raw = decode_observation(packet)
        row = {
            "tick": index,
            "episode_id": packet["episode_id"],
            "sim_step": packet["step"],
            "started_at_s": started,
            "snapshot_ready_at_s": packet["snapshot_ready_at_s"],
            "received_at_s": packet.get("received_at_s"),
            "state": [raw[k] for k in SCALAR_KEYS],
            "camera_frames": packet["camera_frames"],
            "dispatch": "not_sent",
        }
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
        response = client.step(packet, last_action)
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
        if realtime:
            timer.wait()
    return {
        "status": "censored_step_limit",
        "success": None,
        "timeout": False,
        "steps": max_steps,
        "reward": total_reward,
    }


def load_runtime(
    mode: str, device: str, predictor_path: Path | None, seed: int, sink: MemoryMetrics, raw: dict
):
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
    snapshot = snapshot_download(POLICY_REPO_ID, revision=POLICY_REVISION, local_files_only=True)
    config = PreTrainedConfig.from_pretrained(snapshot, local_files_only=True)
    policy, preprocessor, postprocessor = _load_frozen_future_latent_runtime(config, device=device)
    policy.to(device).eval()
    robot = ThreadSafeRobot(SnapshotRobot(raw))
    features = hardware_features()
    if mode == "sync":

        def action(observation: dict) -> list[float]:
            batch = build_dataset_frame(features, observation, prefix="observation")
            batch = prepare_observation_for_inference(batch, torch.device(device), TASK, robot.robot_type)
            batch["task"] = [TASK]
            with torch.inference_mode():
                result = postprocessor(policy.select_action(preprocessor(batch)))
            return result.detach().cpu().reshape(-1).tolist()

        return None, action
    predictor = (
        load_frozen_future_latent_predictor(predictor_path, device=device) if mode == "predicted" else None
    )
    engine = PredictiveAsyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot,
        hw_features=features,
        task=TASK,
        fps=FPS,
        device=device,
        queue_threshold=30,
        latency_quantile=0.9,
        latency_window=50,
        delay_safety_margin_steps=1,
        min_prediction_delay=1,
        max_prediction_delay=8,
        committed_guard_steps=2,
        max_late_steps=2,
        context_mode=mode,
        future_latent_predictor=predictor,
        fallback_mode="identity",
        metrics_sink=sink,
    )
    return engine, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "sync", "identity", "predicted"), required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=1701)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= (30 if args.mode == "smoke" else 750):
        parser.error("This minimal phase permits at most 30 smoke / 750 episode steps")
    if args.mode == "predicted" and args.predictor is None:
        parser.error("predicted requires the frozen portable --predictor")
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
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    client = EnvClient(args.sim_python, args.assets_root, args.leisaac_root, args.device)
    sink, ticks, engine, result = MemoryMetrics(), [], None, {}
    try:
        client.start()
        packet = client.reset(args.seed)
        if args.mode != "smoke":
            engine, sync_action = load_runtime(
                args.mode, args.device, args.predictor, args.policy_seed, sink, decode_observation(packet)
            )
        else:
            sync_action = None
        if engine is not None:
            prepare_engine(engine, decode_observation(packet))
            packet = client.reset(args.seed)
        if args.mode != "smoke":
            from lerobot.utils.random_utils import set_seed

            # Predictor construction and startup consume RNG differently by
            # mode. Seed the measured policy stream only after both are over.
            set_seed(args.policy_seed)
        if engine is not None:
            engine.resume()
        result = drive_episode(
            client,
            packet,
            max_steps=args.max_steps,
            ticks=ticks,
            engine=engine,
            sync_action=sync_action,
            realtime=args.mode != "sync",
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
            metrics_closed=sink.closed,
            subprocess_returncode=None if client.process is None else client.process.returncode,
        )
        if engine is not None:
            result["engine_stats"] = asdict(engine.stats)
        # Control stopped, policy worker joined, sink closed, simulator exited.
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        for name, rows in (("ticks.jsonl", ticks), ("events.jsonl", sink.events)):
            (args.output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "technical_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
