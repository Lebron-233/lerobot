"""Interface fixtures only: these tests produce no LeIsaac task evidence."""

from __future__ import annotations

import math
import multiprocessing
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from eval_leisaac_so101 import (
    EnvClient,
    MemoryMetrics,
    SnapshotRobot,
    decode_observation,
    drive_episode,
    prepare_engine,
    stop_engine,
)
from leisaac_so101_contract import (
    IMAGE_BYTES,
    JOINT_NAMES,
    PROFILE,
    SCALAR_KEYS,
    ContractError,
    action_to_radians,
    hardware_features,
    joint_indices,
    observation_packet,
    state_from_radians,
    validate_observation,
    validate_step,
)
from leisaac_so101_env_server import serve


def packet(*, step=0, episode_id=1, action=None):
    return observation_packet(
        measured=action_to_radians(action or [0, 0, 0, 0, 0, 50]),
        actual_names=JOINT_NAMES,
        images={"front": bytes([21]) * IMAGE_BYTES, "wrist": bytes([83]) * IMAGE_BYTES},
        camera_frames={"front": step + 1, "wrist": step + 1},
        episode_id=episode_id,
        step=step,
        snapshot_ready_at_s=time.perf_counter(),
    )


@pytest.mark.parametrize(
    "action", [[90, -90, 45, -30, 120, 50], [-110, -100, -100, -95, -160, 0], [110, 100, 90, 95, 160, 100]]
)
def test_units_and_shuffled_joint_order(action):
    radians = action_to_radians(action)
    assert radians[0] == pytest.approx(math.radians(action[0]))  # 90 degrees is not 99.
    assert radians[-1] == pytest.approx(math.radians(-10 + 1.1 * action[-1]))
    shuffled = (2, 0, 5, 4, 1, 3)
    result = state_from_radians([radians[i] for i in shuffled], [JOINT_NAMES[i] for i in shuffled])
    assert result == pytest.approx(action)


@pytest.mark.parametrize("action", [[111, 0, 0, 0, 0, 50], [0, 0, 0, 0, 0, 101], [math.nan] * 6, [0] * 5])
def test_bad_actions_fail_instead_of_clipping(action):
    with pytest.raises(ContractError):
        action_to_radians(action)


def test_measured_state_is_not_clipped_and_joint_names_are_required():
    assert state_from_radians([math.pi] * 6, JOINT_NAMES)[0] == pytest.approx(180)
    with pytest.raises(ContractError):
        joint_indices(["gripper"] * 6)


def test_observation_bytes_and_camera_provenance():
    message = packet(action=[90, 0, 0, 0, 0, 50])
    raw = decode_observation(message)
    assert raw[SCALAR_KEYS[0]] == pytest.approx(90)
    assert raw["top"].shape == (480, 640, 3)
    assert raw["top"].dtype == np.uint8
    assert raw["top"][0, 0, 0] == 21 and raw["wrist"][0, 0, 0] == 83
    assert message["camera_sources"]["top"] == "front"
    message["images"]["front"] = b"bad"
    with pytest.raises(ContractError):
        validate_observation(message)


def test_step_identity_frame_update_and_auto_reset_are_distinct():
    before, after = packet(), packet(step=1)
    validate_step(before, after, terminal=False)
    after["camera_frames"] = {"front": 0, "wrist": 0}
    with pytest.raises(ContractError, match="Camera did not advance"):
        validate_step(before, after, terminal=False)
    validate_step(before, after, terminal=True)
    after["episode_id"] += 1
    with pytest.raises(ContractError, match="episode/step"):
        validate_step(before, after, terminal=True)


class FixtureServerEnv:
    metadata = {"profile": PROFILE, "fixture_only": True}

    def reset(self, seed, episode_id):
        return packet(episode_id=episode_id)

    def step(self, action, episode_id, step):
        return {
            "observation": packet(step=step, episode_id=episode_id, action=action),
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "post_reset_observation": True,
        }


def test_real_local_connection_reset_step_close():
    parent, child = multiprocessing.Pipe()
    process = multiprocessing.get_context("fork").Process(target=serve, args=(child, FixtureServerEnv()))
    process.start()
    child.close()
    try:
        assert parent.poll(3)
        assert parent.recv()["metadata"]["fixture_only"]
        parent.send({"op": "reset", "seed": 20260907})
        assert parent.poll(3)
        first = parent.recv()["observation"]
        parent.send(
            {"op": "step", "episode_id": first["episode_id"], "step": 0, "action": [0, 0, 0, 0, 0, 50]}
        )
        assert parent.poll(3)
        result = parent.recv()
        assert result["terminated"] and result["post_reset_observation"]
        parent.send({"op": "reset", "seed": 20260908})
        assert parent.poll(3)
        assert parent.recv()["observation"]["episode_id"] == first["episode_id"] + 1
        parent.send({"op": "close"})
        assert parent.poll(3)
        assert parent.recv() == {"closed": True}
        process.join(3)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(3)


def test_start_failure_closes_connection(tmp_path):
    client = EnvClient(tmp_path / "missing-python", tmp_path, tmp_path, "cpu")
    with pytest.raises(FileNotFoundError):
        client.start()
    client.close()
    assert client.connection.closed


def test_dual_interpreter_client_wire_and_shutdown(tmp_path, monkeypatch):
    # This peer is a transport fixture, not an alternative research simulator.
    # Set LEISAAC_TEST_PYTHON to exercise an actual 3.11 child from 3.12 pytest.
    interpreter = Path(os.environ.get("LEISAAC_TEST_PYTHON", sys.executable))
    folder = str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async")
    program = textwrap.dedent(f"""
        import argparse, sys, time
        from multiprocessing.connection import Connection
        sys.path.insert(0, {folder!r})
        from leisaac_so101_contract import *
        from leisaac_so101_env_server import serve
        class Peer:
            metadata = {{'profile': PROFILE, 'fixture_only': True, 'python': sys.version}}
            def reset(self, seed, episode_id):
                return observation_packet(measured=[0.0]*6, actual_names=JOINT_NAMES,
                    images={{'front': bytes(IMAGE_BYTES), 'wrist': bytes(IMAGE_BYTES)}},
                    camera_frames={{'front': 1, 'wrist': 1}}, episode_id=episode_id,
                    step=0, snapshot_ready_at_s=time.perf_counter())
        parser = argparse.ArgumentParser()
        parser.add_argument('--fd', type=int)
        args, _ = parser.parse_known_args()
        print('transport fixture only', flush=True)
        serve(Connection(args.fd), Peer())
    """)
    popen = subprocess.Popen

    def start_peer(command, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL  # A hidden license prompt must not wait for input.
        return popen([command[0], "-c", program, *command[2:]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", start_peer)
    client = EnvClient(interpreter, tmp_path, tmp_path, "cpu")
    try:
        client.start()
        assert client.metadata["fixture_only"]
        raw = decode_observation(client.reset(20260907))
        assert raw["top"].shape == (480, 640, 3)
        assert raw["gripper.pos"] == pytest.approx(10 / 1.1)
    finally:
        client.close()
    assert client.process.returncode == 0 and client.cleanup_error is None
    assert not client.log_reader.is_alive()
    assert b"transport fixture only" in b"".join(client.logs)


def test_control_order_underflow_and_terminal_observation_not_reused():
    operations, actions = [], []

    class Engine:
        failed = False

        def notify_observation(self, raw):
            operations.append("notify")

        def get_action(self, unused):
            operations.append("get")
            return None

    class Client:
        def step(self, previous, action):
            operations.append("step")
            actions.append(action)
            # A reset observation must not go through decode/notify again.
            successor = packet(step=previous["step"] + 1)
            successor["images"] = {"invalid_after_reset": b""}
            return {
                "observation": successor,
                "reward": 0,
                "terminated": True,
                "truncated": True,
                "post_reset_observation": True,
            }

    ticks = []
    result = drive_episode(Client(), packet(), max_steps=3, ticks=ticks, engine=Engine(), realtime=False)
    assert operations == ["notify", "get", "step"]
    assert actions[0] == pytest.approx([0, 0, 0, 0, 0, 50])
    assert result["success"] is True and result["timeout"] is True
    assert ticks[0]["underflow_hold"] and len(ticks) == 1


def test_lost_control_slot_is_technical_failure_not_slow_realtime():
    class SlowClient:
        def step(self, previous, action):
            time.sleep(0.08)
            return {
                "observation": packet(step=1),
                "reward": 0,
                "terminated": False,
                "truncated": False,
                "post_reset_observation": False,
            }

    ticks = []
    with pytest.raises(ContractError, match="lost_control_slot"):
        drive_episode(SlowClient(), packet(), max_steps=1, ticks=ticks)
    assert ticks[0]["dispatch"] == "completed"


class FixturePipeline:
    steps = ()

    def __init__(self, post=False):
        self.post = post

    def reset(self):
        pass

    def __call__(self, value):
        if self.post:
            result = value * 10
            result[..., 5] += 50
            return result
        value = dict(value)
        for old, new in (("top", "camera1"), ("wrist", "camera2")):
            value[f"observation.images.{new}"] = value.pop(f"observation.images.{old}")
        return value


class FixturePolicy:
    """Fixed CPU tokens/actions. Binding below exists only to test the new seam."""

    def __init__(self):
        self.config = SimpleNamespace(
            type="smolvla",
            max_state_dim=32,
            max_action_dim=32,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
            empty_cameras=0,
            use_peft=False,
            rtc_config=None,
            image_features={
                f"observation.images.camera{i}": SimpleNamespace(shape=(3, 480, 640)) for i in (1, 2)
            },
            robot_state_feature=SimpleNamespace(shape=(6,)),
            action_feature=SimpleNamespace(shape=(6,)),
        )
        self.model = self
        self.entered, self.release = Event(), Event()
        self.block_future = False

    def reset(self):
        assert not self.entered.is_set() or self.release.is_set()

    def prepare_images(self, batch):
        return [torch.zeros(1, 3, 2, 2)] * 2, [torch.ones(1, dtype=torch.bool)] * 2

    def prepare_state(self, batch):
        return torch.nn.functional.pad(batch["observation.state"], (0, 26))

    def encode_image_tokens(self, images, masks):
        return (torch.zeros(1, 2, 960),) * 2, (torch.ones(1, 2, dtype=torch.bool),) * 2

    def predict_action_chunk(self, batch, **kwargs):
        future = "future_image_tokens" in kwargs
        if future and self.block_future:
            self.block_future = False
            self.entered.set()
            assert self.release.wait(3)
        return torch.full((1, 50, 6), 0.02 if future else 0.01)


def test_production_predicted_prefix_late_discard_and_joined_reset():
    from lerobot.policies.smolvla.future_latent import FutureLatentPrediction
    from lerobot.policies.smolvla.future_latent_checkpoint import (
        POLICY_REVISION,
        VLM_REVISION,
        _bind_frozen_candidate,
    )
    from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    seen = []

    class Predictor:
        def __call__(self, tokens, masks, actions, committed, state, delay):
            seen.append((actions.clone(), committed.clone(), state.clone()))
            return FutureLatentPrediction(tuple(torch.zeros_like(t) for t in tokens), torch.zeros(1))

    def wait(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.005)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    policy, pre, post, sink = FixturePolicy(), FixturePipeline(), FixturePipeline(True), MemoryMetrics()
    _bind_frozen_candidate(policy, pre, post, policy_revision=POLICY_REVISION, vlm_revision=VLM_REVISION)
    raw = decode_observation(packet())
    engine = PredictiveAsyncInferenceEngine(
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        robot_wrapper=ThreadSafeRobot(SnapshotRobot(raw)),
        hw_features=hardware_features(),
        task="fixture",
        fps=30,
        device="cpu",
        queue_threshold=49,
        context_mode="predicted",
        future_latent_predictor=Predictor(),
        min_prediction_delay=1,
        max_prediction_delay=8,
        metrics_sink=sink,
    )
    try:
        prepare_engine(engine, raw, timeout=10)
        engine.resume()
        engine.notify_observation(raw)
        wait(lambda: engine.queue.qsize() == 50 and not engine._request_in_flight)
        assert engine.get_action(None)[0].item() == pytest.approx(0.1)
        policy.block_future = True
        engine.notify_observation(raw)
        assert policy.entered.wait(3)
        plan = engine.queue.plan_snapshot()
        assert plan is not None
        for _ in range(plan.planned_delay_steps + 1):
            assert engine.get_action(None)[0].item() == pytest.approx(0.1)
        policy.release.set()
        wait(lambda: engine.stats.deadline_misses == 1)
        assert engine.queue.plan_snapshot() is None
        assert engine.get_action(None)[0].item() == pytest.approx(0.1)  # No partial late takeover.
        actions, committed, state = seen[-1]
        torch.testing.assert_close(actions[committed], torch.full_like(actions[committed], 0.01))
        assert state.shape == (1, 32)
        assert state[0, 5].item() == pytest.approx(50)
    finally:
        policy.release.set()
        stop_engine(engine)
        torch.set_num_threads(old_threads)
    assert sink.closed and engine._worker is None
    assert engine.queue.qsize() == 0
