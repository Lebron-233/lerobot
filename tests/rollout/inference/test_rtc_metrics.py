# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract tests for opt-in RTC inference metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference import factory
from lerobot.rollout.inference.factory import RTCInferenceConfig


class _IdentityPipeline:
    """Minimal processor stub used by the real RTC worker thread."""

    steps = ()

    def __call__(self, batch):
        return batch

    def reset(self) -> None:
        pass


class _GatedChunkPolicy:
    """Deterministic policy stub that lets the test observe one in-flight request."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(rtc_training_max_delay=0)
        self.entered = Event()
        self.release = Event()
        self.actions = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)

    def predict_action_chunk(self, batch, inference_delay=0, prev_chunk_left_over=None):
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release the policy request")
        return self.actions.clone()

    def reset(self) -> None:
        pass

    def supports_text_generation(self) -> bool:
        return False


def _wait_for(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _run_one_request(metrics_path: Path | None) -> torch.Tensor:
    policy = _GatedChunkPolicy()
    config = RTCInferenceConfig(
        rtc=RTCConfig(enabled=True, execution_horizon=4, max_guidance_weight=1.0),
        queue_threshold=0,
        metrics_path=metrics_path,
    )
    engine = factory.create_inference_engine(
        config,
        policy=policy,
        preprocessor=_IdentityPipeline(),
        postprocessor=_IdentityPipeline(),
        robot_wrapper=SimpleNamespace(robot_type="mock", action_features={}),
        hw_features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["joint_a.pos", "joint_b.pos"],
            }
        },
        dataset_features={},
        ordered_action_keys=[],
        task="task A",
        # A request held for milliseconds still has a stable one-step wall delay.
        fps=0.01,
        device="cpu",
    )

    # Give the event non-trivial reset and task provenance before inference starts.
    engine.reset()
    assert engine.set_task("task B") is True
    engine.start()
    try:
        # One underflow precedes the request and one occurs while policy inference is in flight.
        assert engine.get_action(None) is None
        engine.notify_observation({"joint_a.pos": 1.0, "joint_b.pos": 2.0})
        engine.resume()
        assert policy.entered.wait(timeout=2.0)
        assert engine.get_action(None) is None

        policy.release.set()
        assert _wait_for(lambda: engine.action_queue is not None and engine.action_queue.qsize() > 0)
        action = engine.get_action(None)
        assert action is not None
        return action
    finally:
        policy.release.set()
        engine.stop()


def test_rtc_metrics_are_opt_in_and_do_not_change_actions(tmp_path: Path, monkeypatch) -> None:
    """The default path performs no sink I/O; enabling it preserves RTC output."""
    assert RTCInferenceConfig().metrics_path is None

    def _unexpected_sink(_path):
        pytest.fail("default RTC configuration must not construct a metrics sink")

    with monkeypatch.context() as scoped:
        scoped.setattr(factory, "JsonlMetricsSink", _unexpected_sink)
        action_without_metrics = _run_one_request(metrics_path=None)

    metrics_path = tmp_path / "rtc.jsonl"
    action_with_metrics = _run_one_request(metrics_path=metrics_path)

    torch.testing.assert_close(action_with_metrics, action_without_metrics, rtol=0, atol=0)
    assert metrics_path.is_file()


def test_rtc_chunk_request_jsonl_captures_timing_and_provenance(tmp_path: Path) -> None:
    metrics_path = tmp_path / "nested" / "rtc.jsonl"
    action = _run_one_request(metrics_path)
    assert torch.equal(action, torch.tensor([2.0, 3.0]))

    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    chunk_events = [event for event in events if event.get("event") == "chunk_request"]
    assert len(chunk_events) == 1
    event = chunk_events[0]

    assert event["schema_version"] == 1
    assert event["backend"] == "rtc"
    assert event["request_id"] >= 0
    assert event["status"] == "merged"

    assert event["task"] == "task B"
    assert event["task_changed"] is True
    assert event["reset_epoch_at_request"] == 1
    assert event["reset_epoch_at_completion"] == 1

    assert event["queue_size_at_request"] == 0
    assert event["queue_size_after"] == 4
    assert event["next_action_index"] == 0
    assert event["estimated_latency_s"] == 0.0
    assert event["estimated_delay_steps"] == 0
    assert event["measured_latency_s"] > 0
    assert event["measured_delay_steps_wall"] == 1
    assert event["consumed_steps_during_request"] == 0
    assert event["discarded_prefix_steps"] == 1

    assert event["underflow_total"] == 2
    assert event["underflow_during_request"] == 1
    assert event["latency_p50_s"] >= 0
    assert event["latency_p90_s"] >= event["latency_p50_s"]

    outer_stage_fields = {
        "observation_preparation_s",
        "preprocessor_s",
        "rtc_prefix_preparation_s",
        "policy_total_s",
        "postprocessor_s",
        "total_chunk_s",
    }
    policy_stage_fields = {"vision_encode_s", "prefix_prefill_s", "flow_matching_s"}
    for field in outer_stage_fields:
        assert isinstance(event[field], int | float)
        assert event[field] >= 0
    for field in policy_stage_fields:
        assert event[field] is None or event[field] >= 0
