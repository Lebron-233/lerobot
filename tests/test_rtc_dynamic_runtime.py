"""No checkpoints or Env: dynamic weights, CUDA replay and owned request evidence."""

import copy
import os
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from rtc_dynamic_runtime import (  # noqa: E402
    DynamicPrefixRTC,
    DynamicRTCGraphRuntime,
    RTCPrefixOwner,
    RTCRequest,
    config,
    g,
)
from rtc_execution_queue import RequestStamp  # noqa: E402

from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402
from tests.test_rtc_graph_runtime import SyntheticModel  # noqa: E402


@pytest.mark.parametrize("delay", range(9))
@pytest.mark.parametrize("length", [1, 2, 3, 4, 7, 8, 9, 10, 11, 20, 29, 30, 50])
def test_dynamic_padding_and_full_jacobian_equal_original(delay, length):
    p, native = DynamicPrefixRTC(config(), "cpu"), RTCProcessor(config())
    ptr = p.weights.data_ptr()
    p.select(delay, length)
    expected_weights = native.get_prefix_weights(delay, min(length, 10), 50)
    torch.testing.assert_close(p.weights, expected_weights, rtol=0, atol=0)
    assert p.weights.data_ptr() == ptr
    x = torch.linspace(-0.2, 0.3, 1600).reshape(1, 50, 32)
    prefix = torch.linspace(-0.1, 0.4, length * 7).reshape(length, 7)
    padded = torch.zeros(50, 7)
    padded[:length] = prefix
    common = {
        "x_t": x,
        "time": 0.5,
        "original_denoise_step_partial": lambda z: 0.2 * z + 0.03 * torch.sin(z),
        "execution_horizon": 10,
    }
    a = native.denoise_step(**common, inference_delay=delay, prev_chunk_left_over=prefix)
    b = p.denoise_step(**common, inference_delay=3, prev_chunk_left_over=padded)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_snapshot_has_no_shared_storage_and_is_frozen():
    observation = {"index": 20, "returned_at": 0.0, "pixels": torch.zeros(2, 2)}
    prefix = torch.arange(70, dtype=torch.float32).reshape(10, 7)
    ctx = RTCRequest.snapshot(RequestStamp(0, 2, 20, 7), observation, prefix, 1)
    prefix.zero_()
    observation["pixels"].fill_(1)
    assert ctx.prefix.sum() > 0 and ctx.observation["pixels"].sum() == 0
    with pytest.raises(FrozenInstanceError):
        ctx.prefix_source = 2
    with pytest.raises(ValueError, match="Insufficient"):
        RTCRequest.snapshot(RequestStamp(0, 2, 20, 8), observation, prefix[:3], 1)
    with pytest.raises(ValueError, match="identity"):
        RTCRequest.snapshot(RequestStamp(0, 2, 21, 0), observation, prefix, 1)


@pytest.mark.parametrize("close_early", [False, True])
def test_owner_passes_exact_tail_and_keeps_actual_trim(tmp_path, close_early):
    seen, threads = [], []

    def predict(ctx):
        assert isinstance(ctx, RTCRequest)
        seen.append(ctx)
        x = (torch.arange(50, dtype=torch.float32)[:, None] + 100 * ctx.stamp.request_id).repeat(1, 7)
        return {"original": x, "processed": x * 10}

    spec = copy.deepcopy(g.manifest()["rows"][0])
    calls, budget = g.e.Calls(tmp_path / "calls.jsonl"), g.old.Budget()
    budget.begin_episode()
    owner = RTCPrefixOwner(
        predict, calls, spec, budget, on_owner_close=lambda: threads.append(threading.get_ident())
    )
    owner.submit({"index": 0, "returned_at": time.perf_counter()}, 0)
    owner.receive(block=True)
    for _ in range(20):
        owner.queue.pop()
    owner.submit({"index": 20, "returned_at": time.perf_counter()}, 7)
    for _ in range(2):
        owner.queue.pop()
    if close_early:
        owner.close()
        assert not owner.rows[-1]["decision"]["accepted"]
    else:
        row = owner.receive(block=True)
        assert row["decision"]["actual_delay"] == row["decision"]["source_row"] == 2
        assert seen[-1].stamp.expected_delay == 7
        assert torch.equal(seen[-1].prefix, owner.outputs[1]["original"][20:])
        command = owner.queue.pop()
        assert command["action_index"] == 22 and command["source_row"] == 2
        owner.close()
    calls.close()
    assert threads == [owner.thread.ident] and not owner.thread.is_alive()
    assert RTCPrefixOwner.receive is g.old.InferenceOwner.receive
    assert RTCPrefixOwner._serve is g.old.InferenceOwner._serve


@pytest.mark.skipif(os.environ.get("RUN_RTC_GRAPH_CUDA_TESTS") != "1", reason="explicit CUDA opt-in")
def test_dynamic_graph_all_delay_and_lengths_without_recapture():
    model = SyntheticModel()
    original, processor = model.sample_actions, model.rtc_processor
    images = [torch.ones((1, 4, 8), device="cuda") for _ in range(2)]
    masks = [torch.ones((1, 4), dtype=torch.bool, device="cuda") for _ in range(2)]
    tokens = torch.ones((1, 11), dtype=torch.long, device="cuda")
    token_mask = torch.ones_like(tokens, dtype=torch.bool)
    state = torch.zeros((1, 32), device="cuda")
    noise = torch.linspace(-0.5, 0.5, 1600, device="cuda").reshape(1, 50, 32)
    with DynamicRTCGraphRuntime(model) as rt:
        rt.mode = "graph"
        rt(
            images,
            masks,
            tokens,
            token_mask,
            state,
            noise,
            inference_delay=0,
            execution_horizon=10,
            prev_chunk_left_over=None,
        )
        ptr = rt.cached.weights.data_ptr()
        for length in [1, 2, 3, 7, 8, 9, 10, 11, 29, 30, 50]:
            for delay in range(min(length, 8) + 1):
                prefix = torch.full((length, 7), 0.01 * length - 0.015 * delay, device="cuda")
                outputs = []
                for mode in ("native_eager", "graph"):
                    rt.mode = mode
                    outputs.append(
                        rt(
                            images,
                            masks,
                            tokens,
                            token_mask,
                            state,
                            noise,
                            inference_delay=delay,
                            execution_horizon=10,
                            prev_chunk_left_over=prefix,
                        ).clone()
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                assert rt.cached.weights.data_ptr() == ptr and len(rt.graphs) == 2
        rt.mode = "graph"
        with pytest.raises(ValueError, match="signature"):
            rt(
                images,
                masks,
                tokens[:, :10],
                token_mask[:, :10],
                state,
                noise,
                inference_delay=0,
                execution_horizon=10,
            )
        assert len(rt.graphs) == 2
    assert rt.receipt["graphs_released"] and rt.receipt["processor_restored"]
    assert model.sample_actions == original and model.rtc_processor is processor


def test_fixed_input_population_and_optional_cpu_arrays():
    import libero_rtc_dynamic as run
    from rtc_dynamic_runtime import cpu_optional

    rows = run.input_schedule()
    assert len(rows) == 482
    pairs = {(r["delay"], r["length"]) for r in rows if r["length"]}
    assert pairs == {(d, n) for d in range(9) for n in range(d + 2, 31)}
    assert len(pairs) == 225 and len(run.manifest("feedback")["rows"]) == 30
    assert cpu_optional(None) is None
    x = torch.ones(50)
    y = cpu_optional(x)
    assert y.shape == x.shape and torch.equal(x, y) and x.data_ptr() != y.data_ptr()


class FakeDynamicPredictor:
    def __init__(self, policy, pre, post, spec):
        self.spec, self.requests, self.receipt, self.guided = spec, 0, None, 0
        self.owner = None

    def __call__(self, ctx):
        self.owner = threading.get_ident()
        began = time.perf_counter()
        time.sleep(0.065)
        self.requests += 1
        stamp = ctx.stamp
        prefix = ctx.prefix
        used = self.spec["arm"] == "rtc_async" and prefix is not None and len(prefix) > 0
        length = len(prefix) if used else 0
        delay = stamp.expected_delay if used else 0
        x = (torch.arange(50, dtype=torch.float32)[:, None] + 100 * self.requests).repeat(1, 7)
        full, noise = torch.zeros(1, 50, 32), torch.zeros(1, 50, 32)
        full[0, :, :7] = x
        inputs = (
            torch.zeros(1, 4, 8),
            torch.zeros(1, 4, 8),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 11, dtype=torch.long),
            torch.ones(1, 11, dtype=torch.bool),
            torch.zeros(1, 32),
            noise,
        )
        padded = None
        if used:
            inputs += (prefix.clone(),)
            padded = torch.zeros(50, 7)
            padded[:length] = prefix
        self.guided += int(used)
        return {
            "original": x,
            "processed": x * 10,
            "full": full,
            "noise": noise,
            "inputs": inputs,
            "context_prefix": None if prefix is None else prefix.clone(),
            "context_prefix_source": ctx.prefix_source,
            "context_stamp": {
                "epoch": stamp.epoch,
                "request_id": stamp.request_id,
                "observation_index": stamp.observation_index,
                "expected_delay": stamp.expected_delay,
            },
            "rtc_weights": RTCProcessor(config()).get_prefix_weights(delay, min(length, 10), 50)
            if used
            else None,
            "rtc_padded_prefix": padded,
            "owner_thread": self.owner,
            "model_started_at": began,
            "model_returned_at": time.perf_counter(),
            "input_fingerprint": g.observation_fingerprint(ctx.observation),
            "vision_encodes": 1,
            "capture_created": self.requests == 1,
            "rtc_metadata": {
                "mode": "graph",
                "guided": used,
                "replays": 1,
                "expected_delay": delay,
                "prefix_length": length,
                "captured_vjps_per_replay": 10 if used else 0,
                "capture_created": self.requests == 1,
            },
        }

    def close(self):
        assert self.owner == threading.get_ident()
        self.receipt = {
            "sampler_restored": True,
            "processor_restored": True,
            "graphs_released": True,
            "owner": self.owner,
            "close_thread": self.owner,
            "closed_at": time.perf_counter(),
            "requests": self.requests,
            "rgb_encodings": self.requests,
            "live_requests": self.requests,
            "explicit_noise_draws": self.requests,
            "replays": {"no_prefix": self.requests - self.guided, "guided": self.guided},
            "captures": [
                {
                    "status": "captured",
                    "owner": self.owner,
                    "guided": b,
                    "setup": 1,
                    "warmup": 3,
                    "capture": 1,
                    "captured_steps": 10,
                    "captured_vjps": 10 if b else 0,
                    "projection_shapes": [[1, 50, 32]] * 10,
                    "seconds": 0.01,
                }
                for b in (False, True)
            ],
        }


@pytest.mark.parametrize("arm", ["serialized", "aligned_async", "rtc_async"])
def test_full_context_episode_original_trim_and_tamper_audit(tmp_path, monkeypatch, arm):
    import audit_libero_rtc_dynamic as audit
    import libero_rtc_dynamic as run

    from tests.test_libero_graph_feedback import FakeSession

    monkeypatch.setattr(run, "DynamicRTCPredictor", FakeDynamicPredictor)
    monkeypatch.setattr(run.e, "NativeSession", FakeSession)

    def observation(raw, language, index, returned):
        image = torch.full((2, 2, 3), index, dtype=torch.uint8)
        row = {
            "index": index,
            "returned_at": returned,
            "state": torch.zeros(1, 8),
            "eef_quaternion_xyzw": torch.zeros(4),
            "raw_eef_position": torch.zeros(3),
            "raw_gripper_qpos": torch.zeros(2),
            "raw_pixels": {"image": image, "image2": image.clone()},
            "worker_observation": {"image": image, "state_0": float(index)},
        }
        return None, None, row

    monkeypatch.setattr(run.e, "observation", observation)
    spec = copy.deepcopy(next(s for s in run.manifest("feedback")["rows"] if s["arm"] == arm))
    spec["limits"]["measurement"] = 44
    calls, budget = run.e.Calls(tmp_path / "calls.jsonl"), run.Budget()
    rec, initial = run.episode(spec, tmp_path, None, None, None, None, budget, calls)
    calls.close()
    assert rec["status"] == "completed", rec["first_failure"]
    arrays = run.load(tmp_path / f"episode_{spec['ordinal']:03d}/arrays.pt")
    row = audit.inspect_episode(spec, rec, arrays, initial)
    assert row["actions"] == 44
    assert row["guided_requests"] == (len(rec["requests"]) - 1 if arm == "rtc_async" else 0)
    damaged = copy.deepcopy(arrays)
    damaged["outputs"][2]["context_prefix"][0, 0] += 1
    with pytest.raises(ValueError, match="prefix"):
        audit.inspect_episode(spec, rec, damaged, initial)
    damaged = copy.deepcopy(arrays)
    damaged["control"]["dispatches"][23]["source_row"] += 1
    with pytest.raises(ValueError, match="source row"):
        audit.inspect_episode(spec, rec, damaged, initial)
