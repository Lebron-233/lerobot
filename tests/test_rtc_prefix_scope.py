"""Synthetic-only prefix scope tests; no stored images, checkpoints or native Env."""

import copy
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_rtc_prefix_scope as a  # noqa: E402
import libero_rtc_prefix_scope as r  # noqa: E402
from rtc_commitment_owner import CommittedPrefixOwner  # noqa: E402
from rtc_dynamic_runtime import RTCPrefixOwner, config  # noqa: E402
from rtc_execution_queue import RTCExecutionQueue  # noqa: E402
from rtc_prefix_scope_runtime import (  # noqa: E402
    COMMITTED,
    VARIANTS,
    PrefixOnlyGraphRuntime,
    PrefixOnlyWeights,
    PrefixScopeOwner,
    prefix_config,
    schedule_for,
)

from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402
from tests.test_libero_graph_feedback import FakeSession  # noqa: E402
from tests.test_rtc_commitment_owner import PausedPredictor  # noqa: E402
from tests.test_rtc_dynamic_runtime import FakeDynamicPredictor  # noqa: E402
from tests.test_rtc_graph_runtime import SyntheticModel  # noqa: E402


def spec_for(variant):
    return copy.deepcopy(next(s for s in r.manifest("feedback")["rows"] if s["variant"] == variant))


@pytest.mark.parametrize("delay", range(9))
@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 10, 11, 20, 30, 50])
def test_native_binary_weights_and_full_vjp_exact(delay, length):
    native = RTCProcessor(prefix_config())
    cached = PrefixOnlyWeights(prefix_config(), "cpu")
    ptr = cached.weights.data_ptr()
    cached.select(delay, length)
    torch.testing.assert_close(cached.weights, native.get_prefix_weights(delay, min(length, 10), 50), rtol=0, atol=0)
    torch.testing.assert_close(cached.weights, (torch.arange(50) < min(delay, length)).float(), rtol=0, atol=0)
    assert cached.weights.data_ptr() == ptr
    x = torch.linspace(-.2, .3, 1600).reshape(1, 50, 32)
    prefix = torch.linspace(-.1, .4, length*7).reshape(length, 7)
    padded = torch.zeros(50, 7)
    padded[:length] = prefix
    common = {"x_t": x, "time": .5, "execution_horizon": 10,
              "original_denoise_step_partial": lambda z: .2*z+.03*torch.sin(z)}
    p = native.denoise_step(**common, inference_delay=delay, prev_chunk_left_over=prefix)
    q = cached.denoise_step(**common, inference_delay=3, prev_chunk_left_over=padded)
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_zero_tail_targets_do_not_delete_cross_row_jacobian():
    p = RTCProcessor(prefix_config())
    x = torch.zeros(1, 50, 32)
    prefix = torch.ones(30, 7)
    # v[0] depends on x[5]; prefix-only output error must retain its VJP into row5.
    def denoise(z):
        return .25*torch.roll(z, shifts=-5, dims=1)
    result = p.denoise_step(x_t=x, time=.5, inference_delay=1, execution_horizon=10,
                           prev_chunk_left_over=prefix, original_denoise_step_partial=denoise)
    assert p.get_prefix_weights(1, 10, 50)[5] == 0
    torch.testing.assert_close(result[0, 0, :7], torch.full((7,), -2.), rtol=0, atol=0)
    torch.testing.assert_close(result[0, 5, :7], torch.full((7,), .25), rtol=0, atol=0)
    assert torch.count_nonzero(result[..., 7:]) == 0


def test_exp_constructor_and_original_queue_not_weakened():
    with pytest.raises(ValueError, match="configuration"):
        PrefixOnlyGraphRuntime(SimpleNamespace(config=SimpleNamespace(num_steps=10, chunk_size=50, rtc_config=config())))
    with pytest.raises(ValueError):
        PrefixOnlyWeights(config(), "cpu")
    p = PrefixOnlyWeights(prefix_config(), "cpu")
    for d, length in ((True, 30), (9, 30), (1, 0), (1, 51)):
        with pytest.raises(ValueError):
            p.select(d, length)
    with pytest.raises(ValueError):
        p.get_prefix_weights(4, 10, 50)
    assert PrefixScopeOwner.receive is CommittedPrefixOwner.receive
    assert PrefixScopeOwner._serve is RTCPrefixOwner._serve
    assert not torch.cuda.is_initialized()


def test_frozen_schedule_no_seed_or_identity_selection():
    assert len(r.manifest("inputs")["rows"]) == 482
    assert {(s["delay"], s["length"]) for s in r.manifest("inputs")["rows"] if s["length"]} == {
        (d, length) for d in range(9) for length in range(d+2, 31)}
    rows = r.manifest("feedback")["rows"]
    assert len(rows) == 50
    for pair in range(10):
        selected = [s for s in rows if s["pair_index"] == pair]
        shift = pair % 5
        assert [s["variant"] for s in selected] == list(VARIANTS[shift:]+VARIANTS[:shift])
        assert len({s["environment_seed"] for s in selected}) == len({s["policy_seed"] for s in selected}) == 1
    with pytest.raises(ValueError):
        schedule_for("unknown")


@pytest.mark.parametrize("variant", sorted(COMMITTED))
@pytest.mark.parametrize("late", [False, True])
def test_all_committed_controls_use_same_count_and_original_trim(tmp_path, variant, late):
    event = threading.Event()
    if not late:
        event.set()
    budget = r.Budget()
    budget.begin_episode()
    calls = r.e.Calls(tmp_path / "calls.jsonl")
    owner = PrefixScopeOwner(PausedPredictor(event), calls, spec_for(variant), budget)
    timer = None
    try:
        owner.submit({"index": 0, "returned_at": time.perf_counter()}, 0)
        owner.receive(block=True)
        for _ in range(20):
            owner.queue.pop()
        owner.submit({"index": 20, "returned_at": time.perf_counter()}, 3)
        for _ in range(3):
            assert owner.receive() is None
            assert owner.queue.pop()["request_id"] == 1
        if late:
            timer = threading.Timer(.05, event.set)
            timer.start()
        got = owner.receive()
        assert got["decision"]["actual_delay"] == got["decision"]["source_row"] == 3
        assert got["commitment"]["conditioned_steps"] == (0 if variant == "committed_base" else 3)
        assert got["commitment"]["declared_steps"] == 3
        assert type(owner.queue) is RTCExecutionQueue and owner.queue.next_index == 23
        if late:
            assert got["commitment"]["late_compute_wait_s"] > 0
    finally:
        event.set()
        if timer:
            timer.join()
        owner.close()
        calls.close()
    assert not owner.thread.is_alive()


class FakeScopePredictor(FakeDynamicPredictor):
    def __call__(self, ctx):
        out = super().__call__(ctx)
        out["constraint_schedule"] = schedule_for(self.spec["variant"])
        if out["rtc_metadata"]["guided"] and self.spec["variant"] == "committed_prefix":
            out["rtc_weights"] = RTCProcessor(prefix_config()).get_prefix_weights(ctx.stamp.expected_delay, min(10, len(ctx.prefix)), 50)
        return out


@pytest.mark.parametrize("variant", VARIANTS)
def test_full_synthetic_episode_and_tamper_audit(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(r, "PrefixScopePredictor", FakeScopePredictor)
    monkeypatch.setattr(r.e, "NativeSession", FakeSession)
    def observation(raw, language, index, returned):
        image = torch.full((2, 2, 3), index, dtype=torch.uint8)
        return None, None, {"index": index, "returned_at": returned, "state": torch.zeros(1, 8),
            "eef_quaternion_xyzw": torch.zeros(4), "raw_eef_position": torch.zeros(3),
            "raw_gripper_qpos": torch.zeros(2), "raw_pixels": {"image": image, "image2": image.clone()},
            "worker_observation": {"image": image, "state_0": float(index)}}
    monkeypatch.setattr(r.e, "observation", observation)
    spec = spec_for(variant)
    spec["limits"]["measurement"] = 44
    calls, budget = r.e.Calls(tmp_path / "calls.jsonl"), r.Budget()
    try:
        rec, initial = r.episode(spec, tmp_path, None, None, None, None, budget, calls)
    finally:
        calls.close()
    assert rec["status"] == "completed", rec["first_failure"]
    arrays = r.load(tmp_path / f"episode_{spec['ordinal']:03d}/arrays.pt")
    row = a.inspect_episode(spec, rec, arrays, initial)
    assert row["actions"] == 44 and row["variant"] == variant
    bad = copy.deepcopy(arrays)
    bad["outputs"][2]["context_prefix"][0, 0] += 1
    with pytest.raises(ValueError, match="prefix"):
        a.inspect_episode(spec, rec, bad, initial)
    bad = copy.deepcopy(arrays)
    bad["control"]["dispatches"][23]["source_row"] += 1
    with pytest.raises(ValueError, match="source row"):
        a.inspect_episode(spec, rec, bad, initial)
    if variant == "committed_prefix":
        bad = copy.deepcopy(arrays)
        bad["outputs"][2]["rtc_weights"][10] = .1
        with pytest.raises(ValueError, match="weights"):
            a.inspect_episode(spec, rec, bad, initial)
    if variant in COMMITTED:
        bad = copy.deepcopy(rec)
        bad["requests"][1]["commitment"]["declared_steps"] += 1
        with pytest.raises(ValueError, match="declaration"):
            a.inspect_episode(spec, bad, arrays, initial)
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(os.environ.get("RUN_RTC_GRAPH_CUDA_TESTS") != "1", reason="explicit CUDA-only process")
def test_cuda_prefix_scope_refresh_and_full_graph_equivalence():
    model = SyntheticModel()
    model.config.rtc_config = prefix_config()
    model.rtc_processor = RTCProcessor(prefix_config())
    original, processor = model.sample_actions, model.rtc_processor
    images = [torch.ones((1, 4, 8), device="cuda") for _ in range(2)]
    masks = [torch.ones((1, 4), dtype=torch.bool, device="cuda") for _ in range(2)]
    tokens = torch.ones((1, 11), dtype=torch.long, device="cuda")
    token_mask = torch.ones_like(tokens, dtype=torch.bool)
    state = torch.zeros((1, 32), device="cuda")
    noise = torch.linspace(-.5, .5, 1600, device="cuda").reshape(1, 50, 32)
    with PrefixOnlyGraphRuntime(model) as rt:
        rt.mode = "graph"
        rt(images, masks, tokens, token_mask, state, noise, inference_delay=0, execution_horizon=10)
        ptr = rt.cached.weights.data_ptr()
        for length in (1, 2, 7, 8, 9, 10, 11, 30, 50):
            for delay in range(min(length, 8)+1):
                p = torch.full((length, 7), .01*length-.015*delay, device="cuda")
                outputs = []
                for mode in ("native_eager", "graph"):
                    rt.mode = mode
                    outputs.append(rt(images, masks, tokens, token_mask, state, noise,
                        inference_delay=delay, execution_horizon=10, prev_chunk_left_over=p).clone())
                torch.cuda.synchronize()
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                assert rt.cached.weights.data_ptr() == ptr and len(rt.graphs) == 2
        assert rt.graphs["guided"].record["captured_vjps"] == 10
        with pytest.raises(ValueError, match="signature"):
            rt(images, masks, tokens[:, :10], token_mask[:, :10], state, noise,
               inference_delay=0, execution_horizon=10)
    assert model.sample_actions == original and model.rtc_processor is processor
    assert rt.receipt["graphs_released"] and rt.receipt["sampler_restored"]
