"""Synthetic-only tests for binding commitments, late waits, close and evidence tampering."""

import copy
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_rtc_commitment as a  # noqa: E402
import libero_rtc_commitment as r  # noqa: E402
from rtc_commitment_owner import CommittedPrefixOwner  # noqa: E402
from rtc_dynamic_runtime import RTCPrefixOwner  # noqa: E402
from rtc_execution_queue import RTCExecutionQueue  # noqa: E402

from tests.test_libero_graph_feedback import FakeSession  # noqa: E402
from tests.test_rtc_dynamic_runtime import FakeDynamicPredictor  # noqa: E402


def spec_for(variant):
    return copy.deepcopy(next(s for s in r.manifest()["rows"] if s["variant"] == variant))


class PausedPredictor:
    def __init__(self, event):
        self.event = event
        self.calls = 0

    def __call__(self, context):
        self.calls += 1
        if self.calls > 1:
            assert self.event.wait(2)
        x = torch.arange(350, dtype=torch.float32).reshape(50, 7) + self.calls * 1000
        return {"original": x, "processed": x * 10}


def owner_fixture(tmp_path, event):
    budget = r.Budget()
    budget.begin_episode()
    calls = r.e.Calls(tmp_path / "calls.jsonl")
    owner = CommittedPrefixOwner(PausedPredictor(event), calls, spec_for("committed_rtc"), budget)
    owner.submit({"index": 0, "returned_at": time.perf_counter()}, 0)
    owner.receive(block=True)
    for _ in range(20):
        owner.queue.pop()
    return owner, calls


@pytest.mark.parametrize("delay", [1, 3, 4, 7, 8])
def test_early_output_does_not_install_before_declared_count(tmp_path, delay):
    event = threading.Event()
    event.set()
    owner, calls = owner_fixture(tmp_path, event)
    try:
        owner.submit({"index": 20, "returned_at": time.perf_counter()}, delay)
        for _ in range(delay):
            assert owner.receive() is None
            command = owner.queue.pop()
            assert command["request_id"] == 1
        got = owner.receive()
        assert got["decision"]["source_row"] == got["decision"]["actual_delay"] == delay
        assert owner.queue.pop()["request_id"] == 2
        assert owner.rows[-1]["commitment"]["status"] == "installed"
        assert type(owner.queue) is RTCExecutionQueue
    finally:
        owner.close()
        calls.close()
    assert not owner.thread.is_alive()


def test_late_result_waits_without_consuming_extra_action(tmp_path):
    event = threading.Event()
    owner, calls = owner_fixture(tmp_path, event)
    timer = None
    try:
        owner.submit({"index": 20, "returned_at": time.perf_counter()}, 3)
        for _ in range(3):
            owner.queue.pop()
        timer = threading.Timer(.05, event.set)
        timer.start()
        got = owner.receive()
        assert owner.queue.next_index == 23
        assert got["decision"]["source_row"] == 3
        assert not got["commitment"]["result_ready_at_boundary"]
        assert got["commitment"]["late_compute_wait_s"] > 0
    finally:
        event.set()
        if timer:
            timer.join()
        owner.close()
        calls.close()


def test_end_before_commitment_cancels_without_more_actions(tmp_path):
    event = threading.Event()
    owner, calls = owner_fixture(tmp_path, event)
    owner.submit({"index": 20, "returned_at": time.perf_counter()}, 7)
    owner.queue.pop()
    event.set()
    owner.close()
    calls.close()
    assert owner.queue.next_index == 21
    assert owner.rows[-1]["decision"] == {"accepted": False, "reason": "stale_or_closed"}
    assert owner.rows[-1]["commitment"]["status"] == "cancelled_at_stop"
    assert owner.rows[-1]["commitment"]["consumed_at_stop"] == 1


def test_consumption_overshoot_fails_closed(tmp_path):
    event = threading.Event()
    event.set()
    owner, calls = owner_fixture(tmp_path, event)
    try:
        owner.submit({"index": 20, "returned_at": time.perf_counter()}, 3)
        for _ in range(4):
            owner.queue.pop()
        with pytest.raises(RuntimeError, match="overshot"):
            owner.receive()
    finally:
        owner.close()
        calls.close()


def test_original_cap_and_snapshot_are_not_replaced(tmp_path):
    assert CommittedPrefixOwner._serve is RTCPrefixOwner._serve
    q = RTCExecutionQueue(max_delay=8)
    x = torch.zeros(50, 7)
    s, _ = q.begin(0, 0)
    q.finish(s, x, x)
    stamp, _ = q.begin(0, 8)
    for _ in range(9):
        q.pop()
    assert q.finish(stamp, x, x)["reason"] == "expired"
    q.close()


def test_manifest_has_no_identity_specific_decision_or_search():
    m = r.manifest()
    assert len(m["rows"]) == 40 and not m["new_qualification"]
    for pair in range(10):
        rows = [s for s in m["rows"] if s["pair_index"] == pair]
        assert {s["variant"] for s in rows} == set(r.VARIANTS)
        assert len({s["environment_seed"] for s in rows}) == len({s["policy_seed"] for s in rows}) == 1
        assert all(s["limits"]["measurement"] == 280 for s in rows)


@pytest.mark.parametrize("variant", r.VARIANTS)
def test_complete_synthetic_episode_and_original_audit(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(r, "DynamicRTCPredictor", FakeDynamicPredictor)
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
    if variant == "committed_rtc":
        assert row["commitments"] and all(c["declared"] == c["actual"] for c in row["commitments"])
        bad = copy.deepcopy(rec)
        bad["requests"][1]["commitment"]["declared_steps"] += 1
        with pytest.raises(ValueError, match="commitment"):
            a.inspect_episode(spec, bad, arrays, initial)
        bad = copy.deepcopy(rec)
        bad["requests"][1]["commitment"]["late_compute_wait_s"] += 1
        with pytest.raises(ValueError, match="evidence"):
            a.inspect_episode(spec, bad, arrays, initial)
    bad = copy.deepcopy(arrays)
    bad["outputs"][2]["context_prefix"][0, 0] += 1
    with pytest.raises(ValueError, match="prefix"):
        a.inspect_episode(spec, rec, bad, initial)
    assert not torch.cuda.is_initialized()
