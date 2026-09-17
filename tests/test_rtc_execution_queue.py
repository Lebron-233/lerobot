"""Synthetic row-number timing tests; no model/Env or scientific samples."""

import sys
from pathlib import Path
from threading import Event, Thread

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from rtc_execution_queue import RTCExecutionQueue  # noqa: E402


def chunk(n=50, start=0):
    return torch.arange(start, start+n, dtype=torch.float32)[:, None].repeat(1, 7)


def seeded():
    queue = RTCExecutionQueue()
    stamp, prefix = queue.begin(0, 0)
    assert prefix is None
    original = chunk(start=100)
    assert queue.finish(stamp, original, original * 10)["accepted"]
    return queue


@pytest.mark.parametrize("delay", [0, 1, 3, 4, 8])
def test_observation_origin_skip_exactly_once(delay):
    queue = seeded()
    for _ in range(2):
        queue.pop()
    stamp, prefix = queue.begin(2, delay)
    assert torch.equal(prefix[0], chunk(1, 102)[0])
    for _ in range(delay):
        queue.pop()
    result = queue.finish(stamp, chunk(), chunk() * 10)
    assert result["actual_delay"] == result["source_row"] == delay
    first = queue.pop()
    assert first["action_index"] == 2 + delay
    assert first["source_row"] == delay
    assert torch.equal(first["original"], torch.full((7,), float(delay)))
    assert torch.equal(first["command"], first["original"] * 10)


def test_actual_consumption_overrides_prediction_not_elapsed_slots():
    queue = seeded()
    stamp, _ = queue.begin(0, 1)
    for _ in range(3):
        queue.pop()
    assert queue.finish(stamp, chunk(), chunk())["source_row"] == 3
    assert queue.pop()["source_row"] == 3


@pytest.mark.parametrize("consumed,new_length", [(8, 8), (9, 50)])
def test_expired_chunk_never_installed(consumed, new_length):
    queue = seeded()
    stamp, _ = queue.begin(0, 1)
    for _ in range(consumed):
        queue.pop()
    assert queue.finish(stamp, chunk(new_length), chunk(new_length))["reason"] == "expired"
    assert queue.pop()["original"][0] == 100 + consumed


def test_reset_discards_old_epoch_not_new_pending_request():
    queue = seeded()
    old, _ = queue.begin(0, 1)
    queue.reset()
    new, _ = queue.begin(0, 0)
    assert not queue.finish(old, chunk(), chunk())["accepted"]
    assert queue.finish(new, chunk(), chunk())["accepted"]
    assert queue.pop()["request_id"] == new.request_id


def test_underflow_does_not_advance_action_time():
    queue = RTCExecutionQueue()
    assert queue.pop() is None and queue.next_index == 0
    stamp, _ = queue.begin(0, 0)
    for _ in range(5):
        assert queue.pop() is None
    assert queue.finish(stamp, chunk(), chunk())["actual_delay"] == 0


def test_short_prefix_and_obsolete_observation_rejected():
    queue = seeded()
    for _ in range(48):
        queue.pop()
    with pytest.raises(ValueError, match="Insufficient"):
        queue.begin(48, 3)
    with pytest.raises(ValueError, match="Observation"):
        queue.begin(47, 0)


def test_normalized_and_processed_snapshots_are_independent_clones():
    queue = seeded()
    stamp, prefix = queue.begin(0, 0)
    prefix.fill_(-100)
    first = queue.pop()
    assert first["original"][0] == 100 and first["command"][0] == 1000
    with pytest.raises(ValueError, match="7D"):
        queue.finish(stamp, torch.zeros(50, 32), chunk())
    with pytest.raises(ValueError, match="Nonfinite"):
        queue.finish(stamp, chunk() * float("nan"), chunk())
    assert queue.finish(stamp, chunk(), chunk())["accepted"]


def test_close_pending_worker_joins_without_publishing():
    queue = seeded()
    stamp, _ = queue.begin(0, 1)
    release, entered = Event(), Event()
    results = []

    def worker():
        entered.set()
        assert release.wait(2)
        results.append(queue.finish(stamp, chunk(), chunk()))

    thread = Thread(target=worker)
    thread.start()
    assert entered.wait(2)
    queue.close()
    release.set()
    thread.join(2)
    assert not thread.is_alive() and results[0]["reason"] == "stale_or_closed"
    assert queue.pop() is None
    with pytest.raises(RuntimeError, match="Closed"):
        queue.begin(0, 0)


def test_only_one_inflight_request():
    queue = seeded()
    queue.begin(0, 1)
    with pytest.raises(RuntimeError, match="in flight"):
        queue.begin(0, 1)
