"""Observation-origin chunk timing for a single-inflight RTC experiment.

Both unguided and guided arms use the same replacement/time convention.
Only successful pops advance the absolute action index. This is not the
future-token queue's future-origin row-zero convention, nor an online safety gate.
"""

from dataclasses import dataclass
from threading import Lock

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@dataclass(frozen=True)
class RequestStamp:
    epoch: int
    request_id: int
    observation_index: int
    expected_delay: int


class RTCExecutionQueue:
    """Atomic observation/prefix snapshots with stale-result fencing.

    The controller calls begin only at an Env-return boundary with the current
    observation index. Worker completion and action consumption share this lock.
    Inputs are CPU normalized 7D actions and separately postprocessed 7D commands.
    """

    def __init__(self, max_delay: int = 8):
        self._lock = Lock()
        self._queue = ActionQueue(RTCConfig(enabled=True))
        self.max_delay = max_delay
        self.epoch = 0
        self.next_index = 0
        self._serial = 0
        self._pending = None
        self._source = None
        self._origin = 0
        self._closed = False

    def begin(self, observation_index: int, expected_delay: int):
        with self._lock:
            if self._closed or self._pending is not None:
                raise RuntimeError("Closed queue or request already in flight")
            if type(observation_index) is not int or observation_index != self.next_index:
                raise ValueError("Observation must match the next absolute action index")
            if type(expected_delay) is not int or not 0 <= expected_delay <= self.max_delay:
                raise ValueError("Invalid predicted delay")
            prefix = self._queue.get_left_over()
            if prefix is not None and len(prefix) < expected_delay:
                raise ValueError("Insufficient original prefix for predicted delay")
            self._serial += 1
            stamp = RequestStamp(self.epoch, self._serial, observation_index, expected_delay)
            self._pending = stamp
            return stamp, prefix

    def finish(self, stamp: RequestStamp, original: torch.Tensor, processed: torch.Tensor):
        with self._lock:
            if self._closed or stamp != self._pending or stamp.epoch != self.epoch:
                return {"accepted": False, "reason": "stale_or_closed"}
            for value in (original, processed):
                if value.ndim != 2 or value.shape[1] != 7 or value.device.type != "cpu":
                    raise ValueError("Use separate CPU normalized 7D and processed 7D chunks")
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError("Nonfinite or nonfloating chunk")
            if original.shape != processed.shape or not len(original):
                raise ValueError("Chunk lengths differ or are empty")
            actual_delay = self.next_index - stamp.observation_index
            if actual_delay < 0:
                raise RuntimeError("Action index moved backwards")
            self._pending = None
            if actual_delay >= len(original) or actual_delay > self.max_delay:
                return {"accepted": False, "reason": "expired", "actual_delay": actual_delay}
            # Exactly one trim, using actual consumed actions, not wall-clock slots.
            self._queue.merge(original, processed, actual_delay)
            self._source = stamp.request_id
            self._origin = stamp.observation_index
            return {"accepted": True, "reason": "installed", "actual_delay": actual_delay,
                    "takeover_index": self.next_index, "source_row": actual_delay,
                    "request_id": stamp.request_id, "epoch": self.epoch}

    def pop(self):
        with self._lock:
            if self._closed:
                return None
            original = self._queue.get_left_over()
            command = self._queue.get()
            if command is None:
                return None
            row = {"action_index": self.next_index, "request_id": self._source,
                   "source_row": self.next_index - self._origin, "epoch": self.epoch,
                   "original": original[0].clone(), "command": command}
            self.next_index += 1
            return row

    def qsize(self):
        with self._lock:
            return self._queue.qsize()

    def reset(self):
        with self._lock:
            self.epoch += 1
            self.next_index = 0
            self._pending = self._source = None
            self._origin = 0
            self._closed = False
            self._queue.clear()

    def close(self):
        with self._lock:
            self._closed = True
            self.epoch += 1
            self._pending = None
            self._queue.clear()
