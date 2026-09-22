"""Versioned plan-order queue; NOT observation-origin time alignment.

One controller submits/pops/publishes at Env boundaries. A newly accepted plan
starts at row zero at its actual takeover index. Its remaining tail can later be
superseded, but emitted (epoch, request, row) identities never repeat. Numerically
or semantically repeated commands across different plans are not deduplicated.
"""

from threading import Lock

import torch
from rtc_execution_queue import RequestStamp


class SequenceExecutionQueue:
    """Keep new prefixes in order without a backlog of obsolete plan tails."""

    def __init__(self, max_delay=8):
        if type(max_delay) is not int or max_delay < 0:
            raise ValueError('Invalid maximum delay')
        self._lock = Lock()
        self.max_delay = max_delay
        self.epoch = self.next_index = self._serial = 0
        self._pending = self._original = self._processed = self._source = None
        self._cursor = self._origin = self._takeover = 0
        self._closed = False

    def begin(self, observation_index, expected_delay):
        with self._lock:
            if self._closed or self._pending is not None:
                raise RuntimeError('Closed queue or request already in flight')
            if type(observation_index) is not int or observation_index != self.next_index:
                raise ValueError('Observation must match next physical action index')
            if type(expected_delay) is not int or not 0 <= expected_delay <= self.max_delay:
                raise ValueError('Invalid predicted delay')
            prefix = None if self._original is None else self._original[self._cursor:].clone()
            if prefix is not None and len(prefix) < expected_delay:
                raise ValueError('Insufficient original prefix')
            self._serial += 1
            stamp = RequestStamp(self.epoch, self._serial, observation_index, expected_delay)
            self._pending = stamp
            return stamp, prefix

    def finish(self, stamp, original, processed):
        with self._lock:
            if self._closed or stamp != self._pending or stamp.epoch != self.epoch:
                return {'accepted': False, 'reason': 'stale_or_closed'}
            for value in (original, processed):
                if not isinstance(value, torch.Tensor) or value.shape != (50, 7) or value.device.type != 'cpu':
                    raise ValueError('Require full CPU 50x7 chunks')
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError('Nonfinite or nonfloating chunk')
            delay = self.next_index-stamp.observation_index
            if delay < 0:
                raise RuntimeError('Physical action index moved backwards')
            self._pending = None
            if delay > self.max_delay or delay >= 50:
                return {'accepted': False, 'reason': 'expired', 'actual_delay': delay}
            retired = None if self._source is None else {
                'request_id': self._source, 'start_row': self._cursor, 'end_row_exclusive': 50,
                'reason': 'superseded_unexecuted_tail'}
            self._original, self._processed = original.clone(), processed.clone()
            self._cursor, self._source = 0, stamp.request_id
            self._origin, self._takeover = stamp.observation_index, self.next_index
            return {'accepted': True, 'reason': 'installed', 'actual_delay': delay,
                'source_row': 0, 'takeover_index': self.next_index, 'request_id': stamp.request_id,
                'epoch': self.epoch, 'origin_semantics': 'plan_order_at_takeover',
                'nominal_minus_actual': -delay, 'superseded_tail': retired}

    def pop(self):
        with self._lock:
            if self._closed or self._original is None or self._cursor >= 50:
                return None
            row = {'action_index': self.next_index, 'request_id': self._source,
                'source_row': self._cursor, 'epoch': self.epoch,
                'original': self._original[self._cursor].clone(),
                'command': self._processed[self._cursor].clone(),
                'nominal_action_index': self._origin+self._cursor,
                'nominal_minus_actual': self._origin+self._cursor-self.next_index,
                'plan_takeover_index': self._takeover}
            self._cursor += 1
            self.next_index += 1
            return row

    def qsize(self):
        with self._lock:
            return 0 if self._closed or self._original is None else 50-self._cursor

    def reset(self):
        with self._lock:
            self.epoch += 1
            self.next_index = self._cursor = self._origin = self._takeover = 0
            self._pending = self._original = self._processed = self._source = None
            self._closed = False

    def close(self):
        with self._lock:
            self.epoch += 1
            self._closed = True
            self._pending = self._original = self._processed = self._source = None
