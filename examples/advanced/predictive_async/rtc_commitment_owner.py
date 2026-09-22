"""One opt-in receive schedule: fulfill a declared prefix before actual-delay trim.

All model computation, original queue and controller code remain unchanged.
A completed episode cancels its remaining commitment; it never keeps moving.
"""

import time

from rtc_dynamic_runtime import RTCPrefixOwner


class CommittedPrefixOwner(RTCPrefixOwner):
    def __init__(self, *args, **kwargs):
        self.commitment_closing = False
        super().__init__(*args, **kwargs)

    def submit(self, observation, delay):
        stamp = super().submit(observation, delay)
        row = self.pending
        enabled = self.spec["variant"] == "committed_rtc" and stamp.request_id > 1
        row["commitment"] = {
            "enabled": enabled,
            "compute_cover_estimate": delay,
            "declared_steps": delay if enabled else None,
            "conditioned_steps": delay if self.spec["arm"] == "rtc_async" and row["prefix_rows"] else 0,
            "deferred_polls": 0,
            "boundary_at": None,
        }
        return stamp

    def receive(self, block=False):
        row = self.pending
        if row is None:
            return None
        commitment = row["commitment"]
        if not commitment["enabled"] or self.commitment_closing:
            got = super().receive(block=block)
            if got is not None and commitment["enabled"]:
                commitment["status"] = "cancelled_at_stop"
                commitment["consumed_at_stop"] = self.queue.next_index - row["stamp"]["observation_index"]
            return got
        consumed = self.queue.next_index - row["stamp"]["observation_index"]
        target = commitment["declared_steps"]
        if not 0 <= consumed <= target <= self.queue.max_delay:
            raise RuntimeError("Commitment consumption overshot or cap invalid")
        if consumed < target:
            # Do not dequeue a result early; keep executing exactly the owned old prefix.
            commitment["deferred_polls"] += 1
            return None
        began = time.perf_counter()
        commitment["boundary_at"] = began
        got = super().receive(block=True)
        ended = time.perf_counter()
        if not got["decision"]["accepted"] or got["decision"]["actual_delay"] != target:
            raise RuntimeError("Committed installation did not consume exactly the declared prefix")
        commitment.update(
            status="installed",
            consumed=consumed,
            result_ready_at_boundary=got["completed_at"] <= began,
            ready_hold_s=max(0.0, began - got["completed_at"]),
            late_compute_wait_s=max(0.0, got["completed_at"] - began),
            receive_elapsed_s=ended - began,
        )
        return got

    def close(self):
        self.commitment_closing = True
        # Original close fences queue publication, joins the model owner, then drains.
        return super().close()
