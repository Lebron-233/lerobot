"""Observational cyclic-GC timestamps for isolated simulator-pause diagnosis."""

from __future__ import annotations

import gc
import time


class GcTrace:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.started: dict | None = None
        gc.callbacks.append(self.callback)

    def callback(self, phase: str, info: dict) -> None:
        now = time.perf_counter()
        if phase == "start":
            self.started = {"generation": info["generation"], "started_at_s": now}
        elif self.started is not None:
            self.events.append(
                {
                    **self.started,
                    "finished_at_s": now,
                    "duration_s": now - self.started["started_at_s"],
                    "collected": info["collected"],
                    "uncollectable": info["uncollectable"],
                }
            )
            self.started = None

    def close(self) -> None:
        gc.callbacks.remove(self.callback)
