"""The diagnostic observes collection without retaining unreachable cycles."""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from so101_gc_timing import GcTrace  # noqa: E402


def test_gc_trace_records_real_collection_and_removes_callback():
    before = gc.isenabled()
    trace = GcTrace()
    try:
        gc.collect(0)
        assert trace.events
        event = trace.events[-1]
        assert event["generation"] == 0
        assert event["finished_at_s"] >= event["started_at_s"]
        assert event["duration_s"] >= 0
        assert gc.isenabled() == before
    finally:
        trace.close()
    assert trace.callback not in gc.callbacks
