"""CPU-only specification and tail-latency decision tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_rtc_fullpath as r  # noqa: E402


def rows(base=.2, rtc=.3):
    return [{"arm": arm, "repeat": repeat, "complete_s": latency,
             "components": {"preprocess_transfer_s": .01, "public_predict_total_s": latency-.02,
                            "postprocess_transfer_s": .01}}
            for _ in range(16) for repeat in range(6) for arm, latency in (("base", base), ("rtc", rtc))]


def test_exact_request_scope_and_configuration():
    s = r.specification()
    assert s["requests"] == len(r.EXPECTED_KEYS)*2*6 == 192
    assert len(set(r.EXPECTED_KEYS)) == 16
    assert s["rtc"] == {"mode": "guided", "execution_horizon": 10,
                        "max_guidance_weight": 10.0, "schedule": "EXP"}
    assert s["new_env"] == s["training"] == s["qualification_reads"] == 0


def test_tail_latency_not_average_controls_gate():
    data = rows()
    assert r.summary(data)["latency_gate_passed"]
    data[-1]["complete_s"] = .401
    result = r.summary(data)
    assert result["timing"]["rtc"]["n"] == 80
    assert result["timing"]["rtc"]["p99_s"] == .401
    assert not result["latency_gate_passed"]


@pytest.mark.parametrize("arm", ["base", "rtc"])
def test_both_arms_must_fit_cap_with_margin(arm):
    data = rows(base=.351 if arm == "base" else .2, rtc=.351 if arm == "rtc" else .3)
    assert not r.summary(data)["latency_gate_passed"]


def test_cold_records_are_retained_but_not_in_80_denominator():
    data = rows()
    for row in data:
        if row["repeat"] == 0:
            row["complete_s"] = 10.0
    assert r.summary(data)["timing"]["base"]["p99_s"] == .2
    assert len(data) == 192


def test_nearest_rank_explicit():
    assert r.nearest(list(range(1, 81)), .5) == 40
    assert r.nearest(list(range(1, 81)), .95) == 76
    assert r.nearest(list(range(1, 81)), .99) == 80
