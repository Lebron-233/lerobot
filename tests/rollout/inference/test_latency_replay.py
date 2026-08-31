# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.rollout.inference.latency_replay import compute_delay_plan, replay_latencies


def test_replay_uses_only_prior_samples_and_marks_cold_start():
    result = replay_latencies(
        [0.1, 0.2, 0.05],
        fps=10,
        latency_quantile=0.9,
        latency_window=10,
        delay_safety_margin_steps=0,
        min_prediction_delay=0,
        max_prediction_delay=10,
        available_actions=10,
        committed_guard_steps=0,
    )

    cold_start, late, early = result.items
    assert cold_start.planned_delay_steps is None
    assert cold_start.estimated_latency_s is None
    assert cold_start.late_steps is None

    assert late.estimated_latency_s == pytest.approx(0.1)
    assert late.planned_delay_steps == 1
    assert late.actual_delay_steps == 2
    assert late.late_steps == 1

    assert early.estimated_latency_s == pytest.approx(0.19)
    assert early.planned_delay_steps == 2
    assert early.actual_delay_steps == 1
    assert early.early_steps == 1

    assert result.summary.latency_p50_s == pytest.approx(0.1)
    assert result.summary.latency_p90_s == pytest.approx(0.18)
    assert result.summary.calibrated_sample_count == 2
    assert result.summary.late_chunk_count == 1
    assert result.summary.late_chunk_rate == pytest.approx(0.5)


def test_delay_plan_clamps_to_maximum_and_available_actions_after_guard():
    tracker = LatencyTracker(maxlen=10)
    tracker.add(0.5)

    availability_limited = compute_delay_plan(
        tracker,
        fps=10,
        latency_quantile=0.9,
        delay_safety_margin_steps=1,
        min_prediction_delay=2,
        max_prediction_delay=8,
        available_actions=5,
        committed_guard_steps=2,
    )
    assert availability_limited is not None
    assert availability_limited.raw_required_delay_steps == 6
    assert availability_limited.available_after_guard_steps == 3
    assert availability_limited.planned_delay_steps == 3
    assert not availability_limited.prediction_cap_exceeded

    no_committable_actions = compute_delay_plan(
        tracker,
        fps=10,
        latency_quantile=0.9,
        delay_safety_margin_steps=0,
        min_prediction_delay=2,
        max_prediction_delay=8,
        available_actions=1,
        committed_guard_steps=1,
    )
    assert no_committable_actions is not None
    assert no_committable_actions.planned_delay_steps == 0


def test_replay_reports_final_sliding_window_quantiles():
    result = replay_latencies(
        [1.0, 0.1, 0.1],
        fps=10,
        latency_window=2,
        delay_safety_margin_steps=0,
        max_prediction_delay=20,
        available_actions=20,
        committed_guard_steps=0,
    )

    assert result.summary.latency_window_sample_count == 2
    assert result.summary.latency_p50_s == pytest.approx(0.1)
    assert result.summary.latency_p90_s == pytest.approx(0.1)


def test_replay_q_one_forgets_peak_after_it_leaves_sliding_window():
    result = replay_latencies(
        [1.0, 0.1, 0.1, 0.1],
        fps=10,
        latency_quantile=1.0,
        latency_window=2,
        delay_safety_margin_steps=0,
        max_prediction_delay=20,
        available_actions=20,
        committed_guard_steps=0,
    )

    assert result.items[-1].estimated_latency_s == pytest.approx(0.1)
    assert result.items[-1].planned_delay_steps == 1


def test_replay_reports_prediction_cap_without_claiming_full_coverage():
    result = replay_latencies(
        [0.5, 0.5],
        fps=30,
        delay_safety_margin_steps=1,
        max_prediction_delay=8,
        available_actions=30,
        committed_guard_steps=2,
    )

    planned = result.items[1]
    assert planned.raw_required_delay_steps == 16
    assert planned.planned_delay_steps == 8
    assert planned.prediction_cap_exceeded
    assert result.summary.prediction_cap_exceeded_count == 1


def test_replay_counts_underflow_requests_and_ticks_with_per_request_availability():
    result = replay_latencies(
        [0.4, 0.2, 0.1],
        fps=10,
        delay_safety_margin_steps=0,
        available_actions=[2, 1, 4],
        committed_guard_steps=0,
    )

    assert [item.actual_delay_steps for item in result.items] == [4, 2, 1]
    assert [item.underflow_steps for item in result.items] == [2, 1, 0]
    assert result.summary.underflow_count == 2
    assert result.summary.underflow_steps == 3


def test_replay_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least one"):
        replay_latencies([], fps=30)
    with pytest.raises(ValueError, match="one value per latency"):
        replay_latencies([0.1, 0.2], fps=30, available_actions=[5])
    with pytest.raises(ValueError, match="finite and >= 0"):
        replay_latencies([float("nan")], fps=30)


def test_replay_cli_prints_json_without_loading_hardware():
    repo_root = Path(__file__).parents[3]
    script = repo_root / "examples" / "advanced" / "predictive_async" / "replay_latency.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--latencies-ms",
            "100",
            "200",
            "--fps",
            "10",
            "--delay-safety-margin-steps",
            "0",
            "--available-actions",
            "10",
            "--committed-guard-steps",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["items"][0]["planned_delay_steps"] is None
    assert payload["items"][1]["planned_delay_steps"] == 1
    assert payload["summary"]["sample_count"] == 2
