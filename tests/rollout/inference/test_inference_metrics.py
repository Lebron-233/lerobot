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
from concurrent.futures import ThreadPoolExecutor

import pytest

from lerobot.rollout.inference.metrics import JsonlMetricsSink


def test_jsonl_sink_writes_flushed_machine_readable_events(tmp_path):
    output_path = tmp_path / "nested" / "metrics.jsonl"
    sink = JsonlMetricsSink(output_path)

    event = {"event": "chunk_request", "request_id": 3, "task": "抓取\n红色方块"}
    sink.emit(event)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == event

    sink.close()
    assert sink.closed


def test_jsonl_sink_context_manager_closes_and_rejects_late_writes(tmp_path):
    output_path = tmp_path / "metrics.jsonl"
    with JsonlMetricsSink(output_path) as sink:
        sink.emit({"event": "first"})

    assert sink.closed
    sink.close()  # Idempotent lifecycle cleanup.
    with pytest.raises(RuntimeError, match="after the sink is closed"):
        sink.emit({"event": "late"})


def test_jsonl_sink_keeps_concurrent_events_on_separate_lines(tmp_path):
    output_path = tmp_path / "metrics.jsonl"
    sink = JsonlMetricsSink(output_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: sink.emit({"request_id": index}), range(100)))
    sink.close()

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 100
    assert sorted(record["request_id"] for record in records) == list(range(100))
