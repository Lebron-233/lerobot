#!/usr/bin/env python

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

"""Replay a latency sequence through predictive-async delay planning.

Example:

```bash
uv run python examples/advanced/predictive_async/replay_latency.py \
    --latencies-ms 620 590 710 640 \
    --fps 30 \
    --latency-quantile 0.9 \
    --available-actions 30
```

The command prints a single JSON document containing every request and an
aggregate summary. It does not load a policy or connect to robot hardware.
"""

from __future__ import annotations

import argparse
import json

from lerobot.rollout.inference.latency_replay import replay_latencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latencies-ms",
        type=float,
        nargs="+",
        required=True,
        help="Observed end-to-end chunk latencies in milliseconds, in replay order.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--latency-quantile", type=float, default=0.9)
    parser.add_argument("--latency-window", type=int, default=50)
    parser.add_argument("--delay-safety-margin-steps", type=int, default=1)
    parser.add_argument("--min-prediction-delay", type=int, default=0)
    parser.add_argument("--max-prediction-delay", type=int, default=8)
    parser.add_argument("--available-actions", type=int, default=30)
    parser.add_argument("--committed-guard-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = replay_latencies(
        (latency_ms / 1000.0 for latency_ms in args.latencies_ms),
        fps=args.fps,
        latency_quantile=args.latency_quantile,
        latency_window=args.latency_window,
        delay_safety_margin_steps=args.delay_safety_margin_steps,
        min_prediction_delay=args.min_prediction_delay,
        max_prediction_delay=args.max_prediction_delay,
        available_actions=args.available_actions,
        committed_guard_steps=args.committed_guard_steps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
