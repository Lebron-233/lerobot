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

from __future__ import annotations

import torch

from lerobot.rollout.inference.base import InferenceEngine


class _TestInferenceEngine(InferenceEngine):
    @property
    def control_thread_owns_policy(self) -> bool:
        return True

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset(self) -> None: ...

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        return None


def test_task_snapshot_tracks_monotonic_epoch_across_reused_task_strings():
    engine = _TestInferenceEngine(task="task A")

    assert engine.task_snapshot == ("task A", 0)
    assert engine._take_task() == ("task A", False)

    assert engine.set_task("task B") is True
    assert engine.task_snapshot == ("task B", 1)

    # Both updates can land during one inference.  The edge stays coalesced, while the
    # epoch still distinguishes this task A from the task A that started the request.
    assert engine.set_task("task A") is True
    assert engine.task_snapshot == ("task A", 2)
    assert engine._take_task() == ("task A", True)
    assert engine._take_task() == ("task A", False)


def test_repeating_current_task_preserves_epoch_and_changed_edge():
    engine = _TestInferenceEngine(task="task A")

    assert engine.set_task("task A") is False
    assert engine.task_snapshot == ("task A", 0)
    assert engine._take_task() == ("task A", False)

    assert engine.set_task("task B") is True
    assert engine._take_task() == ("task B", True)

    assert engine.set_task("task B") is False
    assert engine.task_snapshot == ("task B", 1)
    assert engine._take_task() == ("task B", False)
