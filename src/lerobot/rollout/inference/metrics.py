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

"""Machine-readable metrics sinks for inference backends."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Any, Protocol, Self


class InferenceMetricsSink(Protocol):
    """Destination for structured inference metric events."""

    def emit(self, event: Mapping[str, Any]) -> None:
        """Write one metric event."""

    def close(self) -> None:
        """Release resources held by the sink."""


class JsonlMetricsSink:
    """Write one JSON object per line to a caller-selected path.

    Constructing the sink is the opt-in switch: inference code that receives
    ``None`` instead performs no serialization or file I/O. Writes and closure
    are guarded by the same lock so the sink can be shared by an inference
    worker and its control thread.

    The destination is truncated when the sink is constructed. Each event is
    flushed immediately so a stopped rollout leaves a readable final record.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._lock = Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        with self._lock:
            return self._closed

    def emit(self, event: Mapping[str, Any]) -> None:
        """Serialize and flush one event atomically."""
        line = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot emit inference metrics after the sink is closed")
            self._stream.write(line)
            self._stream.write("\n")
            self._stream.flush()

    def close(self) -> None:
        """Flush and close the output file; repeated calls are harmless."""
        with self._lock:
            if self._closed:
                return
            self._stream.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
