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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|predictive_async``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Literal

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .metrics import JsonlMetricsSink
from .predictive_async import PredictiveAsyncInferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

if TYPE_CHECKING:
    from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30
    metrics_path: Path | None = None


@InferenceEngineConfig.register_subclass("predictive_async")
@dataclass
class PredictiveAsyncInferenceConfig(InferenceEngineConfig):
    """Predictive asynchronous inference with latency-aware future context."""

    queue_threshold: int = 30
    latency_quantile: float = 0.9
    latency_window: int = 50
    delay_safety_margin_steps: int = 1
    min_prediction_delay: int = 0
    max_prediction_delay: int = 8
    committed_guard_steps: int = 2
    max_late_steps: int = 2
    context_mode: Literal["identity", "oracle", "predicted"] = "identity"
    future_latent_checkpoint: Path | None = None
    fallback_mode: Literal["identity", "discard"] = "identity"

    def __post_init__(self) -> None:
        if self.queue_threshold < 0:
            raise ValueError(f"queue_threshold must be >= 0, got {self.queue_threshold}")
        if not 0.0 <= self.latency_quantile <= 1.0:
            raise ValueError(f"latency_quantile must be in [0, 1], got {self.latency_quantile}")
        if self.latency_window <= 0:
            raise ValueError(f"latency_window must be > 0, got {self.latency_window}")
        if self.delay_safety_margin_steps < 0:
            raise ValueError(f"delay_safety_margin_steps must be >= 0, got {self.delay_safety_margin_steps}")
        if self.min_prediction_delay < 0:
            raise ValueError(f"min_prediction_delay must be >= 0, got {self.min_prediction_delay}")
        if self.max_prediction_delay < self.min_prediction_delay:
            raise ValueError(
                "max_prediction_delay must be >= min_prediction_delay, got "
                f"{self.max_prediction_delay} < {self.min_prediction_delay}"
            )
        if self.max_late_steps < 0:
            raise ValueError(f"max_late_steps must be >= 0, got {self.max_late_steps}")
        if self.committed_guard_steps < self.max_late_steps:
            raise ValueError(
                "committed_guard_steps must be >= max_late_steps, got "
                f"{self.committed_guard_steps} < {self.max_late_steps}"
            )
        if self.context_mode not in ("identity", "oracle", "predicted"):
            raise ValueError(
                f"context_mode must be one of 'identity', 'oracle', or 'predicted', got {self.context_mode!r}"
            )
        if self.context_mode == "predicted":
            if self.future_latent_checkpoint is None:
                raise ValueError("predicted context requires future_latent_checkpoint")
            if not 1 <= self.min_prediction_delay <= self.max_prediction_delay <= 8:
                raise ValueError(
                    "predicted context requires 1 <= min_prediction_delay <= max_prediction_delay <= 8"
                )
        elif self.context_mode == "identity" and self.future_latent_checkpoint is not None:
            raise ValueError("identity context does not accept future_latent_checkpoint")
        if self.fallback_mode not in ("identity", "discard"):
            raise ValueError(
                f"fallback_mode must be one of 'identity' or 'discard', got {self.fallback_mode!r}"
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
    future_latent_predictor: LightweightFutureLatentPredictor | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        metrics_sink = JsonlMetricsSink(config.metrics_path) if config.metrics_path is not None else None
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
            metrics_sink=metrics_sink,
        )
    if isinstance(config, PredictiveAsyncInferenceConfig):
        return PredictiveAsyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            queue_threshold=config.queue_threshold,
            latency_quantile=config.latency_quantile,
            latency_window=config.latency_window,
            delay_safety_margin_steps=config.delay_safety_margin_steps,
            min_prediction_delay=config.min_prediction_delay,
            max_prediction_delay=config.max_prediction_delay,
            committed_guard_steps=config.committed_guard_steps,
            max_late_steps=config.max_late_steps,
            context_mode=config.context_mode,
            future_latent_predictor=future_latent_predictor,
            fallback_mode=config.fallback_mode,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
