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

from threading import Event
from types import SimpleNamespace

import pytest

from lerobot.rollout import context as rollout_context
from lerobot.rollout.inference import factory
from lerobot.rollout.inference.factory import PredictiveAsyncInferenceConfig


def test_predictive_async_config_defaults() -> None:
    config = PredictiveAsyncInferenceConfig()

    assert config.type == "predictive_async"
    assert config.queue_threshold == 30
    assert config.latency_quantile == 0.9
    assert config.latency_window == 50
    assert config.delay_safety_margin_steps == 1
    assert config.min_prediction_delay == 0
    assert config.max_prediction_delay == 8
    assert config.committed_guard_steps == 2
    assert config.max_late_steps == 2
    assert config.context_mode == "identity"
    assert config.fallback_mode == "identity"


@pytest.mark.parametrize("context_mode", ["identity", "oracle", "predicted"])
@pytest.mark.parametrize("fallback_mode", ["identity", "discard"])
def test_predictive_async_config_accepts_supported_modes(context_mode: str, fallback_mode: str) -> None:
    config = PredictiveAsyncInferenceConfig(  # type: ignore[arg-type]
        context_mode=context_mode,
        fallback_mode=fallback_mode,
    )

    assert config.context_mode == context_mode
    assert config.fallback_mode == fallback_mode


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"queue_threshold": -1}, "queue_threshold must be >= 0"),
        ({"latency_quantile": -0.1}, r"latency_quantile must be in \[0, 1\]"),
        ({"latency_quantile": 1.1}, r"latency_quantile must be in \[0, 1\]"),
        ({"latency_window": 0}, "latency_window must be > 0"),
        ({"delay_safety_margin_steps": -1}, "delay_safety_margin_steps must be >= 0"),
        ({"min_prediction_delay": -1}, "min_prediction_delay must be >= 0"),
        (
            {"min_prediction_delay": 4, "max_prediction_delay": 3},
            "max_prediction_delay must be >= min_prediction_delay",
        ),
        ({"max_late_steps": -1}, "max_late_steps must be >= 0"),
        (
            {"committed_guard_steps": 1, "max_late_steps": 2},
            "committed_guard_steps must be >= max_late_steps",
        ),
        ({"context_mode": "unknown"}, "context_mode must be one of"),
        ({"fallback_mode": "rtc_residual"}, "fallback_mode must be one of"),
    ],
)
def test_predictive_async_config_rejects_invalid_values(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        PredictiveAsyncInferenceConfig(**kwargs)  # type: ignore[arg-type]


def test_factory_forwards_predictive_async_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def _build_predictive_engine(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "PredictiveAsyncInferenceEngine", _build_predictive_engine)
    config = PredictiveAsyncInferenceConfig(
        queue_threshold=7,
        latency_quantile=0.8,
        latency_window=12,
        delay_safety_margin_steps=2,
        min_prediction_delay=1,
        max_prediction_delay=6,
        committed_guard_steps=3,
        max_late_steps=2,
        context_mode="identity",
        fallback_mode="discard",
    )

    result = factory.create_inference_engine(
        config,
        policy=object(),  # type: ignore[arg-type]
        preprocessor=object(),  # type: ignore[arg-type]
        postprocessor=object(),  # type: ignore[arg-type]
        robot_wrapper=SimpleNamespace(robot_type="mock"),
        hw_features={"observation.state": {}},
        dataset_features={},
        ordered_action_keys=[],
        task="pick",
        fps=30.0,
        device="cuda",
        use_torch_compile=True,
        compile_warmup_inferences=4,
    )

    assert result is sentinel
    assert captured["queue_threshold"] == 7
    assert captured["latency_quantile"] == 0.8
    assert captured["latency_window"] == 12
    assert captured["delay_safety_margin_steps"] == 2
    assert captured["min_prediction_delay"] == 1
    assert captured["max_prediction_delay"] == 6
    assert captured["committed_guard_steps"] == 3
    assert captured["max_late_steps"] == 2
    assert captured["context_mode"] == "identity"
    assert captured["fallback_mode"] == "discard"
    assert captured["use_torch_compile"] is True
    assert captured["compile_warmup_inferences"] == 4


@pytest.mark.parametrize(
    ("policy_type", "context_mode", "error_type", "error"),
    [
        ("act", "identity", ValueError, "requires a SmolVLA policy"),
        ("smolvla", "oracle", ValueError, "offline-only"),
        ("smolvla", "predicted", NotImplementedError, "learned-predictor milestone"),
    ],
)
def test_invalid_live_predictive_configuration_fails_before_policy_or_hardware_setup(
    policy_type: str,
    context_mode: str,
    error_type: type[Exception],
    error: str,
) -> None:
    cfg = SimpleNamespace(
        inference=PredictiveAsyncInferenceConfig(context_mode=context_mode),  # type: ignore[arg-type]
        policy=SimpleNamespace(pretrained_path="unused", type=policy_type),
    )

    with pytest.raises(error_type, match=error):
        rollout_context.build_rollout_context(cfg, Event())  # type: ignore[arg-type]
