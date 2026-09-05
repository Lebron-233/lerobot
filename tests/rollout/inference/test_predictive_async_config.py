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

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call

import draccus
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig, SO101FollowerConfig
from lerobot.rollout import context as rollout_context
from lerobot.rollout.configs import BaseStrategyConfig
from lerobot.rollout.inference import factory
from lerobot.rollout.inference.factory import InferenceEngineConfig, PredictiveAsyncInferenceConfig

POLICY_REPO = "lerobot/smolvla_base"
POLICY_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
RUNTIME_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


@dataclass
class _CliConfig:
    inference: InferenceEngineConfig = field(default_factory=factory.SyncInferenceConfig)
    robot: RobotConfig | None = None


def _candidate_config():
    return SimpleNamespace(
        inference=PredictiveAsyncInferenceConfig(
            context_mode="predicted",
            future_latent_checkpoint=Path("synthetic-best.pt"),
            min_prediction_delay=1,
        ),
        policy=SmolVLAConfig(
            device="cpu",
            pretrained_path=Path(POLICY_REPO),
            pretrained_revision=POLICY_REVISION,
        ),
        robot=SimpleNamespace(
            type="so100_follower",
            use_degrees=True,
            max_relative_target=None,
            cameras={
                "top": SimpleNamespace(width=640, height=480, color_mode=ColorMode.RGB),
                "wrist": SimpleNamespace(width=640, height=480, color_mode=ColorMode.RGB),
            },
        ),
        fps=30.0,
        interpolation_multiplier=1,
        rename_map=dict(RENAME_MAP),
        device="cpu",
        use_torch_compile=False,
        compile_warmup_inferences=2,
        teleop=None,
        dataset=None,
        strategy=BaseStrategyConfig(),
        task="synthetic pick",
    )


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
    assert config.future_latent_checkpoint is None
    assert config.fallback_mode == "identity"


@pytest.mark.parametrize("context_mode", ["identity", "oracle", "predicted"])
@pytest.mark.parametrize("fallback_mode", ["identity", "discard"])
def test_predictive_async_config_accepts_supported_modes(context_mode: str, fallback_mode: str) -> None:
    config = PredictiveAsyncInferenceConfig(  # type: ignore[arg-type]
        context_mode=context_mode,
        future_latent_checkpoint=Path("synthetic-best.pt") if context_mode == "predicted" else None,
        min_prediction_delay=1 if context_mode == "predicted" else 0,
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
        ({"context_mode": "predicted"}, "requires future_latent_checkpoint"),
        (
            {"context_mode": "identity", "future_latent_checkpoint": Path("synthetic-best.pt")},
            "identity context does not accept future_latent_checkpoint",
        ),
        ({"fallback_mode": "rtc_residual"}, "fallback_mode must be one of"),
    ],
)
def test_predictive_async_config_rejects_invalid_values(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        PredictiveAsyncInferenceConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("minimum", "maximum"), [(0, 8), (1, 9), (9, 9), (4, 3)])
def test_predicted_config_rejects_unsupported_delay_bounds(minimum: int, maximum: int) -> None:
    with pytest.raises(ValueError, match="prediction_delay"):
        PredictiveAsyncInferenceConfig(
            context_mode="predicted",
            future_latent_checkpoint=Path("synthetic-best.pt"),
            min_prediction_delay=minimum,
            max_prediction_delay=maximum,
        )


@pytest.mark.parametrize("maximum", [1, 6, 8])
def test_draccus_parses_predicted_checkpoint_and_runtime_cap(maximum: int) -> None:
    config = draccus.parse(
        _CliConfig,
        args=[
            "--inference.type=predictive_async",
            "--inference.context_mode=predicted",
            "--inference.future_latent_checkpoint=synthetic-best.pt",
            "--inference.min_prediction_delay=1",
            f"--inference.max_prediction_delay={maximum}",
        ],
    ).inference

    assert isinstance(config, PredictiveAsyncInferenceConfig)
    assert config.future_latent_checkpoint == Path("synthetic-best.pt")
    assert config.min_prediction_delay == 1
    assert config.max_prediction_delay == maximum


def test_draccus_predictive_defaults_leave_predictor_disabled() -> None:
    config = draccus.parse(_CliConfig, args=["--inference.type=predictive_async"]).inference

    assert config.context_mode == "identity"
    assert config.future_latent_checkpoint is None


def test_predictive_metrics_default_off_does_not_construct_sink(monkeypatch) -> None:
    config = draccus.parse(_CliConfig, args=["--inference.type=predictive_async"]).inference
    sink_factory = Mock(side_effect=AssertionError("default metrics must not open a sink"))
    engine_factory = Mock(return_value=object())
    monkeypatch.setattr(factory, "JsonlMetricsSink", sink_factory)
    monkeypatch.setattr(factory, "PredictiveAsyncInferenceEngine", engine_factory)

    result = factory.create_inference_engine(
        config,
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        robot_wrapper=SimpleNamespace(robot_type="so100_follower"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=[],
        task="pick",
        fps=30.0,
        device="cpu",
    )

    assert config.metrics_path is None
    sink_factory.assert_not_called()
    engine_factory.assert_called_once()
    assert engine_factory.call_args.kwargs["metrics_sink"] is None
    assert result is engine_factory.return_value


def test_predictive_metrics_path_is_parsed_and_injected_once(monkeypatch, tmp_path) -> None:
    metrics_path = tmp_path / "predictive.jsonl"
    config = draccus.parse(
        _CliConfig,
        args=["--inference.type=predictive_async", f"--inference.metrics_path={metrics_path}"],
    ).inference
    sink = Mock(name="metrics_sink")
    sink_factory = Mock(return_value=sink)
    engine_factory = Mock(return_value=object())
    monkeypatch.setattr(factory, "JsonlMetricsSink", sink_factory)
    monkeypatch.setattr(factory, "PredictiveAsyncInferenceEngine", engine_factory)

    result = factory.create_inference_engine(
        config,
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        robot_wrapper=SimpleNamespace(robot_type="so100_follower"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=[],
        task="pick",
        fps=30.0,
        device="cpu",
    )

    assert config.metrics_path == metrics_path
    sink_factory.assert_called_once_with(metrics_path)
    engine_factory.assert_called_once()
    assert engine_factory.call_args.kwargs["metrics_sink"] is sink
    assert result is engine_factory.return_value


def test_factory_forwards_same_predictor_instance_without_loading(monkeypatch) -> None:
    predictor = Mock(name="already_loaded_predictor")
    engine_factory = Mock(return_value=object())
    monkeypatch.setattr(factory, "PredictiveAsyncInferenceEngine", engine_factory)

    result = factory.create_inference_engine(
        PredictiveAsyncInferenceConfig(
            context_mode="predicted",
            future_latent_checkpoint=Path("synthetic-best.pt"),
            min_prediction_delay=1,
        ),
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        robot_wrapper=SimpleNamespace(robot_type="so100_follower"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=[],
        task="pick",
        fps=30.0,
        device="cpu",
        future_latent_predictor=predictor,
    )

    assert result is engine_factory.return_value
    assert engine_factory.call_count == 1
    assert engine_factory.call_args.kwargs["future_latent_predictor"] is predictor
    predictor.assert_not_called()


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
    assert captured["future_latent_predictor"] is None
    assert captured["fallback_mode"] == "discard"
    assert captured["use_torch_compile"] is True
    assert captured["compile_warmup_inferences"] == 4


@pytest.mark.parametrize(
    ("policy_type", "context_mode", "error_type", "error"),
    [
        ("act", "identity", ValueError, "requires a SmolVLA policy"),
        ("smolvla", "oracle", ValueError, "offline-only"),
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


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("policy", "pretrained_path", Path("someone/another_policy")),
        ("policy", "pretrained_path", Path("/synthetic/local/smolvla_base")),
        ("policy", "pretrained_revision", "another-revision"),
        ("policy", "vlm_model_name", "another/vlm"),
        ("policy", "use_peft", True),
        ("robot", "type", "so101_follower"),
        ("robot", "use_degrees", False),
        ("robot", "max_relative_target", 5.0),
        (None, "fps", 29.97),
        (None, "interpolation_multiplier", 2),
        (None, "rename_map", {}),
        (
            None,
            "rename_map",
            {
                "observation.images.top": "observation.images.camera2",
                "observation.images.wrist": "observation.images.camera1",
            },
        ),
    ],
)
def test_predicted_rejects_incompatible_config_before_resolution_or_hardware(
    monkeypatch, section, field, value
) -> None:
    cfg = _candidate_config()
    setattr(getattr(cfg, section) if section else cfg, field, value)
    resolver = Mock(side_effect=AssertionError("invalid config must not resolve snapshots"))
    robot_factory = Mock(side_effect=AssertionError("invalid config must not construct hardware"))
    predictor_loader = Mock(side_effect=AssertionError("invalid config must not load a predictor"))
    monkeypatch.setattr(rollout_context, "snapshot_download", resolver)
    monkeypatch.setattr(rollout_context, "make_robot_from_config", robot_factory)
    monkeypatch.setattr(rollout_context, "load_frozen_future_latent_predictor", predictor_loader)

    with pytest.raises(ValueError):
        rollout_context.build_rollout_context(cfg, Event())

    resolver.assert_not_called()
    robot_factory.assert_not_called()
    predictor_loader.assert_not_called()


@pytest.mark.parametrize(
    "processor_name",
    ["teleop_action_processor", "robot_action_processor", "robot_observation_processor"],
)
def test_predicted_rejects_each_custom_robot_pipeline_before_resolution(monkeypatch, processor_name) -> None:
    resolver = Mock(side_effect=AssertionError("custom pipeline must fail before model resolution"))
    robot_factory = Mock(side_effect=AssertionError("custom pipeline must fail before hardware"))
    monkeypatch.setattr(rollout_context, "snapshot_download", resolver)
    monkeypatch.setattr(rollout_context, "make_robot_from_config", robot_factory)

    with pytest.raises(ValueError, match="default robot-side processor"):
        rollout_context.build_rollout_context(
            _candidate_config(), Event(), **{processor_name: SimpleNamespace(steps=[])}
        )

    resolver.assert_not_called()
    robot_factory.assert_not_called()


@pytest.mark.parametrize(
    "camera_names",
    [(), ("top",), ("front", "wrist"), ("top", "wrist", "third"), ("top", "wrist", "third", "fourth")],
)
def test_predicted_rejects_wrong_raw_camera_streams_before_resolution(monkeypatch, camera_names) -> None:
    cfg = _candidate_config()
    cfg.robot.cameras = {name: SimpleNamespace(height=480, width=640) for name in camera_names}
    resolver = Mock(side_effect=AssertionError("camera schema must fail before model resolution"))
    monkeypatch.setattr(rollout_context, "snapshot_download", resolver)

    with pytest.raises(ValueError, match="exactly the top and wrist"):
        rollout_context.build_rollout_context(cfg, Event())

    resolver.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [("height", 240), ("width", 320), ("use_rgb", False), ("use_depth", True), ("color_mode", ColorMode.BGR)],
)
def test_predicted_rejects_wrong_raw_camera_shape_or_channels(monkeypatch, field, value) -> None:
    cfg = _candidate_config()
    setattr(cfg.robot.cameras["top"], field, value)
    resolver = Mock(side_effect=AssertionError("camera shape must fail before model resolution"))
    monkeypatch.setattr(rollout_context, "snapshot_download", resolver)

    with pytest.raises(ValueError, match="two RGB cameras"):
        rollout_context.build_rollout_context(cfg, Event())

    resolver.assert_not_called()


@pytest.fixture
def synthetic_runtime(monkeypatch):
    events = []
    policy_snapshot = Path("/synthetic/policy/snapshots") / POLICY_REVISION
    vlm_snapshot = Path("/synthetic/vlm/snapshots") / VLM_REVISION
    snapshot_config = SmolVLAConfig(device="cpu")
    policy = SimpleNamespace(
        config=snapshot_config,
        type="smolvla",
        eval=Mock(),
        requires_grad_=Mock(),
    )
    policy.to = Mock(return_value=policy)
    preprocessor = SimpleNamespace(steps=[])
    postprocessor = SimpleNamespace(steps=[])
    pre_stats = {"so100.buffer.action.mean": torch.arange(6.0), "so100.buffer.action.std": torch.ones(6)}
    post_stats = {
        "so100.buffer.action.mean": torch.arange(6.0) + 10,
        "so100.buffer.action.std": torch.ones(6) * 2,
    }

    def resolve(*, repo_id, revision, local_files_only):
        events.append(f"resolve:{repo_id}")
        assert local_files_only is True
        assert (repo_id, revision) in ((POLICY_REPO, POLICY_REVISION), (VLM_REPO, VLM_REVISION))
        return str(policy_snapshot if repo_id == POLICY_REPO else vlm_snapshot)

    def load_policy(*args, **kwargs):
        events.append("policy")
        policy.config = kwargs["config"]
        return policy

    def make_processors(**kwargs):
        events.append("policy_processors")
        return preprocessor, postprocessor

    predictor = Mock(name="frozen_predictor")

    def load_predictor(*args, **kwargs):
        events.append("predictor")
        return predictor

    robot = SimpleNamespace(
        name="synthetic_so100",
        robot_type="so100_follower",
        observation_features={
            **dict.fromkeys(RUNTIME_KEYS, float),
            "top": (480, 640, 3),
            "wrist": (480, 640, 3),
        },
        action_features=dict.fromkeys(RUNTIME_KEYS, float),
        connect=Mock(side_effect=lambda: events.append("connect")),
        get_observation=Mock(return_value=dict.fromkeys(RUNTIME_KEYS, 0.0)),
    )
    resolver = Mock(side_effect=resolve)
    config_loader = Mock(return_value=snapshot_config)
    policy_loader = Mock(side_effect=load_policy)
    processor_loader = Mock(side_effect=make_processors)
    stats_loader = Mock(side_effect=[pre_stats, post_stats])
    predictor_loader = Mock(side_effect=load_predictor)
    robot_factory = Mock(return_value=robot)
    engine_factory = Mock(return_value=object())
    default_processors = rollout_context.make_default_processors()
    default_processor_factory = Mock(return_value=default_processors)
    regular_policy_loader = Mock(side_effect=AssertionError("predicted must use the exact snapshot loader"))
    monkeypatch.setattr(rollout_context, "snapshot_download", resolver)
    monkeypatch.setattr(rollout_context.PreTrainedConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(
        rollout_context, "get_policy_class", Mock(return_value=SimpleNamespace(from_pretrained=policy_loader))
    )
    monkeypatch.setattr(rollout_context, "make_pre_post_processors", processor_loader)
    monkeypatch.setattr(rollout_context, "load_file", stats_loader)
    monkeypatch.setattr(rollout_context, "load_frozen_future_latent_predictor", predictor_loader)
    monkeypatch.setattr(rollout_context, "make_robot_from_config", robot_factory)
    monkeypatch.setattr(rollout_context, "create_inference_engine", engine_factory)
    monkeypatch.setattr(rollout_context, "make_default_processors", default_processor_factory)
    monkeypatch.setattr(rollout_context, "_load_pretrained_policy", regular_policy_loader)
    return SimpleNamespace(**locals())


def test_predicted_build_resolves_and_loads_once_before_connection(synthetic_runtime) -> None:
    runtime = synthetic_runtime
    cfg = _candidate_config()

    ctx = rollout_context.build_rollout_context(cfg, Event())

    assert runtime.resolver.call_args_list == [
        call(repo_id=POLICY_REPO, revision=POLICY_REVISION, local_files_only=True),
        call(repo_id=VLM_REPO, revision=VLM_REVISION, local_files_only=True),
    ]
    runtime.config_loader.assert_called_once_with(runtime.policy_snapshot, local_files_only=True)
    runtime.policy_loader.assert_called_once_with(
        runtime.policy_snapshot, config=runtime.snapshot_config, local_files_only=True, strict=True
    )
    runtime.predictor_loader.assert_called_once_with(Path("synthetic-best.pt"), device="cpu")
    assert runtime.processor_loader.call_count == 1
    assert runtime.stats_loader.call_args_list == [
        call(
            str(runtime.policy_snapshot / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
            device="cpu",
        ),
        call(
            str(runtime.policy_snapshot / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
            device="cpu",
        ),
    ]
    for event in ("policy", "policy_processors", "predictor"):
        assert runtime.events.index(event) < runtime.events.index("connect")
    runtime.robot.connect.assert_called_once()
    runtime.default_processor_factory.assert_called_once()
    assert ctx.processors.teleop_action_processor is runtime.default_processors[0]
    assert ctx.processors.robot_action_processor is runtime.default_processors[1]
    assert ctx.processors.robot_observation_processor is runtime.default_processors[2]
    runtime.regular_policy_loader.assert_not_called()
    runtime.predictor.assert_not_called()

    kwargs = runtime.processor_loader.call_args.kwargs
    assert kwargs["pretrained_path"] == str(runtime.policy_snapshot)
    assert "dataset_stats" not in kwargs
    pre_overrides = kwargs["preprocessor_overrides"]
    assert pre_overrides["device_processor"] == {"device": "cpu"}
    assert pre_overrides["rename_observations_processor"] == {"rename_map": RENAME_MAP}
    assert pre_overrides["tokenizer_processor"] == {"tokenizer_name": str(runtime.vlm_snapshot)}
    for source, target in [
        (runtime.pre_stats, pre_overrides["normalizer_processor"]["stats"]),
        (runtime.post_stats, kwargs["postprocessor_overrides"]["unnormalizer_processor"]["stats"]),
    ]:
        assert set(target) == {"action"}
        for name in ("mean", "std"):
            assert target["action"][name] is source[f"so100.buffer.action.{name}"]
    assert runtime.policy.config.vlm_model_name == str(runtime.vlm_snapshot)
    assert runtime.policy.config.robot_state_feature.shape == (6,)
    assert runtime.policy.config.action_feature.shape == (6,)
    assert tuple(runtime.policy.config.action_feature_names) == RUNTIME_KEYS
    assert tuple(runtime.policy.config.image_features) == tuple(RENAME_MAP.values())
    runtime.policy.requires_grad_.assert_called_once_with(False)

    injected = runtime.engine_factory.call_args.kwargs
    assert injected["policy"] is ctx.policy.policy is runtime.policy
    assert injected["preprocessor"] is ctx.policy.preprocessor is runtime.preprocessor
    assert injected["postprocessor"] is ctx.policy.postprocessor is runtime.postprocessor
    assert injected["future_latent_predictor"] is runtime.predictor
    assert ctx.data.ordered_action_keys == list(RUNTIME_KEYS)


def test_identity_build_does_not_resolve_load_or_call_predictor(monkeypatch, synthetic_runtime) -> None:
    runtime = synthetic_runtime
    cfg = _candidate_config()
    cfg.inference = PredictiveAsyncInferenceConfig()
    runtime.regular_policy_loader.side_effect = None
    runtime.regular_policy_loader.return_value = runtime.policy
    frozen_loader = Mock(side_effect=AssertionError("identity must not construct the frozen candidate"))
    monkeypatch.setattr(rollout_context, "_load_frozen_future_latent_runtime", frozen_loader)

    ctx = rollout_context.build_rollout_context(cfg, Event())

    runtime.regular_policy_loader.assert_called_once_with(cfg.policy)
    frozen_loader.assert_not_called()
    runtime.resolver.assert_not_called()
    runtime.config_loader.assert_not_called()
    runtime.stats_loader.assert_not_called()
    runtime.predictor_loader.assert_not_called()
    runtime.predictor.assert_not_called()
    assert runtime.engine_factory.call_args.kwargs["future_latent_predictor"] is None
    assert ctx.policy.preprocessor is runtime.preprocessor


@pytest.mark.parametrize("wrong_source", ["policy", "vlm"])
def test_predicted_rejects_wrong_resolved_revision_before_loading(synthetic_runtime, wrong_source) -> None:
    runtime = synthetic_runtime
    runtime.resolver.side_effect = [
        str(runtime.policy_snapshot if wrong_source != "policy" else Path("/synthetic/wrong-policy")),
        str(runtime.vlm_snapshot if wrong_source != "vlm" else Path("/synthetic/wrong-vlm")),
    ]

    with pytest.raises(ValueError, match="different policy or VLM revision"):
        rollout_context.build_rollout_context(_candidate_config(), Event())

    runtime.config_loader.assert_not_called()
    runtime.policy_loader.assert_not_called()
    runtime.predictor_loader.assert_not_called()
    runtime.robot_factory.assert_not_called()


@pytest.mark.parametrize(
    "override", ["num_vlm_layers", "action_feature_names", "input_features", "output_features"]
)
def test_predicted_rejects_semantic_policy_overrides_before_weight_loading(
    synthetic_runtime, override
) -> None:
    runtime = synthetic_runtime
    cfg = _candidate_config()
    if override == "num_vlm_layers":
        cfg.policy.num_vlm_layers += 1
    elif override == "action_feature_names":
        cfg.policy.action_feature_names = list(reversed(RUNTIME_KEYS))
    elif override == "input_features":
        cfg.policy.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        }
    else:
        cfg.policy.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    with pytest.raises(ValueError, match=override):
        rollout_context.build_rollout_context(cfg, Event())

    runtime.policy_loader.assert_not_called()
    runtime.predictor_loader.assert_not_called()
    runtime.robot_factory.assert_not_called()


@pytest.mark.parametrize(
    "mismatch",
    [
        "state_order",
        "state_names",
        "action_order",
        "one_camera",
        "three_cameras",
        "four_cameras",
        "camera_shape",
    ],
)
def test_predicted_rejects_declared_hardware_features_before_connect(synthetic_runtime, mismatch) -> None:
    runtime = synthetic_runtime
    robot = runtime.robot
    if mismatch == "state_order":
        robot.observation_features = {
            **dict.fromkeys(reversed(RUNTIME_KEYS), float),
            "top": (480, 640, 3),
            "wrist": (480, 640, 3),
        }
    elif mismatch == "state_names":
        robot.observation_features["main_shoulder_pan"] = robot.observation_features.pop("shoulder_pan.pos")
    elif mismatch == "action_order":
        robot.action_features = dict.fromkeys(reversed(RUNTIME_KEYS), float)
    elif mismatch == "one_camera":
        del robot.observation_features["wrist"]
    elif mismatch in ("three_cameras", "four_cameras"):
        robot.observation_features["third"] = (480, 640, 3)
        if mismatch == "four_cameras":
            robot.observation_features["fourth"] = (480, 640, 3)
    else:
        robot.observation_features["top"] = (240, 320, 3)

    with pytest.raises(ValueError, match="predicted requires"):
        rollout_context.build_rollout_context(_candidate_config(), Event())

    robot.connect.assert_not_called()
    robot.get_observation.assert_not_called()
    runtime.predictor.assert_not_called()
    runtime.engine_factory.assert_not_called()


def _parse_so_follower(robot_type):
    robot = draccus.parse(
        _CliConfig,
        args=[f"--robot.type={robot_type}", "--robot.port=/synthetic/robot", "--robot.use_degrees=true"],
    ).robot
    robot.cameras = {
        "top": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
        "wrist": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480),
    }
    return robot


@pytest.mark.parametrize(
    ("robot_type", "config_class"),
    [("so100_follower", SO100FollowerConfig), ("so101_follower", SO101FollowerConfig)],
)
def test_so_follower_draccus_roundtrip_preserves_distinct_registered_type(robot_type, config_class) -> None:
    parsed = _parse_so_follower(robot_type)
    encoded = draccus.encode(parsed, declared_type=RobotConfig)
    decoded = draccus.decode(RobotConfig, encoded)

    assert SO100FollowerConfig is not SO101FollowerConfig
    assert type(parsed) is type(decoded) is config_class
    assert parsed.type == decoded.type == encoded["type"] == robot_type
    assert RobotConfig.get_choice_class(robot_type) is config_class
    assert decoded.port == "/synthetic/robot"
    assert decoded.max_relative_target is None
    assert decoded.use_degrees is True
    assert tuple(decoded.cameras) == ("top", "wrist")


def test_parsed_so101_rejected_before_resolution_policy_or_hardware(synthetic_runtime) -> None:
    runtime = synthetic_runtime
    cfg = _candidate_config()
    cfg.robot = _parse_so_follower("so101_follower")

    with pytest.raises(ValueError, match="so100_follower"):
        rollout_context.build_rollout_context(cfg, Event())

    runtime.resolver.assert_not_called()
    runtime.config_loader.assert_not_called()
    runtime.policy_loader.assert_not_called()
    runtime.predictor_loader.assert_not_called()
    runtime.robot_factory.assert_not_called()
    runtime.robot.connect.assert_not_called()


def test_parsed_so100_reaches_supported_mock_construction(synthetic_runtime) -> None:
    runtime = synthetic_runtime
    cfg = _candidate_config()
    cfg.robot = _parse_so_follower("so100_follower")

    ctx = rollout_context.build_rollout_context(cfg, Event())

    runtime.robot_factory.assert_called_once_with(cfg.robot)
    runtime.robot.connect.assert_called_once()
    runtime.policy_loader.assert_called_once()
    runtime.predictor_loader.assert_called_once()
    assert ctx.runtime.cfg.robot.type == "so100_follower"
