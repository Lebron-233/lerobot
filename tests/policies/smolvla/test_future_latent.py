# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

pytest.importorskip(
    "transformers",
    reason="SmolVLA tests require the `smolvla` extra (transformers)",
)

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor


def _config(**overrides) -> FutureLatentConfig:
    values = {
        "token_dim": 8,
        "action_dim": 3,
        "state_dim": 4,
        "rank": 4,
        "action_hidden_dim": 8,
        "state_hidden_dim": 6,
        "delay_embedding_dim": 5,
        "fusion_hidden_dim": 8,
        "max_cameras": 3,
    }
    values.update(overrides)
    return FutureLatentConfig(**values)


def _inputs(
    *,
    batch_size: int = 2,
    token_counts: tuple[int, ...] = (5, 3),
    token_dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    torch.manual_seed(0)
    image_tokens = tuple(
        torch.randn(batch_size, token_count, 8, dtype=token_dtype) for token_count in token_counts
    )
    image_token_masks = tuple(
        torch.ones(batch_size, token_count, dtype=torch.bool) for token_count in token_counts
    )
    delays = torch.tensor(([1, 3] if batch_size == 2 else [2] * batch_size), dtype=torch.long)
    committed_mask = torch.arange(8).unsqueeze(0) < delays.unsqueeze(1)
    return {
        "image_tokens": image_tokens,
        "image_token_masks": image_token_masks,
        "committed_actions": torch.randn(batch_size, 8, 3),
        "committed_mask": committed_mask,
        "state": torch.randn(batch_size, 4),
        "delay_steps": delays,
    }


def _clone_inputs(inputs: dict[str, object]) -> dict[str, object]:
    return {
        key: tuple(item.clone() for item in value) if isinstance(value, tuple) else value.clone()
        for key, value in inputs.items()
    }


def _flatten_deltas(prediction) -> torch.Tensor:
    return torch.cat([delta.flatten() for delta in prediction.delta_tokens])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_dim", 0),
        ("action_dim", True),
        ("rank", 9),
        ("action_hidden_dim", 129),
        ("state_hidden_dim", 65),
        ("fusion_hidden_dim", 129),
        ("max_prediction_delay", 9),
        ("max_cameras", 0),
        ("token_mixer", "attention"),
        ("max_parameter_count", 1_000_001),
    ],
)
def test_config_rejects_values_outside_phase_a_contract(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_config_defaults_are_frozen() -> None:
    config = FutureLatentConfig(token_dim=960, action_dim=32, state_dim=32)

    assert (config.rank, config.action_hidden_dim, config.state_hidden_dim) == (64, 128, 64)
    assert (config.delay_embedding_dim, config.fusion_hidden_dim) == (32, 128)
    assert (config.max_prediction_delay, config.max_cameras) == (8, 4)
    assert (config.enabled, config.token_mixer, config.risk_head) == (False, "depthwise1d", True)
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.rank = 32


def test_zero_init_preserves_shapes_order_dtype_and_masks() -> None:
    model = LightweightFutureLatentPredictor(_config()).eval()
    inputs = _inputs()
    inputs["image_token_masks"][0][0, -1] = False
    inputs["image_tokens"][0][0, -1] = torch.nan

    prediction = model(**inputs)

    assert len(prediction.delta_tokens) == 2
    for delta, tokens, mask in zip(
        prediction.delta_tokens,
        inputs["image_tokens"],
        inputs["image_token_masks"],
        strict=True,
    ):
        assert delta.shape == tokens.shape
        assert delta.dtype == tokens.dtype
        assert delta.device == tokens.device
        assert torch.count_nonzero(delta) == 0
        assert torch.count_nonzero(delta[~mask]) == 0
    assert prediction.predicted_error.shape == (2,)
    assert torch.isfinite(prediction.predicted_error).all()
    assert torch.all(prediction.predicted_error >= 0)


def test_invalid_tokens_do_not_enter_risk_pooling() -> None:
    model = LightweightFutureLatentPredictor(_config()).eval()
    inputs = _inputs()
    inputs["image_token_masks"][0][0, -2:] = False
    changed = _clone_inputs(inputs)
    changed["image_tokens"][0][0, -2:] = torch.tensor([float("nan"), float("inf")])[:, None]

    baseline = model(**inputs)
    modified = model(**changed)

    torch.testing.assert_close(modified.predicted_error, baseline.predicted_error)
    for actual, expected in zip(modified.delta_tokens, baseline.delta_tokens, strict=True):
        torch.testing.assert_close(actual, expected)


def test_invalid_token_delta_stays_zero_after_up_projection_trains() -> None:
    model = LightweightFutureLatentPredictor(_config()).eval()
    inputs = _inputs(batch_size=1, token_counts=(4,))
    inputs["image_token_masks"][0][0, -1] = False
    with torch.no_grad():
        model.up_projection.bias.fill_(1)

    delta = model(**inputs).delta_tokens[0]

    assert torch.count_nonzero(delta[inputs["image_token_masks"][0]]) > 0
    assert torch.count_nonzero(delta[~inputs["image_token_masks"][0]]) == 0


def test_padded_action_values_are_semantically_ignored() -> None:
    model = LightweightFutureLatentPredictor(_config()).eval()
    inputs = _inputs()
    changed = _clone_inputs(inputs)
    changed["committed_actions"][~changed["committed_mask"]] = torch.nan

    baseline = model(**inputs)
    modified = model(**changed)

    torch.testing.assert_close(modified.predicted_error, baseline.predicted_error)
    for actual, expected in zip(modified.delta_tokens, baseline.delta_tokens, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.tensor([[True, False, True, False, False, False, False, False]] * 2),
        torch.tensor([[True, True, False, False, False, False, False, False]] * 2),
    ],
)
def test_committed_mask_must_match_delay_prefix(bad_mask: torch.Tensor) -> None:
    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs()
    inputs["committed_mask"] = bad_mask

    with pytest.raises(ValueError, match="contiguous prefix"):
        model(**inputs)


def test_action_state_and_delay_each_affect_nonzero_residual() -> None:
    torch.manual_seed(3)
    model = LightweightFutureLatentPredictor(_config(token_mixer="none")).eval()
    with torch.no_grad():
        model.up_projection.weight.normal_(std=0.1)
    inputs = _inputs(batch_size=1, token_counts=(4,))
    baseline = _flatten_deltas(model(**inputs))

    changed_action = _clone_inputs(inputs)
    changed_action["committed_actions"][:, 0] += 1
    changed_state = _clone_inputs(inputs)
    changed_state["state"] += 1
    changed_delay = _clone_inputs(inputs)
    changed_delay["delay_steps"][:] = 3
    changed_delay["committed_mask"] = torch.arange(8).unsqueeze(0) < 3

    assert not torch.allclose(_flatten_deltas(model(**changed_action)), baseline)
    assert not torch.allclose(_flatten_deltas(model(**changed_state)), baseline)

    with torch.no_grad():
        for parameter in model.action_gru.parameters():
            parameter.zero_()
    delay_baseline = _flatten_deltas(model(**inputs))
    assert not torch.allclose(_flatten_deltas(model(**changed_delay)), delay_baseline)


@pytest.mark.parametrize("token_counts", [(4,), (5, 3)])
def test_forward_backward_single_and_multi_camera(token_counts: tuple[int, ...]) -> None:
    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs(token_counts=token_counts)
    inputs["image_tokens"] = tuple(tokens.requires_grad_() for tokens in inputs["image_tokens"])

    prediction = model(**inputs)
    loss = prediction.predicted_error.sum() + sum(
        delta.float().square().sum() for delta in prediction.delta_tokens
    )
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(tokens.grad is not None for tokens in inputs["image_tokens"])


@pytest.mark.parametrize("predictor_dtype", [torch.float32, torch.bfloat16])
def test_bfloat16_tokens_accept_fp32_action_and_state_context(
    predictor_dtype: torch.dtype,
) -> None:
    model = LightweightFutureLatentPredictor(_config()).to(dtype=predictor_dtype).eval()
    inputs = _inputs(token_dtype=torch.bfloat16)
    inputs["image_tokens"] = tuple(tokens.requires_grad_() for tokens in inputs["image_tokens"])

    prediction = model(**inputs)
    (
        prediction.predicted_error.sum() + sum(delta.float().sum() for delta in prediction.delta_tokens)
    ).backward()

    assert all(delta.dtype == torch.bfloat16 for delta in prediction.delta_tokens)
    assert inputs["committed_actions"].dtype == torch.float32
    assert inputs["state"].dtype == torch.float32
    assert prediction.predicted_error.dtype == predictor_dtype
    assert all(tokens.grad is not None for tokens in inputs["image_tokens"])


@pytest.mark.parametrize("delay", [0, 9])
def test_delay_must_be_within_primary_horizon(delay: int) -> None:
    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs()
    inputs["delay_steps"][0] = delay

    with pytest.raises(ValueError, match="delay_steps"):
        model(**inputs)


def test_camera_token_dtypes_must_match() -> None:
    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs()
    inputs["image_tokens"] = (inputs["image_tokens"][0], inputs["image_tokens"][1].bfloat16())

    with pytest.raises(ValueError, match="share device and dtype"):
        model(**inputs)


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        (lambda inputs: inputs["image_token_masks"].__setitem__(0, torch.ones(2, 5)), TypeError),
        (
            lambda inputs: inputs.__setitem__("committed_actions", torch.ones(2, 8, 3, dtype=torch.long)),
            TypeError,
        ),
        (lambda inputs: inputs.__setitem__("committed_actions", torch.ones(2, 7, 3)), ValueError),
        (lambda inputs: inputs.__setitem__("delay_steps", torch.ones(2)), TypeError),
        (lambda inputs: inputs.__setitem__("state", torch.ones(2, 5)), ValueError),
    ],
)
def test_input_dtype_and_shape_validation(mutation, error_type: type[Exception]) -> None:
    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs()
    mutable_masks = list(inputs["image_token_masks"])
    inputs["image_token_masks"] = mutable_masks
    mutation(inputs)
    if isinstance(inputs["image_token_masks"], list):
        inputs["image_token_masks"] = tuple(inputs["image_token_masks"])

    with pytest.raises(error_type):
        model(**inputs)


def test_camera_limit_and_cross_device_inputs_fail_clearly() -> None:
    model = LightweightFutureLatentPredictor(_config(max_cameras=1))
    with pytest.raises(ValueError, match="max_cameras"):
        model(**_inputs(token_counts=(2, 2)))

    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs()
    inputs["state"] = torch.empty((2, 4), device="meta")
    with pytest.raises(ValueError, match="share a device"):
        model(**inputs)


def test_nonfinite_valid_inputs_and_outputs_are_rejected() -> None:
    model = LightweightFutureLatentPredictor(_config())
    inputs = _inputs()
    inputs["image_tokens"][0][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        model(**inputs)

    inputs = _inputs()
    with torch.no_grad():
        model.up_projection.weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match="delta_tokens"):
        model(**inputs)

    model = LightweightFutureLatentPredictor(_config())
    with torch.no_grad():
        model.risk_head[-1].weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match="predicted_error"):
        model(**_inputs())


def test_parameter_ceiling_and_state_dict_round_trip() -> None:
    production_shape_config = FutureLatentConfig(token_dim=960, action_dim=32, state_dim=32)
    production_shape_model = LightweightFutureLatentPredictor(production_shape_config)
    parameter_count = sum(parameter.numel() for parameter in production_shape_model.parameters())
    assert parameter_count < 1_000_000

    with pytest.raises(ValueError, match="trainable parameters"):
        LightweightFutureLatentPredictor(_config(max_parameter_count=1))

    config = _config()
    source = LightweightFutureLatentPredictor(config).eval()
    restored = LightweightFutureLatentPredictor(config).eval()
    restored.load_state_dict(source.state_dict())
    inputs = _inputs()
    source_prediction = source(**inputs)
    restored_prediction = restored(**inputs)
    for actual, expected in zip(
        restored_prediction.delta_tokens, source_prediction.delta_tokens, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored_prediction.predicted_error, source_prediction.predicted_error)


def test_risk_head_disabled_returns_zero_error() -> None:
    prediction = LightweightFutureLatentPredictor(_config(risk_head=False))(**_inputs())
    assert torch.count_nonzero(prediction.predicted_error) == 0


def test_frozen_episode_split_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "examples/advanced/predictive_async/m3_episode_split_seed0.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["dataset"] == {
        "repo_id": "lerobot/svla_so100_pickplace",
        "revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
    }
    assert manifest["generation"] == {
        "library": "numpy",
        "generator": "Generator",
        "bit_generator": "PCG64",
        "seed": 0,
        "input_episode_ids": {
            "start_inclusive": 0,
            "stop_exclusive": 50,
            "order": "ascending",
        },
        "split_slices": {"train": [0, 40], "val": [40, 45], "test": [45, 50]},
        "serialized_order": "ascending_within_each_split",
    }
    expected = {
        "train": [
            0,
            1,
            2,
            3,
            4,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            32,
            34,
            35,
            36,
            37,
            38,
            40,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        "val": [5, 7, 14, 39, 49],
        "test": [15, 29, 31, 33, 41],
    }
    actual = {name: manifest["splits"][name]["episode_ids"] for name in expected}
    assert actual == expected
    assert {name: len(ids) for name, ids in actual.items()} == {"train": 40, "val": 5, "test": 5}
    assert {
        name: (manifest["splits"][name]["episode_count"], manifest["splits"][name]["frame_count"])
        for name in expected
    } == {"train": (40, 15_576), "val": (5, 1_822), "test": (5, 2_233)}
    assert all(
        set(actual[left]).isdisjoint(actual[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    assert set().union(*(set(ids) for ids in actual.values())) == set(range(50))
    assert manifest["integrity"] == {
        "episode_count": 50,
        "frame_count": 19_631,
        "pairwise_disjoint": True,
        "episode_id_union": list(range(50)),
    }
