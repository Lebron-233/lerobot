"""Targeted CPU contracts; real-model and native samples require the published fixed protocol."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rollout"))
import libero_graph_native_equivalence as campaign  # noqa: E402
import libero_reference_qualification as q  # noqa: E402
import smolvla_graph_runtime as runtime_module  # noqa: E402
from profile_libero_cuda_graph_compute import copy_inputs, invoke_graph, pack  # noqa: E402
from test_libero_reference_qualification import FakeEnv, IdentityProcessor, fake_policy  # noqa: E402


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(num_steps=10, chunk_size=50, max_action_dim=32)
        self.action_out_proj = torch.nn.Identity()
        self.sampled = []

    def sample_noise(self, shape, device):
        value = torch.randn(shape, device=device)
        self.sampled.append(value.clone())
        return value

    def encode_image_tokens(self, images, masks):
        return tuple(image[:, None, :] for image in images), tuple(mask[:, None] for mask in masks)

    def _validate_image_token_overrides(self, tokens, masks, *, batch_size, device):
        for token, mask in zip(tokens, masks, strict=True):
            if token.ndim != 3 or mask.shape != token.shape[:2] or token.device != device:
                raise ValueError("Invalid image tokens")

    def sample_actions(self, images, masks, tokens, token_masks, state, noise=None, **kwargs):
        if images is None:
            images, masks = kwargs["future_image_tokens"], kwargs["future_image_token_masks"]
        if noise is None:
            noise = self.sample_noise((1, 50, 32), state.device)
        for _ in range(10):
            self.action_out_proj(noise)
        return noise + sum(v.float().sum() for v in (*images, *masks, tokens, token_masks, state))


class FakeGraph:
    def __init__(self, model, original, inputs, projection, capture_record=None):
        torch.rand(19)  # Preparation really consumes RNG; the runtime must restore it.
        self.original = original
        self.inputs = [v.clone() for v in inputs]
        self.latest = invoke_graph(original, inputs)
        self.capture_projection_shapes = [[1, 50, 32]] * 10
        self.replay_calls = 0

    def __call__(self, inputs):
        copy_inputs(self.inputs, inputs)
        self.latest.copy_(invoke_graph(self.original, self.inputs))
        self.replay_calls += 1
        return self.latest


@pytest.fixture
def cpu_graph(monkeypatch):
    monkeypatch.setattr(runtime_module, "TokenGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def inputs(value=0):
    return (
        [torch.full((1, 2), float(value)) for _ in range(2)],
        [torch.full((1,), bool(value)) for _ in range(2)],
        torch.full((1, 3), value, dtype=torch.long),
        torch.full((1, 3), bool(value)),
        torch.full((1, 32), float(value)),
    )


def test_feature_off_never_installs_runtime():
    model = FakeModel()
    original = model.sample_actions
    runtime_module.SmolVLAGraphRuntime(model)
    assert model.sample_actions == original
    torch.manual_seed(123)
    expected = original(*inputs())
    torch.manual_seed(123)
    with runtime_module.SmolVLAGraphRuntime(model):
        actual = model.sample_actions(*inputs())
    assert torch.equal(expected, actual) and model.sample_actions == original


def test_consecutive_noise_output_lifetime_and_all_eight_inputs(cpu_graph):
    sequences = {}
    for mode in ("eager", "graph"):
        model = FakeModel()
        torch.manual_seed(123)
        with runtime_module.SmolVLAGraphRuntime(model) as runtime:
            runtime.begin_episode(mode, "A")
            first = model.sample_actions(*inputs(0))
            first_copy = first.clone()
            second = model.sample_actions(*inputs(1))
            assert torch.equal(first, first_copy)
            assert not torch.equal(first, second)
            assert runtime.noise_draws == 2
            if mode == "graph":
                rgb, masks, *rest = inputs(1)
                tokens, token_masks = model.encode_image_tokens(rgb, masks)
                expected = pack(tokens, token_masks, *rest, model.sampled[-1])
                assert all(torch.equal(a, b) for a, b in zip(runtime.graph.inputs, expected, strict=True))
            sequences[mode] = (model.sampled, [first, second])
    for a, b in zip(sequences["eager"], sequences["graph"], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_reset_reuses_same_task_but_a_b_a_recaptures(cpu_graph):
    model = FakeModel()
    with runtime_module.SmolVLAGraphRuntime(model) as runtime:
        runtime.begin_episode("graph", "A")
        model.sample_actions(*inputs())
        first = runtime.graph
        runtime.begin_episode("graph", "A")
        assert runtime.latest is runtime.latest_noise is runtime.metadata is None
        assert runtime.graph is first
        model.sample_actions(*inputs(1))
        for task in ("B", "A"):
            runtime.begin_episode("graph", task)
            assert runtime.graph is None
            model.sample_actions(*inputs())
        assert [c["capture_id"] for c in runtime.captures] == [1, 2, 3]
    assert runtime.graph is None and runtime.latest is None


@pytest.mark.parametrize("change", ["shape", "dtype", "device"])
def test_incompatible_same_task_inputs_fail_without_replay_and_restore(cpu_graph, change):
    model = FakeModel()
    original = model.sample_actions
    with pytest.raises(ValueError, match="Same-task"), runtime_module.SmolVLAGraphRuntime(model) as runtime:
        runtime.begin_episode("graph", "A")
        model.sample_actions(*inputs())
        graph = runtime.graph
        values = list(inputs())
        values[2] = {
            "shape": torch.zeros((1, 4), dtype=torch.long),
            "dtype": values[2].float(),
            "device": values[2].to("meta"),
        }[change]
        model.sample_actions(*values)
    assert graph.replay_calls == 1 and model.sample_actions == original and runtime.graph is None


def test_preparation_exception_restores_rng_sampler_and_buffers(cpu_graph, monkeypatch):
    def fail(*args):
        torch.rand(100)
        raise RuntimeError("capture failed")

    monkeypatch.setattr(runtime_module, "TokenGraph", fail)
    model = FakeModel()
    original = model.sample_actions
    torch.manual_seed(42)
    noise = torch.randn(1, 50, 32)
    state = torch.random.get_rng_state().clone()
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        runtime_module.SmolVLAGraphRuntime(model) as runtime,
    ):
        runtime.begin_episode("graph", "A")
        model.sample_actions(*inputs(), noise=noise)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.sample_actions == original and runtime.graph is None


def test_graph_audit_requires_capture_and_actual_replay_not_python_hooks():
    capture = [{"capture_id": 1, "projection_shapes": [[1, 50, 32]] * 10}]
    meta = {
        "capture_id": 1,
        "sampler_mode": "graph",
        "replay_count": 1,
        "full_chunk_shape": [1, 50, 32],
        "full_chunk_finite": True,
    }
    assert runtime_module.valid_sampler_evidence("graph", [], meta, capture)
    assert not runtime_module.valid_sampler_evidence("graph", [[1, 50, 32]] * 10, meta, capture)
    assert not runtime_module.valid_sampler_evidence("graph", [], {**meta, "replay_count": 0}, capture)
    assert not runtime_module.valid_sampler_evidence("graph", [], meta, [])


def test_nonfinite_runtime_action_stops_before_native_measurement_and_closes(tmp_path, cpu_graph):
    import gymnasium as gym

    events = []
    spec = campaign.fixed_manifest()["tuples"][0]
    policy = fake_policy(events)
    model = FakeModel()
    policy.model = model
    original = model.sample_actions
    policy.select_action = lambda batch: model.sample_actions(
        *inputs(), noise=torch.full((1, 50, 32), float("nan"))
    )[:, 0, :7]
    pre, post = IdentityProcessor(events, "pre"), IdentityProcessor(events, "post")
    with runtime_module.SmolVLAGraphRuntime(model) as runtime:
        runtime.begin_episode("graph", 0)
        result = q.run_episode(
            spec,
            tmp_path / "episode",
            policy,
            pre,
            post,
            lambda s: gym.wrappers.TimeLimit(FakeEnv(s, events, 1, False), 280),
            runtime=runtime,
        )
    assert result["status"] == "technical_failure" and result["measured_actions"] == 0
    assert result["environment_closed"] and result["settling_actions"] == 10
    assert model.sample_actions == original
    assert (tmp_path / "episode/first_failure_sampler.npz").is_file()


def test_fixed_manifest_and_stage_a_gate(tmp_path, monkeypatch):
    manifest = campaign.read_manifest()
    assert manifest == campaign.fixed_manifest() and len(manifest["tuples"]) == 40
    for index in range(20):
        a, b = manifest["tuples"][index * 2 : index * 2 + 2]
        assert a["task_id"] == b["task_id"] == index // 2
        assert a["initial_state_id"] == b["initial_state_id"] == (41 if index % 2 == 0 else 49)
        assert a["environment_seed"] == b["environment_seed"]
        assert a["policy_seed"] == b["policy_seed"]
        assert [a["sampler_mode"], b["sampler_mode"]] == (
            ["eager", "graph"] if index % 2 == 0 else ["graph", "eager"]
        )
    monkeypatch.setattr(campaign, "REPO", tmp_path)
    with pytest.raises(FileNotFoundError):
        campaign.require_model_pass("fixedhead")


def test_native_crash_accounts_only_started_episode(tmp_path):
    spec = campaign.fixed_manifest()["tuples"][0]
    directory = q.tuple_directory(tmp_path, spec)
    directory.mkdir()
    q.write_json(directory / "started.json", {"tuple": spec})
    with (directory / "events.jsonl").open("w") as file:
        file.write(json.dumps({"event": "native_step_started", "segment": "measurement", "number": 1}) + "\n")
    result = campaign.summarize(tmp_path, "head", 23)
    assert (result["completed"], result["technical_failure"], result["not_run"]) == (0, 1, 39)
    assert result["physical_step_unknown_tuples"] == 1
    assert not result["native_graph_equivalence_passed"]


def test_comparison_catches_full_chunk_difference_outside_selected_action():
    a = {
        "noise": np.zeros((1, 50, 32)),
        "full_chunk": np.zeros((1, 50, 32)),
        "normalized_action": np.zeros((1, 7)),
        "postprocessed_action": np.zeros((1, 7)),
    }
    b = {k: v.copy() for k, v in a.items()}
    b["full_chunk"][0, 49, 31] = 1
    assert not campaign.compare_arrays(a, b)["full_chunk"]["exact_equal"]
    assert campaign.compare_arrays(a, b)["normalized_action"]["exact_equal"]


@pytest.mark.parametrize("failure", ["technical", "divergence"])
def test_worker_stops_dispatch_and_restores_on_first_failure(tmp_path, monkeypatch, cpu_graph, failure):
    model = FakeModel()
    original = model.sample_actions
    policy = SimpleNamespace(model=model)
    calls = []
    monkeypatch.setattr(campaign, "require_model_pass", lambda head: None)
    monkeypatch.setattr(campaign, "load_runtime", lambda *args: (policy, None, None, {}))
    monkeypatch.setattr(q, "make_native_env_factory", lambda: None)

    def run(spec, directory, *args, **kwargs):
        calls.append(spec)
        directory.mkdir()

    monkeypatch.setattr(q, "run_episode", run)
    monkeypatch.setattr(
        q,
        "audit_tuple",
        lambda directory, spec, **kwargs: {
            "tuple": spec,
            "status": "technical_failure" if failure == "technical" else "completed",
            "success": False,
            "measured_actions": 1,
        },
    )
    monkeypatch.setattr(
        campaign,
        "compare_pair",
        lambda *args: {
            "exact_equal": False,
            "compared_requests": 0,
            "first_divergence": {"kind": "runtime_equivalence_failure"},
        },
    )
    args = SimpleNamespace(output=tmp_path, execution_head="head", policy_path=None, vlm_path=None)
    assert campaign.native_worker(args) == 2
    assert len(calls) == (1 if failure == "technical" else 2)
    assert model.sample_actions == original
    result = json.loads((tmp_path / "worker_result.json").read_text())
    assert result["original_sampler_restored"] and result["graph_released"]
