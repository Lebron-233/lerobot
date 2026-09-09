"""CPU-only token and worker contracts, using Events for the tested interleavings."""

import sys
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import smolvla_graph_runtime as runtime_module  # noqa: E402
from smolvla_graph_identity import SmolVLAGraphIdentityEngine  # noqa: E402
from test_smolvla_graph_native import FakeGraph, FakeModel, inputs  # noqa: E402


@pytest.fixture
def fake_graph(monkeypatch):
    monkeypatch.setattr(runtime_module, "TokenGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def test_token_only_never_encodes_or_samples_explicit_noise(fake_graph):
    model = FakeModel()
    rgb, masks, lang, lang_mask, state = inputs(1)
    tokens, token_masks = model.encode_image_tokens(rgb, masks)
    model.encode_image_tokens = lambda *args: pytest.fail("Token-only request encoded RGB")
    with runtime_module.SmolVLAGraphRuntime(model) as runtime:
        runtime.begin_episode("graph", "A")
        explicit = torch.zeros(1, 50, 32)
        actual = runtime(
            None,
            None,
            lang,
            lang_mask,
            state,
            noise=explicit,
            future_image_tokens=tokens,
            future_image_token_masks=token_masks,
        )
        held = actual.clone()
        runtime(
            None,
            None,
            lang,
            lang_mask,
            state,
            future_image_tokens=tokens,
            future_image_token_masks=token_masks,
        )
        assert torch.equal(held, actual)
        assert runtime.noise_draws == 1 and runtime.rgb_encodings == 0
        assert runtime.graph.replay_calls == 2


@pytest.mark.parametrize("bad", ["pair", "shape", "device", "future_state", "rtc", "unknown"])
def test_token_contract_rejects_missing_mismatched_and_unsupported_inputs(fake_graph, bad):
    model = FakeModel()
    rgb, masks, lang, lang_mask, state = inputs()
    tokens, token_masks = model.encode_image_tokens(rgb, masks)
    kwargs = {"future_image_tokens": tokens, "future_image_token_masks": token_masks}
    if bad == "pair":
        kwargs.pop("future_image_token_masks")
    elif bad == "shape":
        kwargs["future_image_tokens"] = (tokens[0].squeeze(1), tokens[1])
    elif bad == "device":
        kwargs["future_image_tokens"] = (tokens[0].to("meta"), tokens[1])
    elif bad == "future_state":
        kwargs["future_state"] = state
    elif bad == "rtc":
        kwargs["inference_delay"] = 0
    else:
        kwargs["misspelled_override"] = tokens
    with runtime_module.SmolVLAGraphRuntime(model) as runtime, pytest.raises((ValueError, TypeError)):
        runtime.begin_episode("graph", "A")
        runtime(None, None, lang, lang_mask, state, **kwargs)
    assert runtime.control_requests == 0


class Processor:
    steps = ()

    def __init__(self):
        self.threads = []

    def __call__(self, value):
        self.threads.append(get_ident())
        return value

    def reset(self):
        self.threads.append(get_ident())


class Policy:
    config = SimpleNamespace(type="smolvla", use_amp=False)

    def __init__(self):
        self.model = FakeModel()
        self.reset_threads = []
        self.inference_modes = []

    def reset(self):
        self.reset_threads.append(get_ident())

    def prepare_images(self, batch):
        return inputs()[:2]

    def predict_action_chunk(self, batch, **kwargs):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        rgb, masks, lang, lang_mask, state = inputs()
        if kwargs:
            rgb = masks = None
        else:
            rgb, masks = self.prepare_images(batch)
        return self.model.sample_actions(rgb, masks, lang, lang_mask, state, **kwargs)[:, :, :7]


class ObservedEngine(SmolVLAGraphIdentityEngine):
    def __init__(self):
        self.policy = Policy()
        self.pre, self.post = Processor(), Processor()
        super().__init__(
            policy=self.policy,
            preprocessor=self.pre,
            postprocessor=self.post,
            robot_wrapper=SimpleNamespace(robot_type="mock"),
            hw_features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": [f"state_{i}" for i in range(8)],
                }
            },
            task="A",
            fps=20,
            device="cpu",
        )
        self.done, self.produced, self.release = Event(), Event(), Event()
        self.block = False
        self.chunks = []
        self.request_keys = []

    def _run_request(self, request):
        self.request_keys.append((request.request_id, request.reset_epoch, request.task_epoch))
        super()._run_request(request)

    def _prepare_queue_actions(self, actions, metrics):
        result = super()._prepare_queue_actions(actions, metrics)
        self.chunks.append(result)
        if self.block:
            self.produced.set()
            assert self.release.wait(3)
        return result

    def _request_finished(self, request):
        self.done.set()

    def request(self):
        self.done.clear()
        self.notify_observation({f"state_{i}": 0.0 for i in range(8)})
        assert self.done.wait(3)
        assert not self.failed, self.failure_traceback

    def startup(self):
        self.start()
        self.resume()
        for _ in range(3):
            self.request()
        assert self.ready

    def drain_to(self, target):
        for _ in range(self.queue.available_steps() - target):
            action = self.get_action(None)
            assert action.device.type == "cpu"


def test_real_loop_owner_cpu_publication_and_ready_reset(fake_graph):
    engine = ObservedEngine()
    original = engine.policy.model.sample_actions
    try:
        engine.startup()
        engine.drain_to(30)
        engine.block = True
        engine.done.clear()
        engine.notify_observation({f"state_{i}": 0.0 for i in range(8)})
        assert engine.produced.wait(3)
        old_epoch = engine.queue.reset_epoch
        reset_count = len(engine.policy.reset_threads)
        engine.reset()
        assert engine.queue.reset_epoch == old_epoch + 1 and engine.queue.available_steps() == 0
        assert len(engine.policy.reset_threads) == reset_count
        engine.release.set()
        assert engine.done.wait(3)
        assert engine.stats.stale_results == 1 and engine.queue.available_steps() == 0
        engine.block = False
        engine.request()
        assert len(engine.policy.reset_threads) == reset_count + 1
        assert all(
            t == engine.owner_thread != get_ident()
            for t in engine.policy.reset_threads + engine.pre.threads + engine.post.threads
        )
        assert all(engine.policy.inference_modes)
        a, b = engine.chunks[-1]
        assert a.device.type == b.device.type == "cpu" and a.data_ptr() != b.data_ptr()
        held = b.clone()
        with torch.inference_mode():
            a.zero_()
        assert torch.equal(b, held)
    finally:
        engine.release.set()
        engine.stop()
    assert engine.worker_joined and engine.graph_released and engine.original_sampler_restored
    assert engine.policy.model.sample_actions == original


def test_reset_before_capture_finishes_preserves_startup_failure(fake_graph, monkeypatch):
    entered, release = Event(), Event()

    class BlockedGraph(FakeGraph):
        def __init__(self, *args):
            entered.set()
            assert release.wait(3)
            super().__init__(*args)

    monkeypatch.setattr(runtime_module, "TokenGraph", BlockedGraph)
    engine = ObservedEngine()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({f"state_{i}": 0.0 for i in range(8)})
        assert entered.wait(3)
        reset_count = len(engine.policy.reset_threads)
        engine.reset()
        assert engine.queue.reset_epoch == 1 and engine.failed and not engine.ready
        assert len(engine.policy.reset_threads) == reset_count
        release.set()
        assert engine.done.wait(3)
    finally:
        release.set()
        engine.stop()
    assert engine.worker_joined and engine.graph_released and engine.original_sampler_restored
    assert len(engine.policy.reset_threads) == reset_count + 1
    assert all(t == engine.owner_thread for t in engine.policy.reset_threads)


def test_task_aba_stale_result_and_new_capture_generation(fake_graph):
    engine = ObservedEngine()
    try:
        engine.startup()
        engine.drain_to(30)
        engine.block = True
        engine.done.clear()
        engine.notify_observation({f"state_{i}": 0.0 for i in range(8)})
        assert engine.produced.wait(3)
        assert engine.set_task("B") and engine.set_task("A")
        engine.release.set()
        assert engine.done.wait(3)
        assert engine.stats.stale_results == 1 and not engine.queue.has_staged_chunk()
        assert engine.get_action(None).device.type == "cpu" and engine.dispatched_task == "A"
        engine.block = False
        previous_captures = len(engine.runtime.captures)
        engine.request()
        assert len(engine.runtime.captures) == previous_captures + 1
        assert engine.runtime.task[1] == 2
    finally:
        engine.release.set()
        engine.stop()


def test_join_timeout_is_not_reported_as_release(fake_graph, monkeypatch):
    from lerobot.rollout.inference import predictive_async

    monkeypatch.setattr(predictive_async, "_JOIN_TIMEOUT_S", 0.01)
    engine = ObservedEngine()
    try:
        engine.startup()
        engine.drain_to(30)
        engine.block = True
        engine.done.clear()
        engine.notify_observation({f"state_{i}": 0.0 for i in range(8)})
        assert engine.produced.wait(3)
        engine.stop()
        assert not engine.worker_joined and not engine.graph_released
        assert engine.queue.available_steps() == 0
        engine.release.set()
        assert engine.done.wait(3)
    finally:
        engine.release.set()
        engine.stop()
    assert engine.worker_joined and engine.graph_released


def test_first_nonstartup_model_error_is_fatal(fake_graph):
    engine = ObservedEngine()
    try:
        engine.startup()
        engine.drain_to(30)

        def fail(*args, **kwargs):
            raise ValueError("first model error")

        engine.policy.model.sample_actions = fail
        engine.done.clear()
        engine.notify_observation({f"state_{i}": 0.0 for i in range(8)})
        assert engine.done.wait(3)
    finally:
        engine.stop()
    assert engine.failed and "first model error" in engine.failure_traceback
    assert len(engine.request_keys) == 4
    assert engine.worker_joined and engine.original_sampler_restored


def test_fixed_twelve_event_controller_with_real_cpu_worker_loop(fake_graph, monkeypatch):
    import validate_smolvla_graph_worker as campaign

    class CPUContract(campaign.ContractEngine):
        def __init__(self, *args):
            super().__init__(*args, device="cpu")

    monkeypatch.setattr(campaign, "ContractEngine", CPUContract)
    budget, result, arrays = campaign.Budget(), {}, {}
    observations = [{f"state_{i}": 0.0 for i in range(8)} for _ in range(10)]
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]}
    }
    campaign.worker_contract(
        Policy(), Processor(), Processor(), observations, features, budget, result, arrays
    )
    assert result["graph_identity_engine_integration_passed"] and result["graph_worker_lifecycle_passed"]
    assert (budget.main_calls, budget.reference_calls, budget.capture_attempts) == (12, 12, 5)
    assert result["worker_stats"]["stale_results"] == 4
    assert [row["label"] for row in result["worker_requests"]] == [row[0] for row in campaign.WORKER_EVENTS]
