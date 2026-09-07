"""Real-worker L13 binding: same committed prefix, future state not renormalized."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from eval_leisaac_so101 import prepare_engine, prime_episode_engine, stop_engine  # noqa: E402
from leisaac_so101_matched import physical_to_motor  # noqa: E402
from leisaac_so101_predicted import SO101PredictiveAsyncInferenceEngine  # noqa: E402
from run_so101_joint_study import ARMS, task_summary  # noqa: E402
from so101_feasible_actions import FeasibleActionProjector  # noqa: E402
from so101_future_state import FutureStateResidual  # noqa: E402
from so101_joint_runtime import STATE_SOURCE, JointContextPredictor, SO101JointAsyncEngine  # noqa: E402
from test_leisaac_so101_predicted import arguments  # noqa: E402


@pytest.mark.parametrize("variant", ["joint", "state_only"])
def test_real_worker_uses_causal_state_override_and_preserves_visual_attribution(variant):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    raw, args = arguments()
    visual = args["future_latent_predictor"]
    visual.config = SimpleNamespace(max_prediction_delay=8)
    state_model = FutureStateResidual().eval().requires_grad_(False)
    state_model.network[-1].bias.fill_(0.25)
    state_model._l12_binding = {"source": STATE_SOURCE, "epoch": 29, "parameters": 25478}
    wrapper = JointContextPredictor(visual, state_model, variant)
    args["future_latent_predictor"] = wrapper
    with pytest.raises(ValueError, match="independently loaded L6"):
        SO101PredictiveAsyncInferenceEngine(**args)
    policy = args["policy"]
    original, seen = policy.predict_action_chunk, []

    def tracked(batch, **kwargs):
        output = original(batch, **kwargs)
        if "future_state" in kwargs:
            seen.append((policy.prepare_state(batch).clone(), kwargs["future_state"].clone()))
            output += kwargs["future_state"][0, 0]
        return output

    policy.predict_action_chunk = tracked
    engine = SO101JointAsyncEngine(
        action_projector=FeasibleActionProjector(torch.zeros(6), torch.full((6,), 10.0)), **args
    )
    try:
        prepare_engine(engine, raw, timeout=10)
        prime_episode_engine(engine, raw, timeout=10)
        torch.testing.assert_close(physical_to_motor(engine.get_action(None)), torch.full((6,), 0.1))
        engine.notify_observation(raw)
        deadline = time.monotonic() + 5
        while not engine.queue.has_staged_chunk():
            assert not engine.failed and time.monotonic() < deadline
            time.sleep(0.005)
        current, future = seen[-1]
        torch.testing.assert_close(future[:, :6], current[:, :6] + 0.25, rtol=0, atol=0)
        torch.testing.assert_close(future[:, 6:], current[:, 6:], rtol=0, atol=0)
        if variant == "joint":
            torch.testing.assert_close(visual.seen[-1][2], current, rtol=0, atol=0)
        else:
            assert not visual.seen
        plan = engine.queue.plan_snapshot()
        for _ in range(plan.planned_delay_steps):
            engine.get_action(None)
        actual = physical_to_motor(engine.get_action(None))
        torch.testing.assert_close(
            actual, torch.full((6,), 3.6 if variant == "joint" else 2.6), atol=1e-5, rtol=1e-5
        )
    finally:
        stop_engine(engine)
        torch.set_num_threads(old_threads)
    planned = [e for e in args["metrics_sink"].events if e.get("request_kind") == "planned"]
    assert planned and all(e["state_predictor_calls"] == 1 for e in planned)
    assert all(e["visual_predictor_calls"] == int(variant == "joint") for e in planned)
    assert args["metrics_sink"].closed


def test_task_gate_keeps_reference_and_state_only_attribution_separate():
    rows = []
    costs = {"joint": 10, "state_only": 5}
    for block in range(8):
        for arm in ARMS:
            rows.append(
                {
                    "block": block,
                    "arm": arm,
                    "first_placement_success": True,
                    "technical_valid": True,
                    "restricted_placement_time_s": costs.get(arm, 40),
                    "initial_state": [0],
                    "initial_objects": {"orange": [0]},
                    "initial_camera_poses": {"front": [0]},
                }
            )
    result = task_summary(rows)
    assert result["benefit_on_reliable_reference"]
    assert not result["contrasts"]["state_only"]["stable_gate"]
    for row in rows:
        if row["arm"] == "sync" and row["block"] < 2:
            row["first_placement_success"] = False
    assert not task_summary(rows)["benefit_on_reliable_reference"]
