"""Check the registered LIBERO wiring without loading weights or creating a renderer."""

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from libero_reference_smoke import (  # noqa: E402
    ENVIRONMENT_SEED,
    action_outside_bounds,
    array_record,
    policy_observation,
    reset_episode,
    terminal_reason,
)

from lerobot.envs.libero import LiberoEnv  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def test_native_camera_mapping_orientation_scale_and_state_semantics():
    """Distinct corners detect a swapped camera or missing/double image rotation."""
    front = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.zeros_like(front)
    front[0, 0] = [255, 0, 0]
    front[-1, -1] = [0, 128, 0]
    wrist[0, -1] = [0, 0, 255]
    native = SimpleNamespace(
        _env=SimpleNamespace(robots=[SimpleNamespace(controller=SimpleNamespace(ee_ori_mat=np.eye(3)))]),
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"},
        obs_type="pixels_agent_pos",
    )
    raw = LiberoEnv._format_raw_obs(
        native,
        {
            "agentview_image": front,
            "robot0_eye_in_hand_image": wrist,
            "robot0_eef_pos": np.array([0.1, 0.2, 0.7]),
            "robot0_eef_quat": np.array([0.0, 0.0, 2**-0.5, 2**-0.5]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
        },
    )
    batch = policy_observation(raw, "pick up the alphabet soup and place it in the basket")
    image = batch["observation.images.image"]
    image2 = batch["observation.images.image2"]
    assert image.shape == image2.shape == (1, 3, 256, 256)
    assert image.dtype == torch.float32
    torch.testing.assert_close(image[0, :, -1, -1], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(image[0, :, 0, 0], torch.tensor([0.0, 128 / 255, 0.0]))
    torch.testing.assert_close(image2[0, :, -1, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(
        batch[OBS_STATE], torch.tensor([[0.1, 0.2, 0.7, 0, 0, np.pi / 2, 0.02, -0.02]])
    )
    policy = SimpleNamespace(config=SimpleNamespace(max_state_dim=32))
    padded = SmolVLAPolicy.prepare_state(policy, batch)
    assert padded.shape == (1, 32)
    torch.testing.assert_close(padded[:, :8], batch[OBS_STATE])
    assert torch.count_nonzero(padded[:, 8:]) == 0


def test_actual_selector_generates_fifty_and_consumes_one_on_every_observation():
    """Exercise the production selector, with only expensive generation replaced."""
    calls = []
    policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=1))
    SmolVLAPolicy.reset(policy)
    policy.eval = lambda: policy
    policy._rtc_enabled = lambda: False
    policy._prepare_batch = lambda batch: batch
    policy._check_get_actions_condition = lambda: not policy._queues[ACTION]

    def generate(batch, noise):
        calls.append(batch["index"])
        return (torch.arange(50) + batch["index"] * 100).float()[None, :, None].expand(1, 50, 7)

    policy._get_action_chunk = generate
    for index in range(20):
        selected = SmolVLAPolicy.select_action(policy, {"index": index})
        assert selected.shape == (1, 7)
        assert torch.all(selected == index * 100)
        assert len(policy._queues[ACTION]) == 0
    assert calls == list(range(20))


def test_episode_reset_clears_the_actual_queue_and_both_processors():
    calls = []
    policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=1), _queues={ACTION: deque(["stale"])})
    policy.reset = lambda: SmolVLAPolicy.reset(policy)
    pre = SimpleNamespace(reset=lambda: calls.append("pre"))
    post = SimpleNamespace(reset=lambda: calls.append("post"))

    def reset(seed):
        assert seed == ENVIRONMENT_SEED
        assert not policy._queues[ACTION]
        assert calls == ["pre", "post"]
        return {"new": True}, {"is_success": False}

    assert reset_episode(SimpleNamespace(reset=reset), policy, pre, post)[0] == {"new": True}


@pytest.mark.parametrize(
    "terminated,truncated,success,reason",
    [
        (True, False, True, "native_success"),
        (True, True, True, "native_success"),
        (False, True, False, "time_limit"),
        (True, False, False, "native_termination_without_success"),
        (False, False, False, None),
    ],
)
def test_success_and_time_limit_are_separate(terminated, truncated, success, reason):
    assert terminal_reason(terminated, truncated, {"is_success": success}) == reason


def test_native_step_keeps_success_and_time_limit_wrapper_reports_timeout():
    """LiberoEnv's length metadata alone does not create a truncation signal."""

    class NativeTask:
        def step(self, action):
            np.testing.assert_array_equal(action, np.arange(7))
            return {"terminal_observation": 1}, 0.0, False, {}

        def check_success(self):
            return False

    raw = SimpleNamespace(
        _env=NativeTask(), task="task0", task_id=0, _ensure_env=lambda: None, _format_raw_obs=lambda obs: obs
    )
    action = np.arange(7)
    observation, _, terminated, truncated, info = LiberoEnv.step(raw, action)
    assert observation == {"terminal_observation": 1}
    assert not terminated and not truncated and not info["is_success"]

    class OneStepEnv(gym.Env):
        def reset(self, **kwargs):
            return {}, {}

        def step(self, action):
            return LiberoEnv.step(raw, action)

    wrapped = gym.wrappers.TimeLimit(OneStepEnv(), max_episode_steps=1)
    wrapped.reset()
    _, _, terminated, truncated, info = wrapped.step(action)
    assert terminal_reason(terminated, truncated, info) == "time_limit"


def test_finite_out_of_box_values_reach_native_handling_unchanged():
    action = np.array([2.0, -3.0, 0.1, 0.2, 0.3, 0.4, -1.2], dtype=np.float32)
    original = action.copy()
    assert action_outside_bounds(action) == [0, 1, 6]
    np.testing.assert_array_equal(action, original)


@pytest.mark.parametrize("action", [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)])
def test_invalid_actions_stop_before_execution_and_remain_recordable(action):
    assert array_record(action)["shape"] == list(action.shape)
    with pytest.raises(ValueError, match="finite seven-dimensional"):
        action_outside_bounds(action)
