"""A documented initial state must not be confused with a task controller."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from leisaac_so101_contract import JOINT_NAMES, ContractError, state_from_radians  # noqa: E402
from run_so101_prepared_start import select_medoid, summarize  # noqa: E402
from so101_prepared_start import (  # noqa: E402
    MOTOR_STATE,
    PROFILE,
    prepared_degrees,
    prepared_joints,
    verify_reset,
)


def test_medoid_uses_all_first_states_and_lowest_episode_breaks_ties():
    points = np.array([[0] * 6, [10] * 6, [11] * 6, [90] * 6], dtype=float)
    assert select_medoid(points) == 1
    with pytest.raises(ValueError, match="finite"):
        select_medoid(np.full((3, 6), np.nan))


def test_native_motor_mapping_is_not_direct_degrees_and_gripper_is_not_double_scaled():
    joints = prepared_joints()
    degrees = prepared_degrees()
    assert degrees[0] == pytest.approx(MOTOR_STATE[0] * 1.1)
    assert degrees[2] == pytest.approx(MOTOR_STATE[2] * 0.95 - 5)
    assert degrees[3] == pytest.approx(MOTOR_STATE[3] * 0.95)
    assert degrees[-1] == pytest.approx(-10 + 1.1 * MOTOR_STATE[-1])
    native = state_from_radians(list(joints.values()), list(joints))
    assert native[:5] == pytest.approx(degrees[:5])
    assert native[-1] == pytest.approx(MOTOR_STATE[-1])


def test_reset_verification_uses_measured_joint_names_and_rejects_the_old_zero_pose():
    expected = prepared_joints()
    names = list(reversed(JOINT_NAMES))
    verify_reset([expected[n] for n in names], names)
    verify_reset([float(np.float32(expected[n])) for n in names], names)
    with pytest.raises(ContractError, match="Measured reset"):
        verify_reset([0.0] * 6, list(JOINT_NAMES))
    with pytest.raises(ContractError, match="Measured reset"):
        verify_reset([math.radians(1)] * 6, list(JOINT_NAMES))
    with pytest.raises(ContractError, match="Measured reset"):
        verify_reset([float("nan")] * 6, list(JOINT_NAMES))


def test_single_object_success_does_not_qualify_the_prepared_native_baseline():
    rows = [
        {
            "block": b,
            "pose": pose,
            "native_success": False,
            "subgoal_success": True,
            "technical_valid": True,
            "native_time_s": 120.0,
            "initial_objects": {},
            "initial_cameras": {"front": {}},
        }
        for b in range(8)
        for pose in ("zero", PROFILE)
    ]
    result = summarize(rows)
    assert all(not r["qualified_development"] for r in result["aggregates"].values())
    assert result["descriptive_paired_bootstrap_95_s"] == [0.0, 0.0]
    assert not result["forecasting_benefit_tested"] and not result["realtime_qualified"]


def test_native_reset_is_measured_not_rewritten_by_adapter():
    from leisaac_so101_env_server import IsaacEnvironment

    measured = torch.tensor([list(prepared_joints().values())])
    obs = {"policy": {"joint_pos": measured}}
    env = IsaacEnvironment.__new__(IsaacEnvironment)
    env.initial_pose = PROFILE
    env.front_reset_anchor = None
    env.joint_names = list(JOINT_NAMES)
    env.env = SimpleNamespace(reset=lambda *, seed: (obs, {}))
    env.packet = lambda observed, episode, step: (observed, episode, step)
    assert env.reset(1, 2) == (obs, 2, 0)
    measured.zero_()
    with pytest.raises(ContractError, match="Measured reset"):
        env.reset(1, 3)
    assert not measured.any()
