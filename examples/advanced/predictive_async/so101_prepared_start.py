"""L19 fixed training-first-state medoid, not an outcome-selected trajectory."""

from __future__ import annotations

import math

from leisaac_so101_contract import JOINT_LIMITS_DEG, JOINT_NAMES, ContractError

PROFILE = "training_medoid_v1"
DATASET = "LightwheelAI/leisaac-pick-orange"
DATASET_REVISION = "fa6e0625d814352b8e6ee1c6d2482194e4da8ed3"
EPISODE = 9
MOTOR_STATE = (
    -11.096244812011719,
    -47.54875946044922,
    55.004150390625,
    50.10224914550781,
    0.10279083251953125,
    8.447356224060059,
)


def prepared_degrees() -> tuple[float, ...]:
    """Convert native motor coordinates to joint degrees, including the gripper."""
    values = tuple(
        (value + 100) * (high - low) / 200 + low
        for value, (low, high) in zip(MOTOR_STATE[:5], JOINT_LIMITS_DEG[:5], strict=True)
    ) + (-10 + 1.1 * MOTOR_STATE[5],)
    if any(not low <= v <= high for v, (low, high) in zip(values, JOINT_LIMITS_DEG, strict=True)):
        raise ContractError("The frozen training medoid is outside the native joint bounds")
    return values


def prepared_joints() -> dict[str, float]:
    return dict(zip(JOINT_NAMES, map(math.radians, prepared_degrees()), strict=True))


def verify_reset(measured: list[float], names: list[str]) -> None:
    """Fail before task dispatch if native reset did not install the registered pose."""
    expected = prepared_joints()
    if set(names) != set(JOINT_NAMES) or len(measured) != 6 or len(names) != 6:
        raise ContractError("Prepared-start measured joints changed")
    if any(
        not math.isfinite(value) or abs(value - expected[name]) > 1e-5
        for value, name in zip(measured, names, strict=True)
    ):
        raise ContractError("Measured reset differs from the frozen training-medoid initial pose")


def provenance() -> dict:
    return {
        "profile": PROFILE,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "episode": EPISODE,
        "frame": 0,
        "selection": "minimum_mean_range_scaled_L1_to_all60_first_states_tie_lowest_episode",
        "motor_state": list(MOTOR_STATE),
        "joint_degrees": list(prepared_degrees()),
        "scope": "initialization_only_no_demonstration_actions_or_task_state_writes",
        "checkpoint_training_dataset_revision_unpinned": True,
    }
