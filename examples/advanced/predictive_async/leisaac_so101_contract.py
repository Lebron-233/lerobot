"""SO101 transfer coordinates and primitive-only wire format (Python 3.11/3.12).

No LeRobot, Isaac, torch or NumPy import belongs in this shared contract.
The profile is a transfer experiment, not SO100 calibration equivalence.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

PROFILE = "leisaac_so101_transfer_v1"
TASK_ID = "LeIsaac-SO101-PickOrange-v0"
TASK = "Pick three oranges and put them into the plate, then reset the arm to rest state."
LEISAAC_REVISION = "24d3bcd3f1e4585740fc79921782c41617237812"
ASSETS_REVISION = "6c35af0af55506eb75c5592930134d4af44e8341"
FPS = 30
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCALAR_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)
JOINT_LIMITS_DEG = ((-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160), (-10, 100))
IMAGE_SHAPE = (480, 640, 3)
IMAGE_BYTES = math.prod(IMAGE_SHAPE)
CAMERA_SOURCES = {"top": "front", "wrist": "wrist"}


class ContractError(ValueError):
    """An interface failure, never a task failure or permission to clip actions."""


def six_finite(values: Sequence[float]) -> list[float]:
    if len(values) != 6:
        raise ContractError("Expected exactly six joint values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ContractError("Joint values must be finite")
    return result


def joint_indices(actual_names: Sequence[str]) -> list[int]:
    if len(actual_names) != 6 or set(actual_names) != set(JOINT_NAMES):
        raise ContractError(f"Unexpected simulator joint names: {actual_names}")
    return [actual_names.index(name) for name in JOINT_NAMES]


def action_to_radians(action: Sequence[float]) -> list[float]:
    """Post-policy physical degrees/gripper -> absolute simulator targets.

    The 1e-4 degree comparison tolerance accommodates float32 round trips at
    endpoints. No value, including a within-tolerance value, is clipped.
    """
    degrees = six_finite(action)
    degrees[5] = -10.0 + 1.1 * degrees[5]
    for name, value, (low, high) in zip(JOINT_NAMES, degrees, JOINT_LIMITS_DEG, strict=True):
        if not low - 1e-4 <= value <= high + 1e-4:
            raise ContractError(f"Out-of-range {name} target: {value} degrees, limits {(low, high)}")
    return [math.radians(value) for value in degrees]


def state_from_radians(measured: Sequence[float], actual_names: Sequence[str]) -> list[float]:
    """Read measured positions by name; do not normalize or clip arm state."""
    values = six_finite(measured)
    degrees = [math.degrees(values[index]) for index in joint_indices(actual_names)]
    degrees[5] = (degrees[5] + 10.0) / 1.1
    return degrees


def observation_packet(
    *,
    measured: Sequence[float],
    actual_names: Sequence[str],
    images: Mapping[str, bytes],
    camera_frames: Mapping[str, int],
    episode_id: int,
    step: int,
    snapshot_ready_at_s: float,
) -> dict[str, Any]:
    packet = {
        "profile": PROFILE,
        "episode_id": episode_id,
        "step": step,
        "sim_time_s": step / FPS,
        "snapshot_ready_at_s": snapshot_ready_at_s,
        "state_f32": struct.pack("<6f", *state_from_radians(measured, actual_names)),
        "images": dict(images),
        "image_shape": IMAGE_SHAPE,
        "image_dtype": "uint8",
        "camera_sources": dict(CAMERA_SOURCES),
        "camera_frames": dict(camera_frames),
    }
    validate_observation(packet)
    return packet


def validate_observation(packet: Mapping[str, Any]) -> list[float]:
    if packet["profile"] != PROFILE or packet["camera_sources"] != CAMERA_SOURCES:
        raise ContractError("Unexpected transfer profile or camera provenance")
    if tuple(packet["image_shape"]) != IMAGE_SHAPE or packet["image_dtype"] != "uint8":
        raise ContractError("Expected uint8 HWC 480x640 RGB")
    if set(packet["images"]) != {"front", "wrist"}:
        raise ContractError("Expected exactly front and wrist images")
    for image in packet["images"].values():
        if not isinstance(image, bytes) or len(image) != IMAGE_BYTES:
            raise ContractError("Invalid RGB byte payload")
    state = packet["state_f32"]
    if not isinstance(state, bytes) or len(state) != 24:
        raise ContractError("Expected six little-endian float32 state values")
    return six_finite(struct.unpack("<6f", state))


def validate_step(previous: Mapping[str, Any], current: Mapping[str, Any], *, terminal: bool) -> None:
    if current["episode_id"] != previous["episode_id"] or current["step"] != previous["step"] + 1:
        raise ContractError("Environment observation episode/step is not aligned with one control tick")
    if not math.isclose(current["sim_time_s"], current["step"] / FPS, abs_tol=1e-9):
        raise ContractError("Environment time differs from the 30 Hz transfer contract")
    # Isaac returns reset observations on terminal steps. They must not be used
    # as the old episode's successor or subjected to a monotone camera test.
    if not terminal:
        for camera in ("front", "wrist"):
            if current["camera_frames"][camera] <= previous["camera_frames"][camera]:
                raise ContractError(f"Camera did not advance: {camera}")


def hardware_features() -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(SCALAR_KEYS)},
        **{f"observation.images.{name}": {"dtype": "video", "shape": IMAGE_SHAPE} for name in CAMERA_SOURCES},
    }
