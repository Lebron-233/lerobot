"""Observational PickOrange evidence, captured before automatic reset."""

from __future__ import annotations

import math
from typing import Any


def placement_flags(witness: dict[str, Any]) -> dict[str, bool]:
    """Strict diagnostic; does not replace the task's box+rest predicate."""
    plate = witness["plate_position"]
    flags = {}
    for name, item in witness["oranges"].items():
        x, y, z = (item["position"][i] - plate[i] for i in range(3))
        speed = math.sqrt(sum(v * v for v in item["linear_velocity"]))
        flags[name] = math.hypot(x, y) < 0.10 and abs(z) < 0.07 and speed <= 0.05
    return flags


class PlacementTracker:
    """Require consecutive occupancy; count simultaneous, not sticky, placements."""

    def __init__(self) -> None:
        self.streaks: dict[str, int] = {}
        self.max_simultaneous = 0
        self.last_simultaneous = 0
        self.first_settled_step: int | None = None

    def update(self, witness: dict[str, Any], step: int) -> int:
        for name, flag in placement_flags(witness).items():
            self.streaks[name] = self.streaks.get(name, 0) + 1 if flag else 0
        self.last_simultaneous = sum(value >= 10 for value in self.streaks.values())
        self.max_simultaneous = max(self.max_simultaneous, self.last_simultaneous)
        if self.last_simultaneous and self.first_settled_step is None:
            self.first_settled_step = step
        return self.last_simultaneous


class NativeTaskWitness:
    """Calls the original success function once and observes its pre-reset scene."""

    def __init__(self, native) -> None:
        self.native = native
        self.latest: dict[str, Any] | None = None

    def __call__(
        self,
        env,
        oranges_cfg,
        plate_cfg,
        x_range=(-0.10, 0.10),
        y_range=(-0.10, 0.10),
        height_range=(-0.07, 0.07),
    ):
        result = self.native(env, oranges_cfg, plate_cfg, x_range, y_range, height_range)
        robot = env.scene["robot"]
        self.latest = {
            "provenance": "native_termination_before_auto_reset",
            "native_success": bool(result[0].item()),
            "plate_position": env.scene[plate_cfg.name].data.root_pos_w[0].detach().cpu().tolist(),
            "oranges": {
                cfg.name: {
                    "position": env.scene[cfg.name].data.root_pos_w[0].detach().cpu().tolist(),
                    "linear_velocity": env.scene[cfg.name].data.root_lin_vel_w[0].detach().cpu().tolist(),
                }
                for cfg in oranges_cfg
            },
            "joint_positions_radians": robot.data.joint_pos[0].detach().cpu().tolist(),
        }
        return result
