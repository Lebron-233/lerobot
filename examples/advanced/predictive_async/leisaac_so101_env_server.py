"""Isolated Python 3.11 LeIsaac process; no import of this LeRobot checkout.

Started only by eval_leisaac_so101.py over an inherited local connection.
The existing Gym task remains the simulator; this file is an interface adapter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from leisaac_so101_contract import (
    ASSETS_REVISION,
    FPS,
    IMAGE_SHAPE,
    JOINT_LIMITS_DEG,
    JOINT_NAMES,
    LEISAAC_REVISION,
    PROFILE,
    TASK_ID,
    ContractError,
    action_to_radians,
    joint_indices,
    observation_packet,
)


class IsaacEnvironment:
    def __init__(self, assets_root: Path, leisaac_root: Path, device: str) -> None:
        if sys.version_info[:2] != (3, 11):
            raise ContractError("The pinned simulator requires a separate Python 3.11 interpreter")
        for name, expected in (("isaaclab", "2.3.0"), ("isaacsim", "5.1.0.0")):
            if importlib.metadata.version(name) != expected:
                raise ContractError(f"The transfer profile requires {name}=={expected}")
        revision = subprocess.check_output(
            ["git", "-C", str(leisaac_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != LEISAAC_REVISION:
            raise ContractError("LeIsaac source revision differs from the pinned profile")
        if (
            ASSETS_REVISION not in assets_root.parts
            or not (assets_root / "robots/so101_follower.usd").is_file()
        ):
            raise ContractError("Expected the pinned Hub snapshot's assets directory")
        os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

        # Isaac requires AppLauncher before imports of environment/render modules.
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=True, device=device).app
        self.env = None
        try:
            import gymnasium as gym
            import leisaac
            import torch
            from isaaclab_tasks.utils import parse_env_cfg
            from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

            source = Path(leisaac.__file__).resolve()
            if not source.is_relative_to(leisaac_root.resolve()):
                raise ContractError("Imported LeIsaac is not the pinned local checkout")
            if tuple(tuple(SO101_FOLLOWER_USD_JOINT_LIMLITS[n]) for n in JOINT_NAMES) != JOINT_LIMITS_DEG:
                raise ContractError("Simulator joint limits changed")
            cfg = parse_env_cfg(TASK_ID, device=device, num_envs=1)
            cfg.use_teleop_device("so101leader")
            cfg.recorders = None
            cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1 / 60, 2, 2
            for term, names in (
                (cfg.actions.arm_action, JOINT_NAMES[:5]),
                (cfg.actions.gripper_action, JOINT_NAMES[5:]),
            ):
                term.joint_names = list(names)
                term.preserve_order = True
                term.use_default_offset = False
                term.scale, term.offset = 1.0, 0.0
            for camera in (cfg.scene.front, cfg.scene.wrist):
                camera.update_period = 1 / FPS
            self.env = gym.make(TASK_ID, cfg=cfg).unwrapped
            self.torch = torch
            self.joint_names = list(self.env.scene["robot"].data.joint_names)
            joint_indices(self.joint_names)
            if set(self.env.termination_manager.active_terms) != {"success", "time_out"}:
                raise ContractError(
                    "Cannot equate terminated with success for this termination configuration"
                )
            if abs(self.env.step_dt - 1 / FPS) > 1e-9:
                raise ContractError("Simulator step_dt is not 1/30 s")
            self.metadata = {
                "profile": PROFILE,
                "task": TASK_ID,
                "leisaac_revision": revision,
                "assets_revision": ASSETS_REVISION,
                "isaaclab": "2.3.0",
                "isaacsim": "5.1.0.0",
                "python": sys.version,
                "joint_names": self.joint_names,
                "camera_sources": {"top": "front", "wrist": "wrist"},
                "camera_config": {
                    name: {
                        "prim_path": getattr(cfg.scene, name).prim_path,
                        "offset_position": tuple(getattr(cfg.scene, name).offset.pos),
                        "offset_quaternion": tuple(getattr(cfg.scene, name).offset.rot),
                        "convention": getattr(cfg.scene, name).offset.convention,
                        "width": getattr(cfg.scene, name).width,
                        "height": getattr(cfg.scene, name).height,
                        "update_period": getattr(cfg.scene, name).update_period,
                    }
                    for name in ("front", "wrist")
                },
                "reset_randomization": "pinned upstream defaults preserved",
                "termination_terms": list(self.env.termination_manager.active_terms),
                "episode_length_s": cfg.episode_length_s,
                "physics_dt": self.env.physics_dt,
                "step_dt": self.env.step_dt,
                "device": str(self.env.device),
            }
        except BaseException:
            self.close()
            raise

    def packet(self, obs: dict, episode_id: int, step: int) -> dict:
        images = {}
        for name in ("front", "wrist"):
            image = obs["policy"][name]
            if tuple(image.shape) != (1, *IMAGE_SHAPE) or image.dtype != self.torch.uint8:
                raise ContractError(f"Actual {name} RGB has shape/dtype {image.shape}/{image.dtype}")
            images[name] = image[0].detach().cpu().contiguous().numpy().tobytes()
        return observation_packet(
            measured=obs["policy"]["joint_pos"][0].detach().cpu().tolist(),
            actual_names=self.joint_names,
            images=images,
            camera_frames={name: int(self.env.scene[name].frame[0].item()) for name in images},
            episode_id=episode_id,
            step=step,
            snapshot_ready_at_s=time.perf_counter(),
        )

    def reset(self, seed: int, episode_id: int) -> dict:
        obs, _ = self.env.reset(seed=seed)
        return self.packet(obs, episode_id, 0)

    def step(self, action: list[float], episode_id: int, step: int) -> dict:
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

        if self.env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(self.env, "so101leader")
        target = self.torch.tensor(
            [action_to_radians(action)], dtype=self.torch.float32, device=self.env.device
        )
        obs, reward, terminated, truncated, _ = self.env.step(target)
        # Copy flags before packet/next reset: returned obs may already be reset.
        success, timeout = bool(terminated[0].item()), bool(truncated[0].item())
        return {
            "observation": self.packet(obs, episode_id, step),
            "reward": float(reward[0].item()),
            "terminated": success,
            "truncated": timeout,
            "post_reset_observation": success or timeout,
        }

    def close(self) -> None:
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.app.close()


def serve(connection: Connection, env: Any) -> None:
    """One environment and one request at a time; require reset after termination."""
    episode_id, step, needs_reset = 0, 0, True
    connection.send({"metadata": env.metadata})
    while True:
        request = connection.recv()
        operation = request["op"]
        if operation == "close":
            connection.send({"closed": True})
            return
        if operation == "reset":
            episode_id += 1
            step, needs_reset = 0, False
            connection.send({"observation": env.reset(int(request["seed"]), episode_id)})
        elif operation == "step":
            if needs_reset or request["episode_id"] != episode_id or request["step"] != step:
                raise ContractError("step requires an explicit reset and the current episode/step")
            step += 1
            result = env.step(request["action"], episode_id, step)
            needs_reset = result["terminated"] or result["truncated"]
            connection.send(result)
        else:
            raise ContractError(f"Unsupported environment operation: {operation}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    connection, env = Connection(args.fd), None
    try:
        env = IsaacEnvironment(args.assets_root, args.leisaac_root, args.device)
        serve(connection, env)
        return 0
    except Exception:
        connection.send({"error": traceback.format_exc()})
        return 1
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
