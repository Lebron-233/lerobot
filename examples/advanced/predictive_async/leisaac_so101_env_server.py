"""Isolated Python 3.11 LeIsaac process; no import of this LeRobot checkout.

Started only by eval_leisaac_so101.py over an inherited local connection.
The existing Gym task remains the simulator; this file is an interface adapter.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import math
import os
import pstats
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
    REST_POSE_DEG,
    TASK_ID,
    ContractError,
    action_to_radians,
    joint_indices,
    observation_packet,
)


class IsaacEnvironment:
    def __init__(
        self,
        assets_root: Path,
        leisaac_root: Path,
        device: str,
        *,
        profile_steps: bool = False,
        episode_seconds: int = 25,
        control_fps: int = 30,
        initial_pose: str = "zero",
        camera_backend: str = "tiled",
        task_evidence: bool = False,
        gc_diagnostics: bool = False,
    ) -> None:
        self.control_fps = control_fps
        self.step_profiler = cProfile.Profile() if profile_steps else None
        self.slowest_step_profiles: list[dict] = []
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

        # This machine has six physical cores. Limit host worker oversubscription;
        # keep the same GPU physics, rendering, sensors and step sizes.
        self.app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device=device,
            kit_args=(
                "--/plugins/carb.tasking.plugin/threadCount=6 "
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=6"
            ),
        ).app
        self.env = None
        self.gc_trace = None
        try:
            import gymnasium as gym
            import leisaac
            import torch
            from isaaclab.sensors import Camera
            from isaaclab_tasks.utils import parse_env_cfg
            from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

            source = Path(leisaac.__file__).resolve()
            if not source.is_relative_to(leisaac_root.resolve()):
                raise ContractError("Imported LeIsaac is not the pinned local checkout")
            if tuple(tuple(SO101_FOLLOWER_USD_JOINT_LIMLITS[n]) for n in JOINT_NAMES) != JOINT_LIMITS_DEG:
                raise ContractError("Simulator joint limits changed")
            cfg = parse_env_cfg(TASK_ID, device=device, num_envs=1)
            cfg.use_teleop_device("so101leader")
            if initial_pose == "rest":
                cfg.scene.robot.init_state.joint_pos = dict(
                    zip(JOINT_NAMES, (math.radians(value) for value in REST_POSE_DEG), strict=True)
                )
            cfg.recorders = None
            cfg.episode_length_s = episode_seconds
            self.task_witness = None
            if task_evidence:
                from so101_task_evidence import NativeTaskWitness

                self.task_witness = NativeTaskWitness(cfg.terminations.success.func)
                cfg.terminations.success.func = self.task_witness
            cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1 / 60, 60 // control_fps, 2
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
                if camera_backend == "standard":
                    camera.class_type = Camera
            self.env = gym.make(TASK_ID, cfg=cfg).unwrapped
            if gc_diagnostics:
                from so101_gc_timing import GcTrace

                self.gc_trace = GcTrace()
            if task_evidence:
                # Isaac managers deepcopy term configs. Read the instance that
                # actually runs, not the pre-construction configuration object.
                self.task_witness = self.env.termination_manager.get_term_cfg("success").func
            self.torch = torch
            self.front_reset_anchor = None
            if camera_backend == "standard":
                self.front_reset_anchor = tuple(
                    value.clone() for value in self.env.scene["front"]._view.get_world_poses()
                )
            self.joint_names = list(self.env.scene["robot"].data.joint_names)
            joint_indices(self.joint_names)
            if set(self.env.termination_manager.active_terms) != {"success", "time_out"}:
                raise ContractError(
                    "Cannot equate terminated with success for this termination configuration"
                )
            if abs(self.env.step_dt - 1 / control_fps) > 1e-9:
                raise ContractError("Simulator step_dt differs from the selected execution time base")
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
                "camera_backend": camera_backend,
                "task_evidence": task_evidence,
                "gc_diagnostics": gc_diagnostics,
                "front_reset_anchor": "nominal" if camera_backend == "standard" else "native_tiled",
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
                "initial_pose": initial_pose,
                "initial_joint_positions_radians": dict(cfg.scene.robot.init_state.joint_pos),
                "physics_dt": self.env.physics_dt,
                "step_dt": self.env.step_dt,
                "control_fps": control_fps,
                "timebase_profile": "native60_diagnostic" if control_fps == 60 else "transfer30",
                "device": str(self.env.device),
                "physics_execution": "cpu_physx_rtx_v1" if device == "cpu" else "gpu_physx_rtx_v1",
                "physics_solver_type": cfg.sim.physx.solver_type,
                "renderer_device": "cuda:0",
                "step_cprofile_enabled": profile_steps,
                "runtime_settings": {
                    key: self.env.sim.carb_settings.get(key)
                    for key in (
                        "/app/runLoops/main/rateLimitEnabled",
                        "/app/runLoops/main/rateLimitFrequency",
                        "/isaaclab/render/active_viewport",
                        "/rtx/ecoMode/enabled",
                        "/plugins/carb.tasking.plugin/threadCount",
                        "/plugins/omni.tbb.globalcontrol/maxThreadCount",
                        "/persistent/physics/numThreads",
                    )
                },
                "torch_num_threads": torch.get_num_threads(),
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
        packet = observation_packet(
            measured=obs["policy"]["joint_pos"][0].detach().cpu().tolist(),
            actual_names=self.joint_names,
            images=images,
            camera_frames={name: int(self.env.scene[name].frame[0].item()) for name in images},
            episode_id=episode_id,
            step=step,
            snapshot_ready_at_s=time.perf_counter(),
            control_fps=self.control_fps,
        )
        packet["task_diagnostics"] = {
            "subtasks": {name: bool(value[0].item()) for name, value in obs.get("subtask_terms", {}).items()},
            "object_positions_world": {
                name: self.env.scene[name].data.root_pos_w[0].detach().cpu().tolist()
                for name in ("Orange001", "Orange002", "Orange003", "Plate")
            },
        }
        # World-transform reads diagnose reset identity, not each control tick.
        # Keep images/frame counters/state at full rate without this USD readback.
        if step == 0:
            packet["camera_world_poses_opengl"] = {}
            for name in ("front", "wrist"):
                position, quaternion = self.env.scene[name]._view.get_world_poses()
                packet["camera_world_poses_opengl"][name] = {
                    "position": position[0].detach().cpu().tolist(),
                    "quaternion_wxyz": quaternion[0].detach().cpu().tolist(),
                }
        return packet

    def reset(self, seed: int, episode_id: int) -> dict:
        if self.front_reset_anchor is not None:
            self.env.scene["front"].set_world_poses(*self.front_reset_anchor, convention="opengl")
        obs, _ = self.env.reset(seed=seed)
        return self.packet(obs, episode_id, 0)

    def step(self, action: list[float], episode_id: int, step: int) -> dict:
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

        started = time.perf_counter()
        gc_start = len(self.gc_trace.events) if self.gc_trace is not None else 0
        if self.env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(self.env, "so101leader")
        effort_finished = time.perf_counter()
        target = self.torch.tensor(
            [action_to_radians(action)], dtype=self.torch.float32, device=self.env.device
        )
        target_finished = time.perf_counter()
        if self.step_profiler is None:
            obs, reward, terminated, truncated, _ = self.env.step(target)
        else:
            self.step_profiler = cProfile.Profile()
            obs, reward, terminated, truncated, _ = self.step_profiler.runcall(self.env.step, target)
        # Copy flags before packet/next reset: returned obs may already be reset.
        success, timeout = bool(terminated[0].item()), bool(truncated[0].item())
        step_finished = time.perf_counter()
        if self.step_profiler is not None:
            duration = step_finished - target_finished
            if len(self.slowest_step_profiles) < 5 or duration > self.slowest_step_profiles[-1]["wall_s"]:
                self.slowest_step_profiles.append(
                    {"step": step, "wall_s": duration, **summarize_step_profile(self.step_profiler)}
                )
                self.slowest_step_profiles.sort(key=lambda row: row["wall_s"], reverse=True)
                del self.slowest_step_profiles[5:]
        result = {
            "observation": self.packet(obs, episode_id, step),
            "reward": float(reward[0].item()),
            "terminated": success,
            "truncated": timeout,
            "post_reset_observation": success or timeout,
        }
        if self.task_witness is not None:
            if self.task_witness.latest is None:
                raise ContractError("Native pre-reset task evidence was not captured")
            result["task_transition"] = self.task_witness.latest
        # Host wall intervals, including any CUDA waits at these boundaries;
        # not claimed to be kernel-only physics/render durations.
        result["server_timing_s"] = {
            "gripper_effort": effort_finished - started,
            "target_creation": target_finished - effort_finished,
            "env_step_and_flags": step_finished - target_finished,
            "observation_packet": time.perf_counter() - step_finished,
        }
        if self.gc_trace is not None:
            result["gc_collections"] = self.gc_trace.events[gc_start:]
        return result

    def profile_report(self) -> dict | None:
        """Materialize diagnostic host-call statistics only after control stops."""
        if self.step_profiler is None:
            return None
        return {"kind": "slowest_five_instrumented_steps", "steps": self.slowest_step_profiles}

    def close(self) -> None:
        try:
            if self.gc_trace is not None:
                self.gc_trace.close()
                self.gc_trace = None
            if self.env is not None:
                self.env.close()
        finally:
            self.app.close()


def summarize_step_profile(profiler: cProfile.Profile) -> dict:
    stats = pstats.Stats(profiler)
    rows = [
        {
            "file": filename,
            "line": line,
            "function": name,
            "primitive_calls": values[0],
            "calls": values[1],
            "self_s": values[2],
            "cumulative_s": values[3],
        }
        for (filename, line, name), values in stats.stats.items()
    ]
    return {
        "kind": "cProfile_host_wall_with_instrumentation_overhead",
        "total_self_s": stats.total_tt,
        "by_cumulative": sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)[:50],
        "by_self": sorted(rows, key=lambda row: row["self_s"], reverse=True)[:30],
    }


def serve(connection: Connection, env: Any) -> None:
    """One environment and one request at a time; require reset after termination."""
    episode_id, step, needs_reset = 0, 0, True
    connection.send({"metadata": env.metadata})
    while True:
        request = connection.recv()
        operation = request["op"]
        if operation == "close":
            response = {"closed": True}
            if hasattr(env, "profile_report"):
                response["step_profile"] = env.profile_report()
            connection.send(response)
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
    parser.add_argument("--episode-seconds", type=int, choices=(25, 60, 120), default=25)
    parser.add_argument("--control-fps", type=int, choices=(30, 60), default=30)
    parser.add_argument("--initial-pose", choices=("zero", "rest"), default="zero")
    parser.add_argument("--camera-backend", choices=("tiled", "standard"), default="tiled")
    parser.add_argument("--profile-steps", action="store_true")
    parser.add_argument("--task-evidence", action="store_true")
    parser.add_argument("--gc-diagnostics", action="store_true")
    args = parser.parse_args()
    connection, env = Connection(args.fd), None
    try:
        env = IsaacEnvironment(
            args.assets_root,
            args.leisaac_root,
            args.device,
            profile_steps=args.profile_steps,
            episode_seconds=args.episode_seconds,
            control_fps=args.control_fps,
            initial_pose=args.initial_pose,
            camera_backend=args.camera_backend,
            task_evidence=args.task_evidence,
            gc_diagnostics=args.gc_diagnostics,
        )
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
