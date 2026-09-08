"""Frozen LIBERO-Object wiring check: task 0, initial state 0, at most 20 actions.

Use the independent environment and pinned local snapshots described in
LIBERO_REFERENCE_SMOKE_PLAN.md. This entry point does not qualify a baseline.
"""

import argparse
import json
import os
import random
import subprocess
import time
import traceback
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

POLICY_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")
MEASUREMENT_LIMIT = 20
SETTLING_ACTIONS = 10
ENVIRONMENT_SEED = 4200
POLICY_SEED = 4201


def write_json(path, value):
    """Save one result with an explicit final newline."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_record(value):
    """Retain non-finite diagnostic values as strings in valid JSON."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "values": [float(x) if np.isfinite(x) else str(x) for x in value.reshape(-1)],
    }


def policy_observation(raw, task):
    """Use the existing environment transforms once, including their image rotation."""
    robot = raw["robot_state"]
    # preprocess_observation batches images, but nested state needs a vector-env batch axis.
    batched = {
        "pixels": raw["pixels"],
        "robot_state": {
            "eef": {key: np.asarray(robot["eef"][key])[None] for key in ("pos", "quat")},
            "gripper": {"qpos": np.asarray(robot["gripper"]["qpos"])[None]},
        },
    }
    batch = LiberoProcessorStep().observation(preprocess_observation(batched))
    batch["task"] = [task]
    return batch


def action_outside_bounds(action):
    """Reject unusable actions; leave finite values unchanged for native OSC/gripper handling."""
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Expected a finite seven-dimensional action, got {action!r}")
    return np.flatnonzero((action < -1) | (action > 1)).tolist()


def terminal_reason(terminated, truncated, info):
    """Keep the native success signal separate from termination and the time limit."""
    if info.get("is_success", False):
        return "native_success"
    if truncated:
        return "time_limit"
    if terminated:
        return "native_termination_without_success"
    return None


def reset_episode(env, policy, pre, post):
    """Clear all policy/processor episode state before the registered environment reset."""
    policy.reset()
    pre.reset()
    post.reset()
    return env.reset(seed=ENVIRONMENT_SEED)


def load_runtime(policy_path, vlm_path):
    """Load the complete pinned checkpoint with LeRobot's strict safetensors loader."""
    if policy_path.name != POLICY_REVISION or vlm_path.name != VLM_REVISION:
        raise ValueError("Use the registered policy and VLM snapshot directories")
    cfg = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    if (
        tuple(cfg.image_features) != CAMERA_KEYS
        or cfg.robot_state_feature.shape != (8,)
        or cfg.action_feature.shape != (7,)
        or (cfg.chunk_size, cfg.n_action_steps, cfg.num_steps) != (50, 1, 10)
        or (cfg.max_state_dim, cfg.max_action_dim) != (32, 32)
        or cfg.use_amp
        or not cfg.load_vlm_weights
        or cfg.adapt_to_pi_aloha
        or cfg.rtc_config is not None
    ):
        raise ValueError("The checkpoint differs from the registered LIBERO contract")
    cfg.vlm_model_name = str(vlm_path)
    cfg.pretrained_path = policy_path
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=cfg, local_files_only=True, strict=True)
    trainable_at_load = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    policy.eval().requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(vlm_path)}},
    )
    vlm = policy.model.vlm_with_expert
    dtypes = Counter()
    for p in policy.parameters():
        dtypes[str(p.dtype)] += p.numel()
    keys = ("state_proj.weight", "action_in_proj.weight", "action_out_proj.weight")
    report = {
        "strict": True,
        # load_model(strict=True) raises on any remaining missing/unexpected key or shape mismatch.
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "vlm_layers": vlm.num_vlm_layers,
        "expert_layers": vlm.num_expert_layers,
        "expert_hidden_size": vlm.expert_hidden_size,
        "parameter_dtypes_numel": dict(dtypes),
        "trainable_parameters_at_load": trainable_at_load,
        "trainable_parameters_for_inference": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "projection_shapes": {k: list(v.shape) for k, v in policy.state_dict().items() if k.endswith(keys)},
        "use_amp": cfg.use_amp,
        "preprocessor_steps": [type(s).__name__ for s in pre.steps],
        "postprocessor_steps": [type(s).__name__ for s in post.steps],
    }
    return policy, pre, post, report


def run_measurements(env, policy, pre, post, raw, task, output, result):
    """Consume the policy's actual select_action queue, logging before each env.step."""
    shapes = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda _module, _args, value: shapes.append(list(value.shape))
    )
    try:
        with (output / "steps.jsonl").open("x") as log, torch.inference_mode():
            for index in range(MEASUREMENT_LIMIT):
                batch = policy_observation(raw, task)
                row = {
                    "measurement_index": index,
                    "observation_after_measured_actions": index,
                    "state": array_record(batch[OBS_STATE]),
                    "eef_quaternion_xyzw": array_record(raw["robot_state"]["eef"]["quat"]),
                }
                write_json(output / "pending_step.json", row)
                processed = pre(batch)
                row["normalized_state"] = array_record(processed[OBS_STATE])
                if index == 0:
                    row["padded_state"] = array_record(policy.prepare_state(processed))
                    row["input_shapes"] = {
                        k: list(v.shape) for k, v in processed.items() if isinstance(v, torch.Tensor)
                    }
                    for key in CAMERA_KEYS:
                        pixels = (batch[key][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                        Image.fromarray(pixels).save(output / f"policy_input_{key.rsplit('.', 1)[1]}.png")
                if processed[OBS_STATE].shape != (1, 8) or not torch.isfinite(processed[OBS_STATE]).all():
                    write_json(output / "failed_step.json", row)
                    raise ValueError("Invalid processed state")
                write_json(output / "pending_step.json", row)
                shapes.clear()
                start = time.perf_counter()
                normalized_action = policy.select_action(processed)
                action = post(normalized_action).detach().cpu().numpy()[0]
                row.update(
                    normalized_action=array_record(normalized_action),
                    action=array_record(action),
                    inference_seconds=time.perf_counter() - start,
                    denoising_projection_shapes=shapes.copy(),
                    queued_actions_after_selection=len(policy._queues[ACTION]),
                    env_step_called=False,
                )
                # Save the original values before any validation or native execution can fail.
                write_json(output / "pending_step.json", row)
                row["outside_box_components"] = action_outside_bounds(action)
                row["bounds_handling"] = (
                    "native OSC clip/scale for EEF; native Panda sign/integration for gripper"
                )
                row["env_step_called"] = True
                write_json(output / "pending_step.json", row)
                raw, reward, terminated, truncated, info = env.step(action)
                result["measured_actions"] += 1
                row.update(
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    native_success=bool(info["is_success"]),
                    native_done=bool(info["done"]),
                )
                native = env.unwrapped._env
                row["controller_goal_position"] = array_record(native.robots[0].controller.goal_pos)
                row["controller_goal_orientation"] = array_record(native.robots[0].controller.goal_ori)
                row["gripper_controller_action"] = array_record(native.robots[0].gripper.current_action)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                (output / "pending_step.json").unlink()
                result["terminal_reason"] = terminal_reason(terminated, truncated, info)
                result["native_success"] = bool(info["is_success"])
                print(
                    f"measurement {index + 1}: success={info['is_success']}, outside={row['outside_box_components']}",
                    flush=True,
                )
                if result["terminal_reason"]:
                    break
            else:
                result["terminal_reason"] = "technical_measurement_bound"
    finally:
        hook.remove()


def main():
    """Run once from committed source and keep failures in their original output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--vlm-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    result = {
        "status": "started",
        "phase": "source",
        "measured_actions": 0,
        "settling_actions": None,
        "environment_closed": None,
        "native_success": None,
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
        "policy_revision": POLICY_REVISION,
        "vlm_revision": VLM_REVISION,
        "assets_revision": ASSETS_REVISION,
        "environment_seed": ENVIRONMENT_SEED,
        "policy_seed": POLICY_SEED,
        "task_id": 0,
        "initial_state_id": 0,
        "measurement_limit": MEASUREMENT_LIMIT,
        "policy_path": str(args.policy_path),
        "vlm_path": str(args.vlm_path),
    }
    env = None
    try:
        result["source_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
            raise RuntimeError("Commit the source and protocol before this one-shot smoke")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("MUJOCO_GL") != "egl":
            raise ValueError(
                "Run with HF_HUB_OFFLINE=1 and MUJOCO_GL=egl using the pinned local LIBERO config"
            )
        result["versions"] = {
            name: version(name)
            for name in (
                "lerobot",
                "hf-libero",
                "mujoco",
                "robosuite",
                "bddl",
                "scipy",
                "PyOpenGL",
                "numpy",
                "torch",
                "torchvision",
                "transformers",
                "huggingface-hub",
                "gymnasium",
                "safetensors",
            )
        }
        torch.set_num_threads(1)
        result["phase"] = "strict_policy_load"
        write_json(args.output / "result.json", result)
        policy, pre, post, report = load_runtime(args.policy_path, args.vlm_path)
        result["policy_load"] = report
        write_json(args.output / "policy_load.json", report)
        # Import only after the isolated environment's local asset path has been prepared.
        import gymnasium as gym
        from libero.libero import benchmark, get_assets_path, get_libero_path
        from OpenGL import GL

        from lerobot.envs.libero import LiberoEnv

        result["phase"] = "environment_reset"
        write_json(args.output / "result.json", result)
        assets = Path(get_assets_path()).resolve()
        if assets.name != ASSETS_REVISION:
            raise ValueError("LIBERO did not use the registered local assets snapshot")
        result["assets_path"] = str(assets)
        result["initial_states_root"] = get_libero_path("init_states")
        result["bddl_root"] = get_libero_path("bddl_files")
        suite = benchmark.get_benchmark_dict()["libero_object"](task_order_index=0)
        result["task"] = suite.get_task(0)._asdict()
        native_env = LiberoEnv(
            suite,
            0,
            "libero_object",
            obs_type="pixels_agent_pos",
            episode_index=0,
            hard_reset=True,
            num_steps_wait=SETTLING_ACTIONS,
            control_freq=20,
            control_mode="relative",
            observation_width=256,
            observation_height=256,
        )
        env = gym.wrappers.TimeLimit(native_env, max_episode_steps=280)
        raw, info = reset_episode(env, policy, pre, post)
        result["settling_actions"] = SETTLING_ACTIONS
        result["reset_info"] = info
        result["queue_length_after_reset"] = len(policy._queues[ACTION])
        result["next_initial_state_id"] = native_env.init_state_id
        native_env._env.sim._render_context_offscreen.gl_ctx.make_current()
        result["renderer"] = {
            name: GL.glGetString(key).decode()
            for name, key in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            )
        }
        # Set policy/noise seeds after constructor and reset RNG use.
        random.seed(POLICY_SEED)
        np.random.seed(POLICY_SEED)
        torch.manual_seed(POLICY_SEED)
        torch.cuda.manual_seed_all(POLICY_SEED)
        result["phase"] = "measurements"
        write_json(args.output / "result.json", result)
        run_measurements(env, policy, pre, post, raw, native_env.task_description, args.output, result)
        result["status"] = "completed"
    except Exception:
        result["status"] = "technical_failure"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        if env is not None:
            try:
                env.close()
                result["environment_closed"] = env.unwrapped._env is None
            except Exception:
                result["environment_closed"] = False
                result["close_exception"] = traceback.format_exc()
                result["status"] = "technical_failure"
        write_json(args.output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "completed":
        raise RuntimeError("The smoke did not close successfully; inspect result.json")


if __name__ == "__main__":
    main()
