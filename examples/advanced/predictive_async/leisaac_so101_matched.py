"""Independent PickOrange candidate; never binds the frozen SO100 predictor."""

from __future__ import annotations

from pathlib import Path

import torch
from leisaac_so101_contract import JOINT_LIMITS_DEG

CANDIDATE_ID = "leisaac_so101_pickorange_edge_v1"
POLICY_REPO = "edge-inference/smolvla-so101-pick-orange"
POLICY_REVISION = "71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d"
SINGLE_RANK_REVISION = "7f19c683128ed07c31240ea2b29fe61afcb1755b"
WSAGI_REVISION = "c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
CANDIDATES = {
    POLICY_REVISION: CANDIDATE_ID,
    SINGLE_RANK_REVISION: "leisaac_so101_pickorange_single_rank_v1",
    WSAGI_REVISION: "leisaac_so101_pickorange_wsagi15k_v1",
}
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TASK = "Grab orange and place into plate"
CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")


def candidate_manifest(snapshot: Path | None = None, *, execution_steps: int = 50) -> dict:
    revision = snapshot.name if snapshot is not None else POLICY_REVISION
    if revision not in CANDIDATES:
        raise ValueError("Unknown task-matched candidate revision")
    return {
        "id": CANDIDATES[revision],
        "policy_repo": "wsagi/SmolVLA-PickOrange" if revision == WSAGI_REVISION else POLICY_REPO,
        "policy_revision": revision,
        "vlm_config_tokenizer_revision": VLM_REVISION,
        "weight_source": "entire task checkpoint; no replacement VLM weights",
        "state_action_coordinates": "LeIsaac arm motor [-100,100], gripper [0,100]",
        "statistics": "checkpoint's own state/action mean/std",
        "task": TASK,
        "predictor": None,
        "generated_chunk_steps": 50,
        "sync_executed_chunk_steps": execution_steps,
        "sync_autocast": {
            "enabled": revision != WSAGI_REVISION,
            "cuda_dtype": "float16" if revision != WSAGI_REVISION else None,
        },
    }


def _coordinates(value: torch.Tensor, *, to_motor: bool) -> torch.Tensor:
    if value.shape[-1] != 6 or not value.is_floating_point():
        raise ValueError("SO101 coordinates require a floating six-dimensional tensor")
    result = value.clone()
    low = value.new_tensor([bounds[0] for bounds in JOINT_LIMITS_DEG[:5]])
    span = value.new_tensor([bounds[1] - bounds[0] for bounds in JOINT_LIMITS_DEG[:5]])
    if to_motor:
        result[..., :5] = (value[..., :5] - low) * 200 / span - 100
    else:
        result[..., :5] = (value[..., :5] + 100) * span / 200 + low
    # The simulator transport and native policy both use gripper RANGE_0_100.
    # Do not clip state or action here. The transport rejects infeasible targets.
    return result


def physical_to_motor(value: torch.Tensor) -> torch.Tensor:
    return _coordinates(value, to_motor=True)


def motor_to_physical(value: torch.Tensor) -> torch.Tensor:
    return _coordinates(value, to_motor=False)


class MotorStatePreprocessor:
    """Transport physical state -> native motor coordinates -> saved processor."""

    def __init__(self, processor) -> None:
        self.processor = processor
        self.steps = processor.steps

    def reset(self) -> None:
        self.processor.reset()

    def __call__(self, batch: dict):
        batch = dict(batch)
        batch["observation.state"] = physical_to_motor(batch["observation.state"])
        return self.processor(batch)


class PhysicalActionPostprocessor:
    """Native saved denormalizer -> motor targets -> transport physical units."""

    def __init__(self, processor) -> None:
        self.processor = processor
        self.steps = processor.steps

    def reset(self) -> None:
        self.processor.reset()

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return motor_to_physical(self.processor(action))


def load_matched_runtime(snapshot: Path, *, device: str, execution_steps: int = 50):
    from huggingface_hub import snapshot_download

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if snapshot.name not in CANDIDATES:
        raise ValueError("Use the exact task-matched HF snapshot, not a different candidate")
    config = PreTrainedConfig.from_pretrained(snapshot, local_files_only=True)
    if config.type != "smolvla" or tuple(config.image_features) != CAMERA_KEYS:
        raise ValueError("Task-matched policy must preserve its native front/wrist schema")
    if tuple(config.robot_state_feature.shape) != (6,) or tuple(config.action_feature.shape) != (6,):
        raise ValueError("Task-matched state/action shape differs")
    if (config.chunk_size, config.n_action_steps, config.num_steps, config.max_state_dim) != (50, 50, 10, 32):
        raise ValueError("Task-matched inference configuration differs")
    if execution_steps not in (25, 50):
        raise ValueError("Only registered 25/50-step synchronous execution profiles are supported")
    config.n_action_steps = execution_steps
    if config.load_vlm_weights != (snapshot.name == WSAGI_REVISION):
        raise ValueError("Candidate VLM initialization configuration differs from its registered snapshot")
    if config.use_amp != (snapshot.name != WSAGI_REVISION):
        raise ValueError("Candidate inference precision differs from its registered snapshot")
    vlm = snapshot_download(VLM_REPO, revision=VLM_REVISION, local_files_only=True)
    config.vlm_model_name = vlm
    config.pretrained_path = snapshot
    config.device = device
    config.compile_model = False
    policy = SmolVLAPolicy.from_pretrained(snapshot, config=config, local_files_only=True, strict=True)
    policy.to(device).eval().requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(snapshot),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": vlm},
            "rename_observations_processor": {
                "rename_map": {"observation.images.top": "observation.images.front"}
            },
        },
    )
    return policy, MotorStatePreprocessor(pre), PhysicalActionPostprocessor(post)
