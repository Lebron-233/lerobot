"""Explicit, logged L8 warm-start setup; no task actions or forecast decisions."""

from __future__ import annotations

import time

import torch
from leisaac_so101_contract import hardware_features, validate_observation, validate_step

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame


def warm_predictor_once(policy, preprocessor, predictor, raw: dict, task: str, device: str) -> dict:
    """Initialize cold kernels once; discard the output and never touch a queue."""
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    started = time.perf_counter()
    with torch.inference_mode():
        batch = build_dataset_frame(hardware_features(), raw, prefix="observation")
        batch = prepare_observation_for_inference(batch, target, task, "so101_follower")
        batch["task"] = [task]
        batch = preprocessor(batch)
        images, image_masks = policy.prepare_images(batch)
        tokens, masks = policy.model.encode_image_tokens(images, image_masks)
        state = policy.prepare_state(batch)
        prefix = torch.zeros(1, 8, 6, device=target, dtype=state.dtype)
        valid = torch.ones(1, 8, device=target, dtype=torch.bool)
        predictor(tokens, masks, prefix, valid, state, torch.tensor([8], device=target))
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    preprocessor.reset()
    return {
        "kind": "discarded_predictor_kernel_initialization",
        "predictor_calls": 1,
        "prefix": "zeros_for_initialization_only",
        "dispatches": 0,
        "wall_s": time.perf_counter() - started,
    }


def warm_environment(client, packet: dict, rows: list[dict]) -> None:
    """Thirty real hold steps with model memory resident; caller resets afterward."""
    action = validate_observation(packet)
    for step in range(30):
        row = {"setup_step": step, "action": list(action), "dispatch": "sent_result_unknown"}
        rows.append(row)
        started = time.perf_counter()
        response = client.step(packet, action)
        row.update(
            dispatch="completed",
            terminated=bool(response["terminated"]),
            truncated=bool(response["truncated"]),
            wall_s=time.perf_counter() - started,
        )
        terminal = row["terminated"] or row["truncated"]
        validate_step(packet, response["observation"], terminal=terminal)
        if terminal:
            raise RuntimeError("Unexpected task terminal during fixed hold-position preparation")
        packet = response["observation"]
