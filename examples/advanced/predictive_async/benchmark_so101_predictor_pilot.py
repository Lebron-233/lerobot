"""Standalone fixed-input L6 predictor/full-RGB-policy host latency, not realtime GO."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from leisaac_so101_contract import SCALAR_KEYS, hardware_features
from leisaac_so101_matched import TASK, candidate_manifest, load_matched_runtime
from PIL import Image

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.random_utils import set_seed


def timed_calls(call):
    for _ in range(50):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(200):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    return {
        "warmup": 50,
        "measured": 200,
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "max_ms": max(values),
        "all_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("Commit source before standalone benchmark")
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cuda:0")
    predictor = LightweightFutureLatentPredictor(FutureLatentConfig(**checkpoint["config"])).to("cuda:0")
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.eval().requires_grad_(False)
    policy, pre, _ = load_matched_runtime(args.snapshot, device="cuda:0")
    with (args.observation_run / "ticks.jsonl").open() as stream:
        first = json.loads(next(stream))
    raw = dict(zip(SCALAR_KEYS, first["state"], strict=True))
    for name in ("top", "wrist"):
        with Image.open(args.observation_run / "observations" / f"step_00000_{name}.png") as image:
            raw[name] = np.array(image.convert("RGB"))
    batch = build_dataset_frame(hardware_features(), raw, prefix="observation")
    batch = prepare_observation_for_inference(batch, torch.device("cuda:0"), TASK, "so101_follower")
    batch["task"] = [TASK]
    batch = pre(batch)
    set_seed(1911)
    with torch.inference_mode():
        images, masks = policy.prepare_images(batch)
        tokens, token_masks = policy.model.encode_image_tokens(images, masks)
        chunk = policy.predict_action_chunk(batch)
        prefix = chunk[:, :8].clone()
        prefix_mask = torch.ones(1, 8, dtype=torch.bool, device="cuda:0")
        state = policy.prepare_state(batch)
        delay = torch.tensor([8], device="cuda:0")
        full = timed_calls(lambda: policy.predict_action_chunk(batch))
        lightweight = timed_calls(lambda: predictor(tokens, token_masks, prefix, prefix_mask, state, delay))
    result = {
        "kind": "standalone_host_wall_completed_cuda_fixed_input_not_realtime",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate": candidate_manifest(args.snapshot),
        "checkpoint_training_source": checkpoint["source_commit"],
        "selected_epoch": checkpoint["epoch"],
        "parameters": checkpoint["parameters"],
        "device": torch.cuda.get_device_name(),
        "input_observation_run": str(args.observation_run.resolve()),
        "full_policy_rgb": full,
        "predictor_forward": lightweight,
        "predictor_to_full_p90_percent": lightweight["p90_ms"] / full["p90_ms"] * 100,
        "simulator_running": False,
        "runtime_go": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        key: value for key, value in result.items() if key not in ("full_policy_rgb", "predictor_forward")
    }
    print(json.dumps(summary, indent=2))
    for name in ("full_policy_rgb", "predictor_forward"):
        print(name, {key: value for key, value in result[name].items() if key != "all_ms"})


if __name__ == "__main__":
    main()
