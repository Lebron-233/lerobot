"""Teacher-forced training-frame diagnostic; not a held-out score or robot run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from leisaac_so101_contract import SCALAR_KEYS, hardware_features
from leisaac_so101_matched import (
    TASK,
    candidate_manifest,
    load_matched_runtime,
    motor_to_physical,
    physical_to_motor,
)

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.random_utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dataset-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = (0, 150, 300)
    images = {i: {} for i in indices}
    for camera in ("front", "wrist"):
        video = args.dataset_snapshot / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
        with av.open(str(video)) as stream:
            for i, frame in enumerate(stream.decode(video=0)):
                if i in images:
                    images[i][camera] = frame.to_ndarray(format="rgb24")
                if i == indices[-1]:
                    break
    data = pq.read_table(args.dataset_snapshot / "data/chunk-000/episode_000000.parquet")
    policy, pre, post = load_matched_runtime(args.snapshot, device="cuda:0")
    rows = []
    for i in indices:
        item = data.slice(i, 1).to_pylist()[0]
        native_state = torch.tensor(item["observation.state"], dtype=torch.float32)
        raw = dict(zip(SCALAR_KEYS, motor_to_physical(native_state).tolist(), strict=True))
        raw.update(top=images[i]["front"], wrist=images[i]["wrist"])
        batch = build_dataset_frame(hardware_features(), raw, prefix="observation")
        batch = prepare_observation_for_inference(batch, torch.device("cuda:0"), TASK, "so101_follower")
        batch["task"] = [TASK]
        policy.reset()
        pre.reset()
        post.reset()
        set_seed(1801)
        with torch.inference_mode(), torch.autocast("cuda", enabled=policy.config.use_amp):
            prepared = pre(batch)
            physical = post(policy.select_action(prepared))
        prediction = physical_to_motor(physical).detach().float().cpu().reshape(-1).numpy()
        target = np.asarray(item["action"])
        model_state = policy.prepare_state(prepared).detach().float().cpu().reshape(-1).tolist()
        rows.append(
            {
                "frame": i,
                "native_state": native_state.tolist(),
                "model_ready_state": model_state,
                "predicted_motor_action": prediction.tolist(),
                "reference_motor_action": target.tolist(),
                "motor_mae": float(np.abs(prediction - target).mean()),
            }
        )
    result = {
        "kind": "training_frame_teacher_forcing_diagnostic_not_held_out",
        "candidate": candidate_manifest(args.snapshot),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_revision": args.dataset_snapshot.name,
        "episode": 0,
        "frames": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
