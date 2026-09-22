"""Offline, fail-closed audit of one demo against an independent corrected index.

Run in the existing CPU data environment that provides PyArrow. Does not fetch
files, use the GPU, or approve physical-frequency compatibility for training.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    target_id, target_task = 814, 25
    original, corrected = root / "matched_demo.parquet", root / "corrected_source/data.parquet"
    meta = pq.read_table(root / "corrected_source/episodes.parquet").to_pylist()
    target = next(r for r in meta if r["episode_index"] == target_id)
    columns = [
        "episode_index",
        "task_index",
        "frame_index",
        "index",
        "timestamp",
        "observation.state",
        "action",
    ]
    left = pq.read_table(original).to_pylist()
    right = pq.read_table(corrected, columns=columns).to_pylist()
    left = [r for r in left if r["episode_index"] == target_id]
    right = [r for r in right if r["episode_index"] == target_id]
    if len(left) != len(right) or len(left) != target["length"]:
        raise ValueError("Episode length does not match corrected metadata")
    for field in columns:
        if not np.array_equal(np.asarray([r[field] for r in left]), np.asarray([r[field] for r in right])):
            raise ValueError("Cross-source numeric mismatch: " + field)
    n = len(left)
    if [r["frame_index"] for r in left] != list(range(n)):
        raise ValueError("Episode frame order differs")
    if [r["index"] for r in left] != list(range(target["dataset_from_index"], target["dataset_to_index"])):
        raise ValueError("Global row interval differs")
    if any(r["task_index"] != target_task for r in left):
        raise ValueError("Wrong task, do not relabel")
    states = np.asarray([r["observation.state"] for r in left], dtype=np.float32)
    actions = np.asarray([r["action"] for r in left], dtype=np.float32)
    if (
        states.shape != (n, 8)
        or actions.shape != (n, 7)
        or not np.isfinite(states).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Malformed state/action arrays")
    anchors = [20, n - 3]
    pixels = []
    for index in anchors:
        cameras = []
        for key in ("observation.images.image", "observation.images.image2"):
            with Image.open(io.BytesIO(left[index][key]["bytes"])) as image:
                x = np.asarray(image.convert("RGB"))
            if x.shape != (256, 256, 3) or x.dtype != np.uint8:
                raise ValueError("Unexpected camera shape/type")
            cameras.append(x)
        pixels.append(cameras)
    packet = root / "demo_packet.npz"
    if packet.exists() or (root / "demo_contract.json").exists():
        raise ValueError("Do not overwrite prepared diagnostic data")
    np.savez_compressed(
        packet, states=states, actions=actions, pixels=np.asarray(pixels), anchors=np.asarray(anchors)
    )
    files = [
        original,
        corrected,
        root / "corrected_source/episodes.parquet",
        root / "corrected_source/tasks.parquet",
    ]
    report = {
        "episode": target_id,
        "task_index": target_task,
        "language": "pick up the milk and place it in the basket",
        "rows": n,
        "diagnostic_anchors": anchors,
        "cross_source_numeric_columns_exact": columns,
        "image_source": "original embedded image bytes; no flip or resampling",
        "numeric_source": "lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        "image_repository": "HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "files": {str(p): sha(p) for p in files},
        "packet_sha256": sha(packet),
        "stored_fps": 10.0,
        "timestamp_step": 0.1,
        "physical_control_step_proven": False,
        "full_training_ready": False,
        "use": "numerical forward/backward interface diagnostic only, no optimizer updates",
        "train_dev_test_split_claimed": False,
        "base_pretraining_unseen_claimed": False,
    }
    with (root / "demo_contract.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
