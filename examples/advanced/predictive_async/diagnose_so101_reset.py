"""Three same-seed real resets, zero policy calls and zero control actions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from eval_leisaac_so101 import EnvClient, decode_observation
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("Commit source before real reset diagnosis")
    client = EnvClient(args.sim_python, args.assets_root, args.leisaac_root, "cpu")
    packets, error = [], None
    try:
        client.start()
        for _ in range(3):
            packets.append(client.reset(args.seed))
    except Exception as exc:
        error = repr(exc)
    finally:
        client.close()
    rows = []
    for i, packet in enumerate(packets):
        row = {
            "reset_index": i,
            "camera_world_poses_opengl": packet["camera_world_poses_opengl"],
            "objects": packet["task_diagnostics"]["object_positions_world"],
        }
        if i:
            for camera in ("front", "wrist"):
                a = np.array(packets[0]["camera_world_poses_opengl"][camera]["position"])
                b = np.array(packet["camera_world_poses_opengl"][camera]["position"])
                row[camera + "_position_difference_m"] = float(np.linalg.norm(b - a))
        rows.append(row)
    result = {
        "source_commit": source,
        "seed": args.seed,
        "resets": rows,
        "error": error,
        "environment": client.metadata,
        "subprocess_returncode": client.process.returncode,
        "cleanup_error": client.cleanup_error,
        "policy_calls": 0,
        "control_steps": 0,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    for i, packet in enumerate(packets):
        raw = decode_observation(packet)
        for camera in ("top", "wrist"):
            Image.fromarray(raw[camera]).save(args.output / f"reset_{i}_{camera}.png")
    print(json.dumps(result, indent=2))
    return int(error is not None or client.cleanup_error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
