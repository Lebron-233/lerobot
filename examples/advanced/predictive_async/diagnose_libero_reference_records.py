"""Describe a closed LIBERO cohort without executing a simulator or a policy."""

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def summary(values):
    """Nearest-rank quantiles retain every measured latency, including outliers."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Expected a nonempty vector of finite nonnegative measurements")
    x = np.sort(x)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "min": float(x[0]),
        **{f"p{p}": float(x[math.ceil(p / 100 * x.size) - 1]) for p in (50, 90, 95, 99)},
        "max": float(x[-1]),
    }


def quaternion_distances(quaternions):
    """Physical orientation changes, invariant to the quaternion's sign."""
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(norms == 0):
        raise ValueError("Invalid quaternion")
    q = q / norms[:, None]
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    return 2 * np.arccos(np.clip(dots, 0, 1))


def read_episode(directory):
    """Pair actual pre-action observations with returned actions; never fill missing rows."""
    result = json.loads((directory / "result.json").read_text())
    observations, actions, returned = {}, {}, {}
    with (directory / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "observation":
                observations[row["index"]] = row
            elif row["event"] == "action_prepared":
                actions[row["number"]] = row
            elif row["event"] == "measurement_returned":
                returned[row["number"]] = row
    n = result["measured_actions"]
    if (
        result["status"] != "completed"
        or set(observations) != set(range(n + 1))
        or set(actions) != set(range(1, n + 1))
        or set(returned) != set(range(1, n + 1))
        or any(actions[k]["observation_index"] != k - 1 for k in actions)
    ):
        raise ValueError(f"Incomplete or misaligned trajectory: {directory}")
    states = np.asarray([observations[k]["state"]["values"] for k in range(n + 1)])
    commands = np.asarray([actions[k]["action"]["values"] for k in range(1, n + 1)])
    latency = np.asarray([actions[k]["inference_seconds"] for k in range(1, n + 1)])
    quaternions = [observations[k]["eef_quaternion_xyzw"]["values"] for k in range(n + 1)]
    if states.shape != (n + 1, 8) or commands.shape != (n, 7):
        raise ValueError(f"Unexpected state/action shape: {directory}")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError(f"Nonfinite trajectory: {directory}")
    angle_jumps = np.linalg.norm(np.diff(states[:, 3:6], axis=0), axis=1)
    physical_rotation = quaternion_distances(quaternions)
    out_of_box = np.abs(commands) > 1
    tail = min(40, n)
    closing = np.flatnonzero(commands[:, 6] > 0)
    # These are continuous descriptors, not substitutes for native grasp/success predicates.
    diagnostics = {
        "tuple": result["tuple"],
        "directory": str(directory),
        "success": result["success"],
        "measured_actions": n,
        "latency_seconds": summary(latency),
        "out_of_box_by_component": out_of_box.sum(axis=0).tolist(),
        "out_of_box_actions": int(out_of_box.any(axis=1).sum()),
        "eef_xyz_initial": states[0, :3].tolist(),
        "eef_xyz_final": states[-1, :3].tolist(),
        "eef_xyz_min": states[:, :3].min(axis=0).tolist(),
        "eef_xyz_max": states[:, :3].max(axis=0).tolist(),
        "eef_path_length_m": float(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum()),
        "last_40_eef_xyz_span_m": np.ptp(states[-tail - 1 :, :3], axis=0).tolist(),
        "last_40_gripper_joint_separation_min_max_m": [
            float(np.min(states[-tail - 1 :, 6] - states[-tail - 1 :, 7])),
            float(np.max(states[-tail - 1 :, 6] - states[-tail - 1 :, 7])),
        ],
        "last_40_mean_abs_translation_command": np.mean(np.abs(commands[-tail:, :3]), axis=0).tolist(),
        "last_40_positive_gripper_commands": int(np.count_nonzero(commands[-tail:, 6] > 0)),
        "first_positive_gripper_action": int(closing[0] + 1) if closing.size else None,
        "gripper_sign_switches": int(np.count_nonzero(np.diff(np.sign(commands[:, 6])))),
        "axis_angle_vector_jump_max_rad": float(angle_jumps.max()),
        "physical_orientation_change_max_rad": float(physical_rotation.max()),
        "axis_angle_wrap_events": [
            {
                "action": int(k + 1),
                "vector_jump_rad": float(angle_jumps[k]),
                "physical_rotation_rad": float(physical_rotation[k]),
            }
            for k in np.flatnonzero((angle_jumps > np.pi) & (physical_rotation < 0.1))
        ],
    }
    return diagnostics, latency, observations, states, commands


def contact_sheet(directory, observations, states, commands, output):
    """Eight fixed temporal positions, actual cameras rotated once for inspection."""
    n = len(commands)
    indices = sorted({min(k, n) for k in range(0, 281, 40)} | {n})
    width, title, header = 256, 34, 52
    image = Image.new("RGB", (width * len(indices), title + header + 512), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (5, 5), str(directory.name) + " | raw cameras rotated 180 degrees | NOT success labels", fill="black"
    )
    for column, index in enumerate(indices):
        x = column * width
        s = states[index]
        next_grip = float(commands[index, 6]) if index < n else None
        text = f"obs {index} | z={s[2]:.3f} | gap={s[6] - s[7]:.3f}\nnext grip={next_grip}"
        draw.text((x + 4, title + 3), text, fill="black")
        with np.load(directory / observations[index]["cameras"], allow_pickle=False) as cameras:
            for row, key in enumerate(("image", "image2")):
                pixels = cameras[key]
                if pixels.shape != (256, 256, 3) or pixels.dtype != np.uint8:
                    raise ValueError(f"Unexpected camera in {directory}/{index}")
                image.paste(Image.fromarray(pixels[::-1, ::-1]), (x, title + header + row * 256))
    image.save(output)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sheets").mkdir()
    cohort = args.source / "development"
    manifest = json.loads((cohort / "registration.json").read_text())
    rows, latencies, sheet_indices = [], [], {}
    for ordinal in range(200):
        directory = cohort / f"tuple_{ordinal:03d}"
        row, latency, observations, states, commands = read_episode(directory)
        if row["tuple"]["ordinal"] != ordinal or row["tuple"]["task_id"] != ordinal // 20:
            raise ValueError("Unexpected frozen cohort order")
        if row["tuple"]["task_id"] == 5 or not row["success"]:
            sheet_indices[str(ordinal)] = contact_sheet(
                directory, observations, states, commands, args.output / "sheets" / f"tuple_{ordinal:03d}.png"
            )
        rows.append(row)
        latencies.append(latency)
    all_latency = np.concatenate(latencies)
    np.savez_compressed(
        args.output / "latency_samples.npz", **{f"tuple_{k:03d}": x for k, x in enumerate(latencies)}
    )
    result = {
        "analysis": "posthoc_readonly_trajectory_descriptors_not_a_new_task_cohort",
        "analysis_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_root": str(args.source),
        "source_registration": manifest,
        "cohort": "development",
        "new_rollouts": 0,
        "model_calls": 0,
        "confirmation_data_read": False,
        "latency_seconds": summary(all_latency),
        "nominal_policy_ticks_ceiling_histogram_not_async_age": dict(
            Counter(np.ceil(all_latency * 20).astype(int).tolist())
        ),
        "latency_exceeding_50ms": int(np.count_nonzero(all_latency > 0.05)),
        "latency_exceeding_400ms": int(np.count_nonzero(all_latency > 0.4)),
        "out_of_box_by_component": np.sum([row["out_of_box_by_component"] for row in rows], axis=0).tolist(),
        "episodes_with_axis_angle_wrap": sum(bool(row["axis_angle_wrap_events"]) for row in rows),
        "native_successes_with_axis_angle_wrap": sum(
            bool(row["axis_angle_wrap_events"]) and row["success"] for row in rows
        ),
        "sheet_observation_indices": sheet_indices,
        "episodes": rows,
        "baseline_qualified": False,
        "realtime_qualified": False,
        "predictor_benefit_tested": False,
        "risk_thresholds": None,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("episodes", "source_registration", "sheet_observation_indices")
            },
            indent=2,
        )
    )
    for row in rows:
        if row["tuple"]["task_id"] == 5 or not row["success"]:
            print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
