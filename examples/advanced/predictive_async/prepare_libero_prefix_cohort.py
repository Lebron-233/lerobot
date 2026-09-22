"""Freeze a bounded four-task raw-RLDS cohort before training outcomes exist.

Reads exactly shards 0..3. Splits complete trajectories by deterministic hash;
never uses task success, loss, or a per-frame random split. Known diagnostic
trajectory is excluded. Stored 10-FPS conversion timestamps are not relabelled.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from libero_rlds_contract import decode_episode, parse_example, records
from PIL import Image

TASKS = {0: "alphabet soup", 2: "salad dressing", 6: "butter", 7: "milk"}
LANGUAGES = {f"pick up the {name} and place it in the basket": task for task, name in TASKS.items()}
KNOWN_ACTION_HASH = "7869df1c85d4661979e587b67c77326aed0117dfbade00f418c227ffc0eca7d7"
REVISION = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def freeze_split(rows):
    selected, excluded = [], []
    seen = set()
    for row in rows:
        if row["action_sha256"] == KNOWN_ACTION_HASH:
            excluded.append({**row, "exclusion": "previous_diagnostic_episode814"})
            continue
        if row["trajectory_id"] in seen:
            raise ValueError("Duplicate trajectory across shards; do not split copies")
        seen.add(row["trajectory_id"])
        selected.append(row)
    split = []
    for task in TASKS:
        group = [r for r in selected if r["task"] == task]
        group.sort(key=lambda r: hashlib.sha256(("prefix-data-v1:" + r["trajectory_id"]).encode()).hexdigest())
        if len(group) < 3:
            raise ValueError(f"Need train/dev/sealed trajectories for task {task}; got {len(group)}")
        for index, row in enumerate(group):
            # Exactly one validation and one sealed trajectory per task. The rest train.
            role = "dev" if index == 0 else "sealed" if index == 1 else "train"
            split.append({**row, "split": role})
    return split, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    out = root / "cohort"
    if out.exists():
        raise ValueError("Cohort already exists; never regenerate split after training")
    rows, scanned, shard_hashes = [], [], {}
    for shard in range(4):
        path = root / f"libero_object-train.tfrecord-{shard:05d}-of-00032"
        shard_hashes[path.name] = sha(path)
        count = 0
        for offset, raw in records(path):
            d = decode_episode(parse_example(raw))
            count += 1
            if d["language"] not in LANGUAGES:
                continue
            identity = hashlib.sha256(d["source_path"].encode() + d["states"].tobytes() + d["actions"].tobytes()).hexdigest()
            rows.append({"trajectory_id": identity, "shard": path.name, "record_offset": offset,
                         "record_bytes": len(raw), "payload_sha256": hashlib.sha256(raw).hexdigest(),
                         "source_path": d["source_path"], "language": d["language"],
                         "task": LANGUAGES[d["language"]], "rows": len(d["actions"]),
                         "action_sha256": hashlib.sha256(d["actions"].tobytes()).hexdigest(),
                         "state_sha256": hashlib.sha256(d["states"].tobytes()).hexdigest()})
        scanned.append({"shard": shard, "episodes": count})
    if [r["episodes"] for r in scanned] != [14, 14, 15, 14]:
        raise ValueError("Published upstream shard lengths differ")
    split, excluded = freeze_split(rows)
    out.mkdir()
    manifest = {"revision": REVISION, "raw_shards": shard_hashes, "scanned": scanned,
                "rows": split, "excluded": excluded, "split_rule": "hash-order per task: first dev, second sealed, rest train",
                "prior_diagnostic_excluded": True, "base_pretraining_unseen_claimed": False,
                "physical_contract_accepted": False, "optimizer_updates": 0}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    packets = []
    for row in split:
        if row["split"] == "sealed":
            continue
        with (root / row["shard"]).open("rb") as f:
            f.seek(row["record_offset"] + 12)
            raw = f.read(row["record_bytes"])
        if hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
            raise ValueError("Payload changed during preparation")
        d = decode_episode(parse_example(raw))
        # Deterministic physical row anchors; not timestamp interpolation or resampling.
        anchors = np.arange(0, len(d["actions"]), 20)
        pixels = []
        for index in anchors:
            camera_pair = []
            for key in ("images", "wrists"):
                with Image.open(io.BytesIO(d[key][index])) as img:
                    value = np.asarray(img.convert("RGB"))
                if value.shape != (256, 256, 3):
                    raise ValueError("Raw camera shape differs")
                camera_pair.append(value)
            pixels.append(camera_pair)
        packet = out / (row["trajectory_id"] + ".npz")
        np.savez_compressed(packet, states=d["states"], actions=d["actions"], joints=d["joints"],
                            anchors=anchors, pixels=np.asarray(pixels))
        packets.append({"trajectory_id": row["trajectory_id"], "split": row["split"],
                        "path": packet.name, "sha256": sha(packet), "anchors": anchors.tolist()})
    (out / "packets.json").write_text(json.dumps(packets, indent=2))
    counts = {str(t): {role: sum(r["task"] == t and r["split"] == role for r in split)
                      for role in ("train", "dev", "sealed")} for t in TASKS}
    print(json.dumps({"cohort": str(out), "counts": counts, "excluded": len(excluded),
                      "packets": len(packets), "manifest_sha256": sha(out / "manifest.json"),
                      "sealed_pixels_decoded": 0, "full_training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
