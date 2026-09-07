"""New, explicitly split L10 action-distillation trajectories; no old test reads."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch
from collect_so101_predictor_pilot import prepared_observation
from eval_leisaac_so101 import EnvClient
from leisaac_so101_contract import action_to_radians, validate_step
from leisaac_so101_matched import candidate_manifest, load_matched_runtime
from so101_feasible_actions import FeasibleActionProjector

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.random_utils import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit source before new split collection")
    if args.split == "test":
        if args.selection is None:
            parser.error("Freeze/publish the selected development model before test collection")
        selected = json.loads(args.selection.read_text())
        if not selected["validation_qualified"]:
            parser.error("Development did not qualify; leave the test split unopened")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    count, first_seed, first_policy = (
        (8, 20270110, 2510) if args.split == "development" else (6, 20270130, 2530)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "L10_new_action_distillation_collection",
        "source_commit": source,
        "candidate": candidate_manifest(snapshot),
        "split": args.split,
        "splits": {"train": list(range(6)), "validation": [6, 7]}
        if args.split == "development"
        else {"test": list(range(6))},
        "environment_seeds": list(range(first_seed, first_seed + count)),
        "policy_seeds": list(range(first_policy, first_policy + count)),
        "action_contract": "feasible_v1",
        "max_actions": 600,
        "realtime": False,
        "selected_model": str(args.selection) if args.selection else None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    policy, pre, post = load_matched_runtime(snapshot, device="cuda:0")
    projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    client = EnvClient(
        base / "leisaac-sim-venv/bin/python",
        base
        / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets",
        base / "leisaac-source",
        "cpu",
    )
    client.episode_seconds, client.camera_backend = 120, "standard"
    results, error = [], None
    episode, ticks = -1, []
    try:
        client.start()
        for episode in range(count):
            set_seed(first_policy + episode)
            policy.reset()
            pre.reset()
            post.reset()
            packet = client.reset(first_seed + episode)
            latents, masks, states, actions, ticks = [], [], [], [], []
            terminal = False
            started = time.perf_counter()
            for step in range(600):
                batch, z, valid, state = prepared_observation(packet, policy, pre)
                latents.append(torch.stack(z, dim=1)[0].detach().cpu())
                masks.append(torch.stack(valid, dim=1)[0].detach().cpu())
                states.append(state[0].detach().cpu())
                language = {
                    key: batch[key].detach().cpu()
                    for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
                }
                with torch.inference_mode():
                    if step % 50 == 0:
                        chunk = projector(
                            policy.predict_action_chunk(
                                batch, future_image_tokens=z, future_image_token_masks=valid
                            )
                        )
                    action = chunk[:, step % 50]
                    physical = post(action).detach().float().cpu().reshape(-1).tolist()
                tick = {"step": step, "action": physical, "dispatch": "not_sent"}
                ticks.append(tick)
                action_to_radians(physical)
                tick["dispatch"] = "sent_result_unknown"
                response = client.step(packet, physical)
                tick.update(
                    dispatch="completed", terminated=response["terminated"], truncated=response["truncated"]
                )
                actions.append(action[0].detach().cpu())
                terminal = bool(response["terminated"] or response["truncated"])
                validate_step(packet, response["observation"], terminal=terminal)
                if terminal:
                    break
                packet = response["observation"]
            if not terminal:
                _, z, valid, state = prepared_observation(packet, policy, pre)
                latents.append(torch.stack(z, dim=1)[0].detach().cpu())
                masks.append(torch.stack(valid, dim=1)[0].detach().cpu())
                states.append(state[0].detach().cpu())
            payload = {
                "tokens": torch.stack(latents),
                "masks": torch.stack(masks),
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "chunk_ids": torch.arange(len(actions)) // 50,
                "language": language,
            }
            torch.save(payload, args.output / f"episode_{episode:02d}.pt")
            (args.output / f"episode_{episode:02d}_ticks.jsonl").write_text(
                "".join(json.dumps(t) + "\n" for t in ticks)
            )
            row = {
                "episode": episode,
                "steps": len(actions),
                "observations": len(latents),
                "terminal": terminal,
                "wall_s": time.perf_counter() - started,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    except Exception:
        error = traceback.format_exc()
        if episode >= 0:
            (args.output / f"episode_{episode:02d}_partial_ticks.jsonl").write_text(
                "".join(json.dumps(t) + "\n" for t in ticks)
            )
            if latents:
                torch.save(
                    {
                        "tokens": torch.stack(latents),
                        "masks": torch.stack(masks),
                        "states": torch.stack(states),
                        "actions": torch.stack(actions) if actions else torch.empty(0, 6),
                        "error": error,
                    },
                    args.output / f"episode_{episode:02d}_partial.pt",
                )
    finally:
        client.close()
        report = {
            "source_commit": source,
            "episodes": results,
            "error": error,
            "environment": client.metadata,
            "cleanup_error": client.cleanup_error,
            "subprocess_returncode": None if client.process is None else client.process.returncode,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    return 1 if error or client.cleanup_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
