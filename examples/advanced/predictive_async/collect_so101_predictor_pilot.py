"""Collect frozen-policy simulator latents for the pre-split L6 forecasting pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch
from eval_leisaac_so101 import EnvClient, decode_observation
from leisaac_so101_contract import ContractError, action_to_radians, hardware_features, validate_step
from leisaac_so101_matched import (
    TASK,
    WSAGI_REVISION,
    candidate_manifest,
    load_matched_runtime,
)

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.random_utils import set_seed


def prepared_observation(packet, policy, pre):
    raw = decode_observation(packet)
    batch = build_dataset_frame(hardware_features(), raw, prefix="observation")
    batch = prepare_observation_for_inference(batch, torch.device("cuda:0"), TASK, "so101_follower")
    batch["task"] = [TASK]
    batch = pre(batch)
    with torch.inference_mode():
        images, masks = policy.prepare_images(batch)
        tokens, token_masks = policy.model.encode_image_tokens(images, masks)
        state = policy.prepare_state(batch)
    return batch, tokens, token_masks, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--continue-from", type=Path, help="Keep completed episodes from the preceding interrupted collection"
    )
    args = parser.parse_args()
    if args.snapshot.name != WSAGI_REVISION:
        parser.error("L6 collection is frozen to the WSAGI candidate")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("Commit source before collecting source-bound trajectories")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "stage": "L6_collection_nonrealtime",
        "source_commit": source,
        "candidate": candidate_manifest(args.snapshot),
        "environment_seeds": list(range(20260920, 20260927)),
        "policy_seeds": list(range(1900, 1907)),
        "splits": {"train": [0, 1, 2, 3], "validation": [4], "test": [5, 6]},
        "max_actions_per_episode": 600,
        "control_fps": 30,
        "camera_backend": "standard",
        "physics": "cpu_physx_rtx_v1",
        "prefix_rule": "all actions from same chunk already committed at t",
    }
    results = []
    manifest["episode_files"] = {
        str(i): str((args.output / f"episode_{i:02d}.pt").resolve()) for i in range(7)
    }
    if args.continue_from is not None:
        previous = json.loads((args.continue_from / "result.json").read_text())
        previous_manifest = json.loads((args.continue_from / "manifest.json").read_text())
        if previous_manifest["candidate"]["policy_revision"] != WSAGI_REVISION:
            raise ValueError("Preceding collection has a different frozen candidate")
        results = previous["episodes"]
        if [r["episode"] for r in results] != list(range(len(results))) or not 0 < len(results) < 7:
            raise ValueError("Continuation requires an ordered completed prefix of collection episodes")
        for r in results:
            i = r["episode"]
            manifest["episode_files"][str(i)] = str((args.continue_from / f"episode_{i:02d}.pt").resolve())
        manifest["previous_collection"] = str(args.continue_from.resolve())
        manifest["previous_source_commit"] = previous_manifest["source_commit"]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    client = EnvClient(args.sim_python, args.assets_root, args.leisaac_root, "cpu")
    client.camera_backend = "standard"
    error = None
    try:
        client.start()
        policy, pre, post = load_matched_runtime(args.snapshot, device="cuda:0")
        for episode in range(len(results), 7):
            packet = client.reset(20260920 + episode)
            policy.reset()
            pre.reset()
            post.reset()
            set_seed(1900 + episode)
            latents, masks_list, states, actions, chunks, ticks = [], [], [], [], [], []
            chunk, terminal, equivalence_max_abs = None, False, None
            action_limit_error = None
            started = time.perf_counter()
            for step in range(601):
                batch, tokens, token_masks, state = prepared_observation(packet, policy, pre)
                latents.append(torch.stack([z[0].detach().cpu() for z in tokens]))
                masks_list.append(torch.stack([m[0].detach().cpu() for m in token_masks]))
                states.append(state[0].detach().cpu())
                if step == 600:
                    break
                with torch.inference_mode():
                    if episode == 0 and step == 0:
                        generator = torch.Generator(device="cuda:0").manual_seed(1977)
                        noise = torch.randn(
                            (1, 50, policy.config.max_action_dim), generator=generator, device="cuda:0"
                        )
                        normal = policy.predict_action_chunk(batch, noise=noise.clone())
                        overridden = policy.predict_action_chunk(
                            batch,
                            noise=noise.clone(),
                            future_image_tokens=tokens,
                            future_image_token_masks=token_masks,
                        )
                        equivalence_max_abs = float((normal - overridden).abs().max().item())
                        torch.testing.assert_close(normal, overridden, rtol=1e-5, atol=1e-5)
                        set_seed(1900)
                    if step % 50 == 0:
                        chunk = policy.predict_action_chunk(
                            batch, future_image_tokens=tokens, future_image_token_masks=token_masks
                        )
                    normalized = chunk[:, step % 50]
                    physical = post(normalized).detach().float().cpu().reshape(-1).tolist()
                row = {
                    "step": step,
                    "chunk_id": step // 50,
                    "dispatch": "not_sent",
                    "action": physical,
                }
                ticks.append(row)
                try:
                    action_to_radians(physical)
                except ContractError as exc:
                    action_limit_error = str(exc)
                    row["technical_error"] = action_limit_error
                    break
                row["dispatch"] = "sent_result_unknown"
                response = client.step(packet, physical)
                row.update(
                    dispatch="completed", terminated=response["terminated"], truncated=response["truncated"]
                )
                terminal = bool(response["terminated"] or response["truncated"])
                validate_step(packet, response["observation"], terminal=terminal)
                actions.append(normalized[0].detach().cpu())
                chunks.append(step // 50)
                if terminal:
                    break  # Returned auto-reset observation is never a target.
                packet = response["observation"]
            episode_result = {
                "episode": episode,
                "environment_seed": 20260920 + episode,
                "policy_seed": 1900 + episode,
                "steps": len(actions),
                "observations": len(latents),
                "terminal": terminal,
                "success": bool(ticks[-1]["terminated"]) if terminal else None,
                "timeout": bool(ticks[-1]["truncated"]) if terminal else False,
                "status": "technical_action_limit"
                if action_limit_error
                else ("terminal" if terminal else "censored_collection_bound"),
                "action_limit_error": action_limit_error,
                "wall_s": time.perf_counter() - started,
                "rgb_token_override_max_abs": equivalence_max_abs,
                "token_shape_per_observation": list(latents[0].shape),
                "token_dtype": str(latents[0].dtype),
            }
            # Episode control is stopped here; never materialize files on its hot path.
            torch.save(
                {
                    "tokens": torch.stack(latents),
                    "masks": torch.stack(masks_list),
                    "states": torch.stack(states),
                    "actions": torch.stack(actions) if actions else torch.empty(0, 6),
                    "chunk_ids": torch.tensor(chunks, dtype=torch.long),
                    "metadata": episode_result,
                },
                args.output / f"episode_{episode:02d}.pt",
            )
            (args.output / f"episode_{episode:02d}.json").write_text(
                json.dumps(episode_result, indent=2) + "\n"
            )
            (args.output / f"episode_{episode:02d}_ticks.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in ticks)
            )
            results.append(episode_result)
            print(json.dumps(episode_result), flush=True)
    except Exception:
        error = traceback.format_exc()
    finally:
        client.close()
        result = {
            "source_commit": source,
            "episodes": results,
            "error": error,
            "environment": client.metadata,
            "cleanup_error": client.cleanup_error,
            "subprocess_returncode": None if client.process is None else client.process.returncode,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    if error:
        print(error, flush=True)
    return int(error is not None or client.cleanup_error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
