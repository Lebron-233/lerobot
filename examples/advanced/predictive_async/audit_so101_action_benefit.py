"""Frozen L8 action audit on six independent, future-blind simulator trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
from collect_so101_predictor_pilot import prepared_observation
from eval_leisaac_so101 import EnvClient
from leisaac_so101_contract import action_to_radians, validate_step
from leisaac_so101_matched import candidate_manifest, load_matched_runtime
from leisaac_so101_predicted import load_so101_l6_predictor
from so101_feasible_actions import FeasibleActionProjector

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.random_utils import set_seed


def summarize_cases(rows: list[dict], missing: list[dict]) -> dict:
    episodes = []
    for episode in range(6):
        group = [row for row in rows if row["episode"] == episode]
        if not group:
            continue
        identity = float(np.mean([row["identity_l1_25"] for row in group]))
        predicted = float(np.mean([row["predicted_l1_25"] for row in group]))
        episodes.append(
            {
                "episode": episode,
                "cases": len(group),
                "identity_l1_25": identity,
                "predicted_l1_25": predicted,
                "reduction_percent": 100 * (1 - predicted / identity) if identity > 0 else None,
            }
        )
    interval = None
    if len(episodes) == 6 and all(row["identity_l1_25"] > 0 for row in episodes):
        values = np.array([[row["identity_l1_25"], row["predicted_l1_25"]] for row in episodes])
        rng = np.random.default_rng(2090)
        bootstrap = values[rng.integers(0, 6, size=(50000, 6))].mean(axis=1)
        interval = np.percentile(100 * (1 - bootstrap[:, 1] / bootstrap[:, 0]), [2.5, 97.5]).tolist()
    improved = sum(row["predicted_l1_25"] < row["identity_l1_25"] for row in rows)
    mean_identity = float(np.mean([row["identity_l1_25"] for row in rows])) if rows else None
    mean_predicted = float(np.mean([row["predicted_l1_25"] for row in rows])) if rows else None
    return {
        "cases": rows,
        "missing_cases": missing,
        "by_episode": episodes,
        "case_count": len(rows),
        "improved_cases": improved,
        "identity_l1_25": mean_identity,
        "predicted_l1_25": mean_predicted,
        "reduction_percent": 100 * (1 - mean_predicted / mean_identity) if mean_identity else None,
        "episode_bootstrap_95_reduction_interval_percent": interval,
        "stable_action_gate": bool(
            len(rows) == 216
            and not missing
            and interval is not None
            and interval[0] > 0
            and improved >= 144
            and sum(row["reduction_percent"] is not None and row["reduction_percent"] > 0 for row in episodes)
            >= 5
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the independent audit before collecting")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    base = repo.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    checkpoint = base / "artifacts/m54l6_predictor_training_v1/best.pt"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "candidate": candidate_manifest(snapshot),
        "task": "independent_action_audit_nonrealtime",
        "action_contract": "feasible_v1",
        "environment_seeds": list(range(20261130, 20261136)),
        "policy_seeds": list(range(2230, 2236)),
        "anchors": list(range(0, 600, 50)),
        "delays": [1, 4, 8],
        "training": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    policy, pre, post = load_matched_runtime(snapshot, device="cuda:0")
    predictor = load_so101_l6_predictor(checkpoint, device="cuda:0")
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
    episodes, error = [], None
    try:
        client.start()
        for episode in range(6):
            packet = client.reset(20261130 + episode)
            policy.reset()
            pre.reset()
            post.reset()
            set_seed(2230 + episode)
            tokens_list, masks_list, states, actions, ticks = [], [], [], [], []
            terminal, language = False, None
            for step in range(600):
                batch, tokens, masks, state = prepared_observation(packet, policy, pre)
                tokens_list.append(torch.stack(tokens, dim=1)[0].detach().cpu())
                masks_list.append(torch.stack(masks, dim=1)[0].detach().cpu())
                states.append(state[0].detach().cpu())
                language = {
                    key: batch[key].detach().cpu()
                    for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
                }
                with torch.inference_mode():
                    if step % 50 == 0:
                        chunk = projector(
                            policy.predict_action_chunk(
                                batch,
                                future_image_tokens=tokens,
                                future_image_token_masks=masks,
                            )
                        )
                    normalized = chunk[:, step % 50]
                    physical = post(normalized).detach().float().cpu().reshape(-1).tolist()
                tick = {"step": step, "action": physical, "dispatch": "not_sent"}
                ticks.append(tick)
                action_to_radians(physical)
                tick["dispatch"] = "sent_result_unknown"
                response = client.step(packet, physical)
                tick.update(
                    dispatch="completed", terminated=response["terminated"], truncated=response["truncated"]
                )
                actions.append(normalized[0].detach().cpu())
                terminal = bool(response["terminated"] or response["truncated"])
                validate_step(packet, response["observation"], terminal=terminal)
                packet = response["observation"]
                if terminal:
                    break
            if not terminal:
                _, tokens, masks, state = prepared_observation(packet, policy, pre)
                tokens_list.append(torch.stack(tokens, dim=1)[0].detach().cpu())
                masks_list.append(torch.stack(masks, dim=1)[0].detach().cpu())
                states.append(state[0].detach().cpu())
            payload = {
                "tokens": torch.stack(tokens_list),
                "masks": torch.stack(masks_list),
                "states": torch.stack(states),
                "actions": torch.stack(actions),
                "chunk_ids": torch.arange(len(actions)) // 50,
                "language": language,
            }
            torch.save(payload, args.output / f"episode_{episode:02d}.pt")
            (args.output / f"episode_{episode:02d}_ticks.jsonl").write_text(
                "".join(json.dumps(tick) + "\n" for tick in ticks)
            )
            episodes.append(
                {
                    "episode": episode,
                    "completed_actions": len(actions),
                    "observations": len(tokens_list),
                    "terminal": terminal,
                    "native_success": bool(ticks[-1]["terminated"]),
                }
            )
            print(json.dumps(episodes[-1]), flush=True)
    except Exception:
        error = traceback.format_exc()
        if "ticks" in locals():
            (args.output / f"episode_{episode:02d}_partial_ticks.jsonl").write_text(
                "".join(json.dumps(tick) + "\n" for tick in ticks)
            )
            if tokens_list and actions:
                torch.save(
                    {
                        "tokens": torch.stack(tokens_list),
                        "masks": torch.stack(masks_list),
                        "states": torch.stack(states),
                        "actions": torch.stack(actions),
                        "error": error,
                    },
                    args.output / f"episode_{episode:02d}_partial.pt",
                )
    finally:
        client.close()
        collection = {
            "source_commit": source,
            "episodes": episodes,
            "error": error,
            "environment": client.metadata,
            "cleanup_error": client.cleanup_error,
            "subprocess_returncode": None if client.process is None else client.process.returncode,
        }
        (args.output / "collection_result.json").write_text(json.dumps(collection, indent=2) + "\n")
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    if error or client.cleanup_error or len(episodes) != 6:
        print(json.dumps(collection, indent=2))
        return 1
    rows, missing = [], []
    with torch.inference_mode():
        for episode in range(6):
            data = torch.load(args.output / f"episode_{episode:02d}.pt", weights_only=True)
            language = {key: value.to("cuda:0") for key, value in data["language"].items()}
            for anchor, delay in ((t, d) for t in range(0, 600, 50) for d in (1, 4, 8)):
                if anchor + delay >= len(data["tokens"]):
                    missing.append({"episode": episode, "anchor": anchor, "delay": delay})
                    continue
                assert bool((data["chunk_ids"][anchor : anchor + delay] == data["chunk_ids"][anchor]).all())
                z = data["tokens"][anchor : anchor + 1].to("cuda:0")
                future = data["tokens"][anchor + delay : anchor + delay + 1].to("cuda:0")
                mask = data["masks"][anchor : anchor + 1].to("cuda:0")
                assert torch.equal(data["masks"][anchor], data["masks"][anchor + delay])
                state = data["states"][anchor : anchor + 1].to("cuda:0")
                prefix = torch.zeros(1, 8, 6, device="cuda:0")
                prefix[0, :delay] = data["actions"][anchor : anchor + delay].to("cuda:0")
                valid = torch.arange(8, device="cuda:0")[None] < delay
                masks = tuple(mask[:, c] for c in range(2))
                prediction = predictor(
                    tuple(z[:, c] for c in range(2)),
                    masks,
                    prefix,
                    valid,
                    state,
                    torch.tensor([delay], device="cuda:0"),
                )
                predicted = z + torch.stack(prediction.delta_tokens, dim=1)
                noise_seed = 230000 + episode * 10000 + anchor * 10 + delay
                generator = torch.Generator(device="cuda:0").manual_seed(noise_seed)
                noise = torch.randn(1, 50, 32, device="cuda:0", generator=generator)
                outputs = {}
                for name, context in (("identity", z), ("predicted", predicted), ("oracle", future)):
                    output = policy.model.sample_actions(
                        None,
                        None,
                        language[OBS_LANGUAGE_TOKENS],
                        language[OBS_LANGUAGE_ATTENTION_MASK],
                        state,
                        noise=noise.clone(),
                        future_image_tokens=tuple(context[:, c] for c in range(2)),
                        future_image_token_masks=masks,
                    )[:, :, :6]
                    outputs[name] = projector(output).float()
                row = {"episode": episode, "anchor": anchor, "delay": delay, "noise_seed": noise_seed}
                for horizon in (25, 50):
                    for name in ("identity", "predicted"):
                        row[f"{name}_l1_{horizon}"] = float(
                            (outputs[name][:, :horizon] - outputs["oracle"][:, :horizon]).abs().mean().item()
                        )
                rows.append(row)
    report = {
        "source_commit": source,
        "kind": "future_visual_oracle_current_state_paired_noise",
        "future_state_used": False,
        "model_updated": False,
        **summarize_cases(rows, missing),
    }
    (args.output / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("cases", "missing_cases")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
