"""Fixed offline action-expert probe, invoked only in the L6 held-out phase."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from leisaac_so101_contract import SCALAR_KEYS, hardware_features
from leisaac_so101_matched import TASK, WSAGI_REVISION, load_matched_runtime

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame


@torch.no_grad()
def action_consistency_probe(predictor, cache: Path, snapshot: Path) -> dict:
    if snapshot.name != WSAGI_REVISION:
        raise ValueError("Action probe must retain the L6 collection policy")
    policy, pre, _ = load_matched_runtime(snapshot, device="cuda:0")
    predictor.eval()
    raw = dict.fromkeys(SCALAR_KEYS, 0.0)
    raw.update({name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("top", "wrist")})
    batch = build_dataset_frame(hardware_features(), raw, prefix="observation")
    batch = prepare_observation_for_inference(batch, torch.device("cuda:0"), TASK, "so101_follower")
    batch["task"] = [TASK]
    # Placeholder RGB only builds the saved language/processor schema. The
    # policy calls below all use recorded/predicted tokens, never these pixels.
    batch = pre(batch)
    rows = []
    for e in [5, 6]:
        manifest = json.loads((cache / "manifest.json").read_text())
        filename = manifest.get("episode_files", {}).get(str(e), cache / f"episode_{e:02d}.pt")
        episode = torch.load(filename, weights_only=True)
        for t in [0, 200, 400]:
            for d in [1, 4, 8]:
                if t + d >= len(episode["tokens"]) or t + d > len(episode["actions"]):
                    continue
                if not bool((episode["chunk_ids"][t : t + d] == episode["chunk_ids"][t]).all()):
                    raise ValueError("The action probe crossed a future chunk-generation boundary")
                z = episode["tokens"][t].unsqueeze(0).to("cuda:0")
                future = episode["tokens"][t + d].unsqueeze(0).to("cuda:0")
                mask = episode["masks"][t].unsqueeze(0).to("cuda:0")
                if not torch.equal(episode["masks"][t], episode["masks"][t + d]):
                    raise ValueError("Action probe camera validity changed")
                state = episode["states"][t].unsqueeze(0).to("cuda:0")
                actions = torch.zeros(1, 8, 6, device="cuda:0", dtype=episode["actions"].dtype)
                actions[0, :d] = episode["actions"][t : t + d].to("cuda:0")
                prefix = torch.arange(8, device="cuda:0")[None] < d
                masks = tuple(mask[:, c] for c in range(2))
                delta = predictor(
                    tuple(z[:, c] for c in range(2)),
                    masks,
                    actions,
                    prefix,
                    state,
                    torch.tensor([d], device="cuda:0"),
                )
                predicted = (z.float() + torch.stack(delta.delta_tokens, 1).float()).to(z.dtype)
                # Preserve current normalized state for all three contexts.
                batch["observation.state"] = state[:, :6]
                torch.testing.assert_close(policy.prepare_state(batch), state, rtol=0, atol=0)
                noise_seed = 1910 + e * 1000 + t * 10 + d
                generator = torch.Generator(device="cuda:0").manual_seed(noise_seed)
                noise = torch.randn(
                    (1, 50, policy.config.max_action_dim), generator=generator, device="cuda:0"
                )
                outputs = {}
                for name, tokens in (("identity", z), ("predicted", predicted), ("oracle_visual", future)):
                    policy.reset()
                    outputs[name] = policy.predict_action_chunk(
                        batch,
                        noise=noise.clone(),
                        future_image_tokens=tuple(tokens[:, c] for c in range(2)),
                        future_image_token_masks=masks,
                    ).float()
                baseline = float((outputs["identity"] - outputs["oracle_visual"]).abs().mean().item())
                proposed = float((outputs["predicted"] - outputs["oracle_visual"]).abs().mean().item())
                rows.append(
                    {
                        "episode": e,
                        "anchor": t,
                        "delay": d,
                        "noise_seed": noise_seed,
                        "identity_action_l1": baseline,
                        "predicted_action_l1": proposed,
                    }
                )
    identity = float(np.mean([r["identity_action_l1"] for r in rows]))
    predicted = float(np.mean([r["predicted_action_l1"] for r in rows]))
    return {
        "kind": "offline_visual_oracle_current_state_paired_noise",
        "cases": rows,
        "identity_action_l1": identity,
        "predicted_action_l1": predicted,
        "action_l1_reduction_percent": 100 * (1 - predicted / identity),
        "future_state_used": False,
        "task_success_claim": False,
    }
