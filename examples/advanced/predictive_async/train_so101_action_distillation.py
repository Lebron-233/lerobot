"""Action-aware L10 predictor fitting and one independent frozen evaluation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from leisaac_so101_matched import load_matched_runtime
from leisaac_so101_predicted import load_so101_l6_predictor
from so101_feasible_actions import FeasibleActionProjector
from train_so101_predictor_pilot import per_sample_errors

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def causal_pairs(data: dict, *, stride: int, delays: tuple[int, ...]) -> list[tuple[int, int]]:
    return [
        (t, d)
        for t in range(0, min(600, len(data["actions"])), stride)
        for d in delays
        if t + d < len(data["tokens"])
        and t + d <= len(data["actions"])
        and bool((data["chunk_ids"][t : t + d] == data["chunk_ids"][t]).all())
    ]


def query_from_data(data: dict, t: int, d: int, *, noise_seed: int, device: str = "cuda:0") -> dict:
    if (
        t + d >= len(data["tokens"])
        or t + d > len(data["actions"])
        or not bool((data["chunk_ids"][t : t + d] == data["chunk_ids"][t]).all())
    ):
        raise ValueError("Forecast input crosses a future replan or an episode boundary")
    if not torch.equal(data["masks"][t], data["masks"][t + d]):
        raise ValueError("Future visual target validity changed")
    actions = torch.zeros(1, 8, 6, device=device)
    actions[0, :d] = data["actions"][t : t + d].to(device)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    return {
        "z": data["tokens"][t : t + 1].to(device),
        "future": data["tokens"][t + d : t + d + 1].to(device),
        "masks": data["masks"][t : t + 1].to(device),
        "state": data["states"][t : t + 1].to(device),
        "actions": actions,
        "valid": torch.arange(8, device=device)[None] < d,
        "delay": torch.tensor([d], device=device),
        "language": {key: value.to(device) for key, value in data["language"].items()},
        "noise": torch.randn((1, 50, 32), generator=generator, device=device),
    }


def forecast(predictor, query: dict) -> torch.Tensor:
    z, masks = query["z"], query["masks"]
    prediction = predictor(
        tuple(z[:, c] for c in range(2)),
        tuple(masks[:, c] for c in range(2)),
        query["actions"],
        query["valid"],
        query["state"],
        query["delay"],
    )
    return (z.float() + torch.stack(prediction.delta_tokens, dim=1).float()).to(z.dtype)


def action_outputs(policy, projector, query: dict, tokens: torch.Tensor) -> torch.Tensor:
    language, mask = query["language"], query["masks"]
    outputs = policy.model.sample_actions(
        None,
        None,
        language[OBS_LANGUAGE_TOKENS],
        language[OBS_LANGUAGE_ATTENTION_MASK],
        query["state"],
        noise=query["noise"].clone(),
        future_image_tokens=tuple(tokens[:, c] for c in range(2)),
        future_image_token_masks=tuple(mask[:, c] for c in range(2)),
    )[:, :, :6]
    return projector(outputs).float()


def training_loss(policy, projector, predictor, query: dict) -> tuple[torch.Tensor, dict]:
    # no_grad rather than inference_mode: the fixed teacher is a normal tensor
    # saved by the differentiable loss, but it has no backward graph of its own.
    with torch.no_grad():
        teacher = action_outputs(policy, projector, query, query["future"])
    predicted = forecast(predictor, query)
    student = action_outputs(policy, projector, query, predicted)
    action_loss = (student[:, :25] - teacher[:, :25]).abs().mean()
    latent, cosine = per_sample_errors(predicted, query["future"], query["masks"])
    loss = action_loss + 0.001 * latent.mean() + 0.01 * cosine.mean()
    return loss, {"action_loss": float(action_loss.detach()), "latent_loss": float(latent.detach().mean())}


def case_specs(data: dict[int, dict], *, stride: int, delays: tuple[int, ...]) -> list[tuple[int, int, int]]:
    return [
        (e, t, d)
        for e, episode in data.items()
        for t, d in causal_pairs(episode, stride=stride, delays=delays)
    ]


@torch.no_grad()
def reference_cases(policy, projector, parent, data: dict[int, dict], *, seed_base: int) -> list[dict]:
    result = []
    for e, t, d in case_specs(data, stride=50, delays=(1, 4, 8)):
        query = query_from_data(data[e], t, d, noise_seed=seed_base + e * 10000 + t * 10 + d)
        teacher = action_outputs(policy, projector, query, query["future"])
        identity = action_outputs(policy, projector, query, query["z"])
        parent_actions = action_outputs(policy, projector, query, forecast(parent, query))
        row = {"episode": e, "anchor": t, "delay": d, "query": query, "teacher": teacher}
        for horizon in (25, 50):
            for name, output in (("identity", identity), ("parent", parent_actions)):
                row[f"{name}_l1_{horizon}"] = float((output[:, :horizon] - teacher[:, :horizon]).abs().mean())
        result.append(row)
    return result


@torch.no_grad()
def evaluate(policy, projector, predictor, references: list[dict]) -> dict:
    predictor.eval()
    rows = []
    for reference in references:
        query = reference["query"]
        predicted = forecast(predictor, query)
        actions = action_outputs(policy, projector, query, predicted)
        row = {key: value for key, value in reference.items() if key not in ("query", "teacher")}
        for horizon in (25, 50):
            row[f"predicted_l1_{horizon}"] = float(
                (actions[:, :horizon] - reference["teacher"][:, :horizon]).abs().mean()
            )
        for name, tokens in (("identity", query["z"]), ("predicted", predicted)):
            l1, cos = per_sample_errors(tokens, query["future"], query["masks"])
            row[name + "_latent_smooth_l1"] = float(l1.mean())
            row[name + "_latent_cosine"] = float(cos.mean())
        rows.append(row)
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key not in ("episode", "anchor", "delay")
    }
    metrics["improved_cases"] = sum(row["predicted_l1_25"] < row["identity_l1_25"] for row in rows)
    metrics["cases"] = rows
    return metrics


def independent_summary(metrics: dict) -> dict:
    rows = metrics["cases"]
    episodes = []
    for e in sorted({row["episode"] for row in rows}):
        group = [row for row in rows if row["episode"] == e]
        i, p = (float(np.mean([row[key] for row in group])) for key in ("identity_l1_25", "predicted_l1_25"))
        episodes.append(
            {
                "episode": e,
                "count": len(group),
                "identity_l1_25": i,
                "predicted_l1_25": p,
                "reduction_percent": 100 * (1 - p / i) if i else None,
            }
        )
    interval = None
    if len(episodes) == 6 and all(row["identity_l1_25"] > 0 for row in episodes):
        values = np.array([[row["identity_l1_25"], row["predicted_l1_25"]] for row in episodes])
        rng = np.random.default_rng(2501)
        means = values[rng.integers(0, 6, size=(50000, 6))].mean(axis=1)
        interval = np.percentile(100 * (1 - means[:, 1] / means[:, 0]), [2.5, 97.5]).tolist()
    stable = bool(
        len(rows) == 216
        and interval is not None
        and interval[0] > 0
        and metrics["improved_cases"] >= 144
        and sum(row["reduction_percent"] is not None and row["reduction_percent"] > 0 for row in episodes)
        >= 5
    )
    return {
        **metrics,
        "by_episode": episodes,
        "bootstrap_95_reduction_percent": interval,
        "reduction_percent": 100 * (1 - metrics["predicted_l1_25"] / metrics["identity_l1_25"]),
        "stable_action_gate": stable,
        "task_benefit_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit source before fitting or testing")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    required = "development" if args.phase == "train" else "test"
    if manifest["split"] != required:
        parser.error("This phase cannot open the other split")
    base = repo.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    parent_path = base / "artifacts/m54l6_predictor_training_v1/best.pt"
    if args.phase == "train":
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        selection = json.loads((args.output / "selection.json").read_text())
        if not selection["validation_qualified"] or (args.output / "test_report.json").exists():
            parser.error("Unqualified selection or an already opened test")
    torch.set_num_threads(1)
    torch.manual_seed(2500)
    torch.cuda.manual_seed_all(2500)
    policy, _, _ = load_matched_runtime(snapshot, device="cuda:0")
    projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
    parent = load_so101_l6_predictor(parent_path, device="cuda:0")
    predictor = LightweightFutureLatentPredictor(parent.config).to("cuda:0")
    predictor.load_state_dict(parent.state_dict(), strict=True)
    started = time.perf_counter()
    try:
        if args.phase == "train":
            data = {e: torch.load(args.cache / f"episode_{e:02d}.pt", weights_only=True) for e in range(8)}
            train_data = {e: data[e] for e in range(6)}
            specs = case_specs(train_data, stride=25, delays=tuple(range(1, 9)))
            references = reference_cases(
                policy, projector, parent, {e: data[e] for e in (6, 7)}, seed_base=270000
            )
            if len(references) != 72 or not specs:
                raise ValueError("Development collection incomplete; do not silently select on fewer cases")
            rng = random.Random(2500)
            optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4, weight_decay=1e-4)
            history, best, selected_update, selected_metric = [], float("inf"), None, None
            gradient_proof = None
            for update in range(401):
                if update:
                    predictor.train()
                    e, t, d = specs[rng.randrange(len(specs))]
                    query = query_from_data(data[e], t, d, noise_seed=280000 + update)
                    optimizer.zero_grad(set_to_none=True)
                    loss, losses = training_loss(policy, projector, predictor, query)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        predictor.parameters(), 1.0, error_if_nonfinite=True
                    )
                    if update == 1:
                        if not norm > 0 or any(
                            p.requires_grad or p.grad is not None for p in policy.parameters()
                        ):
                            raise ValueError("Backward path failed to isolate nonzero predictor gradients")
                        gradient_proof = {"predictor_gradient_norm": float(norm), "policy_gradients": 0}
                    optimizer.step()
                    history.append(
                        {
                            "update": update,
                            "loss": float(loss.detach()),
                            "gradient_norm": float(norm),
                            **losses,
                        }
                    )
                    if update % 25 == 0:
                        print(json.dumps(history[-1]), flush=True)
                if update % 100 == 0:
                    metric = evaluate(policy, projector, predictor, references)
                    score = metric["predicted_l1_25"]
                    if score < best:
                        best, selected_update, selected_metric = score, update, metric
                        torch.save(
                            {
                                "kind": "so101_action_distilled_v1",
                                "state_dict": predictor.state_dict(),
                                "config": asdict(predictor.config),
                                "source_commit": source,
                                "selected_update": update,
                                "collection_manifest": manifest,
                                "parent_checkpoint": str(parent_path),
                                "validation": metric,
                            },
                            args.output / "best.pt",
                        )
                    (args.output / f"validation_{update:04d}.json").write_text(
                        json.dumps(metric, indent=2) + "\n"
                    )
                    (args.output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "validation_update": update,
                                "identity": metric["identity_l1_25"],
                                "parent": metric["parent_l1_25"],
                                "student": score,
                                "improved_cases": metric["improved_cases"],
                            }
                        ),
                        flush=True,
                    )
            qualified = bool(
                selected_update > 0
                and best < min(selected_metric["identity_l1_25"], selected_metric["parent_l1_25"])
                and selected_metric["improved_cases"] > 36
            )
            selection = {
                "source_commit": source,
                "selected_update": selected_update,
                "validation_qualified": qualified,
                "selected_validation": selected_metric,
                "gradient_proof": gradient_proof,
                "training_cases": len(specs),
                "updates": 400,
                "test_opened": False,
                "wall_s": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
            print(json.dumps({k: v for k, v in selection.items() if k != "selected_validation"}), flush=True)
        else:
            checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
            if (
                checkpoint["kind"] != "so101_action_distilled_v1"
                or FutureLatentConfig(**checkpoint["config"]) != parent.config
            ):
                raise ValueError("Selected student configuration changed")
            predictor.load_state_dict(checkpoint["state_dict"], strict=True)
            data = {e: torch.load(args.cache / f"episode_{e:02d}.pt", weights_only=True) for e in range(6)}
            with (args.output / "test_report.json").open("x") as file:
                references = reference_cases(policy, projector, parent, data, seed_base=290000)
                report = {
                    "source_commit": source,
                    "selected_update": checkpoint["selected_update"],
                    "future_state_used": False,
                    **independent_summary(evaluate(policy, projector, predictor, references)),
                }
                json.dump(report, file, indent=2)
                file.write("\n")
            print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    except Exception:
        (args.output / f"{args.phase}_failure.json").write_text(
            json.dumps({"error": traceback.format_exc(), "source_commit": source}, indent=2)
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
