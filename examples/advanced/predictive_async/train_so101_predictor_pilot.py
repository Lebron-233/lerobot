"""Train/validate, then once evaluate the pre-split independent L6 predictor."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor


def valid_pairs(observations: int, chunk_ids: torch.Tensor) -> list[tuple[int, int]]:
    return [
        (t, d)
        for t in range(min(observations - 1, len(chunk_ids)))
        for d in range(1, 9)
        if t + d < observations
        and t + d <= len(chunk_ids)
        and bool((chunk_ids[t : t + d] == chunk_ids[t]).all())
    ]


class ForecastPairs(Dataset):
    def __init__(self, root: Path, episodes: list[int]) -> None:
        self.episodes = [torch.load(root / f"episode_{i:02d}.pt", weights_only=True) for i in episodes]
        self.index = [
            (e, t, d)
            for e, episode in enumerate(self.episodes)
            for t, d in valid_pairs(len(episode["tokens"]), episode["chunk_ids"])
        ]
        if not self.index:
            raise ValueError("No causal within-chunk forecasting samples")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        e, t, d = self.index[index]
        episode = self.episodes[e]
        actions = torch.zeros(8, 6, dtype=episode["actions"].dtype)
        actions[:d] = episode["actions"][t : t + d]
        return (
            episode["tokens"][t],
            episode["masks"][t],
            episode["tokens"][t + d],
            episode["masks"][t + d],
            episode["states"][t],
            actions,
            torch.tensor(d),
        )


def per_sample_errors(prediction, target, masks):
    weight = masks.unsqueeze(-1).float()
    l1 = (F.smooth_l1_loss(prediction.float(), target.float(), reduction="none") * weight).sum((1, 2, 3))
    l1 = l1 / (weight.sum((1, 2, 3)) * prediction.shape[-1]).clamp_min(1)
    cosine = ((1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)) * masks).sum((1, 2))
    cosine = cosine / masks.sum((1, 2)).clamp_min(1)
    return l1, cosine


def forward_batch(model, batch):
    z, masks, target, future_masks, state, actions, delay = [x.to("cuda:0", non_blocking=True) for x in batch]
    if not torch.equal(masks, future_masks):
        raise ValueError("Camera/token validity changed across the future target")
    prefix = torch.arange(8, device=delay.device)[None] < delay[:, None]
    prediction = model(
        tuple(z[:, c] for c in range(2)), tuple(masks[:, c] for c in range(2)), actions, prefix, state, delay
    )
    predicted = z.float() + torch.stack(prediction.delta_tokens, dim=1).float()
    return z, predicted, target, masks, delay


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    sums = {
        d: {
            "count": 0,
            "identity_smooth_l1": 0.0,
            "predicted_smooth_l1": 0.0,
            "identity_cosine_distance": 0.0,
            "predicted_cosine_distance": 0.0,
        }
        for d in range(1, 9)
    }
    for batch in DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True):
        z, predicted, target, masks, delays = forward_batch(model, batch)
        i_l1, i_cos = per_sample_errors(z, target, masks)
        p_l1, p_cos = per_sample_errors(predicted, target, masks)
        values = torch.stack((i_l1, p_l1, i_cos, p_cos), 1).double().cpu()
        for d, row in zip(delays.cpu().tolist(), values.tolist(), strict=True):
            sums[d]["count"] += 1
            for key, value in zip(list(sums[d])[1:], row, strict=True):
                sums[d][key] += value
    total = {key: sum(s[key] for s in sums.values()) for key in sums[1]}

    def means(row):
        n = row["count"]
        result = {key: value / n if key != "count" else value for key, value in row.items()}
        result["smooth_l1_reduction_percent"] = 100 * (
            1 - result["predicted_smooth_l1"] / result["identity_smooth_l1"]
        )
        return result

    return {"aggregate": means(total), "by_delay": {str(d): means(row) for d, row in sums.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("Commit source before training or test evaluation")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if manifest["splits"] != {"train": [0, 1, 2, 3], "validation": [4], "test": [5, 6]}:
        raise ValueError("The pre-registered episode split changed")
    torch.set_num_threads(1)
    torch.manual_seed(1907)
    torch.cuda.manual_seed_all(1907)
    if args.phase == "train":
        args.output.mkdir(parents=True, exist_ok=False)
        train = ForecastPairs(args.cache, [0, 1, 2, 3])
        validation = ForecastPairs(args.cache, [4])
        config = FutureLatentConfig(
            token_dim=train.episodes[0]["tokens"].shape[-1],
            action_dim=6,
            state_dim=32,
            max_cameras=2,
            risk_head=False,
        )
        model = LightweightFutureLatentPredictor(config).to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loader = DataLoader(
            train, batch_size=32, shuffle=True, pin_memory=True, generator=torch.Generator().manual_seed(1907)
        )
        log, best = [], float("inf")
        for epoch in range(1, 21):
            model.train()
            total_loss, count = 0.0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                _, prediction, target, masks, _ = forward_batch(model, batch)
                smooth, cosine = per_sample_errors(prediction, target, masks)
                loss = (smooth + 0.1 * cosine).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                count += len(target)
                total_loss += float(loss.detach()) * len(target)
            metric = evaluate(model, validation)
            row = {"epoch": epoch, "train_loss": total_loss / count, "validation": metric}
            log.append(row)
            value = metric["aggregate"]["predicted_smooth_l1"]
            if value < best:
                best = value
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(config),
                        "epoch": epoch,
                        "validation": metric,
                        "source_commit": source,
                        "collection_manifest": manifest,
                        "parameters": sum(p.numel() for p in model.parameters()),
                    },
                    args.output / "best.pt",
                )
            (args.output / "training.json").write_text(json.dumps(log, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": row["train_loss"],
                        "validation_reduction_percent": metric["aggregate"]["smooth_l1_reduction_percent"],
                    }
                ),
                flush=True,
            )
        selected = torch.load(args.output / "best.pt", weights_only=True, map_location="cpu")
        result = {
            "source_commit": source,
            "selected_epoch": selected["epoch"],
            "parameters": selected["parameters"],
            "training_pairs": len(train),
            "validation_pairs": len(validation),
            "test_opened": False,
            "selected_validation": selected["validation"],
        }
        (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        # Exclusive report path prevents accidental test reruns through this entry.
        with (args.output / "test_report.json").open("x") as report:
            checkpoint = torch.load(args.output / "best.pt", weights_only=True, map_location="cuda:0")
            model = LightweightFutureLatentPredictor(FutureLatentConfig(**checkpoint["config"])).to("cuda:0")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            episodes = {str(e): evaluate(model, ForecastPairs(args.cache, [e])) for e in [5, 6]}
            total_count = sum(value["aggregate"]["count"] for value in episodes.values())
            aggregate = {
                key: sum(value["aggregate"][key] * value["aggregate"]["count"] for value in episodes.values())
                / total_count
                for key in (
                    "identity_smooth_l1",
                    "predicted_smooth_l1",
                    "identity_cosine_distance",
                    "predicted_cosine_distance",
                )
            }
            aggregate["count"] = total_count
            aggregate["smooth_l1_reduction_percent"] = 100 * (
                1 - aggregate["predicted_smooth_l1"] / aggregate["identity_smooth_l1"]
            )
            result = {
                "source_commit": source,
                "selected_epoch": checkpoint["epoch"],
                "parameters": checkpoint["parameters"],
                "aggregate": aggregate,
                "by_episode": episodes,
                "task_benefit_tested": False,
                "runtime_application_tested": False,
                "risk_thresholds": None,
            }
            report.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
