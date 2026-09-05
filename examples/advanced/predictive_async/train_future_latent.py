#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the offline SmolVLA future-latent predictor on approved caches."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from future_latent_training import (
    CACHE_PRODUCER_SHA,
    FutureLatentBatch,
    FutureLatentCacheDataset,
    SelectionState,
    accumulation_windows,
    bounded_run_markers,
    collate_future_latent_pairs,
    compute_future_latent_objective,
    compute_identity_baseline_metrics,
    deterministic_train_indices,
    deterministic_val_indices_by_delay,
    forward_predictor,
    load_last_checkpoint,
    make_predictor_optimizer,
    move_future_latent_batch,
    optimizer_step,
    save_predictor_checkpoint,
    update_selection,
)
from torch.utils.data import DataLoader, Subset

from lerobot.policies.smolvla.configuration_future_latent import FutureLatentConfig
from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor

SEED = 0
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 0
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5
MIN_RELATIVE_IMPROVEMENT = 0.001


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _epoch_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 30:
        raise argparse.ArgumentTypeError(f"max epochs must be in [1, 30], got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument(
        "--learning-rate",
        type=float,
        choices=(1e-4, 3e-4, 1e-3),
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        choices=(0.0, 1e-4, 1e-2),
        default=1e-4,
    )
    parser.add_argument("--lambda-cos", type=float, choices=(0.05, 0.1, 0.2), default=0.1)
    parser.add_argument("--lambda-risk", type=float, choices=(0.05, 0.1, 0.5), default=0.1)
    parser.add_argument("--max-epochs", type=_epoch_count, default=30)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-optimizer-steps", type=_positive_int)
    parser.add_argument("--max-val-pairs-per-delay", type=_positive_int)
    return parser.parse_args()


def _git_sha() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _jsonl_append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _is_bounded(args: argparse.Namespace) -> bool:
    return args.max_optimizer_steps is not None or args.max_val_pairs_per_delay is not None


def _cache_provenance(cache_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(cache_dir.resolve()),
        "split": manifest["split"],
        "complete_split": manifest["complete_split"],
        "producer_git_sha": manifest["producer"]["git_sha"],
        "inputs": manifest["inputs"],
        "policy_camera_order": manifest["policy_camera_order"],
        "episode_count": manifest["episode_count"],
        "frame_count": manifest["frame_count"],
        "valid_pair_count_by_delay": manifest["valid_pair_count_by_delay"],
        "semantics": manifest["semantics"],
        "token_scaling_convention": manifest["token_scaling_convention"],
    }


def _cache_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in provenance.items() if key != "path"}


def _make_run_config(
    args: argparse.Namespace,
    *,
    model_config: FutureLatentConfig,
    trainer_git_sha: str,
    train_provenance: dict[str, Any],
    val_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classification": "offline_future_latent_predictor_training_not_task_capability",
        "created_at_utc": datetime.now(UTC).isoformat(),
        **bounded_run_markers(_is_bounded(args)),
        "trainer_git_sha": trainer_git_sha,
        "cache_producer_sha": CACHE_PRODUCER_SHA,
        "train_cache": train_provenance,
        "val_cache": val_provenance,
        "future_latent_config": asdict(model_config),
        "device": str(args.device),
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // args.micro_batch_size,
        "num_workers": NUM_WORKERS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "weight_decay": args.weight_decay,
        },
        "scheduler": None,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "lambda_cos": args.lambda_cos,
        "lambda_risk": args.lambda_risk,
        "max_epochs": args.max_epochs,
        "seed": SEED,
        "max_optimizer_steps": args.max_optimizer_steps,
        "max_val_pairs_per_delay": args.max_val_pairs_per_delay,
        "resume_from": None if args.resume_from is None else str(args.resume_from.resolve()),
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def _seed_everything(device: torch.device) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def _data_loader(
    dataset: FutureLatentCacheDataset,
    indices: tuple[int, ...] | list[int],
    *,
    micro_batch_size: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_future_latent_pairs,
    )


def _training_window_batches(
    dataset: FutureLatentCacheDataset,
    ordered_indices: tuple[int, ...],
    *,
    start: int,
    stop: int,
    micro_batch_size: int,
    device: torch.device,
) -> list[FutureLatentBatch]:
    loader = _data_loader(
        dataset,
        ordered_indices[start:stop],
        micro_batch_size=micro_batch_size,
    )
    return [move_future_latent_batch(batch, device) for batch in loader]


def _validation_indices(
    dataset: FutureLatentCacheDataset,
    max_pairs_per_delay: int | None,
) -> tuple[int, ...]:
    if max_pairs_per_delay is not None:
        for delay in range(1, 9):
            available = len(dataset.indices_for_delay(delay))
            if available < max_pairs_per_delay:
                raise ValueError(
                    f"validation delay {delay} has {available} pairs, fewer than requested "
                    f"{max_pairs_per_delay}"
                )
    indices = deterministic_val_indices_by_delay(
        dataset,
        max_pairs_per_delay=max_pairs_per_delay,
    )
    if max_pairs_per_delay is not None and len(indices) != 8 * max_pairs_per_delay:
        raise RuntimeError("bounded validation did not select the requested count for every delay")
    return indices


def _evaluate(
    model: LightweightFutureLatentPredictor,
    dataset: FutureLatentCacheDataset,
    *,
    device: torch.device,
    micro_batch_size: int,
    max_pairs_per_delay: int | None,
    lambda_cos: float,
    lambda_risk: float,
) -> dict[str, Any]:
    indices = _validation_indices(dataset, max_pairs_per_delay)
    loader = _data_loader(dataset, indices, micro_batch_size=micro_batch_size)
    totals = {
        delay: {
            "count": 0,
            "predicted_smoothl1_sum": 0.0,
            "predicted_mse_sum": 0.0,
            "predicted_cosine_sum": 0.0,
            "identity_smoothl1_sum": 0.0,
            "identity_mse_sum": 0.0,
            "identity_cosine_sum": 0.0,
            "risk_smoothl1_sum": 0.0,
        }
        for delay in range(1, 9)
    }

    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_future_latent_batch(cpu_batch, device)
            prediction = forward_predictor(model, batch)
            objective = compute_future_latent_objective(
                prediction,
                batch,
                lambda_cos=lambda_cos,
                lambda_risk=lambda_risk,
            )
            identity = compute_identity_baseline_metrics(batch)
            risk_per_sample = F.smooth_l1_loss(
                prediction.predicted_error.float(),
                objective.risk_target,
                beta=1.0,
                reduction="none",
            )
            for delay in range(1, 9):
                selected = batch.delay_steps == delay
                selected_count = int(selected.sum().item())
                if selected_count == 0:
                    continue
                bucket = totals[delay]
                bucket["count"] += selected_count
                bucket["predicted_smoothl1_sum"] += float(objective.per_sample_smoothl1[selected].sum())
                bucket["predicted_mse_sum"] += float(objective.per_sample_mse[selected].sum())
                bucket["predicted_cosine_sum"] += float(objective.per_sample_cosine[selected].sum())
                bucket["identity_smoothl1_sum"] += float(identity.smoothl1[selected].sum())
                bucket["identity_mse_sum"] += float(identity.mse[selected].sum())
                bucket["identity_cosine_sum"] += float(identity.cosine[selected].sum())
                bucket["risk_smoothl1_sum"] += float(risk_per_sample[selected].sum())

    per_delay: dict[str, Any] = {}
    for delay, bucket in totals.items():
        count = int(bucket["count"])
        if count == 0:
            raise RuntimeError(f"validation produced no pairs for delay {delay}")
        metrics = {
            "sample_count": count,
            "predicted": {
                "smoothl1": bucket["predicted_smoothl1_sum"] / count,
                "mse": bucket["predicted_mse_sum"] / count,
                "cosine": bucket["predicted_cosine_sum"] / count,
            },
            "identity": {
                "smoothl1": bucket["identity_smoothl1_sum"] / count,
                "mse": bucket["identity_mse_sum"] / count,
                "cosine": bucket["identity_cosine_sum"] / count,
            },
            "risk_smoothl1": bucket["risk_smoothl1_sum"] / count,
        }
        metric_values = (
            *metrics["predicted"].values(),
            *metrics["identity"].values(),
            metrics["risk_smoothl1"],
        )
        if not all(math.isfinite(float(value)) for value in metric_values):
            raise RuntimeError(f"validation metrics are non-finite for delay {delay}")
        per_delay[str(delay)] = metrics

    val_macro_smoothl1 = sum(per_delay[str(delay)]["predicted"]["smoothl1"] for delay in range(1, 9)) / 8
    val_macro_mse = sum(per_delay[str(delay)]["predicted"]["mse"] for delay in range(1, 9)) / 8
    macro = {
        "predicted_smoothl1": val_macro_smoothl1,
        "predicted_mse": val_macro_mse,
        "predicted_cosine": sum(per_delay[str(delay)]["predicted"]["cosine"] for delay in range(1, 9)) / 8,
        "identity_smoothl1": sum(per_delay[str(delay)]["identity"]["smoothl1"] for delay in range(1, 9)) / 8,
        "identity_mse": sum(per_delay[str(delay)]["identity"]["mse"] for delay in range(1, 9)) / 8,
        "identity_cosine": sum(per_delay[str(delay)]["identity"]["cosine"] for delay in range(1, 9)) / 8,
        "risk_smoothl1": sum(per_delay[str(delay)]["risk_smoothl1"] for delay in range(1, 9)) / 8,
    }
    return {
        "sample_count": len(indices),
        "val_macro_smoothl1": val_macro_smoothl1,
        "val_macro_mse": val_macro_mse,
        "macro": macro,
        "per_delay": per_delay,
    }


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    run_config: dict[str, Any],
) -> None:
    prior = payload.get("train_config")
    if not isinstance(prior, dict):
        raise ValueError("last checkpoint is missing train_config")
    fixed_fields = (
        "run_kind",
        "protocol_complete",
        "eligible_for_checkpoint_selection",
        "eligible_for_test",
        "trainer_git_sha",
        "cache_producer_sha",
        "future_latent_config",
        "device",
        "micro_batch_size",
        "effective_batch_size",
        "num_workers",
        "optimizer",
        "scheduler",
        "grad_clip_norm",
        "lambda_cos",
        "lambda_risk",
        "seed",
        "max_val_pairs_per_delay",
    )
    for field in fixed_fields:
        if prior.get(field) != run_config[field]:
            raise ValueError(f"resume checkpoint train config differs at {field!r}")
    cache_provenance = payload.get("cache_provenance")
    if not isinstance(cache_provenance, dict):
        raise ValueError("last checkpoint is missing cache_provenance")
    for split in ("train", "val"):
        if _cache_identity(cache_provenance.get(split, {})) != _cache_identity(run_config[f"{split}_cache"]):
            raise ValueError(f"resume checkpoint uses a different {split} cache identity")
    cursor = prior.get("resume_cursor")
    if not isinstance(cursor, dict) or set(cursor) != {"next_epoch", "next_window"}:
        raise ValueError("last checkpoint is missing the exact training resume cursor")


def _verify_checkpoints(
    best_path: Path,
    last_path: Path,
    *,
    model_config: FutureLatentConfig,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, bool]:
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("checkpoint_kind") != "best":
        raise RuntimeError("best.pt has the wrong checkpoint kind")
    best_model = LightweightFutureLatentPredictor(model_config)
    best_model.load_state_dict(best["predictor_state_dict"], strict=True)

    last_model = LightweightFutureLatentPredictor(model_config)
    last_optimizer = make_predictor_optimizer(
        last_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    load_last_checkpoint(
        last_path,
        predictor=last_model,
        optimizer=last_optimizer,
        restore_rng=False,
    )
    return {"best": True, "last": True}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.resume_from is not None and (
        args.resume_from.name != "last.pt" or not args.resume_from.is_file()
    ):
        raise ValueError("--resume-from must point to an existing last.pt")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(device)

    train_dataset = FutureLatentCacheDataset(args.train_cache, expected_split="train")
    val_dataset = FutureLatentCacheDataset(args.val_cache, expected_split="val")
    train_provenance = _cache_provenance(args.train_cache, train_dataset.manifest)
    val_provenance = _cache_provenance(args.val_cache, val_dataset.manifest)
    cache_provenance = {"train": train_provenance, "val": val_provenance}

    model_config = FutureLatentConfig(token_dim=960, action_dim=6, state_dim=32, enabled=True)
    model = LightweightFutureLatentPredictor(model_config).to(device=device, dtype=torch.float32)
    optimizer = make_predictor_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_config = _make_run_config(
        args,
        model_config=model_config,
        trainer_git_sha=_git_sha(),
        train_provenance=train_provenance,
        val_provenance=val_provenance,
    )

    resume_payload: dict[str, Any] | None = None
    if args.resume_from is not None:
        resume_payload = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        _validate_resume_payload(resume_payload, run_config=run_config)

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    _json_dump(args.output_dir / "run_config.json", run_config)
    metrics_path.touch()

    global_step = 0
    start_epoch = 0
    start_window = 0
    selection_state = SelectionState()
    best_val_metrics: dict[str, Any] | None = None
    if args.resume_from is not None:
        loaded = load_last_checkpoint(
            args.resume_from,
            predictor=model,
            optimizer=optimizer,
            restore_rng=True,
        )
        global_step = int(loaded["global_step"])
        cursor = loaded["train_config"]["resume_cursor"]
        start_epoch = int(cursor["next_epoch"])
        start_window = int(cursor["next_window"])
        selection_state = loaded["selection_state"]
        if not isinstance(selection_state, SelectionState):
            raise ValueError("last checkpoint has an invalid selection_state")
        best_val_metrics = loaded["best_val_metrics"]
        prior_best_path = args.resume_from.parent / "best.pt"
        if not prior_best_path.is_file():
            raise ValueError("resume checkpoint must have a sibling best.pt")
        prior_best = torch.load(prior_best_path, map_location="cpu", weights_only=False)
        if prior_best.get("checkpoint_kind") != "best":
            raise ValueError("resume checkpoint sibling best.pt has the wrong checkpoint kind")
        shutil.copy2(prior_best_path, best_path)

    stop_reason = "max_epochs"
    last_validation: dict[str, Any] | None = None
    maximum_parameter_delta = 0.0
    optimizer_steps_this_run = 0
    epochs_evaluated = 0
    if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
        stop_reason = "max_optimizer_steps"

    for epoch in range(start_epoch, args.max_epochs):
        if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
            break
        model.train()
        train_indices = deterministic_train_indices(len(train_dataset), seed=SEED, epoch=epoch)
        windows = accumulation_windows(
            len(train_indices),
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
        )
        stopped_mid_epoch = False
        next_window = 0
        for window_index, (start, stop) in enumerate(windows):
            if epoch == start_epoch and window_index < start_window:
                continue
            micro_batches = _training_window_batches(
                train_dataset,
                train_indices,
                start=start,
                stop=stop,
                micro_batch_size=args.micro_batch_size,
                device=device,
            )
            result = optimizer_step(
                model,
                optimizer,
                micro_batches,
                lambda_cos=args.lambda_cos,
                lambda_risk=args.lambda_risk,
                grad_clip_norm=GRAD_CLIP_NORM,
            )
            result_metrics = asdict(result)
            numeric_values = (
                result.total,
                result.latent_smoothl1,
                result.cosine,
                result.risk_smoothl1,
                result.pre_clip_grad_norm,
                result.post_clip_grad_norm,
                result.parameter_delta,
            )
            if not all(math.isfinite(value) for value in numeric_values):
                raise RuntimeError("optimizer step produced non-finite evidence")
            if result.parameter_delta <= 0:
                raise RuntimeError("optimizer step did not update predictor parameters")
            if result.sample_count != stop - start:
                raise RuntimeError("optimizer step sample count does not match its accumulation window")

            global_step += result.optimizer_step_count
            optimizer_steps_this_run += result.optimizer_step_count
            maximum_parameter_delta = max(maximum_parameter_delta, result.parameter_delta)
            next_window = window_index + 1
            _jsonl_append(
                metrics_path,
                {
                    "event": "train_optimizer_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    **result_metrics,
                },
            )
            if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
                stop_reason = "max_optimizer_steps"
                stopped_mid_epoch = next_window < len(windows)
                break

        last_validation = _evaluate(
            model,
            val_dataset,
            device=device,
            micro_batch_size=args.micro_batch_size,
            max_pairs_per_delay=args.max_val_pairs_per_delay,
            lambda_cos=args.lambda_cos,
            lambda_risk=args.lambda_risk,
        )
        last_validation["epoch"] = epoch
        last_validation["global_step"] = global_step
        epochs_evaluated += 1
        _jsonl_append(metrics_path, {"event": "validation", **last_validation})

        selection_state, is_best, should_stop = update_selection(
            selection_state,
            epoch=epoch,
            val_macro_smoothl1=last_validation["val_macro_smoothl1"],
            val_macro_mse=last_validation["val_macro_mse"],
            min_relative_improvement=MIN_RELATIVE_IMPROVEMENT,
            patience=EARLY_STOPPING_PATIENCE,
        )
        if is_best:
            best_val_metrics = last_validation
            save_predictor_checkpoint(
                best_path,
                predictor=model,
                optimizer=None,
                train_config=run_config,
                epoch=epoch,
                global_step=global_step,
                best_val_metrics=best_val_metrics,
                cache_provenance=cache_provenance,
                trainer_git_sha=run_config["trainer_git_sha"],
                kind="best",
                selection_state=selection_state,
            )
        if best_val_metrics is None:
            raise RuntimeError("validation did not produce best metrics")

        epoch_complete = not stopped_mid_epoch
        cursor = {
            "next_epoch": epoch + 1 if epoch_complete else epoch,
            "next_window": 0 if epoch_complete else next_window,
        }
        save_predictor_checkpoint(
            last_path,
            predictor=model,
            optimizer=optimizer,
            train_config={**run_config, "resume_cursor": cursor},
            epoch=epoch,
            global_step=global_step,
            best_val_metrics=best_val_metrics,
            cache_provenance=cache_provenance,
            trainer_git_sha=run_config["trainer_git_sha"],
            kind="last",
            selection_state=selection_state,
        )

        if stopped_mid_epoch:
            break
        start_window = 0
        if run_config["eligible_for_checkpoint_selection"] and should_stop:
            stop_reason = "early_stopping"
            break

    if last_validation is None or best_val_metrics is None:
        raise RuntimeError("training stopped before producing validation/checkpoint evidence")
    if args.max_optimizer_steps is not None and global_step != args.max_optimizer_steps:
        raise RuntimeError(
            f"bounded run requested {args.max_optimizer_steps} optimizer steps but finished at {global_step}"
        )
    checkpoint_reload = _verify_checkpoints(
        best_path,
        last_path,
        model_config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    summary = {
        "schema_version": 1,
        **bounded_run_markers(_is_bounded(args)),
        "stop_reason": stop_reason,
        "train_pair_count": len(train_dataset),
        "val_pair_count": len(val_dataset),
        "global_step": global_step,
        "optimizer_steps_this_run": optimizer_steps_this_run,
        "epochs_evaluated": epochs_evaluated,
        "best_epoch": best_val_metrics["epoch"],
        "best_global_step": best_val_metrics["global_step"],
        "best_val_metrics": best_val_metrics,
        "last_val_metrics": last_validation,
        "maximum_parameter_delta": maximum_parameter_delta,
        "checkpoint_reload": checkpoint_reload,
        "test_data_read": False,
    }
    _json_dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
