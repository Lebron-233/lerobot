"""L15: fit only incremental visual residuals on the frozen state-only context."""

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
from train_so101_action_distillation import action_outputs, forecast
from train_so101_predictor_pilot import per_sample_errors

from lerobot.policies.smolvla.future_latent import LightweightFutureLatentPredictor


def device_query(pair: dict, noise_seed: int, device: str = "cuda:0") -> dict:
    keys = ("z", "future", "masks", "state", "predicted_state", "actions", "valid", "delay")
    query = {key: pair[key].to(device) for key in keys}
    query["language"] = {key: value.to(device) for key, value in pair["language"].items()}
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    query["noise"] = torch.randn((1, 50, 32), generator=generator, device=device)
    if query["delay"].item() != 7 or pair["target_step"] != pair["request_step"] + 7:
        raise ValueError("L15 does not change its fixed seven-step prediction target")
    expected = torch.arange(8, device=device)[None] < 7
    if not torch.equal(query["valid"], expected):
        raise ValueError("The recorded prefix is not exactly seven committed actions")
    return query


def conditional_actions(policy, projector, query: dict, tokens: torch.Tensor) -> torch.Tensor:
    # Forecast inputs retain current state. Only the frozen decoder uses this
    # predicted model-ready state, identically in teacher and all comparisons.
    return action_outputs(policy, projector, {**query, "state": query["predicted_state"]}, tokens)


@torch.no_grad()
def references(policy, projector, parent, data: dict[int, dict], seed_base: int) -> list[dict]:
    rows = []
    for episode, saved in data.items():
        for index, pair in enumerate(saved["pairs"]):
            for noise_index in range(2):
                seed = seed_base + episode * 10000 + index * 2 + noise_index
                query = device_query(pair, seed)
                teacher = conditional_actions(policy, projector, query, query["future"]).detach()
                baseline = conditional_actions(policy, projector, query, query["z"]).detach()
                parent_actions = conditional_actions(policy, projector, query, forecast(parent, query))
                rows.append(
                    {
                        "episode": episode,
                        "query_index": index,
                        "request_step": pair["request_step"],
                        "noise_index": noise_index,
                        "noise_seed": seed,
                        "query": query,
                        "teacher": teacher,
                        "baseline": baseline,
                        "state_only_l1": float((baseline[:, :25] - teacher[:, :25]).abs().mean()),
                        "parent_joint_l1": float((parent_actions[:, :25] - teacher[:, :25]).abs().mean()),
                    }
                )
        print(json.dumps({"reference_episode": episode, "cumulative_cases": len(rows)}), flush=True)
    return rows


def summarize(rows: list[dict], *, held_out: bool = False) -> dict:
    by_episode = []
    keys = ("state_only_l1", "parent_joint_l1", "student_l1")
    for episode in sorted({row["episode"] for row in rows}):
        group = [row for row in rows if row["episode"] == episode]
        by_episode.append(
            {
                "episode": episode,
                "cases": len(group),
                **{key: float(np.mean([row[key] for row in group])) for key in keys},
                "improved_cases": sum(row["student_l1"] < row["state_only_l1"] for row in group),
            }
        )
    if not by_episode:
        raise ValueError("No complete action-reference cases")
    means = {key: float(np.mean([row[key] for row in by_episode])) for key in keys}
    improved = sum(row["student_l1"] < row["state_only_l1"] for row in rows)
    episode_wins = sum(row["student_l1"] < row["state_only_l1"] for row in by_episode)
    coverage = 3 * improved >= 2 * len(rows)
    lower_than_both = means["student_l1"] < min(means["state_only_l1"], means["parent_joint_l1"])
    interval = None
    if held_out and len(by_episode) == 6:
        values = np.array([[row["state_only_l1"], row["student_l1"]] for row in by_episode])
        rng = np.random.default_rng(3501)
        boot = values[rng.integers(0, 6, size=(50000, 6))].mean(axis=1)
        interval = np.percentile(100 * (1 - boot[:, 1] / boot[:, 0]), [2.5, 97.5]).tolist()
    return {
        "equal_episode_means": means,
        "by_episode": by_episode,
        "cases": rows,
        "case_count": len(rows),
        "improved_cases": improved,
        "improved_episodes": episode_wins,
        "reduction_percent": 100 * (1 - means["student_l1"] / means["state_only_l1"]),
        "cluster_bootstrap_95_percent": interval,
        "validation_eligible": bool(
            len(by_episode) == 2 and lower_than_both and coverage and episode_wins == 2
        ),
        "stable_incremental_action_gate": bool(
            held_out
            and len(by_episode) == 6
            and coverage
            and episode_wins >= 5
            and interval is not None
            and interval[0] > 0
        ),
    }


@torch.no_grad()
def evaluate(policy, projector, student, refs: list[dict], *, held_out: bool = False) -> dict:
    student.eval()
    rows = []
    for ref in refs:
        query = ref["query"]
        output = conditional_actions(policy, projector, query, forecast(student, query))
        row = {key: value for key, value in ref.items() if key not in ("query", "teacher", "baseline")}
        row["student_l1"] = float((output[:, :25] - ref["teacher"][:, :25]).abs().mean())
        rows.append(row)
    return summarize(rows, held_out=held_out)


def train_loss(policy, projector, student, ref: dict):
    query = ref["query"]
    prediction = forecast(student, query)
    actions = conditional_actions(policy, projector, query, prediction)
    error = (actions[:, :25] - ref["teacher"][:, :25]).abs().mean()
    base_error = ref["state_only_l1"]
    scaled = error / (base_error + 0.02)
    harm = torch.relu((error - base_error) / (base_error + 0.02))
    latent, cosine = per_sample_errors(prediction, query["future"], query["masks"])
    loss = scaled + 0.5 * harm + 0.0005 * latent.mean() + 0.005 * cosine.mean()
    return loss, {"action_mae": float(error.detach()), "baseline_mae": base_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        parser.error("Commit the bounded study before any fitting or testing")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    collection = json.loads((args.cache / "result.json").read_text())
    training = args.phase == "train"
    expected_seeds = list(range(20270610, 20270618)) if training else list(range(20270630, 20270636))
    if (
        manifest.get("kind") != "l15_state_only_onpolicy_pairs"
        or manifest["split"] != ("development" if training else "test")
        or manifest["environment_seeds"] != expected_seeds
        or not collection["complete"]
        or collection["subprocess_returncode"] != 0
    ):
        parser.error("Wrong data split or incomplete native collection")
    if not training:
        if args.model_folder is None:
            parser.error("Test requires the frozen selected student")
        selection = json.loads((args.model_folder / "selection.json").read_text())
        if selection.get("kind") != "l15_incremental_visual_selection" or not selection["qualified"]:
            parser.error("Do not open a test for an ineligible student")
        if Path(manifest["selection"]).resolve() != (args.model_folder / "selection.json").resolve():
            parser.error("Collection was authorized for a different selection")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps({"source_commit": source, "phase": args.phase, "data_manifest": manifest}, indent=2) + "\n"
    )
    torch.set_num_threads(1)
    torch.manual_seed(3500)
    torch.cuda.manual_seed_all(3500)
    base = root.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    started = time.perf_counter()
    try:
        policy, _, _ = load_matched_runtime(snapshot, device="cuda:0")
        parent = load_so101_l6_predictor(
            base / "artifacts/m54l6_predictor_training_v1/best.pt", device="cuda:0"
        )
        projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
        student = LightweightFutureLatentPredictor(parent.config).to("cuda:0")
        student.load_state_dict(parent.state_dict(), strict=True)
        data = {
            e: torch.load(args.cache / f"episode_{e:02d}.pt", weights_only=True)
            for e in range(8 if training else 6)
        }
        if any(len(value["pairs"]) < 8 for value in data.values()):
            raise ValueError("Insufficient complete pairs; do not exclude an episode")
        if training:
            with torch.no_grad():
                student.up_projection.weight.zero_()
                student.up_projection.bias.zero_()
            train_refs = references(policy, projector, parent, {e: data[e] for e in range(6)}, 350000)
            val_refs = references(policy, projector, parent, {e: data[e] for e in (6, 7)}, 360000)
            with torch.no_grad():
                probe = train_refs[0]
                zero_actions = conditional_actions(
                    policy, projector, probe["query"], forecast(student, probe["query"])
                )
                zero_error = float((zero_actions - probe["baseline"]).abs().max())
                if zero_error != 0:
                    raise ValueError("Zero visual residual is not exactly the frozen state-only reference")
            groups = {e: [ref for ref in train_refs if ref["episode"] == e] for e in range(6)}
            rng = random.Random(3500)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
            history, selected, eligible_selected = [], None, False
            best, gradient_proof = float("inf"), None
            for update in range(1601):
                if update:
                    student.train()
                    group = groups[rng.randrange(6)]
                    ref = group[rng.randrange(len(group))]
                    optimizer.zero_grad(set_to_none=True)
                    loss, detail = train_loss(policy, projector, student, ref)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
                    if update == 1:
                        if not norm > 0 or any(
                            p.requires_grad or p.grad is not None for p in policy.parameters()
                        ):
                            raise ValueError("Real action gradients did not isolate the visual student")
                        gradient_proof = {"student_norm": float(norm), "policy_gradients": 0}
                    optimizer.step()
                    history.append(
                        {
                            "update": update,
                            "loss": float(loss.detach()),
                            "gradient_norm": float(norm),
                            **detail,
                        }
                    )
                if update % 200 == 0:
                    metric = evaluate(policy, projector, student, val_refs)
                    eligible = bool(update > 0 and metric["validation_eligible"])
                    score = metric["equal_episode_means"]["student_l1"]
                    if (
                        selected is None
                        or (eligible and not eligible_selected)
                        or (eligible == eligible_selected and score < best)
                    ):
                        selected, eligible_selected, best = update, eligible, score
                        torch.save(
                            {
                                "kind": "l15_state_conditional_visual_v1",
                                "source_commit": source,
                                "selected_update": update,
                                "state_dict": student.state_dict(),
                                "config": asdict(student.config),
                                "validation": metric,
                                "state_forecaster": "L12_epoch29",
                                "data_manifest": manifest,
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
                                "update": update,
                                "means": metric["equal_episode_means"],
                                "wins": metric["improved_cases"],
                                "cases": metric["case_count"],
                                "eligible": eligible,
                            }
                        ),
                        flush=True,
                    )
            saved = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
            result = {
                "kind": "l15_incremental_visual_selection",
                "source_commit": source,
                "qualified": eligible_selected,
                "selected_update": selected,
                "updates_complete": 1600,
                "parameters": sum(p.numel() for p in student.parameters()),
                "zero_residual_max_action_difference": zero_error,
                "gradient_proof": gradient_proof,
                "selected_validation": saved["validation"],
                "test_opened": False,
                "wall_s": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "task_benefit_established": False,
                "risk_thresholds": None,
            }
            (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "selected_validation"}), flush=True)
        else:
            saved = torch.load(args.model_folder / "best.pt", map_location="cpu", weights_only=True)
            if (
                saved["kind"] != "l15_state_conditional_visual_v1"
                or saved["selected_update"] != selection["selected_update"]
            ):
                raise ValueError("Selected student identity changed after collection")
            student.load_state_dict(saved["state_dict"], strict=True)
            refs = references(policy, projector, parent, data, 370000)
            result = {
                "source_commit": source,
                "selected_update": saved["selected_update"],
                "training_source": saved["source_commit"],
                "teacher": "actual_future_visual_with_same_causal_predicted_state",
                "task_benefit_established": False,
                "realtime_qualified": False,
                **evaluate(policy, projector, student, refs, held_out=True),
            }
            (args.output / "test_report.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2), flush=True)
    except Exception:
        (args.output / "failure.json").write_text(
            json.dumps({"source_commit": source, "error": traceback.format_exc()}, indent=2) + "\n"
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
