"""L12 state fitting and offline coherent-teacher context factorial; no rollout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from leisaac_so101_matched import load_matched_runtime
from leisaac_so101_predicted import load_so101_l6_predictor
from so101_feasible_actions import FeasibleActionProjector
from so101_future_state import FutureStateResidual, state_pairs
from torch.utils.data import DataLoader, TensorDataset
from train_so101_action_distillation import action_outputs, forecast, query_from_data


def state_mae(model, arrays) -> tuple[float, float]:
    total, baseline, count = 0.0, 0.0, 0
    with torch.no_grad():
        for s, a, m, d, target in DataLoader(TensorDataset(*arrays), batch_size=512):
            s, a, m, d, target = (v.to("cuda:0") for v in (s, a, m, d, target))
            predicted = model(s, a, m, d)
            total += float((predicted[:, :6] - target[:, :6]).abs().sum())
            baseline += float((s[:, :6] - target[:, :6]).abs().sum())
            count += s.shape[0] * 6
    return total / count, baseline / count


def fit_state(data, output: Path, source: str) -> tuple[FutureStateResidual, dict]:
    parts = [state_pairs(data[e]) for e in range(6)]
    arrays = tuple(torch.cat([p[k] for p in parts]) for k in range(5))
    validation = {e: state_pairs(data[e]) for e in (6, 7)}
    model = FutureStateResidual().to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(
        TensorDataset(*arrays), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(3100)
    )
    history, best = [], float("inf")
    for epoch in range(31):
        if epoch:
            model.train()
            for s, a, m, d, target in loader:
                s, a, m, d, target = (v.to("cuda:0") for v in (s, a, m, d, target))
                optimizer.zero_grad(set_to_none=True)
                predicted = model(s, a, m, d)
                loss = torch.nn.functional.smooth_l1_loss(predicted[:, :6], target[:, :6])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        model.eval()
        metrics = {
            str(e): dict(zip(("predicted_mae", "persistence_mae"), state_mae(model, a), strict=True))
            for e, a in validation.items()
        }
        score = float(np.mean([r["predicted_mae"] for r in metrics.values()]))
        row = {"epoch": epoch, "validation_state_mae": score, "by_episode": metrics}
        history.append(row)
        if score < best:
            best = score
            torch.save(
                {
                    "kind": "so101_future_state_l12_v1",
                    "state_dict": model.state_dict(),
                    "source_commit": source,
                    "epoch": epoch,
                    "validation": metrics,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_episodes": list(range(6)),
                    "validation_episodes": [6, 7],
                },
                output / "state_best.pt",
            )
        print(json.dumps(row), flush=True)
    (output / "state_training.json").write_text(json.dumps(history, indent=2) + "\n")
    selected = torch.load(output / "state_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["state_dict"], strict=True)
    (output / "state_selection.json").write_text(
        json.dumps({k: v for k, v in selected.items() if k != "state_dict"}, indent=2) + "\n"
    )
    return model.eval().requires_grad_(False), selected


def summarize_contexts(rows: list[dict], expected: int) -> dict:
    arms = ("identity", "visual_only", "state_only", "joint", "oracle_visual_only", "oracle_state_only")
    metrics = {arm: float(np.mean([r[arm + "_l1_25"] for r in rows])) for arm in arms}
    by_episode = []
    for e in sorted({r["episode"] for r in rows}):
        group = [r for r in rows if r["episode"] == e]
        by_episode.append(
            {
                "episode": e,
                "cases": len(group),
                **{arm: float(np.mean([r[arm + "_l1_25"] for r in group])) for arm in arms},
            }
        )
    contrasts = {}
    for baseline in ("identity", "visual_only", "state_only"):
        improving = sum(r["joint_l1_25"] < r[baseline + "_l1_25"] for r in rows)
        interval = None
        if len(by_episode) == 6:
            values = np.array([[r[baseline], r["joint"]] for r in by_episode])
            rng = np.random.default_rng(3101)
            means = values[rng.integers(0, 6, size=(50000, 6))].mean(1)
            interval = np.percentile(100 * (1 - means[:, 1] / means[:, 0]), [2.5, 97.5]).tolist()
        ep_improved = sum(r["joint"] < r[baseline] for r in by_episode)
        contrasts[baseline] = {
            "reduction_percent": 100 * (1 - metrics["joint"] / metrics[baseline]),
            "improving_cases": improving,
            "improving_episodes": ep_improved,
            "episode_bootstrap_95_percent": interval,
            "stable_gate": bool(
                len(rows) == expected == 288
                and improving >= 192
                and ep_improved >= 5
                and interval is not None
                and interval[0] > 0
            ),
        }
    return {
        "means_l1_25": metrics,
        "by_episode": by_episode,
        "contrasts": contrasts,
        "cases": len(rows),
        "complete": len(rows) == expected,
    }


@torch.no_grad()
def factorial(policy, projector, visual, state_model, data, *, seed_base: int) -> list[dict]:
    rows = []
    for e, episode in data.items():
        for t in range(0, 600, 50):
            for d in (1, 4, 7, 8):
                q = query_from_data(episode, t, d, noise_seed=seed_base + e * 10000 + t * 10 + d)
                future_state = episode["states"][t + d : t + d + 1].to("cuda:0")
                predicted_state = state_model(q["state"], q["actions"], q["valid"], q["delay"])
                predicted_visual = forecast(visual, q)
                teacher = action_outputs(policy, projector, {**q, "state": future_state}, q["future"])
                row = {
                    "episode": e,
                    "anchor": t,
                    "delay": d,
                    "persistence_state_mae": float((q["state"][:, :6] - future_state[:, :6]).abs().mean()),
                    "predicted_state_mae": float((predicted_state[:, :6] - future_state[:, :6]).abs().mean()),
                }
                contexts = (
                    ("identity", q["z"], q["state"]),
                    ("visual_only", predicted_visual, q["state"]),
                    ("state_only", q["z"], predicted_state),
                    ("joint", predicted_visual, predicted_state),
                    ("oracle_visual_only", q["future"], q["state"]),
                    ("oracle_state_only", q["z"], future_state),
                )
                for name, z, s in contexts:
                    output = action_outputs(policy, projector, {**q, "state": s}, z)
                    for horizon in (25, 50):
                        row[f"{name}_l1_{horizon}"] = float(
                            (output[:, :horizon] - teacher[:, :horizon]).abs().mean()
                        )
                rows.append(row)
        print(json.dumps({"episode_finished": e, "cases": len(rows)}), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "test"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit source before fitting or evaluating")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if args.phase == "development":
        if manifest["split"] != "development" or manifest["environment_seeds"] != list(
            range(20270110, 20270118)
        ):
            parser.error("Only L10 designated development cache is authorized for this fitting")
    else:
        if args.selection is None or manifest.get("protocol") != "l12_joint":
            parser.error("Freeze eligible L12 development before the independent test")
        selection = json.loads(args.selection.read_text())
        if not selection["validation_qualified"] or manifest["environment_seeds"] != list(
            range(20270320, 20270326)
        ):
            parser.error("Wrong independent seeds or unqualified state model")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "source_commit": source,
                "phase": args.phase,
                "cache": str(args.cache.resolve()),
                "selection": str(args.selection) if args.selection else None,
                "teacher": "future_visual_and_future_model_ready_state",
                "risk_thresholds": None,
            },
            indent=2,
        )
        + "\n"
    )
    torch.set_num_threads(1)
    torch.manual_seed(3100)
    torch.cuda.manual_seed_all(3100)
    if args.phase == "development":
        data = {e: torch.load(args.cache / f"episode_{e:02d}.pt", weights_only=True) for e in range(8)}
        state_model, selected = fit_state(data, args.output, source)
        data = {e: data[e] for e in (6, 7)}
    else:
        selected = torch.load(args.selection.parent / "state_best.pt", map_location="cpu", weights_only=True)
        if (
            selected["kind"] != "so101_future_state_l12_v1"
            or selected["epoch"] != selection["state_epoch"]
            or selected["source_commit"] != selection["source_commit"]
            or selected["epoch"] != 29
            or selected["source_commit"] != "bf4e025dbf0cacf4777289b90498a8b70d4974fa"
        ):
            raise ValueError("Wrong state predictor identity")
        state_model = FutureStateResidual().to("cuda:0").eval().requires_grad_(False)
        state_model.load_state_dict(selected["state_dict"], strict=True)
        data = {e: torch.load(args.cache / f"episode_{e:02d}.pt", weights_only=True) for e in range(6)}
    base = repo.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    policy, _, _ = load_matched_runtime(snapshot, device="cuda:0")
    projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
    visual = load_so101_l6_predictor(base / "artifacts/m54l6_predictor_training_v1/best.pt", device="cuda:0")
    rows = factorial(
        policy,
        projector,
        visual,
        state_model,
        data,
        seed_base=310000 if args.phase == "development" else 320000,
    )
    summary = summarize_contexts(rows, expected=96 if args.phase == "development" else 288)
    state_pass = all(r["predicted_mae"] < r["persistence_mae"] for r in selected["validation"].values())
    qualified = bool(
        summary["complete"]
        and state_pass
        and all(r["joint"] < min(r["identity"], r["visual_only"]) for r in summary["by_episode"])
        and summary["contrasts"]["identity"]["improving_cases"] >= 64
    )
    report = {
        "source_commit": source,
        "state_epoch": selected["epoch"],
        "state_parameters": selected["parameters"],
        "validation_qualified": qualified if args.phase == "development" else None,
        "teacher": "future_visual_and_future_model_ready_state",
        "rows": rows,
        **summary,
        "task_benefit_tested": False,
        "runtime_enabled": False,
    }
    name = "selection.json" if args.phase == "development" else "test_report.json"
    (args.output / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
