"""L17 noise x visual-information factorial with native-task follow-through."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
from collect_so101_predictor_pilot import prepared_observation
from eval_leisaac_so101 import EnvClient
from leisaac_so101_matched import load_matched_runtime
from run_so101_privileged_reference import drive, generate, geometry
from so101_feasible_actions import FeasibleActionProjector
from so101_joint_runtime import load_future_state
from so101_noise_coupling import ActionIndexedNoise
from so101_startup_preparation import warm_environment
from so101_task_evidence import PlacementTracker

from lerobot.utils.random_utils import set_seed

ARMS = ("fresh_state", "fresh_oracle", "coupled_state", "coupled_oracle")


def outcomes(ticks: list[dict], result: dict) -> dict:
    completed = [t for t in ticks if t["dispatch"] == "completed"]
    tracker = PlacementTracker()
    for tick in completed:
        witness = tick["task_transition_after_action"]
        if witness is None or witness["native_success"] != tick["terminated"]:
            raise ValueError("Native pre-auto-reset witness missing or inconsistent")
        tracker.update(witness, tick["tick"])
    valid = result["status"] != "technical_failure" and bool(ticks) and len(completed) == len(ticks)
    first = tracker.first_settled_step
    subgoal = bool(valid and first is not None)
    native = bool(valid and result.get("native_success") is True)
    return {
        "technical_valid": valid,
        "first_occupancy_observed_step": first,
        "subgoal_success": subgoal,
        "subgoal_time_s": (first + 1) / 30 if subgoal else 120.0,
        "native_success": native,
        "native_time_s": len(completed) / 30 if native else 120.0,
        "completed_actions": len(completed),
    }


def contrast(differences: list[float]) -> dict:
    result = {"differences_s": differences, "complete": len(differences) == 8}
    if len(differences) == 8:
        x = np.asarray(differences)
        rng = np.random.default_rng(3701)
        ci = np.percentile(x[rng.integers(0, 8, (50000, 8))].mean(1), [2.5, 97.5]).tolist()
        result.update(
            mean_s=float(x.mean()),
            bootstrap_95_s=ci,
            first_half_s=float(x[:4].mean()),
            second_half_s=float(x[4:].mean()),
            consistent_negative=bool(ci[1] < 0 and x[:4].mean() < 0 and x[4:].mean() < 0),
        )
    return result


def summarize(rows: list[dict]) -> dict:
    aggregates = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        aggregates[arm] = {
            "trials": len(group),
            "subgoal_successes": sum(r["subgoal_success"] for r in group),
            "native_successes": sum(r["native_success"] for r in group),
            "technical_failures": sum(not r["technical_valid"] for r in group),
            "mean_subgoal_time_s": float(np.mean([r["subgoal_time_s"] for r in group])) if group else None,
            "mean_native_time_s": float(np.mean([r["native_time_s"] for r in group])) if group else None,
        }
    pairs = (
        ("coupled_oracle", "coupled_state"),
        ("fresh_oracle", "fresh_state"),
        ("coupled_state", "fresh_state"),
        ("coupled_oracle", "fresh_oracle"),
    )
    contrasts = {}
    for a, b in pairs:
        delta, native, matches = [], [], []
        for block in range(8):
            group = {r["arm"]: r for r in rows if r["block"] == block}
            if a in group and b in group:
                delta.append(group[a]["subgoal_time_s"] - group[b]["subgoal_time_s"])
                native.append(group[a]["native_time_s"] - group[b]["native_time_s"])
                matches.append(
                    all(
                        group[arm]["initial_geometry_match"] and group[arm]["bootstrap_matches"]
                        for arm in (a, b)
                    )
                )
        contrasts[a + "_minus_" + b] = {
            **contrast(delta),
            "matched": matches,
            "native_descriptive": contrast(native),
        }
    interaction = []
    for block in range(8):
        group = {r["arm"]: r for r in rows if r["block"] == block}
        if set(group) == set(ARMS):
            interaction.append(
                group["coupled_oracle"]["subgoal_time_s"]
                - group["coupled_state"]["subgoal_time_s"]
                - group["fresh_oracle"]["subgoal_time_s"]
                + group["fresh_state"]["subgoal_time_s"]
            )
    primary = contrasts["coupled_oracle_minus_coupled_state"]
    ref, candidate = aggregates["coupled_state"], aggregates["coupled_oracle"]
    eligible = ref["trials"] == 8 and ref["subgoal_successes"] >= 7 and ref["technical_failures"] == 0
    return {
        "aggregates": aggregates,
        "contrasts": contrasts,
        "interaction": contrast(interaction),
        "reference_eligible": eligible,
        "privileged_visual_value_pass": bool(
            len(rows) == 32
            and eligible
            and primary.get("consistent_negative", False)
            and all(primary["matched"])
            and candidate["subgoal_successes"] >= ref["subgoal_successes"]
            and candidate["technical_failures"] <= ref["technical_failures"]
        ),
        "learned_visual_benefit": False,
        "realtime_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "study"), required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("Commit the noise protocol before outcome collection")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    smoke = args.phase == "smoke"
    if not smoke and (
        args.smoke is None or not json.loads((args.smoke / "summary.json").read_text())["smoke_pass"]
    ):
        parser.error("The single coupled-oracle wiring smoke must qualify first")
    orders = [ARMS[i:] + ARMS[:i] for i in range(4)]
    reverse = tuple(reversed(ARMS))
    orders += [reverse[i:] + reverse[:i] for i in range(4)]
    random.Random(3700).shuffle(orders)
    if smoke:
        orders = [("coupled_oracle",)]
    first_seed, first_policy = (20270801, 3701) if smoke else (20270810, 3710)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_commit": source,
        "phase": args.phase,
        "orders": orders,
        "first_seed": first_seed,
        "first_policy_seed": first_policy,
        "delay_steps": 7,
        "follow_native_task": True,
        "realtime": False,
        "risk_thresholds": None,
        "no_training": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    base = repo.parent
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    client = EnvClient(
        base / "leisaac-sim-venv/bin/python",
        base
        / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets",
        base / "leisaac-source",
        "cpu",
    )
    client.camera_backend, client.episode_seconds, client.task_evidence = "standard", 120, True
    rows, total_setup, global_error = [], 0, None
    try:
        policy, pre, post = load_matched_runtime(snapshot, device="cuda:0")
        state_model = load_future_state(
            base / "artifacts/m54l12_state_development_v1/state_best.pt", "cuda:0"
        )
        client.start()
        for block, order in enumerate(orders):
            seed, policy_seed = first_seed + block, first_policy + block
            field = ActionIndexedNoise(policy_seed, device="cuda:0")
            folder = args.output / f"block{block:02d}_bootstrap"
            folder.mkdir()
            setup = []
            try:
                packet = client.reset(seed)
                warm_environment(client, packet, setup)
                packet = client.reset(seed)
                policy.reset()
                pre.reset()
                post.reset()
                set_seed(policy_seed)
                batch, z, mask, state = prepared_observation(packet, policy, pre)
                projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
                normal, physical = generate(
                    policy, post, projector, batch, z, mask, state, policy_seed * 100000
                )
                fresh = torch.randn(
                    (1, 50, 32),
                    generator=torch.Generator(device="cuda:0").manual_seed(policy_seed * 100000),
                    device="cuda:0",
                )
                if not torch.equal(fresh, field.window(0)):
                    raise ValueError("The coupled field changed the real-policy bootstrap noise")
                bootstrap = {
                    "normalized": normal,
                    "physical": physical,
                    "geometry": geometry(packet),
                    "noise_seed": policy_seed * 100000,
                    "noise_field": field.field.cpu(),
                }
                torch.save(bootstrap, folder / "bootstrap.pt")
            finally:
                (folder / "setup_ticks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in setup))
                total_setup += sum(r["dispatch"] == "completed" for r in setup)
            for arm in order:
                folder = args.output / f"block{block:02d}_{arm}"
                folder.mkdir()
                ticks, events, setup, initial = [], [], [], None
                projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
                try:
                    packet = client.reset(seed)
                    warm_environment(client, packet, setup)
                    packet = client.reset(seed)
                    initial = geometry(packet)
                    if initial != bootstrap["geometry"]:
                        raise ValueError("Initial geometry does not match; never resample a scene")
                    policy.reset()
                    pre.reset()
                    post.reset()
                    set_seed(policy_seed)
                    context = "oracle_visual" if arm.endswith("oracle") else "state_only"
                    result = drive(
                        client,
                        packet,
                        policy,
                        pre,
                        post,
                        state_model,
                        None,
                        projector,
                        context,
                        policy_seed,
                        bootstrap,
                        60 if smoke else 3600,
                        False,
                        ticks,
                        events,
                        noise_provider=field if arm.startswith("coupled") else None,
                    )
                except Exception:
                    result = {"status": "technical_failure", "error": traceback.format_exc()}
                setup_count = sum(r["dispatch"] == "completed" for r in setup)
                total_setup += setup_count
                # Preserve partial dispatch evidence even if endpoint validation fails.
                for name, values in (("ticks", ticks), ("events", events), ("setup_ticks", setup)):
                    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
                measured = outcomes(ticks, result)
                prefix = [r["normalized_action"] for r in ticks[:27] if r["dispatch"] == "completed"]
                jumps = [
                    float(
                        np.linalg.norm(
                            (np.array(t["action"]) - np.array(ticks[i - 1]["action"]))
                            / np.array([220, 200, 190, 190, 320, 100])
                        )
                    )
                    for i, t in enumerate(ticks)
                    if i > 0 and t["dispatch"] == "completed" and t["queue_outcome"] == "takeover"
                ]
                row = {
                    **result,
                    **measured,
                    "block": block,
                    "arm": arm,
                    "seed": seed,
                    "policy_seed": policy_seed,
                    "initial_geometry_match": initial == bootstrap["geometry"],
                    "bootstrap_matches": bool(prefix)
                    and prefix == bootstrap["normalized"][: len(prefix)].tolist(),
                    "setup_actions": setup_count,
                    "takeovers": len(jumps),
                    "takeover_jump_mean_range_normalized_l2": float(np.mean(jumps)) if jumps else None,
                    "projection_components": sum(p["projected_components"] for p in projector.records),
                }
                rows.append(row)
                (folder / "result.json").write_text(json.dumps(row, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "block",
                                "arm",
                                "status",
                                "completed_actions",
                                "subgoal_success",
                                "native_success",
                                "subgoal_time_s",
                                "native_time_s",
                            )
                        }
                    ),
                    flush=True,
                )
    except Exception:
        global_error = traceback.format_exc()
    finally:
        client.close()
        cleanup = (
            client.cleanup_error is None and client.process is not None and client.process.returncode == 0
        )
        report = {
            **manifest,
            "rows": rows,
            "complete": global_error is None and len(rows) == sum(map(len, orders)),
            "shared_cleanup_complete": cleanup,
            "simulator_returncode": None if client.process is None else client.process.returncode,
            "global_error": global_error,
            "cleanup_error": client.cleanup_error,
            "setup_actions": total_setup,
            "completed_actions": sum(r["completed_actions"] for r in rows),
            "environment": client.metadata,
            **summarize(rows),
        }
        if not cleanup:
            report["privileged_visual_value_pass"] = False
        if smoke:
            report["smoke_pass"] = bool(
                cleanup
                and len(rows) == 1
                and rows[0]["technical_valid"]
                and rows[0]["completed_actions"] == 60
                and rows[0]["takeovers"] == 2
                and rows[0]["bootstrap_matches"]
            )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("rows", "environment")}, indent=2),
        flush=True,
    )
    return int(not report["complete"] or not cleanup or (smoke and not report["smoke_pass"]))


if __name__ == "__main__":
    raise SystemExit(main())
