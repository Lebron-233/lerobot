"""L15 causal pairs on new state-only controlled-delay trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import torch
from collect_so101_predictor_pilot import prepared_observation
from eval_leisaac_so101 import EnvClient
from eval_so101_controlled_delay import run_episode
from leisaac_so101_matched import candidate_manifest, load_matched_runtime
from leisaac_so101_predicted import load_so101_l6_predictor
from so101_feasible_actions import FeasibleActionProjector
from so101_joint_runtime import load_future_state
from so101_startup_preparation import warm_environment

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.random_utils import set_seed


class CausalVisualRecorder:
    """Read targets only once the exact committed prefix has physically executed."""

    def __init__(self) -> None:
        self.pending: dict | None = None
        self.pairs: list[dict] = []
        self.request_count = 0

    def request(self, step, plan, batch, tokens, masks, state, predicted_state):
        if self.pending is not None or predicted_state is None or plan.planned_delay_steps != 7:
            raise ValueError("L15 requires disjoint planned d7 requests with causal predicted state")
        if plan.next_action_index != step or plan.takeover_index != step + 7:
            raise ValueError("Request snapshot and target tick disagree")
        self.pending = {
            "request_step": step,
            "target_step": step + 7,
            "z": torch.stack(tokens, dim=1).detach().cpu().clone(),
            "masks": torch.stack(masks, dim=1).detach().cpu().clone(),
            "state": state.detach().cpu().clone(),
            "predicted_state": predicted_state.detach().cpu().clone(),
            "actions": plan.committed_policy_actions[None].detach().cpu().clone(),
            "valid": plan.committed_mask[None].detach().cpu().clone(),
            "delay": torch.tensor([7]),
            "language": {
                key: batch[key].detach().cpu().clone()
                for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
            },
        }
        self.request_count += 1

    def observe(self, packet, step, policy, pre):
        if self.pending is None:
            return
        if step > self.pending["target_step"]:
            raise ValueError("Skipped the intended future observation")
        if step != self.pending["target_step"]:
            return
        _, tokens, masks, state = prepared_observation(packet, policy, pre)
        if not torch.equal(torch.stack(masks, dim=1).cpu(), self.pending["masks"]):
            raise ValueError("Visual validity changed at the future target")
        self.pending["future"] = torch.stack(tokens, dim=1).detach().cpu().clone()
        self.pending["future_state"] = state.detach().cpu().clone()
        self.pairs.append(self.pending)
        self.pending = None

    def audit(self, ticks: list[dict]) -> dict:
        completed = {row["tick"]: row for row in ticks if row["dispatch"] == "completed"}
        for pair in self.pairs:
            start = pair["request_step"]
            actual = torch.tensor([completed[i]["normalized_action"] for i in range(start, start + 7)])
            if not torch.equal(actual, pair["actions"][0, :7]):
                raise ValueError("Recorded action prefix differs from the seven physical actions")
            if any(completed[i]["terminated"] or completed[i]["truncated"] for i in range(start, start + 7)):
                raise ValueError("A recorded target is after a native terminal/reset")
        if len(self.pairs) + int(self.pending is not None) != self.request_count:
            raise ValueError("Request-to-target accounting is incomplete")
        return {
            "requests": self.request_count,
            "complete_pairs": len(self.pairs),
            "censored_terminal_or_bound_prefixes": int(self.pending is not None),
            "audited_old_actions": 7 * len(self.pairs),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        parser.error("Commit the collection implementation before execution")
    if args.split == "test":
        if args.selection is None:
            parser.error("Publish the L15 selection before opening its independent scenes")
        selection = json.loads(args.selection.read_text())
        if selection.get("kind") != "l15_incremental_visual_selection" or not selection["qualified"]:
            parser.error("The incremental-vision student did not qualify")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    base = root.parent
    count, first_seed, first_policy = (
        (8, 20270610, 3510) if args.split == "development" else (6, 20270630, 3530)
    )
    snapshot = (
        base
        / "matched-candidate-cache/models--wsagi--SmolVLA-PickOrange/snapshots/c8c3318dba152b0ba671ff07b4314418d5aa4b4a"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "l15_state_only_onpolicy_pairs",
        "source_commit": source,
        "split": args.split,
        "candidate": candidate_manifest(snapshot),
        "state_model": "L12_epoch29",
        "environment_seeds": list(range(first_seed, first_seed + count)),
        "policy_seeds": list(range(first_policy, first_policy + count)),
        "max_actions": 1200,
        "simulated_delay": 7,
        "realtime": False,
        "selection": str(args.selection.resolve()) if args.selection else None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    client = EnvClient(
        base / "leisaac-sim-venv/bin/python",
        base
        / "simulator-assets/models--LightwheelAI--leisaac_env/snapshots/6c35af0af55506eb75c5592930134d4af44e8341/assets",
        base / "leisaac-source",
        "cpu",
    )
    client.camera_backend, client.episode_seconds, client.task_evidence = "standard", 120, True
    results, error = [], None
    try:
        policy, pre, post = load_matched_runtime(snapshot, device="cuda:0")
        visual = load_so101_l6_predictor(
            base / "artifacts/m54l6_predictor_training_v1/best.pt", device="cuda:0"
        )
        state_model = load_future_state(
            base / "artifacts/m54l12_state_development_v1/state_best.pt", "cuda:0"
        )
        client.start()
        for episode in range(count):
            ticks, events, setup, frames = [], [], [], []
            recorder = CausalVisualRecorder()
            trial = {}
            try:
                policy.reset()
                pre.reset()
                post.reset()
                packet = client.reset(first_seed + episode)
                warm_environment(client, packet, setup)
                packet = client.reset(first_seed + episode)
                set_seed(first_policy + episode)
                projector = FeasibleActionProjector.from_snapshot(snapshot, "cuda:0")
                trial = run_episode(
                    client,
                    packet,
                    policy,
                    pre,
                    post,
                    visual,
                    state_model,
                    projector,
                    "state_only",
                    first_policy + episode,
                    1200,
                    ticks,
                    events,
                    frames,
                    False,
                    recorder,
                )
                trial.update(recorder.audit(ticks))
                if len(recorder.pairs) < 8:
                    raise ValueError("Too few complete causal pairs; no replacement episode is allowed")
            finally:
                trial.update(
                    episode=episode, completed_actions=sum(t["dispatch"] == "completed" for t in ticks)
                )
                torch.save(
                    {"pairs": recorder.pairs, "metadata": trial}, args.output / f"episode_{episode:02d}.pt"
                )
                for name, rows in (("ticks", ticks), ("events", events), ("setup_ticks", setup)):
                    (args.output / f"episode_{episode:02d}_{name}.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                results.append(trial)
                print(json.dumps(trial), flush=True)
    except Exception:
        error = traceback.format_exc()
    finally:
        client.close()
        report = {
            "source_commit": source,
            "episodes": results,
            "error": error,
            "environment": client.metadata,
            "cleanup_error": client.cleanup_error,
            "subprocess_returncode": None if client.process is None else client.process.returncode,
            "complete": len(results) == count and error is None and client.cleanup_error is None,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "simulator.log").write_bytes(b"".join(client.logs))
    if error:
        print(error, flush=True)
    return int(not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
