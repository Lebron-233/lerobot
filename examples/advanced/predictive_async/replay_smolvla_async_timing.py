"""Stage C: drive the existing engine with fixed native completion traces and a CPU fake policy."""

import argparse
import json
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot.policies.rtc.scheduled_action_queue import ScheduledActionQueue, StageOutcome
from lerobot.rollout.inference import predictive_async
from lerobot.rollout.inference.predictive_async import PredictiveAsyncInferenceEngine

REPO = Path(__file__).resolve().parents[3]
NATIVE_HEAD = "5d45353ff98fc448bc514b284c1a19609405585b"
SOURCE = REPO / "outputs/libero_graph_native_5d45353f"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class IdentityPipeline:
    steps = ()

    def __call__(self, value):
        return value

    def reset(self):
        pass


class FakePolicy:
    """Seven marker columns, not physical commands; state and context remain identity."""

    def __init__(self):
        self.config = SimpleNamespace(type="smolvla")
        self.model = SimpleNamespace(encode_image_tokens=self.encode)
        self.complete = None

    def reset(self):
        pass

    def prepare_images(self, batch):
        return [torch.zeros(1, 3, 4, 4)] * 2, [torch.ones(1, dtype=torch.bool)] * 2

    def encode(self, images, masks):
        return tuple(torch.zeros(1, 2, 4) for _ in images), tuple(
            torch.ones(1, 2, dtype=torch.bool) for _ in masks
        )

    def predict_action_chunk(self, batch, **kwargs):
        return self.complete(batch, kwargs)


class VirtualControl:
    """Deterministic driver of _run_request, not a replacement inference engine.

    Tick callbacks run while the fake model is in flight. Completion precedes a
    coincident tick. No actual sleeping, renderer, model weights or native action.
    """

    def __init__(self, durations, task, output=None):
        self.durations = durations
        self.output = output
        self.now = 0.0
        self.tick_number = 0
        self.used = 0
        self.request = None
        self.records = []
        self.no_action_ticks = []
        self.action_indices = []
        self.policy = FakePolicy()
        self.policy.complete = self.complete
        self.engine = PredictiveAsyncInferenceEngine(
            policy=self.policy,
            preprocessor=IdentityPipeline(),
            postprocessor=IdentityPipeline(),
            robot_wrapper=SimpleNamespace(robot_type="mock"),
            hw_features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": [f"state_{i}" for i in range(8)],
                }
            },
            task=task,
            fps=20.0,
            device="cpu",
            queue_threshold=30,
            latency_quantile=0.9,
            latency_window=50,
            delay_safety_margin_steps=1,
            min_prediction_delay=0,
            max_prediction_delay=8,
            committed_guard_steps=2,
            max_late_steps=2,
            context_mode="identity",
        )
        self.engine.resume()

    def perf_counter(self):
        return self.now

    def tick(self):
        self.now = self.tick_number / 20
        index = self.engine.queue.next_action_index
        observation = {f"state_{i}": 0.0 for i in range(8)}
        observation["state_7"] = float(index)
        observation["control_tick"] = self.tick_number
        self.engine.notify_observation(observation)
        pending = self.engine._pending_request
        if self.request is not None and pending is not None:
            raise AssertionError("A second request appeared while the model was in flight")
        if (
            pending is not None
            and pending.plan is not None
            and pending.observation["state_7"] != pending.plan.next_action_index
        ):
            raise AssertionError("Observation and next-action snapshot differ")
        if self.engine.ready:
            action = self.engine.get_action(None)
            if action is None:
                self.no_action_ticks.append({"tick": self.tick_number, "kind": "underflow"})
            else:
                if int(action[0]) != index:
                    raise AssertionError("Action marker no longer matches its absolute control index")
                self.action_indices.append(index)
        else:
            self.no_action_ticks.append({"tick": self.tick_number, "kind": "startup_not_ready"})
        self.tick_number += 1

    def complete(self, batch, kwargs):
        request = self.request
        history = list(self.engine._latency_tracker._values)
        if batch["observation.state"][0, 7].item() != request.observation["state_7"]:
            raise AssertionError("The model received a different observation snapshot")
        # Reveal this completion duration only after notify_observation has chosen the plan.
        duration = self.durations[self.used]
        self.used += 1
        target = request.requested_at + duration
        prefix = request.plan.committed_policy_actions.clone() if request.plan else None
        base = request.plan.takeover_index if request.plan else self.engine.queue.next_action_index
        while self.tick_number / 20 < target:
            self.tick()
        self.now = target
        if history != list(self.engine._latency_tracker._values):
            raise AssertionError("Current completion leaked into planner history while in flight")
        if prefix is not None and not torch.equal(prefix, request.plan.committed_policy_actions):
            raise AssertionError("Committed prefix changed during inference")
        self.records.append(
            {
                "source_request_index": self.used - 1,
                "request_id": request.request_id,
                "kind": request.kind,
                "requested_at": request.requested_at,
                "completed_at": target,
                "duration_seconds": duration,
                "history_count_before": len(history),
                "planned_delay_steps": request.plan.planned_delay_steps if request.plan else None,
                "next_action_index_snapshot": request.observation["state_7"],
                "takeover_index": request.plan.takeover_index if request.plan else None,
                "current_duration_revealed_after_planning": True,
            }
        )
        return torch.arange(base, base + 50, dtype=torch.float32).view(1, 50, 1).expand(1, 50, 7).clone()

    def run(self):
        status = "completed"
        error = None
        try:
            with patch.object(predictive_async, "time", self):
                while self.used < len(self.durations):
                    self.tick()
                    request = self.engine._pending_request
                    if request is None:
                        if self.engine.queue.available_steps() == 0 and self.engine.ready:
                            raise RuntimeError("Empty queue and no request: fixed planner cannot recover")
                        continue
                    self.engine._pending_request = None
                    self.engine._request_ready.clear()
                    self.engine._request_in_flight = True
                    self.request = request
                    try:
                        self.engine._run_request(request)
                        self.records[-1]["publication_completed"] = True
                    finally:
                        self.request = None
                        self.engine._request_in_flight = False
            if self.action_indices != list(range(len(self.action_indices))):
                raise AssertionError("An absolute action index was repeated or skipped")
        except Exception:
            status, error = "technical_failure", traceback.format_exc()
        finally:
            self.engine.stop()
        result = {
            "status": status,
            "exception": error,
            "trace_requests_scheduled": len(self.durations),
            "trace_requests_completed": sum(r.get("publication_completed", False) for r in self.records),
            "wall_ticks": self.tick_number,
            "virtual_seconds": self.now,
            "actions_consumed": len(self.action_indices),
            "no_action_ticks": self.no_action_ticks,
            "engine_stats": asdict(self.engine.stats),
            "one_in_flight": True if status == "completed" else None,
            "absolute_indices_unique": len(self.action_indices) == len(set(self.action_indices)),
            "records": self.records,
        }
        if self.output is not None:
            write_json(self.output, result)
        return result


def queue_boundary_cases():
    def setup():
        queue = ScheduledActionQueue()
        old = torch.arange(12, dtype=torch.float32).view(12, 1)
        queue.install_active_chunk(old, old, task="A", reset_epoch=0, task_epoch=0)
        plan = queue.create_takeover_plan(
            request_id=1,
            planned_delay_steps=3,
            max_prediction_delay=8,
            committed_guard_steps=2,
            reset_epoch=0,
            task_epoch=0,
            task="A",
        ).plan
        return queue, plan

    def stage(queue):
        new = torch.arange(100, 110, dtype=torch.float32).view(10, 1)
        return queue.stage_chunk(new, new, request_id=1, reset_epoch=0, task_epoch=0, task="A")

    cases = []
    queue, plan = setup()
    plan.committed_policy_actions.fill_(-99)
    cases.append(
        {
            "case": "committed_prefix_private",
            "passed": bool(
                torch.equal(
                    queue.plan_snapshot().committed_policy_actions[:3, 0],
                    torch.arange(3, dtype=torch.float32),
                )
            ),
        }
    )
    staged = stage(queue)
    actions = [queue.get_with_task().post_policy_action.item() for _ in range(4)]
    cases.append(
        {
            "case": "early_staging_and_exact_takeover",
            "passed": staged.outcome is StageOutcome.STAGED_EARLY and actions == [0, 1, 2, 100],
            "actions": actions,
        }
    )
    for late in (1, 2, 3):
        queue, plan = setup()
        for _ in range(plan.takeover_index + late):
            queue.get_with_task()
        actual = stage(queue)
        matching_plan_cleared = queue.plan_snapshot() is None
        no_staged_chunk = not queue.has_staged_chunk()
        next_action = queue.get_with_task().post_policy_action.item()
        expected_next_action = plan.takeover_index + late
        cases.append(
            {
                "case": f"late_{late}_whole_discard",
                "passed": (
                    actual.late_steps == late
                    and actual.outcome is StageOutcome.DEADLINE_MISS
                    and next_action == expected_next_action
                    and matching_plan_cleared
                    and no_staged_chunk
                ),
                "late_steps": actual.late_steps,
                "actual_outcome": actual.outcome.value,
                "actual_next_action": next_action,
                "expected_next_action": expected_next_action,
                "matching_plan_cleared": matching_plan_cleared,
                "no_staged_chunk": no_staged_chunk,
            }
        )
    for change in ("reset", "task"):
        queue, _ = setup()
        queue.reset(1, task_epoch=0) if change == "reset" else queue.invalidate_task(1)
        actual = stage(queue)
        cases.append(
            {
                "case": change + "_invalidates_old_request",
                "passed": actual.outcome is StageOutcome.STALE,
                "actual_outcome": actual.outcome.value,
            }
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((SOURCE / "summary.json").read_text())
    if summary["execution_head"] != NATIVE_HEAD or not summary["native_graph_equivalence_passed"]:
        raise RuntimeError("Stage B must have passed at its fixed execution HEAD")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = {
        "source": str(SOURCE),
        "source_execution_head": NATIVE_HEAD,
        "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "new_native_episodes": 0,
        "model_weights_loaded": False,
        "fps": 20,
        "episodes": [],
        "status": "running",
        "scope": "cpu_fake_policy_timing_not_native_control",
        "late_policy": "whole_discard_any_late",
        "late_policy_decisions": [
            "https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5471357164",
            "https://github.com/Lebron-233/lerobot/issues/1#issuecomment-5554561794",
        ],
    }
    rows = json.loads((SOURCE / "tuple_accounting.json").read_text())
    for row in rows:
        spec = row["tuple"]
        events = SOURCE / f"tuple_{spec['ordinal']:03d}/events.jsonl"
        durations = [
            e["engineering_seconds"]
            for line in events.read_text().splitlines()
            if (e := json.loads(line))["event"] == "request_timing"
        ]
        replay = VirtualControl(
            durations, spec["task_name"], args.output / f"tuple_{spec['ordinal']:03d}.json"
        )
        episode = replay.run()
        result["episodes"].append(
            {
                "tuple": spec,
                **{k: v for k, v in episode.items() if k not in ("records", "no_action_ticks")},
                "no_action_ticks": len(episode["no_action_ticks"]),
                "startup_no_action_ticks": sum(
                    e["kind"] == "startup_not_ready" for e in episode["no_action_ticks"]
                ),
            }
        )
        print(
            f"REPLAY {spec['ordinal']} {episode['status']} requests={episode['trace_requests_completed']} "
            f"ticks={episode['wall_ticks']} no_action={len(episode['no_action_ticks'])}",
            flush=True,
        )
        if episode["status"] != "completed":
            result["first_failure"] = {"tuple": spec, "exception": episode["exception"]}
            break
    result["boundary_cases"] = queue_boundary_cases()
    failed = [c for c in result["boundary_cases"] if not c["passed"]]
    if failed and "first_failure" not in result:
        result["first_failure"] = failed[0]
    result["status"] = (
        "passed" if len(result["episodes"]) == 40 and "first_failure" not in result else "contract_gap"
    )
    write_json(args.output / "result.json", result)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
