"""Exercise qualification accounting and episode sequencing without weights or a renderer."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
import libero_reference_qualification as q  # noqa: E402
import libero_single_step_native as single  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


def test_committed_manifest_has_exact_full_cohorts_and_disjoint_identities():
    manifest = q.read_manifest(q.MANIFEST)
    dev, confirm = (manifest["cohorts"][p] for p in ("development", "confirmation"))
    assert len(dev) == len(confirm) == 200
    assert len({r["tuple_id"] for r in dev + confirm}) == 400
    for rows, state_start, env_base, policy_base in (
        (dev, 1, 510000, 610000),
        (confirm, 21, 710000, 810000),
    ):
        assert [r["ordinal"] for r in rows] == list(range(200))
        assert [r["task_id"] for r in rows] == [t for t in range(10) for _ in range(20)]
        for t in range(10):
            task_rows = rows[t * 20 : (t + 1) * 20]
            assert [r["initial_state_id"] for r in task_rows] == list(range(state_start, state_start + 20))
            assert all(r["task_name"] == q.TASK_NAMES[t] for r in task_rows)
            assert all(r["environment_seed"] == env_base + 100 * t + r["initial_state_id"] for r in task_rows)
            assert all(r["policy_seed"] == policy_base + 100 * t + r["initial_state_id"] for r in task_rows)
    for key in ("environment_seed", "policy_seed", "initial_state_id"):
        assert {r[key] for r in dev}.isdisjoint({r[key] for r in confirm})
    assert len({r["environment_seed"] for r in dev + confirm}) == 400
    assert len({r["policy_seed"] for r in dev + confirm}) == 400


@pytest.mark.parametrize(
    "field", ["task_id", "task_name", "initial_state_id", "environment_seed", "policy_seed", "ordinal"]
)
def test_misaligned_manifest_field_is_rejected(tmp_path, field):
    manifest = q.registered_manifest()
    manifest["cohorts"]["development"][19][field] = "wrong"
    file = tmp_path / "manifest.json"
    q.write_json(file, manifest)
    with pytest.raises(ValueError, match="Manifest differs"):
        q.read_manifest(file)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order"])
def test_missing_duplicate_or_reordered_tuple_is_rejected(tmp_path, mutation):
    manifest = q.registered_manifest()
    rows = manifest["cohorts"]["confirmation"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = rows[0]
    else:
        rows.reverse()
    file = tmp_path / "manifest.json"
    q.write_json(file, manifest)
    with pytest.raises(ValueError, match="Manifest differs"):
        q.read_manifest(file)


def raw_observation():
    return {
        "pixels": {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "image2": np.ones((4, 4, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef": {"pos": np.array([0.1, 0.2, 0.7]), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
            "gripper": {"qpos": np.array([0.02, -0.02])},
        },
    }


def fake_policy(events, denoising_steps=10):
    policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=1))
    policy.eval = lambda: policy
    policy._rtc_enabled = lambda: False
    policy._prepare_batch = lambda batch: batch
    policy._check_get_actions_condition = lambda: not policy._queues[ACTION]
    policy.model = SimpleNamespace(action_out_proj=torch.nn.Identity())

    def reset():
        events.append("policy_reset")
        SmolVLAPolicy.reset(policy)

    def generate(batch, noise):
        events.append("generate")
        for _ in range(denoising_steps):
            policy.model.action_out_proj(torch.zeros(1, 50, 32))
        actions = torch.zeros(1, 50, 7)
        actions[:, :, 6] = 1.03
        return actions

    policy.reset = reset
    policy._get_action_chunk = generate
    policy.select_action = lambda batch: SmolVLAPolicy.select_action(policy, batch)
    return policy


class IdentityProcessor:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def reset(self):
        self.events.append(self.name + "_reset")

    def __call__(self, value):
        return value


class FakeNative:
    def __init__(self, success_at):
        self.steps = 0
        self.success_at = success_at
        self.actions = []

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        done = self.success_at is not None and self.steps == self.success_at
        return raw_observation(), 0.0, done, {}


class FakeEnv(gym.Env):
    def __init__(self, spec, events, success_at, close_fails):
        self.spec, self.events, self.close_fails = spec, events, close_fails
        self._env = FakeNative(success_at)
        self.init_state_id = spec["initial_state_id"]

    def _ensure_env(self):
        pass

    def reset(self, *, seed=None, options=None):
        self.events.append(("environment_seed", seed))
        assert seed == self.spec["environment_seed"]
        for _ in range(10):
            self._env.step([0, 0, 0, 0, 0, 0, -1])
            self.events.append("settled")
        self._env.steps = 0
        self.init_state_id += 1
        return raw_observation(), {"is_success": False}

    def step(self, action):
        obs, reward, done, _ = self._env.step(action)
        return obs, reward, done, False, {"is_success": done, "done": done}

    def close(self):
        self.events.append("close")
        if self.close_fails:
            raise RuntimeError("Required cleanup failed")
        self._env = None


def run_fake(tmp_path, *, success_at=1, close_fails=False, events=None, spec=None, denoising_steps=10):
    events = [] if events is None else events
    spec = spec or q.registered_manifest()["cohorts"]["development"][0]
    policy = fake_policy(events, denoising_steps)
    pre, post = IdentityProcessor(events, "pre"), IdentityProcessor(events, "post")

    def factory(spec):
        return gym.wrappers.TimeLimit(FakeEnv(spec, events, success_at, close_fails), max_episode_steps=280)

    output = tmp_path / f"tuple_{spec['ordinal']:03d}"
    result = q.run_episode(spec, output, policy, pre, post, factory, denoising_steps=denoising_steps)
    return result, output, events


def test_action_280_success_keeps_simultaneous_timeout_and_is_credited(tmp_path):
    result, output, _ = run_fake(tmp_path, success_at=280)
    assert result["success"] and result["terminated"] and result["truncated"]
    assert result["first_success_action"] == result["measured_actions"] == 280
    assert result["restricted_simulated_seconds"] == 14
    audited = q.audit_tuple(output, result["tuple"])
    assert audited["audit_errors"] == []
    assert audited["success"] and audited["native_measurement_returns"] == 280
    assert audited["out_of_box_actions"] == 280


def test_cleanup_failure_revokes_success_without_losing_native_signal(tmp_path):
    result, output, _ = run_fake(tmp_path, close_fails=True)
    assert result["native_success_observed"] is True
    assert result["status"] == "technical_failure" and result["success"] is False
    assert result["environment_closed"] is False
    assert q.audit_tuple(output, result["tuple"])["success"] is False


def test_reset_settling_then_policy_seed_and_one_consumed_action_every_episode(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(q, "seed_policy", lambda seed: events.append(("policy_seed", seed)))
    for spec in q.registered_manifest()["cohorts"]["development"][:2]:
        begin = len(events)
        result, output, _ = run_fake(tmp_path, success_at=3, events=events, spec=spec)
        order = events[begin:]
        assert order[:4] == [
            "policy_reset",
            "pre_reset",
            "post_reset",
            ("environment_seed", spec["environment_seed"]),
        ]
        assert order[4:14] == ["settled"] * 10
        assert order[14] == ("policy_seed", spec["policy_seed"])
        assert order[15:] == ["generate"] * 3 + ["close"]
        actions = [e for e in q.read_events(output / "events.jsonl") if e["event"] == "action_prepared"]
        assert len(actions) == 3 and all(e["queue_length"] == 0 for e in actions)
        assert result["success"] and q.audit_tuple(output, spec)["audit_errors"] == []


def test_nonfinite_state_is_a_technical_failure_before_policy_action(tmp_path, monkeypatch):
    original = q.policy_observation

    def invalid(raw, task):
        batch = original(raw, task)
        batch["observation.state"][0, 0] = float("nan")
        return batch

    monkeypatch.setattr(q, "policy_observation", invalid)
    result, _, events = run_fake(tmp_path)
    assert result["status"] == "technical_failure" and "generate" not in events
    assert result["environment_closed"] is True and result["measured_actions"] == 0


def outcome_records(success_counts):
    records = []
    for spec in q.registered_manifest()["cohorts"]["development"]:
        success = spec["ordinal"] % 20 < success_counts[spec["task_id"]]
        records.append(
            {
                "tuple": spec,
                "status": "completed",
                "success": success,
                "native_success_observed": success,
                "environment_closed": True,
                "terminated": success,
                "truncated": not success,
                "restricted_simulated_seconds": 5.0 if success else 14.0,
                "wall_seconds": 25.0,
            }
        )
    return records


def test_high_total_score_cannot_hide_one_weak_task():
    summary = q.summarize(outcome_records([15] + [20] * 9), "development", 0, "head")
    assert summary["successes"] == 195 and summary["completed"] == 200
    assert not summary["passed"]
    passing = q.summarize(outcome_records([16, 19, 19] + [18] * 7), "development", 0, "head")
    assert passing["successes"] == 180 and passing["passed"]
    assert all(t["wilson_denominator"] == 20 for t in passing["tasks"])
    assert len(passing["bootstrap_macro_95"]) == 2


def test_development_failure_cannot_create_confirmation_output_or_worker(tmp_path, monkeypatch):
    summary = q.summarize(outcome_records([15] + [20] * 9), "development", 0, "head")
    (tmp_path / "development").mkdir()
    q.write_json(tmp_path / "development/summary.json", summary)
    monkeypatch.setattr(q.subprocess, "Popen", lambda *a, **k: pytest.fail("Confirmation worker opened"))
    args = SimpleNamespace(output=tmp_path, phase="confirmation", manifest=q.MANIFEST, execution_head="head")
    with pytest.raises(RuntimeError, match="Development did not"):
        q.supervise(args)
    assert not (tmp_path / "confirmation").exists()


def test_worker_exit_without_final_record_only_fails_launched_tuple(tmp_path):
    specs = q.registered_manifest()["cohorts"]["development"]
    directory = q.tuple_directory(tmp_path, specs[0])
    directory.mkdir()
    q.write_json(directory / "started.json", {"tuple": specs[0]})
    journal = q.Journal(directory / "events.jsonl")
    journal.emit("native_step_started", segment="measurement", number=1, action=q.array_record(np.zeros(7)))
    journal.close()
    process = subprocess.run([sys.executable, "-c", "import os; os._exit(23)"], check=False)
    records = [q.audit_tuple(q.tuple_directory(tmp_path, s), s) for s in specs]
    summary = q.summarize(records, "development", process.returncode, "head")
    assert (summary["completed"], summary["technical_failure"], summary["not_run"]) == (0, 1, 199)
    assert summary["physical_step_unknown_tuples"] == 1 and summary["measured_actions_returned"] == 0
    assert summary["tasks"][0]["wilson_denominator"] == 1
    assert summary["tasks"][1]["observed_wilson_95"] is None
    assert summary["bootstrap_macro_95"] is None and not summary["passed"]
    assert not q.tuple_directory(tmp_path, specs[1]).exists()


def test_process_setup_failure_has_no_invented_episode_failures(tmp_path):
    records = [
        q.audit_tuple(q.tuple_directory(tmp_path, s), s)
        for s in q.registered_manifest()["cohorts"]["development"]
    ]
    summary = q.summarize(records, "development", 1, "head")
    assert summary["observed"] == summary["technical_failure"] == summary["completed"] == 0
    assert summary["not_run"] == 200 and not summary["passed"]


def test_incomplete_intervals_use_observed_denominator_and_keep_all_decision_slots():
    rows = outcome_records([2] * 10)
    for r in rows[3:]:
        r.update(status="not_run", success=False)
    summary = q.summarize(rows, "development", 1, "head")
    task = summary["tasks"][0]
    assert task["decision_success_rate"] == 2 / 20
    assert task["wilson_denominator"] == 3 and task["observed_wilson_95"] == q.wilson(2, 3)
    assert summary["not_run"] == 197 and summary["bootstrap_macro_95"] is None


def test_step_log_disagreement_revokes_credit(tmp_path):
    result, output, _ = run_fake(tmp_path)
    events = q.read_events(output / "events.jsonl")
    for e in events:
        if e["event"] == "native_step_started" and e["segment"] == "measurement":
            e["action"]["values"][0] = 0.5
    (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    audited = q.audit_tuple(output, result["tuple"])
    assert audited["status"] == "technical_failure" and not audited["success"]
    assert "native action" in audited["audit_errors"][0]


def single_step_records(counts):
    rows = []
    for spec in single.registered_tuples():
        success = spec["ordinal"] % 9 < counts[spec["task_id"]]
        rows.append(
            {
                "tuple": spec,
                "status": "completed",
                "success": success,
                "truncated": not success,
                "environment_closed": True,
                "restricted_simulated_seconds": 5.0 if success else 14.0,
                "audit_errors": [],
            }
        )
    return rows


def test_single_step_manifest_uses_all_and_only_unused_states_on_all_tasks():
    rows = single.registered_tuples()
    old = q.registered_manifest()["cohorts"]
    assert len(rows) == 90 and [r["ordinal"] for r in rows] == list(range(90))
    assert len({r["tuple_id"] for r in rows}) == 90
    for task in range(10):
        task_rows = rows[9 * task : 9 * (task + 1)]
        assert [r["initial_state_id"] for r in task_rows] == list(range(41, 50))
        assert all(r["task_id"] == task and r["task_name"] == q.TASK_NAMES[task] for r in task_rows)
        assert all(r["environment_seed"] == 940000 + 100 * task + r["initial_state_id"] for r in task_rows)
        assert all(r["policy_seed"] == 950000 + 100 * task + r["initial_state_id"] for r in task_rows)
    for key in ("initial_state_id", "environment_seed", "policy_seed"):
        assert {r[key] for r in rows}.isdisjoint({r[key] for r in old["development"] + old["confirmation"]})


def test_single_step_observed_projection_and_action_280_success(tmp_path):
    result, output, _ = run_fake(
        tmp_path, success_at=280, spec=single.registered_tuples()[0], denoising_steps=1
    )
    audit = q.audit_tuple(output, result["tuple"], denoising_steps=1)
    assert audit["success"] and audit["terminated"] and audit["truncated"]
    assert audit["first_success_action"] == 280 and not audit["audit_errors"]
    assert not q.audit_tuple(output, result["tuple"])["success"]


def test_single_step_screen_requires_both_total_and_every_task_without_full_qualification():
    passing = single.summarize(single_step_records([9] + [8] * 9), 0, "head")
    assert passing["successes"] == 81 and passing["native_screen_passed"]
    assert not passing["baseline_qualified"] and not passing["realtime_qualified"]
    assert not single.summarize(single_step_records([8] * 10), 0, "head")["native_screen_passed"]
    assert not single.summarize(single_step_records([7] + [9] * 9), 0, "head")["native_screen_passed"]
    assert not single.summarize(single_step_records([9] * 10), 23, "head")["native_screen_passed"]


def test_single_step_missing_slots_remain_unobserved():
    rows = single_step_records([9] * 10)
    for row in rows[2:]:
        row.update(status="not_run", success=False, environment_closed=None)
    result = single.summarize(rows, 2, "head")
    assert result["observed"] == result["completed"] == 2
    assert result["not_run"] == 88 and result["technical_failure"] == 0
    assert result["macro_bootstrap_95"] is None and not result["native_screen_passed"]
    assert result["tasks"][0]["wilson_95_observed"] == q.wilson(2, 2)
    assert result["tasks"][1]["wilson_95_observed"] is None


def test_single_step_no_old_confirmation_or_reordered_record_accepted():
    rows = single_step_records([9] * 10)
    rows[0]["tuple"] = q.registered_manifest()["cohorts"]["confirmation"][0]
    with pytest.raises(ValueError, match="90-slot"):
        single.summarize(rows, 0, "head")


def test_single_step_cleanup_failure_not_credited(tmp_path):
    result, output, _ = run_fake(tmp_path, close_fails=True, denoising_steps=1)
    audited = q.audit_tuple(output, result["tuple"], denoising_steps=1)
    assert audited["native_success_observed"] is True and not audited["success"]
    assert audited["status"] == "technical_failure" and audited["environment_closed"] is False
