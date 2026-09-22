"""Synthetic evidence/partition tests; no simulator or pretrained policy."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import audit_libero_demo_time as audit  # noqa: E402
from prepare_libero_prefix_cohort import KNOWN_ACTION_HASH, TASKS, freeze_split  # noqa: E402


def test_trajectory_split_no_overlap_and_known_probe_excluded():
    rows = [{"task": t, "trajectory_id": f"{t}-{j}", "action_sha256": f"action-{t}-{j}"}
            for t in TASKS for j in range(5)]
    rows.append({"task": 7, "trajectory_id": "old-probe", "action_sha256": KNOWN_ACTION_HASH})
    split, excluded = freeze_split(rows)
    assert len(split) == 20 and len(excluded) == 1
    groups = {role: {r["trajectory_id"] for r in split if r["split"] == role}
              for role in ("train", "dev", "sealed")}
    assert len(groups["train"]) == 12 and len(groups["dev"]) == len(groups["sealed"]) == 4
    assert not groups["train"] & groups["dev"] and not groups["dev"] & groups["sealed"]
    assert split == freeze_split(list(reversed(rows)))[0]
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_split(rows + rows[:1])
    with pytest.raises(ValueError, match="Need"):
        freeze_split(rows[:2])


def evidence(tmp_path, monkeypatch):
    ref = np.ones((175, 8), dtype=np.float32)
    commands = np.zeros((175, 7), dtype=np.float32)
    np.savez(tmp_path / "raw_milk_7.npz", states=ref, actions=commands)
    (tmp_path / "replay_preparation.json").write_text(json.dumps({"head": "test", "sources": {}}))
    out = tmp_path / "out"
    out.mkdir()
    rows = []
    for hz, delta in [(20, 0.001), (10, 0.1)]:
        states = ref + delta
        error = states.astype(float) - ref.astype(float)
        np.savez(out / f"replay_{hz}.npz", states=states, reference=ref, commands=commands,
                 sim_times=np.arange(176) / hz, done=np.ones(175, dtype=bool))
        rows.append({"state_mse": float((error ** 2).mean()),
                     "position_rmse_m": float(np.sqrt((error[:, :3] ** 2).mean())), "terminal_success": True})
    (out / "result.json").write_text(json.dumps({"head": "test", "status": "completed", "first_failure": None,
        "env_created": 2, "env_closed": 2, "attempts": 1, "retries": 0,
        "policy_forwards": 0, "optimizer_updates": 0, "arms": rows}))
    monkeypatch.setattr(audit, "DATA", tmp_path)
    monkeypatch.setattr(audit, "sources", lambda: {})
    return out


def test_audit_recomputes_time_and_metrics(tmp_path, monkeypatch):
    out = evidence(tmp_path, monkeypatch)
    result = audit.audit(out)
    assert result["independent_contract_accepted"] and result["twenty_hz_more_consistent"]
    assert result["native_actions_checked"] == 350


@pytest.mark.parametrize("field", ["commands", "sim_times", "states"])
def test_audit_rejects_tampered_arrays(tmp_path, monkeypatch, field):
    out = evidence(tmp_path, monkeypatch)
    path = out / "replay_20.npz"
    data = dict(np.load(path, allow_pickle=False))
    data[field] = data[field].copy()
    data[field].flat[1] += 0.1
    np.savez(path, **data)
    with pytest.raises(ValueError):
        audit.audit(out)
