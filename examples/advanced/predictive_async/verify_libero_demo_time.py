"""F-DTC1: two fixed replay diagnostics for source-row physical time, not VLA eval.

Same initial full simulator state and same 175 recorded actions, at 20 then 10 Hz.
Original production controller, policy and all previous outcomes are unchanged.
"""

import argparse
import hashlib
import json
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/smolvla_prefix_data_contract_development_20260922"
FILES = ("hdf5_matched_state.npz", "raw_milk_7.npz", "hdf5_physical_probe.json", "raw_conversion_crosscheck.json")
CODE = ("examples/advanced/predictive_async/audit_libero_demo_time.py", "examples/advanced/predictive_async/verify_libero_demo_time.py", "examples/advanced/predictive_async/libero_rlds_contract.py", "examples/advanced/predictive_async/prepare_libero_prefix_cohort.py", "tests/test_libero_rlds_contract.py", "tests/test_libero_demo_time.py", "docs/experiments/SMOLVLA_DEMO_TIME_CONTRACT_PLAN.md")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, value):
    with Path(p).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sources():
    lock = json.loads((ROOT / "docs/experiments/SMOLVLA_GRAPH_CANDIDATE_LOCK.json").read_text())
    paths = [*(DATA / n for n in FILES), *(ROOT / n for n in CODE),
             *(ROOT / n for n in lock["core_sha256"])]
    package = ROOT.parent / "libero-reference-venv/lib/python3.12/site-packages"
    paths += [package / "libero/libero/envs/env_wrapper.py", package / "robosuite/controllers/config/osc_pose.json"]
    return {str(p): sha(p) for p in paths}


def guard(head):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != head:
        raise ValueError("Execution HEAD changed")
    b = json.loads((DATA / "baseline.json").read_text())["pending"]
    current = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    got = {s[3:]: {"status": s[:2], "sha256": sha(ROOT / s[3:])} for s in current.splitlines()}
    if got != b:
        raise ValueError("Old pending changed or execution files not committed")


def prepare(head):
    guard(head)
    prep, out = DATA / "replay_preparation.json", ROOT / f"outputs/smolvla_demo_time_{head[:8]}"
    if prep.exists() or out.exists():
        raise ValueError("One preparation and one output only")
    write(prep, {"head": head, "sources": sources(), "out": str(out), "frequencies": [20, 10],
                 "task": "pick_up_the_milk_and_place_it_in_the_basket", "source_demo": "demo_32",
                 "actions_per_arm": 175, "settling_per_arm": 10, "env_seed": 0,
                 "new_env": 2, "policy_forwards": 0, "optimizer_updates": 0,
                 "total_budget_s": 180, "row_policy": "no interpolation or action repetition"})
    print(json.dumps({"prep": str(prep), "out": str(out), "sha256": sha(prep)}), flush=True)


def execute(head):
    guard(head)
    saved = json.loads((DATA / "replay_preparation.json").read_text())
    if saved["head"] != head or saved["sources"] != sources():
        raise ValueError("Prepared source changed")
    body = (DATA / "replay_registration.md").read_text()
    rb = json.loads((DATA / "replay_registration_readback.json").read_text())
    if (not isinstance(rb.get("id"), int) or rb.get("body") != body
        or rb.get("issue_url") != "https://api.github.com/repos/Lebron-233/lerobot/issues/1"
        or f"F-DTC1-REGISTER:{head}" not in body or sha(DATA / "replay_preparation.json") not in body):
        raise ValueError("Actual registration missing")
    out = Path(saved["out"])
    out.mkdir()
    result = {"experiment": "F-DTC1", "head": head, "registration_id": rb["id"], "status": "technical_failure",
              "first_failure": None, "arms": [], "attempts": 1, "retries": 0,
              "policy_forwards": 0, "optimizer_updates": 0, "env_created": 0, "env_closed": 0}
    start = time.perf_counter()
    def alarm(_signum, _frame):
        raise TimeoutError("F-DTC1 total budget exceeded")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(180)
    env = None
    try:
        import robosuite.utils.transform_utils as transforms
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        h5 = np.load(DATA / "hdf5_matched_state.npz", allow_pickle=False)
        raw = np.load(DATA / "raw_milk_7.npz", allow_pickle=False)
        if h5["actions"].shape != (175, 7) or not np.array_equal(h5["actions"], raw["actions"]):
            raise ValueError("Action source differs")
        suite = benchmark.get_benchmark_dict()["libero_object"]()
        tasks = [suite.get_task(i) for i in range(suite.n_tasks)]
        task = next(t for t in tasks if t.name == saved["task"])
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        for hz in saved["frequencies"]:
            began = time.perf_counter()
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     control_freq=hz)
            result["env_created"] += 1
            env.seed(0)
            env.reset()
            env.set_init_state(h5["sim_states"][0])
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            states, sim_times, commands, outcomes = [], [], [], []
            for action in raw["actions"]:
                states.append(np.concatenate([obs["robot0_eef_pos"], transforms.quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"]]).astype(np.float32))
                sim_times.append(float(env.sim.data.time))
                commands.append(action.copy())
                obs, reward, done, info = env.step(action.tolist())
                outcomes.append(bool(done))
            sim_times.append(float(env.sim.data.time))
            states = np.asarray(states)
            np.savez_compressed(out / f"replay_{hz}.npz", states=states, sim_times=sim_times,
                                commands=commands, done=outcomes, reference=raw["states"])
            error = states.astype(np.float64) - raw["states"].astype(np.float64)
            steps = np.diff(sim_times)
            row = {"control_freq": hz, "actions": len(commands), "settling": 10,
                   "state_mse": float(np.square(error).mean()), "position_rmse_m": float(np.sqrt(np.square(error[:, :3]).mean())),
                   "max_abs_state": float(np.abs(error).max()), "exact_state_rows": int(np.all(error == 0, axis=1).sum()),
                   "simulation_dt_minmax": [float(steps.min()), float(steps.max())],
                   "terminal_success": outcomes[-1], "any_success": any(outcomes), "wall_s": time.perf_counter() - began}
            result["arms"].append(row)
            env.close()
            env = None
            result["env_closed"] += 1
            print(json.dumps(row), flush=True)
        if saved["sources"] != sources():
            raise ValueError("Source mutation")
        result["status"] = "completed"
    except BaseException:
        result["first_failure"] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
                result["env_closed"] += 1
            except BaseException:
                result["close_failure"] = traceback.format_exc()
        signal.alarm(0)
        result["wall_s"] = time.perf_counter() - start
        result["sources_unchanged"] = saved["sources"] == sources()
        write(out / "result.json", result)
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "completed" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution-head", required=True)
    p.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    if a.prepare:
        prepare(a.execution_head)
        return 0
    return execute(a.execution_head)


if __name__ == "__main__":
    raise SystemExit(main())
