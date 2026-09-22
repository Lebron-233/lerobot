"""Independent saved-array audit of F-DTC1. Never creates an environment."""

import argparse
import json
from pathlib import Path

import numpy as np
from verify_libero_demo_time import DATA, sha, sources, write


def audit(out):
    result = json.loads((out / "result.json").read_text())
    prep = json.loads((DATA / "replay_preparation.json").read_text())
    if (result["status"] != "completed" or result["first_failure"] is not None
        or result["env_created"] != 2 or result["env_closed"] != 2
        or result["attempts"] != 1 or result["retries"] != 0
        or result["policy_forwards"] != 0 or result["optimizer_updates"] != 0
        or result["head"] != prep["head"] or sources() != prep["sources"]):
        raise ValueError("Run/source/exit incomplete")
    reference = np.load(DATA / "raw_milk_7.npz", allow_pickle=False)
    rows = []
    for hz, original in zip((20, 10), result["arms"], strict=True):
        p = out / f"replay_{hz}.npz"
        data = np.load(p, allow_pickle=False)
        if not np.array_equal(data["commands"], reference["actions"]):
            raise ValueError("Commands differ from original 175 rows")
        if not np.array_equal(data["reference"], reference["states"]):
            raise ValueError("Saved reference differs")
        if data["states"].shape != (175, 8) or data["sim_times"].shape != (176,):
            raise ValueError("Replay population differs")
        dt = np.diff(data["sim_times"])
        if not np.allclose(dt, 1 / hz, rtol=0, atol=1e-9):
            raise ValueError("Physical clock differs from declared frequency")
        error = data["states"].astype(np.float64) - reference["states"].astype(np.float64)
        mse = float(np.square(error).mean())
        rmse = float(np.sqrt(np.square(error[:, :3]).mean()))
        if mse != original["state_mse"] or rmse != original["position_rmse_m"]:
            raise ValueError("Independent metric differs")
        if bool(data["done"][-1]) != original["terminal_success"]:
            raise ValueError("Terminal outcome differs")
        rows.append({"hz": hz, "state_mse": mse, "position_rmse_m": rmse,
                     "exact_state_rows": int(np.all(error == 0, axis=1).sum()),
                     "success": bool(data["done"][-1]), "sha256": sha(p)})
    supports = rows[0]["state_mse"] < rows[1]["state_mse"]
    return {"independent_contract_accepted": True, "native_actions_checked": 350,
            "arms": rows, "twenty_hz_more_consistent": supports,
            "state_mse_ratio_20_to_10": rows[0]["state_mse"] / rows[1]["state_mse"] if rows[1]["state_mse"] else None,
            "full_four_task_training_ready": False, "audit_env": 0, "audit_model_forwards": 0,
            "limitations": ["One known demonstration, not VLA task retention.",
                            "HDF5 and RLDS match actions but not every state; do not claim bitwise replay.",
                            "Published converter timestamps remain 10 FPS; no relabelling or interpolation."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output.resolve()
    result = audit(out)
    write(out / "independent_audit.json", result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
