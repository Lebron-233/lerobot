"""CPU feasibility probe, not a trained policy or a new task qualification.

Uses tiny synthetic projections with the actual SmolVLA suffix method. Reads only
local dataset metadata; never downloads, loads policy weights or creates an Env.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

REPO = Path(__file__).resolve().parents[3]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def suffix_interface_probe():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260922)
        stub = SimpleNamespace(
            action_in_proj=torch.nn.Linear(32, 8),
            action_time_mlp_in=torch.nn.Linear(16, 8),
            action_time_mlp_out=torch.nn.Linear(8, 8),
            vlm_with_expert=SimpleNamespace(expert_hidden_size=8),
            config=SimpleNamespace(min_period=0.004, max_period=4.0, chunk_size=50),
        )
        actions = torch.linspace(-1, 1, 2 * 50 * 32).reshape(2, 50, 32).requires_grad_(True)
        scalar_time = torch.tensor([0.25, 0.75])
        scalar, pad, attention = VLAFlowMatching.embed_suffix(stub, actions, scalar_time)
        assert scalar.shape == (2, 50, 8) and pad.shape == attention.shape == (2, 50)
        grad = torch.autograd.grad(scalar.square().mean(), actions)[0]
        assert torch.isfinite(scalar).all() and torch.isfinite(grad).all() and grad.abs().sum() > 0
        times = scalar_time[:, None].expand(2, 50).clone()
        times[0, :3], times[1, :8] = 0, 0
        embeddings = create_sinusoidal_pos_embedding(times, 8, 0.004, 4.0, device=torch.device("cpu"))
        assert embeddings.shape == (2, 50, 8) and torch.isfinite(embeddings).all()
        try:
            VLAFlowMatching.embed_suffix(stub, actions.detach(), times)
        except RuntimeError as error:
            error_text = str(error)
            assert "expand" in error_text
        else:
            raise AssertionError("Per-action support changed; review the probe rather than reuse its conclusion")
    return {
        "scalar_time_shape": [2],
        "scalar_suffix_shape": list(scalar.shape),
        "scalar_gradient_finite_nonzero": True,
        "per_action_time_shape": [2, 50],
        "shared_time_helper_shape": list(embeddings.shape),
        "current_smolvla_per_action_time_supported": False,
        "observed_error": error_text,
        "scope": "Actual suffix method on tiny random CPU projections, not pretrained inference",
    }


def suffix_loss_probe():
    results = []
    batch, horizon, dims = 2, 50, 32
    actions = torch.linspace(-0.5, 0.5, batch * horizon * dims).reshape(batch, horizon, dims)
    noise = actions.flip(1) + 0.125
    times = torch.tensor([0.25, 0.75])[:, None, None]
    for count in (0, 3, 8, 49):
        prefix_mask = torch.arange(horizon)[None, :, None] < count
        mixed = times * noise + (1 - times) * actions
        conditioned = torch.where(prefix_mask, actions, mixed)
        target = noise - actions
        valid = (~prefix_mask).expand(batch, horizon, dims) & (torch.arange(dims)[None, None, :] < 7)
        prediction = torch.zeros_like(actions, requires_grad=True)
        loss = ((prediction - target).square() * valid).sum() / valid.sum()
        grad = torch.autograd.grad(loss, prediction)[0]
        assert torch.equal(conditioned[:, :count], actions[:, :count])
        assert torch.equal(conditioned[:, count:], mixed[:, count:])
        assert (grad[~valid] == 0).all() and grad[valid].abs().sum() > 0
        assert int(valid.sum()) == batch * (horizon - count) * 7
        results.append({"C": count, "valid_suffix_coordinates": int(valid.sum()), "loss": float(loss.detach())})
    return {
        "cases": results,
        "clean_prefix_preserved": True,
        "masked_prefix_and_padding_prediction_gradients_zero": True,
        "suffix_prediction_gradient_nonzero": True,
        "scope": "Synthetic objective arithmetic only; not model-gradient or task-quality validation",
    }


def dataset_inventory():
    cfg_path = REPO.parent / "libero-reference-cache/config/config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    roots = [Path(cfg["datasets"]), REPO.parent / "libero-reference-cache/hub",
             Path.home() / ".cache/huggingface/lerobot", Path.home() / ".cache/huggingface/hub"]
    records = []
    for root in roots:
        row = {"root": str(root), "exists": root.is_dir(), "data_file_count": 0,
               "metadata": [], "examples": [], "visited_dirs": 0, "scan_truncated": False}
        if root.is_dir():
            for folder, dirs, files in os.walk(root, followlinks=False):
                depth = len(Path(folder).relative_to(root).parts)
                eligible = sorted(d for d in dirs if d not in {".git", "blobs", "locks", ".locks", "__pycache__"}
                                  and not d.startswith("models--"))
                if depth >= 8 and eligible:
                    row["scan_truncated"] = True
                dirs[:] = eligible if depth < 8 else []
                row["visited_dirs"] += 1
                if row["visited_dirs"] > 2000:
                    row["scan_truncated"] = True
                    break
                for name in sorted(files):
                    path = Path(folder) / name
                    if name.endswith((".h5", ".hdf5", ".parquet")):
                        row["data_file_count"] += 1
                        if len(row["examples"]) < 12:
                            row["examples"].append(str(path))
                    if name == "info.json":
                        info = json.loads(path.read_text())
                        row["metadata"].append({"path": str(path), "sha256": digest(path),
                                                **{k: info.get(k) for k in ("robot_type", "fps", "total_episodes", "total_frames", "features")}})
        records.append(row)
    return {"config_path": str(cfg_path), "config_sha256": digest(cfg_path),
            "configured_demo_root": cfg["datasets"], "roots": records,
            "scope": "Only configured/shared experiment and standard HF caches; no full-machine or remote scan"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if not output.is_relative_to(REPO / "outputs") or output.exists():
        raise ValueError("Use one new output file under this project's outputs directory")
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    core = [REPO / "src/lerobot/policies/smolvla/modeling_smolvla.py",
            REPO / "src/lerobot/policies/common/vla_utils.py"]
    hashes = {str(p.relative_to(REPO)): digest(p) for p in core}
    before = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    report = {"probe": "F-PTF0", "checked_utc": datetime.now(UTC).isoformat(),
              "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "probe_sha256": digest(Path(__file__)), "core_sha256": hashes,
              "suffix_interface": suffix_interface_probe(), "suffix_objective": suffix_loss_probe(),
              "dataset_inventory": dataset_inventory(),
              "pretrained_model_loads": 0, "policy_forwards": 0, "new_env": 0,
              "optimizer_updates": 0, "network_requests": 0, "scientific_qualification_claimed": False}
    assert hashes == {str(p.relative_to(REPO)): digest(p) for p in core}
    assert before == subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    assert not torch.cuda.is_initialized()
    report.update(core_unchanged=True, cuda_initialized=False, completed=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "interface": report["suffix_interface"], "loss_cases": report["suffix_objective"]["cases"],
                      "data_roots": [{k: v[k] for k in ("root", "exists", "data_file_count", "scan_truncated")}
                                     for v in report["dataset_inventory"]["roots"]],
                      "pretrained_model_loads": 0, "new_env": 0, "optimizer_updates": 0,
                      "cuda_initialized": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
