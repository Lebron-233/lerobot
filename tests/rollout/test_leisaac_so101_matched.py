"""New candidate coordinates and processor boundary, not task outcomes."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from leisaac_so101_contract import action_to_radians
from leisaac_so101_matched import (
    POLICY_REVISION,
    SINGLE_RANK_REVISION,
    WSAGI_REVISION,
    MotorStatePreprocessor,
    PhysicalActionPostprocessor,
    candidate_manifest,
    motor_to_physical,
    physical_to_motor,
)


def test_candidate_variants_have_distinct_exact_manifest_identities():
    main = candidate_manifest(Path(POLICY_REVISION))
    single = candidate_manifest(Path(SINGLE_RANK_REVISION))
    assert main["id"] != single["id"]
    assert main["policy_revision"] == POLICY_REVISION
    assert single["policy_revision"] == SINGLE_RANK_REVISION
    assert main["predictor"] is None and single["predictor"] is None
    wsagi = candidate_manifest(Path(WSAGI_REVISION))
    assert wsagi["policy_repo"] == "wsagi/SmolVLA-PickOrange"
    assert wsagi["sync_autocast"]["enabled"] is False
    assert main["sync_autocast"]["enabled"] is True
    short = candidate_manifest(Path(SINGLE_RANK_REVISION), execution_steps=25)
    assert short["generated_chunk_steps"] == 50 and short["sync_executed_chunk_steps"] == 25
    with pytest.raises(ValueError, match="Unknown"):
        candidate_manifest(Path("not-registered"))


@pytest.mark.parametrize("shape", [(6,), (1, 6), (1, 50, 6)])
def test_motor_units_match_official_affine_formula_without_clipping(shape):
    motor = torch.tensor([90.0, -100.0, 100.0, 0.0, 50.0, 75.0]).expand(shape)
    physical = motor_to_physical(motor)
    assert physical.reshape(-1, 6)[0].tolist() == pytest.approx([99, -100, 90, 0, 80, 75])
    torch.testing.assert_close(physical_to_motor(physical), motor)
    assert motor.reshape(-1, 6)[0, 0] == 90  # no input mutation


def test_candidate_postprocessing_remains_separate_from_policy_space():
    class SavedPost:
        steps = ()

        def __call__(self, x):
            return x * 10 + 20

        def reset(self):
            self.reset_called = True

    normalized = torch.zeros(1, 6)
    saved = SavedPost()
    post = PhysicalActionPostprocessor(saved)
    physical = post(normalized)
    torch.testing.assert_close(physical_to_motor(physical), torch.full_like(normalized, 20))
    torch.testing.assert_close(normalized, torch.zeros_like(normalized))
    post.reset()
    assert saved.reset_called


def test_native_state_coordinates_precede_saved_mean_std_normalizer():
    seen = []

    class SavedPre:
        steps = ()

        def __call__(self, batch):
            seen.append(batch["observation.state"].clone())
            return {**batch, "observation.state": (batch["observation.state"] - 3) / 2}

    pre = MotorStatePreprocessor(SavedPre())
    raw = torch.tensor([[99.0, -100.0, 90.0, 0.0, 80.0, 75.0]])
    result = pre({"observation.state": raw})
    torch.testing.assert_close(seen[0], torch.tensor([[90.0, -100.0, 100.0, 0.0, 50.0, 75.0]]))
    torch.testing.assert_close(result["observation.state"], (seen[0] - 3) / 2)
    assert raw[0, 0] == 99


def test_infeasible_motor_target_is_not_hidden_by_new_mapping():
    physical = motor_to_physical(torch.tensor([0.0, 105, 0, 0, 0, 50]))
    with pytest.raises(ValueError, match="Out-of-range shoulder_lift"):
        action_to_radians(physical.tolist())
    assert physical_to_motor(physical)[1] > 100


def test_matched_sync_honors_checkpoint_autocast(monkeypatch):
    import leisaac_so101_matched as matched
    import numpy as np
    from eval_leisaac_so101 import MemoryMetrics, load_runtime
    from leisaac_so101_contract import SCALAR_KEYS

    class Policy:
        config = SimpleNamespace(use_amp=True)

        def to(self, device):
            return self

        def eval(self):
            return self

        def select_action(self, batch):
            assert torch.is_autocast_enabled("cpu")
            return torch.zeros(1, 6)

    monkeypatch.setattr(matched, "load_matched_runtime", lambda *a, **k: (Policy(), lambda x: x, lambda x: x))
    raw = dict.fromkeys(SCALAR_KEYS, 0.0)
    raw.update({name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("top", "wrist")})
    engine, action = load_runtime(
        "sync", "cpu", None, 1801, MemoryMetrics(), raw, matched_snapshot=Path("fixture")
    )
    assert engine is None and action(raw) == [0.0] * 6


def test_native60_diagnostic_preserves_camera30_without_weakening_transfer30():
    from leisaac_so101_contract import ContractError, validate_step

    def observation(step, frame, fps):
        return {
            "episode_id": 1,
            "step": step,
            "sim_time_s": step / fps,
            "control_fps": fps,
            "camera_frames": {"front": frame, "wrist": frame},
        }

    validate_step(observation(0, 1, 60), observation(1, 1, 60), terminal=False)
    validate_step(observation(1, 1, 60), observation(2, 2, 60), terminal=False)
    with pytest.raises(ContractError, match="Camera did not advance"):
        validate_step(observation(1, 1, 60), observation(2, 1, 60), terminal=False)
    with pytest.raises(ContractError, match="Camera did not advance"):
        validate_step(observation(0, 1, 30), observation(1, 1, 30), terminal=False)
    with pytest.raises(ContractError, match="time base"):
        validate_step(observation(0, 1, 30), observation(1, 1, 60), terminal=False)


def test_official_rest_pose_centers_respect_unchanged_transport_limits():
    import math

    from leisaac_so101_contract import REST_POSE_DEG

    target = list(REST_POSE_DEG[:5]) + [0.0]
    assert action_to_radians(target) == pytest.approx([math.radians(x) for x in REST_POSE_DEG])
