import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from leisaac_so101_contract import IMAGE_BYTES, JOINT_NAMES, SCALAR_KEYS, observation_packet
from so101_startup_preparation import warm_environment, warm_predictor_once


def test_predictor_warmup_has_no_policy_sampling_or_dispatch():
    calls = []

    class Policy:
        model = SimpleNamespace(
            encode_image_tokens=lambda images, masks: (
                (torch.zeros(1, 64, 960),) * 2,
                (torch.ones(1, 64, dtype=torch.bool),) * 2,
            )
        )

        def prepare_images(self, batch):
            return None, None

        def prepare_state(self, batch):
            return torch.zeros(1, 32)

    class Pre:
        def __call__(self, batch):
            return batch

        def reset(self):
            calls.append("reset")

    def predictor(tokens, masks, prefix, valid, state, delay):
        assert prefix.shape == (1, 8, 6) and torch.count_nonzero(prefix) == 0
        assert valid.all() and delay.tolist() == [8] and state.shape == (1, 32)
        calls.append("predictor")

    raw = dict.fromkeys(SCALAR_KEYS, 0.0)
    raw.update({name: np.zeros((480, 640, 3), dtype=np.uint8) for name in ("top", "wrist")})
    result = warm_predictor_once(Policy(), Pre(), predictor, raw, "fixture", "cpu")
    assert calls == ["predictor", "reset"] and result["dispatches"] == 0


def test_environment_preparation_is_exactly_thirty_logged_hold_steps():
    def packet(step):
        return observation_packet(
            measured=[0.0] * 6,
            actual_names=JOINT_NAMES,
            images={name: bytes(IMAGE_BYTES) for name in ("front", "wrist")},
            camera_frames={"front": step + 1, "wrist": step + 1},
            episode_id=1,
            step=step,
            snapshot_ready_at_s=0.0,
        )

    class Client:
        def step(self, previous, action):
            assert action[:5] == [0.0] * 5
            return {"observation": packet(previous["step"] + 1), "terminated": False, "truncated": False}

    rows = []
    warm_environment(Client(), packet(0), rows)
    assert len(rows) == 30 and all(row["dispatch"] == "completed" for row in rows)
