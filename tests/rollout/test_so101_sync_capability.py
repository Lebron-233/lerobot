"""A task-text diagnostic must reach the synchronous policy and never change async binding."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
import leisaac_so101_matched as matched  # noqa: E402
from eval_leisaac_so101 import MemoryMetrics, load_runtime  # noqa: E402
from leisaac_so101_contract import SCALAR_KEYS  # noqa: E402
from run_so101_sync_capability import PROMPTS, summarize  # noqa: E402


def test_task_override_is_rejected_before_loading_any_async_policy():
    with pytest.raises(ValueError, match="matched synchronous"):
        load_runtime(
            "identity",
            "cpu",
            None,
            1,
            MemoryMetrics(),
            {},
            matched_snapshot=Path("unused"),
            sync_task_text=PROMPTS["model_card_literal"],
        )


def test_exact_recorded_text_reaches_preprocessor_and_frozen_sync_policy(monkeypatch):
    seen = []
    literal = PROMPTS["model_card_literal"]

    class Policy:
        config = SimpleNamespace(use_amp=False)

        def to(self, device):
            return self

        def eval(self):
            return self

        def select_action(self, batch):
            assert batch["task"] == [literal]
            seen.append("policy")
            return torch.zeros(1, 6)

    def pre(batch):
        assert batch["task"] == [literal]
        seen.append("preprocessor")
        return batch

    monkeypatch.setattr(matched, "load_matched_runtime", lambda *a, **k: (Policy(), pre, lambda x: x))
    raw = dict.fromkeys(SCALAR_KEYS, 0.0)
    raw.update({key: np.zeros((480, 640, 3), np.uint8) for key in ("top", "wrist")})
    engine, action = load_runtime(
        "sync",
        "cpu",
        None,
        1,
        MemoryMetrics(),
        raw,
        matched_snapshot=Path("unused"),
        sync_task_text=literal,
    )
    assert engine is None and action(raw) == [0.0] * 6
    assert seen == ["preprocessor", "policy"]


def test_subgoal_success_cannot_qualify_native_task_reference():
    rows = [
        {
            "prompt": prompt,
            "native_success": False,
            "subgoal_success": True,
            "technical_valid": True,
            "native_time_s": 120.0,
        }
        for prompt in PROMPTS
        for _ in range(8)
    ]
    assert all(not r["qualified_development"] for r in summarize(rows).values())
