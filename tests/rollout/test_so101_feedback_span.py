"""Use SmolVLA's actual selector to verify consumed span, not a replacement queue."""

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from leisaac_so101_matched import WSAGI_REVISION, candidate_manifest  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


@pytest.mark.parametrize("span,expected_calls", [(10, 6), (50, 2)])
def test_actual_selector_generates_fifty_but_reobserves_at_the_registered_span(span, expected_calls):
    calls = []
    policy = SimpleNamespace(config=SimpleNamespace(n_action_steps=span), _queues={ACTION: deque()})
    policy.eval = lambda: policy
    policy._rtc_enabled = lambda: False
    policy._prepare_batch = lambda batch: batch
    policy._check_get_actions_condition = lambda: not policy._queues[ACTION]

    def generate(batch, noise):
        base = len(calls) * 100
        calls.append(batch["step"])
        return (torch.arange(50) + base).float()[None, :, None].expand(1, 50, 6)

    policy._get_action_chunk = generate
    values = [float(SmolVLAPolicy.select_action(policy, {"step": t})[0, 0]) for t in range(60)]
    assert calls == list(range(0, 60, span)) and len(calls) == expected_calls
    assert values[9] == 9
    assert values[10] == (100 if span == 10 else 10)
    assert candidate_manifest(Path(WSAGI_REVISION), execution_steps=span)["generated_chunk_steps"] == 50
    assert candidate_manifest(Path(WSAGI_REVISION), execution_steps=span)["sync_executed_chunk_steps"] == span
