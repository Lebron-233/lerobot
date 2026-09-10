"""F-LAT1 alignment and model-boundary tests; CPU only, no Env/model load."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
import libero_future_latent_pilot as pilot  # noqa: E402


def fixture():
    actions = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    posts = actions / 4
    record = {
        "requests": [{"request_id": 3, "kind": "planned", "observation_index": 20}],
        "native_steps": [
            {
                "action_index": 20 + i,
                "segment": "measurement",
                "action": posts[i].tolist(),
                "returned_at": 100 + i,
                "started_at": 99.5 + i,
            }
            for i in range(3)
        ],
    }
    prefix = {
        "next_action_index": 20,
        "takeover_index": 23,
        "planned_delay_steps": 3,
        "committed_policy_actions": actions.clone(),
        "committed_post_policy_actions": posts.clone(),
        "committed_mask": torch.ones(3, dtype=torch.bool),
    }
    arrays = {
        "observations": [{"index": 20, "returned_at": 99}, {"index": 23, "returned_at": 103}],
        "requests": {
            "prefix_3": prefix,
            "request_2": {"policy_chunk": actions, "post_chunk": posts},
            "request_3": {"inputs": ()},
        },
        "control": {
            "dispatches": [
                {
                    "action_index": 20 + i,
                    "source_request_id": 2,
                    "source_row_offset": i,
                    "command": posts[i].clone(),
                }
                for i in range(3)
            ]
        },
    }
    return record, arrays


def test_task_split_is_fixed_and_disjoint():
    assert [pilot.split_for(i) for i in range(10)] == ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
    with pytest.raises(ValueError):
        pilot.split_for(10)


def test_actual_prefix_maps_to_pre_takeover_observation():
    record, arrays = fixture()
    pairs, excluded = pilot.aligned_pairs(record, arrays)
    assert not excluded and len(pairs) == 1
    pair = pairs[0]
    assert pair["current_index"] == 20 and pair["future_index"] == 23
    assert pair["actions"].shape == (1, 8, 7)
    assert pair["mask"].tolist() == [[True, True, True, False, False, False, False, False]]
    assert pair["actions"][0, 3:].count_nonzero() == 0


@pytest.mark.parametrize("difference", ["index", "policy", "post", "mask", "time"])
def test_misalignment_stops_instead_of_silently_selecting_another_sample(difference):
    record, arrays = fixture()
    if difference == "index":
        record["requests"][0]["observation_index"] = 21
    elif difference == "policy":
        arrays["requests"]["prefix_3"]["committed_policy_actions"][0, 0] += 1
    elif difference == "post":
        arrays["control"]["dispatches"][0]["command"][0] += 1
    elif difference == "mask":
        arrays["requests"]["prefix_3"]["committed_mask"][0] = False
    else:
        arrays["observations"][-1]["returned_at"] = 101
    with pytest.raises(ValueError):
        pilot.aligned_pairs(record, arrays)


def test_terminal_truncated_prefix_is_explicitly_excluded():
    record, arrays = fixture()
    arrays["observations"].pop()
    pairs, excluded = pilot.aligned_pairs(record, arrays)
    assert not pairs
    assert excluded == [{"request_id": 3, "reason": "episode_ended_before_complete_target"}]


def test_no_action_arm_only_masks_actions_and_never_uses_supervision():
    seen = []

    def model(tokens, masks, actions, action_mask, state, delay):
        seen.append((actions.clone(), state.clone(), delay.clone()))
        return SimpleNamespace(delta_tokens=tuple(torch.zeros_like(v) for v in tokens))

    batch = {
        "tokens": (torch.ones(1, 2, 3),),
        "masks": (torch.ones(1, 2, dtype=torch.bool),),
        "actions": torch.ones(1, 8, 7),
        "action_mask": torch.ones(1, 8, dtype=torch.bool),
        "state": torch.ones(1, 32),
        "delay": torch.tensor([8]),
        "future": (torch.full((1, 2, 3), 99),),
    }
    original = copy.deepcopy(batch)
    pilot.prediction(model, batch, "conditioned")
    pilot.prediction(model, batch, "no_action")
    assert seen[0][0].count_nonzero() == 56 and seen[1][0].count_nonzero() == 0
    assert torch.equal(seen[0][1], seen[1][1])
    assert torch.equal(batch["actions"], original["actions"])


def test_small_existing_predictor_starts_at_identity_without_risk_calibration():
    model = pilot.LightweightFutureLatentPredictor(pilot.config()).eval()
    assert sum(p.numel() for p in model.parameters()) < 1_000_000
    assert model.risk_head is None
    z = (torch.randn(1, 64, 960), torch.randn(1, 64, 960))
    masks = (torch.ones(1, 64, dtype=torch.bool),) * 2
    prediction = model(
        z, masks, torch.zeros(1, 8, 7), (torch.arange(8) < 3)[None], torch.zeros(1, 32), torch.tensor([3])
    )
    assert all(v.count_nonzero() == 0 for v in prediction.delta_tokens)


def test_error_is_sample_mean_not_a_token_magnitude_proxy():
    p = (torch.tensor([[[1.0, 1.0], [99.0, 99.0]], [[2.0, 2.0], [2.0, 2.0]]]),)
    t = (torch.zeros_like(p[0]),)
    m = (torch.tensor([[True, False], [True, True]]),)
    assert pilot.per_sample_mse(p, t, m).tolist() == [1, 4]
