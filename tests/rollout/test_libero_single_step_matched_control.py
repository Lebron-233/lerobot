"""Check pairing, the failed-screen prerequisite and actual initial-observation comparisons."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
import libero_single_step_matched_control as control  # noqa: E402
import libero_single_step_native as single  # noqa: E402


def rows(successes, *, ten_step=False):
    specs = control.registered_control_tuples() if ten_step else single.registered_tuples()
    return [
        {"tuple": spec, "status": "completed", "success": i < successes, "audit_errors": []}
        for i, spec in enumerate(specs)
    ]


def test_complete_control_matches_all_candidate_starts_and_seeds_not_only_failures():
    assert len(control.registered_control_tuples()) == 90
    for left, right in zip(single.registered_tuples(), control.registered_control_tuples(), strict=True):
        assert left["tuple_id"] != right["tuple_id"]
        assert {k: v for k, v in left.items() if k != "tuple_id"} == {
            k: v for k, v in right.items() if k != "tuple_id"
        }


@pytest.mark.parametrize("left,right,expected", [(0, 0, 1), (8, 0, 0.0078125), (7, 1, 0.0703125)])
def test_exact_discordance_probability(left, right, expected):
    assert control.exact_mcnemar(left, right) == expected
    assert control.exact_mcnemar(right, left) == expected


def test_paired_counts_and_effect_direction():
    result = control.paired_statistics(rows(64), rows(80, ten_step=True))
    assert (result["both_success"], result["single_only"], result["ten_only"], result["both_failure"]) == (
        64,
        0,
        16,
        10,
    )
    assert result["single_minus_ten_success_rate"] == -16 / 90
    assert result["exact_two_sided_mcnemar_p"] == 2**-15
    assert sum(t["ten_step_successes"] for t in result["tasks"]) == 80
    assert (
        result["paired_stratified_bootstrap_95"][0] < -16 / 90 < result["paired_stratified_bootstrap_95"][1]
    )
    same = control.paired_statistics(rows(64), rows(64, ten_step=True))
    assert same["paired_stratified_bootstrap_95"] == [0, 0]
    assert same["exact_two_sided_mcnemar_p"] == 1


def test_incomplete_control_has_no_fake_paired_inference():
    records = rows(80, ten_step=True)
    records[-1]["status"] = "not_run"
    assert control.paired_statistics(rows(64), records) is None
    records[-1] = rows(80)[-1]
    with pytest.raises(ValueError, match="Control tuple"):
        control.paired_statistics(rows(64), records)


def test_failed_but_complete_screen_is_eligible_for_diagnosis(tmp_path):
    summary = {
        "source_head": control.CANDIDATE_HEAD,
        "completed": 90,
        "technical_failure": 0,
        "worker_exit_code": 0,
        "records_verified": True,
        "native_screen_passed": False,
        "successes": 64,
    }
    for name, value in (
        ("summary.json", summary),
        ("registration.json", {"control": {"denoising": 1}, "tuples": single.registered_tuples()}),
        ("tuple_accounting.json", rows(64)),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    returned, _, records = control.require_closed_candidate(tmp_path)
    assert returned["successes"] == 64 and not returned["native_screen_passed"]
    assert len(records) == 90


def test_initial_camera_mismatch_is_retained(tmp_path):
    paths = [tmp_path / "candidate", tmp_path / "control"]
    for path in paths:
        path.mkdir()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        np.savez_compressed(path / "000.npz", image=image, image2=image)
        event = {
            "event": "observation",
            "index": 0,
            "cameras": "000.npz",
            "state": {"values": [0.0] * 8},
            "eef_quaternion_xyzw": {"values": [0, 0, 0, 1]},
        }
        (path / "events.jsonl").write_text(json.dumps(event) + "\n")
    assert control.compare_initial_observations(*paths)["all_equal"]
    image[0, 0, 0] = 1
    np.savez_compressed(paths[1] / "000.npz", image=image, image2=np.zeros_like(image))
    comparison = control.compare_initial_observations(*paths)
    assert comparison["state_equal"] and comparison["image2_equal"]
    assert not comparison["all_equal"] and not comparison["image_equal"]
