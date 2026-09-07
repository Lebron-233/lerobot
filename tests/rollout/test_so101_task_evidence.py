import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))

from so101_task_evidence import NativeTaskWitness, PlacementTracker, placement_flags


def witness():
    return {
        "plate_position": [0, 0, 0.9],
        "oranges": {
            name: {"position": [0.05, 0, 0.93], "linear_velocity": [0, 0, 0]}
            for name in ("Orange001", "Orange002", "Orange003")
        },
    }


def test_settled_placement_is_consecutive_not_sticky():
    tracker = PlacementTracker()
    sample = witness()
    for step in range(9):
        assert tracker.update(sample, step) == 0
    assert tracker.update(sample, 9) == 3
    sample["oranges"]["Orange001"]["position"][0] = 0.2
    assert tracker.update(sample, 10) == 2
    assert tracker.max_simultaneous == 3 and tracker.last_simultaneous == 2


def test_placement_excludes_motion_height_and_square_corners():
    sample = witness()
    sample["oranges"]["Orange001"]["linear_velocity"] = [0.1, 0, 0]
    sample["oranges"]["Orange002"]["position"] = [0.09, 0.09, 0.93]
    sample["oranges"]["Orange003"]["position"][2] = 1.1
    assert not any(placement_flags(sample).values())


def test_native_result_unmodified_and_snapshot_precedes_reset():
    returned = torch.tensor([True])
    calls = []

    def native(*args):
        calls.append(args)
        return returned

    def asset():
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2, 3]]),
                root_lin_vel_w=torch.zeros(1, 3),
            )
        )

    env = SimpleNamespace(
        scene={
            "Plate": asset(),
            "Orange001": asset(),
            "robot": SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 6))),
        }
    )
    monitor = NativeTaskWitness(native)
    assert monitor(env, [SimpleNamespace(name="Orange001")], SimpleNamespace(name="Plate")) is returned
    captured = copy.deepcopy(monitor.latest)
    env.scene["Orange001"].data.root_pos_w.zero_()
    assert monitor.latest == captured and len(calls) == 1
    assert monitor.latest["native_success"]
