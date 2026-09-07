"""Full-task completion is independently distinct from one object in the plate."""

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/advanced/predictive_async"))
from audit_so101_noise_reference import native_predicate  # noqa: E402


def test_original_task_requires_three_objects_and_rest_not_only_occupancy():
    witness = {
        "plate_position": [2.0, -0.3, 0.92],
        "oranges": {
            name: {"position": [2.01, -0.32, 0.94]} for name in ("Orange001", "Orange002", "Orange003")
        },
        "joint_positions_radians": [math.radians(v) for v in (0, -90, 80, 50, 0, 0)],
    }
    assert native_predicate(witness)
    displaced = copy.deepcopy(witness)
    displaced["oranges"]["Orange003"]["position"][0] = 2.2
    assert not native_predicate(displaced)
    no_rest = copy.deepcopy(witness)
    no_rest["joint_positions_radians"][1] = 0
    assert not native_predicate(no_rest)
