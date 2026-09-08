"""Catch stale inputs and changing shapes before a bounded CUDA-graph experiment."""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "graph_probe",
    Path(__file__).resolve().parents[1]
    / "examples/advanced/predictive_async/profile_libero_graph_compute.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_eight_inputs_are_updated_without_replacing_storage():
    static = tuple(torch.zeros(2) for _ in range(8))
    addresses = [x.data_ptr() for x in static]
    current = tuple(torch.full((2,), float(k + 1)) for k in range(8))
    MODULE.copy_inputs(static, current)
    assert [x.data_ptr() for x in static] == addresses
    assert all(torch.equal(a, b) for a, b in zip(static, current, strict=True))
    MODULE.copy_inputs(static, tuple(-x for x in current))
    assert all(torch.equal(a, -b) for a, b in zip(static, current, strict=True))


def test_shape_changes_do_not_silently_replay_stale_buffers():
    with pytest.raises(ValueError, match="shape"):
        MODULE.copy_inputs((torch.zeros(2),), (torch.zeros(3),))
    with pytest.raises(ValueError, match="count"):
        MODULE.copy_inputs((torch.zeros(2),), ())


def test_explicit_noise_and_both_cameras_are_required():
    with pytest.raises(ValueError, match="explicit noise"):
        MODULE.flat_inputs((1, 2), (3, 4), 5, 6, 7, None)
    assert MODULE.flat_inputs((1, 2), (3, 4), 5, 6, 7, 8) == tuple(range(1, 9))


def test_synthetic_inputs_change_both_cameras_and_state():
    import numpy as np

    a, b = MODULE.synthetic_raw(0), MODULE.synthetic_raw(1)
    assert all(not np.array_equal(a["pixels"][k], b["pixels"][k]) for k in ("image", "image2"))
    assert not np.array_equal(a["robot_state"]["eef"]["pos"], b["robot_state"]["eef"]["pos"])
