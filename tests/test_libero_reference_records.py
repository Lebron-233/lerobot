"""Regression checks for read-only latency and orientation interpretation."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "diagnose_records",
    Path(__file__).resolve().parents[1]
    / "examples/advanced/predictive_async/diagnose_libero_reference_records.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecordDiagnosticsTest(unittest.TestCase):
    def test_nearest_rank_and_tail_are_retained(self):
        value = MODULE.summary(list(range(1, 101)))
        self.assertEqual(value["p95"], 95)
        self.assertEqual(value["p99"], 99)
        self.assertEqual(value["max"], 100)
        self.assertEqual(value["mean"], 50.5)

    def test_invalid_latencies_are_not_dropped(self):
        for values in ([], [float("nan")], [-1], [float("inf")]):
            with self.assertRaises(ValueError):
                MODULE.summary(values)

    def test_quaternion_sign_is_not_physical_rotation(self):
        result = MODULE.quaternion_distances([[1, 0, 0, 0], [-1, 0, 0, 0]])
        np.testing.assert_allclose(result, [0], atol=1e-12)

    def test_known_quarter_turn(self):
        result = MODULE.quaternion_distances([[0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]])
        np.testing.assert_allclose(result, [np.pi / 2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
