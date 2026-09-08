"""Small deterministic checks for the bounded compute profile's reported statistics."""

import runpy
import unittest
from pathlib import Path

MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples/advanced/predictive_async/profile_libero_reference_compute.py"
    )
)


class TimingSummaryTest(unittest.TestCase):
    def test_twenty_samples_and_nearest_rank(self):
        result = MODULE["timing_summary"]([x / 1000 for x in range(20, 0, -1)])
        self.assertEqual(result["n"], 20)
        self.assertAlmostEqual(result["mean_seconds"], 0.0105)
        self.assertAlmostEqual(result["median_seconds"], 0.0105)
        self.assertEqual(result["empirical_p95_seconds"], 0.019)
        self.assertEqual(result["min_seconds"], 0.001)
        self.assertEqual(result["max_seconds"], 0.020)

    def test_single_sample(self):
        result = MODULE["timing_summary"]([0.05])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["empirical_p95_seconds"], 0.05)

    def test_invalid_measurements_are_not_silently_dropped(self):
        for samples in ([], [0], [-1], [float("nan")], [float("inf")], [0.1, 0]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                MODULE["timing_summary"](samples)


if __name__ == "__main__":
    unittest.main()
