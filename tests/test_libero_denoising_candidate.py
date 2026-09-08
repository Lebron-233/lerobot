"""A candidate setting must execute its declared number of full-chunk projections."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from profile_libero_denoising_candidate import check_projection_shapes  # noqa: E402


class DenoisingContractTest(unittest.TestCase):
    def test_declared_step_counts(self):
        for count in (1, 10):
            check_projection_shapes([[1, 50, 32]] * count, count)

    def test_unchanged_sampler_is_not_a_one_step_candidate(self):
        with self.assertRaises(ValueError):
            check_projection_shapes([[1, 50, 32]] * 10, 1)

    def test_shorter_chunk_is_not_accepted_as_acceleration(self):
        with self.assertRaises(ValueError):
            check_projection_shapes([[1, 1, 32]], 1)


if __name__ == "__main__":
    unittest.main()
