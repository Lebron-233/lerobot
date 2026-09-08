"""CPU-only checks for fixed-address refresh and preservation of all ten denoising steps."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/advanced/predictive_async"))
from profile_libero_cuda_graph_compute import check_capture_shapes, compare, copy_inputs  # noqa: E402


class GraphComputeContractTest(unittest.TestCase):
    def test_refresh_keeps_addresses_and_changes_every_input(self):
        destination = [torch.zeros(2), torch.zeros(3, dtype=torch.long), torch.zeros(1, dtype=torch.bool)]
        source = [torch.ones(2), torch.ones(3, dtype=torch.long), torch.ones(1, dtype=torch.bool)]
        addresses = [value.data_ptr() for value in destination]
        copy_inputs(destination, source)
        self.assertEqual(addresses, [value.data_ptr() for value in destination])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(destination, source, strict=True)))

    def test_changed_shape_or_dtype_is_not_silently_replayed(self):
        for value in (torch.zeros(3), torch.zeros(2, dtype=torch.long)):
            with self.assertRaises(ValueError):
                copy_inputs([torch.zeros(2)], [value])
        with self.assertRaises(ValueError):
            copy_inputs([torch.zeros(2)], [])

    def test_capture_requires_ten_full_projections(self):
        check_capture_shapes([[1, 50, 32]] * 10)
        for shapes in ([[1, 50, 32]], [[1, 1, 32]] * 10, []):
            with self.assertRaises(ValueError):
                check_capture_shapes(shapes)

    def test_comparison_detects_change_outside_selected_action(self):
        full = np.zeros((1, 50, 32), dtype=np.float32)
        changed = full.copy()
        changed[0, 49, 31] = 1.0
        action = np.zeros((1, 7), dtype=np.float32)
        result = compare((0.2, full, action, action), (0.1, changed, action, action))
        self.assertFalse(result["full_padded_chunk"]["exact_equal"])
        self.assertTrue(result["selected_normalized_action"]["exact_equal"])


if __name__ == "__main__":
    unittest.main()
