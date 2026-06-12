import ctypes
import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.stream import destroy_sparse_batch, get_sparse_batch_from_fens


class TestSparseBatchSingleBlock(unittest.TestCase):
    def test_single_block_metadata_and_tensor_views(self):
        fens = [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1",
        ]
        scores = [13, -7]
        plies = [1, 2]
        results = [1, -1]

        batch_ptr = get_sparse_batch_from_fens("HalfKAv2_hm", fens, scores, plies, results)
        self.assertIsNotNone(batch_ptr)

        try:
            batch = batch_ptr.contents
            base_address = ctypes.addressof(batch.data.contents)

            self.assertGreater(batch.total_bytes, 0)
            self.assertEqual(base_address + batch.is_white_offset, ctypes.addressof(batch.is_white.contents))
            self.assertEqual(base_address + batch.outcome_offset, ctypes.addressof(batch.outcome.contents))
            self.assertEqual(base_address + batch.score_offset, ctypes.addressof(batch.score.contents))
            self.assertEqual(base_address + batch.white_values_offset, ctypes.addressof(batch.white_values.contents))
            self.assertEqual(base_address + batch.black_values_offset, ctypes.addressof(batch.black_values.contents))
            self.assertEqual(base_address + batch.white_offset, ctypes.addressof(batch.white.contents))
            self.assertEqual(base_address + batch.black_offset, ctypes.addressof(batch.black.contents))
            self.assertEqual(base_address + batch.psqt_indices_offset, ctypes.addressof(batch.psqt_indices.contents))
            self.assertEqual(
                base_address + batch.layer_stack_indices_offset,
                ctypes.addressof(batch.layer_stack_indices.contents),
            )

            expected_total_bytes = batch.layer_stack_indices_offset + batch.size * ctypes.sizeof(ctypes.c_int)
            self.assertEqual(batch.total_bytes, expected_total_bytes)

            with patch(
                "data_loader._native._pin_and_move",
                side_effect=lambda t, device, use_pinned_memory=False, dtype=None: t.clone(),
            ) as mock_move:
                (
                    us,
                    them,
                    white_indices,
                    white_values,
                    black_indices,
                    black_values,
                    outcome,
                    score,
                    psqt_indices,
                    layer_stack_indices,
                ) = batch.get_tensors("cpu")

            mock_move.assert_called_once()

            self.assertTrue(torch.equal(us[:, 0], torch.tensor([1.0, 0.0], dtype=torch.float32)))
            self.assertTrue(torch.equal(them[:, 0], torch.tensor([0.0, 1.0], dtype=torch.float32)))
            self.assertTrue(torch.equal(outcome[:, 0], torch.tensor([1.0, 0.0], dtype=torch.float32)))
            self.assertTrue(torch.equal(score[:, 0], torch.tensor(scores, dtype=torch.float32)))
            self.assertEqual(white_indices.dtype, torch.int32)
            self.assertEqual(black_indices.dtype, torch.int32)
            self.assertEqual(white_values.dtype, torch.float32)
            self.assertEqual(black_values.dtype, torch.float32)
            self.assertEqual(psqt_indices.dtype, torch.int64)
            self.assertEqual(layer_stack_indices.dtype, torch.int64)
            self.assertTrue(torch.equal(psqt_indices, layer_stack_indices))

            self.assertEqual(white_indices.shape, (batch.size, batch.max_active_features))
            self.assertEqual(black_indices.shape, (batch.size, batch.max_active_features))
            self.assertEqual(white_values.shape, (batch.size, batch.max_active_features))
            self.assertEqual(black_values.shape, (batch.size, batch.max_active_features))

            white_padding = white_indices == -1
            black_padding = black_indices == -1
            self.assertTrue(bool(white_padding.any()))
            self.assertTrue(bool(black_padding.any()))
            self.assertTrue(torch.all(white_values[white_padding] == 0.0).item())
            self.assertTrue(torch.all(black_values[black_padding] == 0.0).item())
        finally:
            destroy_sparse_batch(batch_ptr)


if __name__ == "__main__":
    unittest.main()
