import unittest

import numpy as np

from data_loader.compact import CompactBatch


class TestCompactBatchTargets(unittest.TestCase):
    def test_signed_scores_outcomes_and_flags(self):
        records = np.zeros((4, 32), dtype=np.uint8)
        scores = np.array([-32768, -1, 0, 32767], dtype=np.int16)
        bits = scores.view(np.uint16)
        records[:, 24] = bits & 255
        records[:, 25] = bits >> 8
        records[:, 26] = [0, 1, 2, 0]
        records[:, 30] = [0, 1, 0, 1]
        records[:, 31] = [0, 0, 1, 1]
        is_white, outcome, decoded = CompactBatch(records).training_targets()
        np.testing.assert_array_equal(is_white, [1, 0, 1, 0])
        np.testing.assert_array_equal(outcome, [0, 0.5, 2, 2])
        np.testing.assert_array_equal(decoded, scores)
        self.assertEqual(records.shape[1], 32)


if __name__ == "__main__":
    unittest.main()
