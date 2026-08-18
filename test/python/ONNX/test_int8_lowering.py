"""Unit tests for test-only INT8 lowering helpers."""

import unittest

import int8_lowering as lowering
import numpy as np


class Int8LoweringTest(unittest.TestCase):
    def test_conv_to_matrix_preserves_patch_order(self):
        input_value = np.arange(1 * 1 * 3 * 4, dtype=np.int8).reshape(
            1, 1, 3, 4
        )
        actual = lowering.conv_to_matrix(
            input_value, (2, 2), (1, 2), (1, 0, 0, 0)
        )
        expected = np.asarray(
            [
                [0, 0, 0, 1],
                [0, 0, 2, 3],
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [4, 5, 8, 9],
                [6, 7, 10, 11],
            ],
            dtype=np.int8,
        ).T
        np.testing.assert_array_equal(actual, expected)

    def test_fold_bias_exact_k_inserts_one_lane(self):
        weight = np.arange(2 * 64, dtype=np.int8).reshape(2, 64)
        input_value = np.arange(64 * 3, dtype=np.int8).reshape(64, 3)
        bias = np.asarray([7, -9], dtype=np.int32)
        folded_weight, folded_input = lowering.fold_bias(
            weight, input_value, bias, exact_k=True
        )
        self.assertEqual(folded_weight.shape, (2, 65))
        self.assertEqual(folded_input.shape, (65, 3))
        np.testing.assert_array_equal(
            folded_weight[:, 63], bias.astype(np.int8)
        )
        np.testing.assert_array_equal(folded_input[63], 1)

    def test_wrap_i32(self):
        values = np.asarray([2**31 + 3, -(2**31) - 4], dtype=np.int64)
        np.testing.assert_array_equal(
            lowering.wrap_i32(values), [-(2**31) + 3, 2**31 - 4]
        )


if __name__ == "__main__":
    unittest.main()
