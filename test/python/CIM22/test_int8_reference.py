"""Unit tests for the test-private CIM22 INT8 reference model."""

import sys
import unittest

import int8_reference as reference
import numpy as np


def cache_lines(values: np.ndarray) -> list[tuple[int, str]]:
    return [
        (
            row,
            "".join(
                "00000000" + f"{reference.encode_signed_i8(value):08b}"
                for value in values[row]
            ),
        )
        for row in range(reference.CACHE_ROWS)
    ]


class SignedIntegerTest(unittest.TestCase):
    def test_i8_boundaries(self):
        for value in (-128, -1, 0, 1, 127):
            self.assertEqual(
                reference.decode_signed_i8(reference.encode_signed_i8(value)),
                value,
            )
        for value in (-129, 128):
            with self.assertRaises(reference.ReferenceError):
                reference.encode_signed_i8(value)

    def test_i21_boundaries(self):
        for value in (reference.I21_MIN, -1, 0, 1, reference.I21_MAX):
            self.assertEqual(
                reference.decode_signed_i21(reference.encode_signed_i21(value)),
                value,
            )
        for value in (reference.I21_MIN - 1, reference.I21_MAX + 1):
            with self.assertRaises(reference.ReferenceError):
                reference.encode_signed_i21(value)


class WeightLayoutTest(unittest.TestCase):
    def test_zero_one_hot_and_asymmetric_round_trip(self):
        cases = [
            np.zeros((16, 64), dtype=np.int8),
            np.full((16, 64), -1, dtype=np.int8),
            np.arange(16 * 64, dtype=np.uint16)
            .astype(np.uint8)
            .view(np.int8)
            .reshape(16, 64),
        ]
        one_hot = np.zeros((16, 64), dtype=np.int8)
        one_hot[0, 0] = 1
        one_hot[15, 31] = 2
        one_hot[0, 32] = 4
        one_hot[15, 63] = -128
        cases.append(one_hot)
        for weight in cases:
            words = reference.encode_int8_weight_words(weight)
            np.testing.assert_array_equal(
                reference.decode_int8_weight_words(words), weight
            )

    def test_input_contract_is_strict(self):
        with self.assertRaises(reference.ReferenceError):
            reference.encode_int8_weight_words(
                np.zeros((16, 64), dtype=np.int16)
            )
        with self.assertRaises(reference.ReferenceError):
            reference.decode_int8_weight_words(np.zeros(256, dtype=np.int32))


class CacheLayoutTest(unittest.TestCase):
    def setUp(self):
        self.values = (
            np.arange(8 * 64, dtype=np.uint16)
            .astype(np.uint8)
            .view(np.int8)
            .reshape(8, 64)
        )

    def test_decode_signed_slots(self):
        np.testing.assert_array_equal(
            reference.decode_int8_cache_lines(cache_lines(self.values)),
            self.values,
        )

    def test_rejects_bad_addresses_width_and_high_byte(self):
        valid = cache_lines(self.values)
        for broken in (
            valid[:-1],
            [(0, valid[0][1]), *valid[:-1]],
            [(8, valid[0][1]), *valid[1:]],
            [(0, valid[0][1][:-1]), *valid[1:]],
            [(0, "1" + valid[0][1][1:]), *valid[1:]],
        ):
            with (
                self.subTest(broken=broken[0][0] if broken else None),
                self.assertRaises(reference.ReferenceError),
            ):
                reference.decode_int8_cache_lines(broken)


class TileSimulationTest(unittest.TestCase):
    def test_zero_and_signed_lane_markers(self):
        activation = np.arange(64, dtype=np.int8) - np.int8(32)
        weight = np.zeros((16, 64), dtype=np.int8)
        for lane in range(16):
            weight[lane, lane] = -1 if lane % 2 else 1
        expected = np.asarray(
            [
                int(activation[lane]) * (-1 if lane % 2 else 1)
                for lane in range(16)
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(
            reference.simulate_int8_tile(activation, weight), expected
        )
        np.testing.assert_array_equal(
            reference.simulate_int8_tile(np.zeros(64, dtype=np.int8), weight),
            np.zeros(16, dtype=np.int64),
        )

    def test_rejects_i21_overflow_and_dynamic_shape(self):
        with self.assertRaises(reference.ReferenceError):
            reference.simulate_int8_tile(
                np.full(64, -128, dtype=np.int8),
                np.full((16, 64), -128, dtype=np.int8),
            )
        with self.assertRaises(reference.ReferenceError):
            reference.simulate_int8_tile(
                np.zeros(63, dtype=np.int8), np.zeros((16, 64), dtype=np.int8)
            )


class OutputResponseTest(unittest.TestCase):
    def test_lane_order_flit_order_and_round_trip(self):
        result = np.asarray(
            [
                reference.I21_MIN,
                -1000,
                -2,
                -1,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                1000,
                reference.I21_MAX,
            ],
            dtype=np.int64,
        )
        flits = reference.encode_int8_output_response(result)
        payload = sum(
            reference.encode_signed_i21(value) << (21 * lane)
            for lane, value in enumerate(result)
        )
        response = payload << 48
        expected = tuple(
            (response >> (64 * (5 - index))) & ((1 << 64) - 1)
            for index in range(6)
        )
        self.assertEqual(flits, expected)
        np.testing.assert_array_equal(
            reference.decode_int8_output_response(flits), result
        )

    def test_rejects_padding_lane_overflow_and_bad_flits(self):
        flits = list(reference.encode_int8_output_response([0] * 16))
        flits[-1] |= 1
        with self.assertRaises(reference.ReferenceError):
            reference.decode_int8_output_response(flits)
        with self.assertRaises(reference.ReferenceError):
            reference.encode_int8_output_response(
                [reference.I21_MAX + 1] + [0] * 15
            )
        with self.assertRaises(reference.ReferenceError):
            reference.decode_int8_output_response([0] * 5)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout))
