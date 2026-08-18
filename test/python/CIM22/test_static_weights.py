import sys
import unittest

import int8_reference as reference
import numpy as np
import verify_static_weights as verify


class StaticWeightVerificationTest(unittest.TestCase):
    def test_zero_and_all_one(self):
        tiles = np.stack(
            (
                np.zeros((16, 64), dtype=np.uint8),
                np.full((16, 64), 0xFF, dtype=np.uint8),
            )
        )
        words = verify.map_tiles(tiles).view(np.uint32)
        self.assertTrue(np.all(words[0] == 0))
        self.assertTrue(np.all(words[1] == 0xFFFFFFFF))

    def test_one_hot_boundaries(self):
        for lane in (0, 15):
            for k_value in (0, 31, 32, 63):
                for byte_bit in (0, 7):
                    with self.subTest(lane=lane, k=k_value, bit=byte_bit):
                        tile = np.zeros((1, 16, 64), dtype=np.uint8)
                        tile[0, lane, k_value] = 1 << byte_bit
                        words = verify.map_tiles(tile).view(np.uint32)[0]
                        upper = k_value >= 32
                        q_value = (63 if upper else 31) - k_value
                        word_bit = 2 * lane + (0 if upper else 1)
                        expected = np.zeros(256, dtype=np.uint32)
                        expected[8 * q_value + byte_bit] = np.uint32(
                            1 << word_bit
                        )
                        np.testing.assert_array_equal(words, expected)

    def test_asymmetric_round_trip_and_determinism(self):
        lane, k_value = np.indices((16, 64), dtype=np.uint16)
        tile = (0x5A + 67 * lane + 29 * k_value + 7 * (lane ^ k_value)).astype(
            np.uint8
        )
        first = verify.map_tiles(tile[None, ...])
        decoded = reference.decode_int8_weight_words(first[0].view(np.uint32))
        np.testing.assert_array_equal(decoded.view(np.uint8), tile)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout))
