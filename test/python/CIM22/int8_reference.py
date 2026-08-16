"""Bit-exact, software-only INT8 reference semantics for one CIM22 tile."""

from collections.abc import Sequence

import numpy as np


LANES = 16
TILE_K = 64
WEIGHT_WORDS = 256
CACHE_ROWS = 8
CACHE_LINE_BITS = 1024
OUTPUT_FLITS = 6
I21_MIN = -(1 << 20)
I21_MAX = (1 << 20) - 1


class ReferenceError(ValueError):
    """Raised when data is outside the evidenced CIM22 INT8 contract."""


def _encode_signed(value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ReferenceError(f"expected signed i{bits} integer")
    value = int(value)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not minimum <= value <= maximum:
        raise ReferenceError(f"{value} is outside signed i{bits}")
    return value & ((1 << bits) - 1)


def _decode_signed(value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ReferenceError(f"expected unsigned i{bits} bit pattern")
    value = int(value)
    if not 0 <= value < (1 << bits):
        raise ReferenceError(f"{value} is outside unsigned i{bits}")
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def encode_signed_i8(value: int) -> int:
    return _encode_signed(value, 8)


def decode_signed_i8(value: int) -> int:
    return _decode_signed(value, 8)


def encode_signed_i21(value: int) -> int:
    return _encode_signed(value, 21)


def decode_signed_i21(value: int) -> int:
    return _decode_signed(value, 21)


def _require_i8_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.int8 or value.shape != shape:
        dtype = getattr(value, "dtype", type(value).__name__)
        actual_shape = getattr(value, "shape", None)
        raise ReferenceError(
            f"{name}: expected np.int8{shape}, actual {dtype}{actual_shape}")


def encode_int8_weight_words(weight: np.ndarray) -> np.ndarray:
    """Encode canonical ``weight[lane, k]`` as 256 uint32 hardware words."""
    _require_i8_array(weight, (LANES, TILE_K), "weight")
    words = np.zeros(WEIGHT_WORDS, dtype=np.uint32)
    unsigned = weight.view(np.uint8)
    for lane in range(LANES):
        for k in range(TILE_K):
            upper_half = k >= 32
            q = 63 - k if upper_half else 31 - k
            word_bit = 2 * lane + (0 if upper_half else 1)
            for byte_bit in range(8):
                bit = (int(unsigned[lane, k]) >> byte_bit) & 1
                words[8 * q + byte_bit] |= np.uint32(bit << word_bit)
    return words


def decode_int8_weight_words(words: np.ndarray) -> np.ndarray:
    """Decode 256 uint32 hardware words to canonical ``weight[lane, k]``."""
    if not isinstance(words, np.ndarray) or words.dtype != np.uint32 or words.shape != (WEIGHT_WORDS,):
        dtype = getattr(words, "dtype", type(words).__name__)
        shape = getattr(words, "shape", None)
        raise ReferenceError(
            f"words: expected np.uint32({WEIGHT_WORDS},), actual {dtype}{shape}")
    weight = np.zeros((LANES, TILE_K), dtype=np.uint8)
    for lane in range(LANES):
        for k in range(TILE_K):
            upper_half = k >= 32
            q = 63 - k if upper_half else 31 - k
            word_bit = 2 * lane + (0 if upper_half else 1)
            value = 0
            for byte_bit in range(8):
                value |= ((int(words[8 * q + byte_bit]) >> word_bit) & 1) << byte_bit
            weight[lane, k] = value
    return weight.view(np.int8)


def decode_int8_cache_lines(
        lines: Sequence[tuple[int, str]]) -> np.ndarray:
    """Decode eight address-tagged 1024-bit Cache lines to ``input[8,64]``."""
    if len(lines) != CACHE_ROWS:
        raise ReferenceError(f"cache: expected {CACHE_ROWS} lines, actual {len(lines)}")
    ordered: list[str | None] = [None] * CACHE_ROWS
    for address, bits in lines:
        if isinstance(address, bool) or not isinstance(address, int):
            raise ReferenceError("cache address must be an integer")
        if not 0 <= address < CACHE_ROWS:
            raise ReferenceError(f"cache address {address} is outside 0..7")
        if ordered[address] is not None:
            raise ReferenceError(f"duplicate cache address {address}")
        if len(bits) != CACHE_LINE_BITS or set(bits) - {"0", "1"}:
            raise ReferenceError(
                f"cache address {address}: expected {CACHE_LINE_BITS} binary bits")
        ordered[address] = bits
    if any(bits is None for bits in ordered):
        raise ReferenceError("cache addresses must cover 0..7")

    values = np.zeros((CACHE_ROWS, TILE_K), dtype=np.int8)
    for row, bits in enumerate(ordered):
        assert bits is not None
        for k in range(TILE_K):
            slot = bits[16 * k:16 * (k + 1)]
            if slot[:8] != "00000000":
                raise ReferenceError(f"cache address {row} slot {k}: high byte is nonzero")
            values[row, k] = decode_signed_i8(int(slot[8:], 2))
    return values


def simulate_int8_tile(input_: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Compute one native 64x16 tile and reject results outside signed i21."""
    _require_i8_array(input_, (TILE_K,), "input")
    _require_i8_array(weight, (LANES, TILE_K), "weight")
    result = weight.astype(np.int64) @ input_.astype(np.int64)
    if np.any(result < I21_MIN) or np.any(result > I21_MAX):
        minimum = int(result.min())
        maximum = int(result.max())
        raise ReferenceError(f"tile result outside signed i21: [{minimum},{maximum}]")
    return result


def encode_int8_output_response(result: Sequence[int]) -> tuple[int, ...]:
    """Encode lanes 0..15 into six uint64 flits sent most-significant first."""
    if len(result) != LANES:
        raise ReferenceError(f"output: expected {LANES} lanes, actual {len(result)}")
    payload = 0
    for lane, value in enumerate(result):
        payload |= encode_signed_i21(value) << (21 * lane)
    response = payload << 48
    return tuple(
        (response >> (64 * (OUTPUT_FLITS - 1 - index))) & ((1 << 64) - 1)
        for index in range(OUTPUT_FLITS))


def decode_int8_output_response(flits: Sequence[int]) -> np.ndarray:
    """Decode six most-significant-first uint64 flits to canonical lanes 0..15."""
    if len(flits) != OUTPUT_FLITS:
        raise ReferenceError(f"response: expected {OUTPUT_FLITS} flits, actual {len(flits)}")
    response = 0
    for index, flit in enumerate(flits):
        if isinstance(flit, bool) or not isinstance(flit, (int, np.integer)):
            raise ReferenceError(f"flit {index}: expected uint64 integer")
        flit = int(flit)
        if not 0 <= flit < (1 << 64):
            raise ReferenceError(f"flit {index}: outside uint64")
        response = (response << 64) | flit
    if response & ((1 << 48) - 1):
        raise ReferenceError("response has nonzero low 48-bit padding")
    payload = response >> 48
    return np.asarray([
        decode_signed_i21((payload >> (21 * lane)) & ((1 << 21) - 1))
        for lane in range(LANES)
    ], dtype=np.int64)
