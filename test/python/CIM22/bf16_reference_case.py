"""Prepare and check the fixed dual-Macro BF16 reference demo.

Lane-zero expectations were evaluated with ExpPrealign/int21_to_bf16 from
test_data_1_17.zip, independently of the C++ exporter. Other lanes alternate
weight sign and increase the exponent by one; no BF16 arithmetic is reimplemented.
The raw weights are defined by Integration/ExecutionPlan/bf16-runner.mlir.
"""

import sys
from pathlib import Path

import int8_reference as reference
import numpy as np
import supplier_fixture_replay as supplier
from verify_rtl_artifacts import read_source, read_words

INPUT_PAIRS = (
    (0x3F80, 0xBFC0),  # 1, -1.5
    (0x3F00, 0xBF80),  # 0.5, -1
    (0x3F80, 0x3F80),  # Cancellation; supplier zero retains its exponent.
    (0xBF80, 0x3F80),  # -1, 1
    (0x4000, 0xC040),  # 2, -3
    (0x3F80, 0),
    (0, 0),
    (0x3F80, 0x3B80),  # Exponent gap eight drops the second input.
)
LANE_ZERO = (
    (
        (10240, 0x43A0),
        (6144, 0x4340),
        (0, 0x4080),
        (-8192, 0xC380),
        (10240, 0x4420),
        (4096, 0x4300),
        (0, 0x0100),
        (4096, 0x4300),
    ),
    (
        (15360, 0x4BF0),
        (9216, 0x4B90),
        (0, 0x4880),
        (-12288, 0xCBC0),
        (15360, 0x4C70),
        (6144, 0x4B40),
        (0, 0x0900),
        (6144, 0x4B40),
    ),
)


def input_rows(profile="normal"):
    for row, pair in enumerate(INPUT_PAIRS):
        values = list(pair) + [0] * 62
        if profile == "inf" and row == 0:
            values[0] = 0x7F80
        if profile == "overflow":
            values = [0xBFFF] * 64
        yield "".join(f"{value:016b}" for value in values)


def verify(directory):
    for macro, fixed_rows in enumerate(LANE_ZERO):
        suffix = macro + 1
        weights = reference.decode_int8_weight_words(
            supplier._parse_weight(directory / "sources" / f"cim{suffix}_w.txt")
        )
        fixed_weights = np.zeros((16, 64), dtype=np.int8)
        for lane in range(16):
            value = (64 if macro == 0 else 96) * (-1 if lane % 2 else 1)
            fixed_weights[lane, :2] = (value, -value)
        np.testing.assert_array_equal(weights, fixed_weights)
        assert read_words(
            directory / "sources" / f"cim{suffix}_w_exp.txt", 8
        ) == list(range(1 + 16 * macro, 17 + 16 * macro))
        assert read_source(
            directory / "sources" / f"cache{suffix}_in.txt", 1024, 8
        ) == {row: int(bits, 2) for row, bits in enumerate(input_rows())}
        intermediate = supplier._parse_expected(
            directory / "expected" / f"int21_{suffix}.txt"
        )
        output = supplier._parse_expected(
            directory / "expected" / f"output_{suffix}.txt"
        )
        for row, (total, bits) in enumerate(fixed_rows):
            for lane in range(16):
                assert intermediate[row, lane] == total * (
                    -1 if lane % 2 else 1
                )
                sign = (
                    ((bits & 0x8000) ^ (0x8000 if lane % 2 else 0))
                    if total
                    else 0
                )
                expected = sign | ((bits & 0x7FFF) + 128 * lane)
                assert output[row, lane] == expected, (macro, row, lane)
    print(
        "PASS BF16 reference weights, inputs, exponents, 256 INT21 and 256 BF16 outputs"
    )


if __name__ == "__main__":
    if sys.argv[1] == "input":
        for address, bits in enumerate(
            input_rows(sys.argv[2] if len(sys.argv) > 2 else "normal")
        ):
            print(f"{address},{bits}")
    else:
        verify(Path(sys.argv[2]))
