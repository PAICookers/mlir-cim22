import re
import sys
import unittest
from collections import Counter
from pathlib import Path

import verify_schedule as verify

MATMULINTEGER_FIXTURE = (
    Path(__file__).parents[2] / "Inputs/ONNX/int8_matmul_m1_k128_n320.onnx"
)


SMALL_SCHEDULE = [
    verify.Work(0, 0, 0, 0, 0),
    verify.Work(1, 0, 0, 1, 0),
    verify.Work(2, 0, 1, 0, 1),
    verify.Work(3, 0, 1, 1, 1),
    verify.Work(4, 1, 0, 0, 2),
    verify.Work(5, 1, 0, 1, 2),
    verify.Work(6, 1, 1, 0, 3),
    verify.Work(7, 1, 1, 1, 3),
]


def render_dump(work, m, k, n):
    lines = []
    for item in work:
        attrs = "cim.transaction_idx = 0 : i64, " + ", ".join(
            f"{name} = {value} : i64"
            for name, value in zip(verify.ATTRS, item, strict=True)
        )
        lines.append(
            f"%v{item.work_id} = cim.vmm %input, %weight {{{attrs}}} : "
            "tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>"
        )
    k_tiles = verify.ceil_div(k, 64)
    lines.extend(
        "%e = arith.extsi %v : tensor<16xi21> to tensor<16xi32>"
        for _ in range(len(work) if k_tiles > 1 else 0)
    )
    lines.extend(
        "%a = arith.addi %lhs, %rhs : tensor<16xi32>"
        for _ in range(m * verify.ceil_div(n, 16) * (k_tiles - 1))
    )
    return "\n".join(lines)


def replace_attr(text, line_index, attr, value):
    lines = text.splitlines()
    lines[line_index] = re.sub(
        rf"\b{attr}\s*=\s*-?\d+", f"{attr} = {value}", lines[line_index]
    )
    return "\n".join(lines)


class ScheduleVerificationTest(unittest.TestCase):
    def test_m2_k65_n17_exact_schedule_and_operation_counts(self):
        self.assertEqual(verify.expected_schedule(2, 65, 17), SMALL_SCHEDULE)
        verify.validate_dump(render_dump(SMALL_SCHEDULE, 2, 65, 17), 2, 65, 17)

    def test_matmulinteger_boundaries_dense_ids_and_group_size(self):
        work = verify.expected_schedule(1, 65, 337)
        self.assertEqual(len(work), 44)
        self.assertEqual(work[0], verify.Work(0, 0, 0, 0, 0))
        self.assertEqual(
            work[38:40],
            [
                verify.Work(38, 0, 19, 0, 19),
                verify.Work(39, 0, 19, 1, 19),
            ],
        )
        self.assertEqual(work[40], verify.Work(40, 0, 20, 0, 20))
        self.assertEqual(
            work[-2:],
            [
                verify.Work(42, 0, 21, 0, 21),
                verify.Work(43, 0, 21, 1, 21),
            ],
        )
        self.assertEqual([item.work_id for item in work], list(range(44)))
        self.assertEqual(
            max(Counter(item.group_id for item in work).values()), 2
        )
        verify.validate_dump(render_dump(work, 1, 65, 337), 1, 65, 337)

    def test_fixed_seed_int32_direct_equals_tiled_reconstruction(self):
        shape = (2, 65, 17)
        result_shape, partial_min, partial_max = verify.verify_numeric(
            *shape, seed=2205
        )
        self.assertEqual(result_shape, (shape[0], shape[2]))
        self.assertLessEqual(verify.I21_MIN, partial_min)
        self.assertLessEqual(partial_min, partial_max)
        self.assertLessEqual(partial_max, verify.I21_MAX)

    def test_real_matmulinteger_fixture_int32_direct_equals_tiled_reconstruction(
        self,
    ):
        weight = verify.load_onnx_weight(MATMULINTEGER_FIXTURE, 128, 320)
        self.assertEqual(weight.shape, (128, 320))
        self.assertEqual(weight.dtype.name, "int8")
        result_shape, partial_min, partial_max = verify.verify_numeric(
            1, 128, 320, seed=2205, weight=weight
        )
        self.assertEqual(result_shape, (1, 320))
        self.assertLessEqual(verify.I21_MIN, partial_min)
        self.assertLessEqual(partial_min, partial_max)
        self.assertLessEqual(partial_max, verify.I21_MAX)

    def test_fault_injection_is_rejected(self):
        faults = ("work", "tile", "group", "duplicate", "missing", "trunci")
        for fault in faults:
            with self.subTest(fault=fault):
                dump = render_dump(SMALL_SCHEDULE, 2, 65, 17)
                if fault == "work":
                    dump = replace_attr(dump, 1, "work_id", 0)
                elif fault == "tile":
                    dump = replace_attr(dump, 1, "k_tile", 0)
                elif fault == "group":
                    dump = replace_attr(dump, 1, "group_id", 1)
                elif fault == "duplicate":
                    dump = dump.splitlines()[0] + "\n" + dump
                elif fault == "missing":
                    dump = "\n".join(dump.splitlines()[1:])
                else:
                    dump += (
                        "\n%bad = arith.trunci %value : "
                        "tensor<16xi32> to tensor<16xi21>"
                    )
                expected_error = {
                    "duplicate": "VMM count",
                    "missing": "VMM count",
                    "trunci": "arith.trunci count",
                }.get(fault, "first divergent work 1")
                with self.assertRaisesRegex(
                    verify.VerificationError, expected_error
                ):
                    verify.validate_dump(dump, 2, 65, 17)


if __name__ == "__main__":
    unittest.main(
        testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    )
