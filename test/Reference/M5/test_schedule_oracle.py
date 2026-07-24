from collections import Counter
from pathlib import Path
import re
import sys
import unittest

import schedule_oracle as oracle


F0_FIXTURE = Path(__file__).parents[2] / "Inputs/ONNX/int8_linear_model.onnx"


SMALL_SCHEDULE = [
    oracle.Work(0, 0, 0, 0, 0),
    oracle.Work(1, 0, 0, 1, 0),
    oracle.Work(2, 0, 1, 0, 1),
    oracle.Work(3, 0, 1, 1, 1),
    oracle.Work(4, 1, 0, 0, 2),
    oracle.Work(5, 1, 0, 1, 2),
    oracle.Work(6, 1, 1, 0, 3),
    oracle.Work(7, 1, 1, 1, 3),
]


def render_dump(work, m, k, n):
    lines = []
    for item in work:
        attrs = ", ".join(f"{name} = {value} : i64" for name, value in zip(oracle.ATTRS, item))
        lines.append(
            f"%v{item.work_id} = cim.vmm %input, %weight {{{attrs}}} : "
            "tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>")
    k_tiles = oracle.ceil_div(k, 64)
    lines.extend("%e = arith.extsi %v : tensor<16xi21> to tensor<16xi32>"
                 for _ in range(len(work) if k_tiles > 1 else 0))
    lines.extend("%a = arith.addi %lhs, %rhs : tensor<16xi32>"
                 for _ in range(m * oracle.ceil_div(n, 16) * (k_tiles - 1)))
    return "\n".join(lines)


def replace_attr(text, line_index, attr, value):
    lines = text.splitlines()
    lines[line_index] = re.sub(rf"\b{attr}\s*=\s*-?\d+", f"{attr} = {value}", lines[line_index])
    return "\n".join(lines)


class ScheduleOracleTest(unittest.TestCase):
    def test_m2_k65_n17_exact_schedule_and_operation_counts(self):
        self.assertEqual(oracle.expected_schedule(2, 65, 17), SMALL_SCHEDULE)
        oracle.validate_dump(render_dump(SMALL_SCHEDULE, 2, 65, 17), 2, 65, 17)

    def test_f0_boundaries_dense_ids_and_group_size(self):
        work = oracle.expected_schedule(32, 512, 1024)
        self.assertEqual(len(work), 16384)
        self.assertEqual(work[0], oracle.Work(0, 0, 0, 0, 0))
        self.assertEqual(work[38:40], [
            oracle.Work(38, 0, 4, 6, 19),
            oracle.Work(39, 0, 4, 7, 19),
        ])
        self.assertEqual(work[40], oracle.Work(40, 0, 5, 0, 20))
        self.assertEqual(work[-2:], [
            oracle.Work(16382, 31, 63, 6, 8191),
            oracle.Work(16383, 31, 63, 7, 8191),
        ])
        self.assertEqual([item.work_id for item in work], list(range(16384)))
        self.assertEqual(max(Counter(item.group_id for item in work).values()), 2)
        oracle.validate_dump(render_dump(work, 32, 512, 1024), 32, 512, 1024)

    def test_fixed_seed_int32_direct_equals_tiled_reconstruction(self):
        for shape in ((2, 65, 17), (32, 512, 1024)):
            with self.subTest(shape=shape):
                result_shape, partial_min, partial_max = oracle.verify_numeric(
                    *shape, seed=2205)
                self.assertEqual(result_shape, (shape[0], shape[2]))
                self.assertLessEqual(oracle.I21_MIN, partial_min)
                self.assertLessEqual(partial_min, partial_max)
                self.assertLessEqual(partial_max, oracle.I21_MAX)

    def test_real_f0_fixture_int32_direct_equals_tiled_reconstruction(self):
        weight = oracle.load_onnx_weight(F0_FIXTURE, 512, 1024)
        self.assertEqual(weight.shape, (512, 1024))
        self.assertEqual(weight.dtype.name, "int8")
        result_shape, partial_min, partial_max = oracle.verify_numeric(
            32, 512, 1024, seed=2205, weight=weight)
        self.assertEqual(result_shape, (32, 1024))
        self.assertLessEqual(oracle.I21_MIN, partial_min)
        self.assertLessEqual(partial_min, partial_max)
        self.assertLessEqual(partial_max, oracle.I21_MAX)

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
                    dump += ("\n%bad = arith.trunci %value : "
                             "tensor<16xi32> to tensor<16xi21>")
                expected_error = {
                    "duplicate": "VMM count",
                    "missing": "VMM count",
                    "trunci": "arith.trunci count",
                }.get(fault, "first divergent work 1")
                with self.assertRaisesRegex(oracle.OracleError, expected_error):
                    oracle.validate_dump(dump, 2, 65, 17)


if __name__ == "__main__":
    unittest.main(
        testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
