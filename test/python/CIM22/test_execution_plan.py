import sys
import unittest

import verify_execution_plan as verify

ZERO_MAPPING = ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0, 0, 0, 0, 0))
NEXT_MAPPING = ((0, 1), (0, 0), (0, 0), (0, 1), (0, 1, 0, 0, 0, 0))
WORK0 = verify.Work(0, 0, 0, 0, 0, 0, 0, ZERO_MAPPING)
WORK1 = verify.Work(1, 0, 0, 0, 1, 0, 1, ZERO_MAPPING)
WORK2 = verify.Work(2, 1, 0, 1, 0, 1, 0, NEXT_MAPPING)


def array(values):
    return "array<i64: " + ", ".join(map(str, values)) + ">"


def mapping_text(mapping):
    core, ingress, source, destination, route = mapping
    return (
        f"{{core_coord = {array(core)}, destination = {array(destination)}, "
        f"ingress = {array(ingress)}, route = {array(route)}, "
        f"source = {array(source)}}}"
    )


def profile_attrs():
    return (
        f'cim.target_profile = "{verify.PROFILE_ID}", '
        f"cim.target_profile_version = {verify.PROFILE_VERSION} : i64"
    )


def work_attrs(work):
    return (
        f"cim.mapping = {mapping_text(work.mapping)}, core_slot = {work.core_slot} : i64, "
        f"group_id = {work.group_id} : i64, k_tile = {work.k_tile} : i64, "
        f"m_tile = {work.m_tile} : i64, macro_slot = {work.macro_slot} : i64, "
        f"n_tile = {work.n_tile} : i64, "
        f"work_id = {work.work_id} : i64"
    )


def render_dump(groups=((WORK0, WORK1),)):
    function_attrs = (
        f"cim.execution_plan_schema_version = {verify.SCHEMA_VERSION} : i64, "
        f'cim.placement_policy = "{verify.PLACEMENT_POLICY}", '
        f'cim.route_policy = "{verify.ROUTE_POLICY}", '
        f"{profile_attrs()}"
    )
    lines = [
        "module {",
        (
            '  "cim.static_weight"() {sym_name = "w0", '
            "value = dense<0> : tensor<16x64xi8>} : () -> ()"
        ),
        (
            f"  func.func @invoke(%input: tensor<64xi8>) -> tensor<16xi32> "
            f"attributes {{{function_attrs}}} {{"
        ),
    ]
    readbacks = []
    for group in groups:
        for work in group:
            attrs = work_attrs(work)
            lines.append(
                f'    "cim.configure_input"(%input) {{{attrs}}} : (tensor<64xi8>) -> ()'
            )
            lines.append(
                f'    "cim.configure_weight"() {{{attrs}, weight = @w0}} : () -> ()'
            )
        lines.extend(
            f'    "cim.dispatch"() {{{work_attrs(work)}}} : () -> ()'
            for work in group
        )
        first = group[0]
        once_attrs = (
            f"cim.mapping = {mapping_text(first.mapping)}, "
            f"core_slot = {first.core_slot} : i64, "
            f"group_id = {first.group_id} : i64"
        )
        lines.append(f'    "cim.once"() {{{once_attrs}}} : () -> ()')
        for work in group:
            result = f"%read{work.work_id}"
            lines.append(
                f'    {result} = "cim.readback"() {{{work_attrs(work)}}} '
                ": () -> tensor<16xi21>"
            )
            readbacks.append(result)
        barrier_attrs = f"group_id = {first.group_id} : i64"
        lines.append(
            f'    "cim.group_barrier"() {{{barrier_attrs}}} : () -> ()'
        )
    lines.append(
        f"    %wide0 = arith.extsi {readbacks[0]} : tensor<16xi21> to tensor<16xi32>"
    )
    if len(readbacks) > 1:
        lines.append(
            f"    %wide1 = arith.extsi {readbacks[1]} : tensor<16xi21> to tensor<16xi32>"
        )
        lines.append("    %sum = arith.addi %wide0, %wide1 : tensor<16xi32>")
        result = "%sum"
    else:
        result = "%wide0"
    lines.extend((f"    return {result} : tensor<16xi32>", "  }", "}"))
    return "\n".join(lines)


def mutate_line(text, token, old, new):
    lines = text.splitlines()
    index = next(index for index, line in enumerate(lines) if token in line)
    lines[index] = lines[index].replace(old, new)
    return "\n".join(lines)


class ExecutionPlanVerificationTest(unittest.TestCase):
    def test_valid_dual_macro_and_single_work_groups(self):
        dual = verify.validate_dump(render_dump())
        single = verify.validate_dump(render_dump(((WORK0,),)))
        self.assertEqual(sum(op.kind == "dispatch" for op in dual), 2)
        self.assertEqual(sum(op.kind == "configure_input" for op in dual), 2)
        self.assertEqual(sum(op.kind == "once" for op in dual), 1)
        self.assertEqual(sum(op.kind == "dispatch" for op in single), 1)

    def test_deterministic_parse(self):
        text = render_dump()
        self.assertEqual(verify.validate_dump(text), verify.validate_dump(text))

    def test_function_and_static_resource_faults(self):
        text = render_dump()
        faults = (
            (
                text.replace(
                    "cim.execution_plan_schema_version = 1 : i64, ", "", 1
                ),
                "missing cim.execution_plan_schema_version",
            ),
            (
                text.replace("tensor<16x64xi8>", "tensor<8x64xi8>", 1),
                "static weight must be tensor<16x64xi8>",
            ),
            (
                text.replace(
                    "  func.func",
                    '  "cim.static_weight"() {sym_name = "w0", value = dense<1> : tensor<16x64xi8>} : () -> ()\n  func.func',
                    1,
                ),
                "duplicate static weight w0",
            ),
            (
                text.replace("weight = @w0", "weight = @missing", 1),
                "configure_weight must reference one static weight",
            ),
        )
        for faulty, error in faults:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(verify.VerificationError, error),
            ):
                verify.validate_dump(faulty)

    def test_configuration_faults(self):
        text = render_dump()
        missing_work = mutate_line(
            text,
            "cim.configure_input",
            "work_id = 0 : i64",
            "missing_work = 0 : i64",
        )
        invalid_macro = mutate_line(
            text,
            "cim.configure_input",
            "macro_slot = 0 : i64",
            "macro_slot = 2 : i64",
        )
        mismatched_mapping = mutate_line(
            text,
            "cim.configure_input",
            "route = array<i64: 0, 0, 0, 0, 0, 0>",
            "route = array<i64: 0, 0, 0, 1, 0, 0>",
        )
        same_macro = text.replace(
            "macro_slot = 1 : i64", "macro_slot = 0 : i64"
        )
        missing_config = "\n".join(
            line
            for line in text.splitlines()
            if not ("cim.configure_weight" in line and "work_id = 1" in line)
        )
        single = render_dump(((WORK0,),))
        wrong_single_macro = single.replace(
            "macro_slot = 0 : i64", "macro_slot = 1 : i64"
        )
        skipped_work = single.replace("work_id = 0 : i64", "work_id = 1 : i64")
        faults = (
            (missing_work, "missing work_id"),
            (invalid_macro, "invalid macro_slot 2"),
            (mismatched_mapping, "configuration provenance mismatch"),
            (same_macro, "Macro selectors do not match scheduled work"),
            (missing_config, "each Macro needs separate"),
            (wrong_single_macro, "Macro selectors do not match scheduled work"),
            (skipped_work, "work IDs do not preserve two-wide schedule"),
        )
        for faulty, error in faults:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(verify.VerificationError, error),
            ):
                verify.validate_dump(faulty)

    def test_once_readback_barrier_and_group_faults(self):
        text = render_dump()
        once_macro = text.replace(
            '"cim.once"() {', '"cim.once"() {macro_slot = 0 : i64, ', 1
        )
        once_line = next(
            line for line in text.splitlines() if "cim.once" in line
        )
        duplicate_once = text.replace(
            once_line, once_line + "\n" + once_line, 1
        )
        wrong_readback = text.replace(
            "-> tensor<16xi21>", "-> tensor<16xi32>", 1
        )
        barrier_lines = text.splitlines()
        barrier_index = next(
            index
            for index, line in enumerate(barrier_lines)
            if "cim.group_barrier" in line
        )
        read_index = next(
            index
            for index, line in enumerate(barrier_lines)
            if "cim.readback" in line
        )
        barrier_lines[barrier_index], barrier_lines[read_index] = (
            barrier_lines[read_index],
            barrier_lines[barrier_index],
        )
        barrier_early = "\n".join(barrier_lines)
        two_groups = render_dump(((WORK0,), (WORK2,)))
        group0_barrier = next(
            line
            for line in two_groups.splitlines()
            if "cim.group_barrier" in line and "group_id = 0" in line
        )
        nonmonotonic = two_groups + "\n" + group0_barrier
        faults = (
            (once_macro, "once must not carry macro_slot"),
            (duplicate_once, "expected one once operation"),
            (wrong_readback, "readback must return tensor<16xi21>"),
            (barrier_early, "readback work order mismatches dispatch"),
            (nonmonotonic, "group sequence is not monotonic"),
        )
        for faulty, error in faults:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(verify.VerificationError, error),
            ):
                verify.validate_dump(faulty)


if __name__ == "__main__":
    unittest.main(
        testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    )
