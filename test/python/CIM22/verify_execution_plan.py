"""Verify the canonical M2.5 execution-plan operation family."""

import argparse
import re
from pathlib import Path
from typing import NamedTuple

PROFILE_ID = "cim22-4x5-v1"
PROFILE_VERSION = 1
SCHEMA_VERSION = 1
PLACEMENT_POLICY = "core-major-dual-macro-v1"
ROUTE_POLICY = "lower-left-maximal-xy-v1"
OPS = (
    "static_weight",
    "configure_input",
    "configure_weight",
    "dispatch",
    "once",
    "readback",
    "group_barrier",
)
COMMON = (
    "work_id",
    "group_id",
    "m_tile",
    "n_tile",
    "k_tile",
    "core_idx",
    "macro_idx",
)


class VerificationError(ValueError):
    pass


class Op(NamedTuple):
    kind: str
    body: str
    line: str
    line_number: int
    transaction_idx: int = -1


class Work(NamedTuple):
    work_id: int
    group_id: int
    m_tile: int
    n_tile: int
    k_tile: int
    core_idx: int
    macro_idx: int
    mapping: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int, int, int, int, int],
    ]


def _balanced_body(line: str, start: int, line_number: int) -> str:
    opening = line.find("{", start)
    if opening < 0:
        raise VerificationError(
            f"line {line_number}: {line[start:].strip()} missing attributes"
        )
    depth = 0
    for index in range(opening, len(line)):
        if line[index] == "{":
            depth += 1
        elif line[index] == "}":
            depth -= 1
            if depth == 0:
                return line[opening + 1 : index]
    raise VerificationError(
        f"line {line_number}: unterminated attribute dictionary"
    )


def parse_ops(text: str) -> tuple[Op, ...]:
    pattern = re.compile(
        r'(?<![A-Za-z0-9_.])"?cim\.(' + "|".join(OPS) + r')"?(?![A-Za-z0-9_.])'
    )
    operations = []
    transaction_idx = -1
    transaction_start = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        if re.search(r'"cim\.transaction"\s*\([^)]*\)\s*\(\{', line):
            transaction_start = len(operations)
        match = pattern.search(line)
        if match:
            body = (
                line[match.end() :]
                if match.group(1) == "static_weight"
                and "{" not in line[match.end() :]
                else _balanced_body(line, match.start(), line_number)
            )
            operations.append(
                Op(match.group(1), body, line, line_number, transaction_idx)
            )
        if re.match(r"\s*\}\)\s*\{cim\.transaction_idx", line):
            closing_transaction = re.search(
                r"cim\.transaction_idx\s*=\s*(-?\d+)", line
            )
            if closing_transaction and transaction_start is not None:
                transaction_idx = int(closing_transaction.group(1))
                for index in range(transaction_start, len(operations)):
                    operations[index] = operations[index]._replace(
                        transaction_idx=transaction_idx
                    )
            transaction_idx = -1
            transaction_start = None
    return tuple(operations)


def _function_attrs(text: str) -> str:
    start = text.find("func.func")
    if start < 0:
        raise VerificationError("missing func.func execution plan")
    attrs = text.find("attributes", start)
    if attrs < 0:
        raise VerificationError("func.func missing execution-plan attributes")
    return _balanced_body(text, attrs, text[:attrs].count("\n") + 1)


def _integer(body: str, name: str, line_number: int) -> int:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)\s*:\s*i64\b", body)
    if not match:
        raise VerificationError(f"line {line_number}: missing {name}")
    return int(match.group(1))


def _integer_optional(body: str, name: str) -> int | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)\s*:\s*i64\b", body)
    return int(match.group(1)) if match else None


def _string(body: str, name: str, line_number: int) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\"([^\"]+)\"", body)
    if not match:
        raise VerificationError(f"line {line_number}: missing {name}")
    return match.group(1)


def _array(
    body: str, name: str, size: int, line_number: int
) -> tuple[int, ...]:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*array<i64:\s*([^>]*)>", body)
    if not match:
        raise VerificationError(
            f"line {line_number}: missing mapping field {name}"
        )
    try:
        values = tuple(int(item.strip()) for item in match.group(1).split(","))
    except ValueError as error:
        raise VerificationError(
            f"line {line_number}: non-integer mapping field {name}"
        ) from error
    if len(values) != size:
        raise VerificationError(
            f"line {line_number}: {name} expects {size} values"
        )
    return values


def _mapping(body: str, line_number: int):
    match = re.search(r"\bcim\.mapping\s*=\s*\{([^{}]*)\}", body)
    if not match:
        raise VerificationError(f"line {line_number}: missing cim.mapping")
    return (
        _array(match.group(1), "core_coord", 2, line_number),
        _array(match.group(1), "ingress", 2, line_number),
        _array(match.group(1), "source", 2, line_number),
        _array(match.group(1), "destination", 2, line_number),
        _array(match.group(1), "route", 6, line_number),
    )


def _work(op: Op) -> Work:
    values = [_integer(op.body, name, op.line_number) for name in COMMON[:-1]]
    macro = _integer(op.body, "macro_idx", op.line_number)
    if macro not in (0, 1):
        raise VerificationError(
            f"line {op.line_number}: invalid macro_idx {macro}"
        )
    mapping = _mapping(op.body, op.line_number)
    return Work(*values, macro, mapping)


def _static_symbols(ops: tuple[Op, ...]) -> set[str]:
    symbols = set()
    for op in ops:
        if op.kind != "static_weight":
            continue
        match = re.search(r"\bsym_name\s*=\s*\"([^\"]+)\"", op.body)
        if not match:
            match = re.search(r"@([A-Za-z_][A-Za-z0-9_.$-]*)\s*=", op.body)
        if not match:
            raise VerificationError(
                f"line {op.line_number}: static weight missing sym_name"
            )
        symbol = match.group(1)
        if symbol in symbols:
            raise VerificationError(
                f"line {op.line_number}: duplicate static weight {symbol}"
            )
        if not re.search(r"tensor<16x64xi8>", op.body):
            raise VerificationError(
                f"line {op.line_number}: static weight must be tensor<16x64xi8>"
            )
        if "cim.mapping" in op.body or "cim.transaction_idx" in op.body or any(
            _integer_optional(op.body, name) is not None for name in COMMON
        ):
            raise VerificationError(
                f"line {op.line_number}: static weight has work identity"
            )
        symbols.add(symbol)
    if not symbols:
        raise VerificationError("missing cim.static_weight")
    return symbols


def _weight_symbol(op: Op, symbols: set[str]) -> None:
    refs = re.findall(r"@([A-Za-z_][A-Za-z0-9_.$-]*)", op.line)
    if len(refs) != 1 or refs[0] not in symbols:
        raise VerificationError(
            f"line {op.line_number}: configure_weight must reference one static weight"
        )


def _validate_function_attrs(text: str) -> None:
    body = _function_attrs(text)
    expected = (
        ("cim.target_profile", PROFILE_ID),
        ("cim.target_profile_version", PROFILE_VERSION),
        ("cim.execution_plan_schema_version", SCHEMA_VERSION),
        ("cim.placement_policy", PLACEMENT_POLICY),
        ("cim.route_policy", ROUTE_POLICY),
    )
    for name, value in expected:
        actual = (
            _string(body, name, 1)
            if isinstance(value, str)
            else _integer(body, name, 1)
        )
        if actual != value:
            raise VerificationError(f"function attribute {name} mismatch")


def validate_dump(text: str) -> tuple[Op, ...]:
    _validate_function_attrs(text)
    program_text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )
    if re.search(r"\bcim\.vmm\b|\bcimframe\.", program_text):
        raise VerificationError(
            "execution-plan dump contains a pre-execution-plan or frame operation"
        )
    ops = parse_ops(text)
    symbols = _static_symbols(ops)
    effectful = tuple(op for op in ops if op.kind != "static_weight")
    if not effectful:
        raise VerificationError("missing function execution-plan operations")
    groups: dict[tuple[int, int], list[Op]] = {}
    for op in effectful:
        group = _integer(op.body, "group_id", op.line_number)
        transaction = op.transaction_idx
        if transaction < 0:
            transaction = _integer(op.body, "cim.transaction_idx", op.line_number)
        groups.setdefault((transaction, group), []).append(op)
    transactions = sorted({transaction for transaction, _ in groups})
    if transactions != list(range(len(transactions))):
        raise VerificationError(
            f"transaction sequence is not dense: {transactions}"
        )

    for (transaction_idx, group_id), group_ops in groups.items():
        dispatches = [op for op in group_ops if op.kind == "dispatch"]
        reads = [op for op in group_ops if op.kind == "readback"]
        if not dispatches:
            raise VerificationError(f"group {group_id}: missing dispatch")
        works = [_work(op) for op in dispatches]
        work_ids = [work.work_id for work in works]
        if work_ids != sorted(set(work_ids)):
            raise VerificationError(
                f"group {group_id}: dispatch work order is not strict"
            )
        expected_work_ids = list(range(group_id * 2, group_id * 2 + len(works)))
        if len(works) > 2 or work_ids != expected_work_ids:
            raise VerificationError(
                f"group {group_id}: work IDs do not preserve two-wide schedule"
            )
        if any(work.macro_idx != work.work_id % 2 for work in works):
            raise VerificationError(
                f"group {group_id}: Macro selectors do not match scheduled work"
            )
        if [
            _integer(op.body, "work_id", op.line_number) for op in reads
        ] != work_ids:
            raise VerificationError(
                f"group {group_id}: readback work order mismatches dispatch"
            )
        by_work = {work.work_id: work for work in works}
        configs = [
            op
            for op in group_ops
            if op.kind in ("configure_input", "configure_weight")
        ]
        config_pairs: dict[int, list[Op]] = {}
        for op in configs:
            work = _work(op)
            if work.work_id not in by_work or work != by_work[work.work_id]:
                raise VerificationError(
                    f"group {group_id}: configuration identity mismatch"
                )
            if op.kind == "configure_input" and "tensor<64xi8>" not in op.line:
                raise VerificationError(
                    f"line {op.line_number}: configure_input must use tensor<64xi8>"
                )
            if op.kind == "configure_weight":
                _weight_symbol(op, symbols)
            config_pairs.setdefault(work.macro_idx, []).append(op)
        expected_macros = {work.macro_idx for work in works}
        if set(config_pairs) != expected_macros or any(
            sorted(op.kind for op in pair)
            != ["configure_input", "configure_weight"]
            for pair in config_pairs.values()
        ):
            raise VerificationError(
                f"group {group_id}: each Macro needs separate input and weight configuration"
            )
        config_order = [(_work(op).macro_idx, op.kind) for op in configs]
        expected_config_order = [
            (macro, kind)
            for macro in sorted(expected_macros)
            for kind in ("configure_input", "configure_weight")
        ]
        if config_order != expected_config_order:
            raise VerificationError(
                f"group {group_id}: Macro configuration order mismatch"
            )

        once = [op for op in group_ops if op.kind == "once"]
        if len(once) != 1:
            raise VerificationError(
                f"group {group_id}: expected one once operation"
            )
        once_op = once[0]
        once_core = _integer(once_op.body, "core_idx", once_op.line_number)
        if _integer_optional(once_op.body, "macro_idx") is not None:
            raise VerificationError(
                f"line {once_op.line_number}: once must not carry macro_idx"
            )
        once_mapping = _mapping(once_op.body, once_op.line_number)
        if once_core != works[0].core_idx or any(
            work.core_idx != once_core for work in works
        ):
            raise VerificationError(
                f"group {group_id}: once core mismatches work core"
            )
        if once_mapping != works[0].mapping:
            raise VerificationError(
                f"group {group_id}: once mapping mismatches work mapping"
            )

        for op in reads:
            work = _work(op)
            if "tensor<16xi21>" not in op.line:
                raise VerificationError(
                    f"line {op.line_number}: readback must return tensor<16xi21>"
                )
            if work.work_id not in by_work or work != by_work[work.work_id]:
                raise VerificationError(
                    f"line {op.line_number}: readback identity mismatch"
                )

        barrier = [op for op in group_ops if op.kind == "group_barrier"]
        if len(barrier) != 1:
            raise VerificationError(
                f"group {group_id}: expected one group_barrier"
            )
        expected = []
        for _macro in sorted(expected_macros):
            expected.extend(("configure_input", "configure_weight"))
        expected.extend(("dispatch",) * len(works))
        expected.append("once")
        expected.extend(("readback",) * len(works))
        expected.append("group_barrier")
        if [op.kind for op in group_ops] != expected:
            raise VerificationError(
                f"group {group_id}: operation order mismatch"
            )

    return ops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical M2.5 execution-plan order and identity"
    )
    parser.add_argument("mlir", type=Path)
    args = parser.parse_args()
    ops = validate_dump(args.mlir.read_text())
    groups = len(
        {
            (
                op.transaction_idx,
                _integer(op.body, "group_id", op.line_number),
            )
            for op in ops
            if op.kind != "static_weight"
        }
    )
    works = sum(op.kind == "dispatch" for op in ops)
    configs = sum(
        op.kind in ("configure_input", "configure_weight") for op in ops
    )
    print(
        f"PASS software-only profile={PROFILE_ID} schema={SCHEMA_VERSION} "
        f"groups={groups} works={works} configs={configs}"
    )


if __name__ == "__main__":
    main()
