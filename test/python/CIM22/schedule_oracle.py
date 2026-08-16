#!/usr/bin/env python3
"""Independent software-only oracle for canonical M5 schedule MLIR dumps."""

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import NamedTuple

import numpy as np

import int8_reference as reference


ATTRS = (
    "work_id", "m_tile", "n_tile", "k_tile",
    "group_id",
)
OP_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(cim\.vmm|arith\.extsi|arith\.addi|arith\.trunci)"
    r"(?![A-Za-z0-9_.])")
I21_MIN = -(1 << 20)
I21_MAX = (1 << 20) - 1


class OracleError(ValueError):
    pass


class Work(NamedTuple):
    work_id: int
    m_tile: int
    n_tile: int
    k_tile: int
    group_id: int


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def expected_schedule(m: int, k: int, n: int) -> list[Work]:
    if min(m, k, n) <= 0:
        raise OracleError("M, K, and N must be positive")
    work = []
    for m_tile in range(m):
        for n_tile in range(ceil_div(n, 16)):
            for k_tile in range(ceil_div(k, 64)):
                work_id = len(work)
                group_id = work_id // 2
                work.append(Work(work_id, m_tile, n_tile, k_tile, group_id))
    return work


def _operation_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        counts.update(OP_PATTERN.findall(line))
    return counts


def parse_schedule(text: str) -> list[Work]:
    """Parse only one-line canonical cim.vmm operations and their frozen attrs."""
    work = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("//") or not re.search(r"\bcim\.vmm\b", line):
            continue
        values = []
        for attr in ATTRS:
            match = re.search(rf"\b{attr}\s*=\s*(-?\d+)\s*:\s*i64\b", line)
            if not match:
                raise OracleError(f"line {line_number}: missing canonical i64 attribute {attr}")
            values.append(int(match.group(1)))
        if not re.search(r"->\s*tensor<16xi21>(?![A-Za-z0-9_])", line):
            raise OracleError(f"line {line_number}: cim.vmm partial is not tensor<16xi21>")
        work.append(Work(*values))
    return work


def validate_dump(text: str, m: int, k: int, n: int) -> list[Work]:
    expected = expected_schedule(m, k, n)
    actual = parse_schedule(text)
    if len(actual) != len(expected):
        raise OracleError(f"VMM count: expected {len(expected)}, actual {len(actual)}")

    for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if actual_item != expected_item:
            raise OracleError(
                f"first divergent work {index}: expected {expected_item}, actual {actual_item}")

    ids = [item.work_id for item in actual]
    if ids != list(range(len(expected))):
        raise OracleError(f"work_id values are not dense and ordered: {ids[:8]}")
    tiles = [(item.m_tile, item.n_tile, item.k_tile) for item in actual]
    if len(set(tiles)) != len(tiles):
        raise OracleError("duplicate logical tile identity")

    groups: dict[int, list[Work]] = {}
    for item in actual:
        groups.setdefault(item.group_id, []).append(item)
    if list(groups) != list(range(ceil_div(len(actual), 2))):
        raise OracleError("group_id values are not dense and strictly ordered")
    if max(map(len, groups.values()), default=0) > 2:
        raise OracleError("schedule group contains more than two work items")
    counts = _operation_counts(text)
    k_tiles = ceil_div(k, 64)
    expected_counts = {
        "cim.vmm": len(expected),
        "arith.extsi": len(expected) if k_tiles > 1 else 0,
        "arith.addi": m * ceil_div(n, 16) * (k_tiles - 1),
        "arith.trunci": 0,
    }
    for operation, expected_count in expected_counts.items():
        if counts[operation] != expected_count:
            raise OracleError(
                f"{operation} count: expected {expected_count}, actual {counts[operation]}")
    return actual


def load_onnx_weight(model_path: Path, k: int, n: int) -> np.ndarray:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "MatMulInteger":
        raise OracleError("ONNX fixture must contain one MatMulInteger node")
    node = model.graph.node[0]
    if len(node.input) != 2:
        raise OracleError("ONNX fixture must omit MatMulInteger zero points")
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    if node.input[1] not in initializers:
        raise OracleError("ONNX MatMulInteger B must be an embedded initializer")
    initializer = initializers[node.input[1]]
    if initializer.data_location == onnx.TensorProto.EXTERNAL or initializer.external_data:
        raise OracleError("ONNX MatMulInteger B must not use external data")
    weight = numpy_helper.to_array(initializer)
    if weight.dtype != np.int8 or weight.shape != (k, n):
        raise OracleError(
            f"ONNX weight: expected int8[{k},{n}], actual {weight.dtype}{weight.shape}")
    return weight


def verify_numeric(m: int, k: int, n: int, seed: int,
                   weight: np.ndarray | None = None) -> tuple[tuple[int, int], int, int]:
    rng = np.random.default_rng(seed)
    activation = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    if weight is None:
        weight = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    elif weight.dtype != np.int8 or weight.shape != (k, n):
        raise OracleError(f"weight: expected int8[{k},{n}], actual {weight.dtype}{weight.shape}")
    expected = activation.astype(np.int32) @ weight.astype(np.int32)
    actual = np.zeros((m, n), dtype=np.int32)
    partial_min = I21_MAX
    partial_max = I21_MIN
    for item in expected_schedule(m, k, n):
        k_begin = item.k_tile * 64
        n_begin = item.n_tile * 16
        valid_k = min(64, k - k_begin)
        valid_n = min(16, n - n_begin)
        input_tile = np.zeros(64, dtype=np.int8)
        weight_tile = np.zeros((16, 64), dtype=np.int8)
        input_tile[:valid_k] = activation[
            item.m_tile, k_begin:k_begin + valid_k]
        weight_tile[:valid_n, :valid_k] = weight[
            k_begin:k_begin + valid_k,
            n_begin:n_begin + valid_n].T
        partial = reference.simulate_int8_tile(
            input_tile, weight_tile)[:valid_n]
        partial_min = min(partial_min, int(partial.min()))
        partial_max = max(partial_max, int(partial.max()))
        actual[item.m_tile, n_begin:n_begin + len(partial)] += partial.astype(np.int32)
    if not I21_MIN <= partial_min <= partial_max <= I21_MAX:
        raise OracleError(f"random partial outside i21: [{partial_min},{partial_max}]")
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype != np.int32 or actual.shape != (m, n):
        raise OracleError(f"numeric result has dtype={actual.dtype}, shape={actual.shape}")
    return actual.shape, partial_min, partial_max


def _format(item: Work) -> str:
    return (f"{item.work_id}:({item.m_tile},{item.n_tile},{item.k_tile})/"
            f"g{item.group_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a canonical scheduled MLIR dump and an independent INT32 oracle")
    parser.add_argument("mlir", type=Path)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2205)
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()

    work = validate_dump(args.mlir.read_text(), args.m, args.k, args.n)
    weight = load_onnx_weight(args.onnx, args.k, args.n) if args.onnx else None
    shape, partial_min, partial_max = verify_numeric(
        args.m, args.k, args.n, args.seed, weight)
    groups = ceil_div(len(work), 2)
    group19 = [item for item in work if item.group_id == 19]
    next_group = next((item for item in work if item.group_id == 20), None)
    last_pair = work[-2:] if len(work) > 1 else work
    print(f"PASS software-only M={args.m} K={args.k} N={args.n} "
          f"work={len(work)} groups={groups} dtype=int32 shape={shape} "
          f"seed={args.seed} weight={'onnx' if args.onnx else 'random'} "
          f"partial=[{partial_min},{partial_max}]")
    print(f"boundaries first={_format(work[0])} "
          f"group19=[{','.join(map(_format, group19)) or 'NA'}] "
          f"next={_format(next_group) if next_group else 'NA'} "
          f"last=[{','.join(map(_format, last_pair))}]")


if __name__ == "__main__":
    main()
