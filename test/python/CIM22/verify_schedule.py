"""Verify canonical M5 schedule MLIR dumps and tiled INT8 execution."""

import argparse
import itertools
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import int8_reference as reference
import numpy as np

ATTRS = ("work_id", "m_tile", "n_tile", "k_tile", "group_id")
VMM_PATTERN = re.compile(r"\bcim\.vmm\b")
ATTR_PATTERN = re.compile(
    r"\b(work_id|m_tile|n_tile|k_tile|group_id)\s*=\s*(-?\d+)\s*:\s*i64\b"
)
RESULT_PATTERN = re.compile(r"->\s*tensor<16xi21>(?![A-Za-z0-9_])")
OP_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(cim\.vmm|arith\.extsi|arith\.addi|arith\.trunci)"
    r"(?![A-Za-z0-9_.])"
)
I21_MIN = -(1 << 20)
I21_MAX = (1 << 20) - 1


class VerificationError(ValueError):
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
        raise VerificationError("M, K, and N must be positive")
    return [
        Work(index, m_tile, n_tile, k_tile, index // 2)
        for index, (m_tile, n_tile, k_tile) in enumerate(
            itertools.product(
                range(m), range(ceil_div(n, 16)), range(ceil_div(k, 64))
            )
        )
    ]


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
        if line.lstrip().startswith("//") or not VMM_PATTERN.search(line):
            continue
        attributes = dict(ATTR_PATTERN.findall(line))
        if set(attributes) != set(ATTRS):
            raise VerificationError(
                f"line {line_number}: missing canonical i64 attribute"
            )
        if not RESULT_PATTERN.search(line):
            raise VerificationError(
                f"line {line_number}: cim.vmm partial is not tensor<16xi21>"
            )
        work.append(Work(*(int(attributes[attr]) for attr in ATTRS)))
    return work


def validate_dump(text: str, m: int, k: int, n: int) -> list[Work]:
    expected = expected_schedule(m, k, n)
    actual = parse_schedule(text)
    if len(actual) != len(expected):
        raise VerificationError(
            f"VMM count: expected {len(expected)}, actual {len(actual)}"
        )

    for index, (expected_item, actual_item) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if actual_item != expected_item:
            raise VerificationError(
                f"first divergent work {index}: expected {expected_item}, actual {actual_item}"
            )

    ids = [item.work_id for item in actual]
    if ids != list(range(len(expected))):
        raise VerificationError(
            f"work_id values are not dense and ordered: {ids[:8]}"
        )
    tiles = [(item.m_tile, item.n_tile, item.k_tile) for item in actual]
    if len(set(tiles)) != len(tiles):
        raise VerificationError("duplicate logical tile identity")

    groups = {}
    for group_id, items in itertools.groupby(
        actual, key=lambda item: item.group_id
    ):
        groups[group_id] = tuple(items)
    if list(groups) != list(range(ceil_div(len(actual), 2))):
        raise VerificationError(
            "group_id values are not dense and strictly ordered"
        )
    if max(map(len, groups.values()), default=0) > 2:
        raise VerificationError(
            "schedule group contains more than two work items"
        )
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
            raise VerificationError(
                f"{operation} count: expected {expected_count}, actual {counts[operation]}"
            )
    return actual


def load_onnx_weight(model_path: Path, k: int, n: int) -> np.ndarray:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    if (
        len(model.graph.node) != 1
        or model.graph.node[0].op_type != "MatMulInteger"
    ):
        raise VerificationError(
            "ONNX fixture must contain one MatMulInteger node"
        )
    node = model.graph.node[0]
    if len(node.input) != 2:
        raise VerificationError(
            "ONNX fixture must omit MatMulInteger zero points"
        )
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    if node.input[1] not in initializers:
        raise VerificationError(
            "ONNX MatMulInteger B must be an embedded initializer"
        )
    initializer = initializers[node.input[1]]
    if (
        initializer.data_location == onnx.TensorProto.EXTERNAL
        or initializer.external_data
    ):
        raise VerificationError(
            "ONNX MatMulInteger B must not use external data"
        )
    weight = numpy_helper.to_array(initializer)
    if weight.dtype != np.int8 or weight.shape != (k, n):
        raise VerificationError(
            f"ONNX weight: expected int8[{k},{n}], actual {weight.dtype}{weight.shape}"
        )
    return weight


def verify_numeric(
    m: int, k: int, n: int, seed: int, weight: np.ndarray | None = None
) -> tuple[tuple[int, int], int, int]:
    rng = np.random.default_rng(seed)
    activation = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    if weight is None:
        weight = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    elif weight.dtype != np.int8 or weight.shape != (k, n):
        raise VerificationError(
            f"weight: expected int8[{k},{n}], actual {weight.dtype}{weight.shape}"
        )
    full_result = activation.astype(np.int32) @ weight.astype(np.int32)
    k_tiles = ceil_div(k, 64)
    n_tiles = ceil_div(n, 16)
    padded_activation = np.pad(
        activation, ((0, 0), (0, k_tiles * 64 - k))
    ).reshape(m, k_tiles, 64)
    padded_weight = (
        np.pad(weight, ((0, k_tiles * 64 - k), (0, n_tiles * 16 - n)))
        .reshape(k_tiles, 64, n_tiles, 16)
        .transpose(0, 2, 3, 1)
    )
    partials = reference.simulate_int8_tiles(
        padded_activation[:, :, None, :], padded_weight[None, :, :, :, :]
    )
    partial_min = int(partials.min())
    partial_max = int(partials.max())
    tiled_result = (
        partials.sum(axis=1).reshape(m, n_tiles * 16)[:, :n].astype(np.int32)
    )
    if not I21_MIN <= partial_min <= partial_max <= I21_MAX:
        raise VerificationError(
            f"random partial outside i21: [{partial_min},{partial_max}]"
        )
    np.testing.assert_array_equal(tiled_result, full_result)
    if tiled_result.dtype != np.int32 or tiled_result.shape != (m, n):
        raise VerificationError(
            f"numeric result has dtype={tiled_result.dtype}, shape={tiled_result.shape}"
        )
    return tiled_result.shape, partial_min, partial_max


def _format(item: Work) -> str:
    return f"{item.work_id}:({item.m_tile},{item.n_tile},{item.k_tile})/g{item.group_id}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a canonical schedule against tiled INT8 execution"
    )
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
        args.m, args.k, args.n, args.seed, weight
    )
    groups = ceil_div(len(work), 2)
    group19 = [item for item in work if item.group_id == 19]
    next_group = next((item for item in work if item.group_id == 20), None)
    last_pair = work[-2:] if len(work) > 1 else work
    print(
        f"PASS software-only M={args.m} K={args.k} N={args.n} "
        f"work={len(work)} groups={groups} dtype=int32 shape={shape} "
        f"seed={args.seed} weight={'onnx' if args.onnx else 'random'} "
        f"partial=[{partial_min},{partial_max}]"
    )
    print(
        f"boundaries first={_format(work[0])} "
        f"group19=[{','.join(map(_format, group19)) or 'NA'}] "
        f"next={_format(next_group) if next_group else 'NA'} "
        f"last=[{','.join(map(_format, last_pair))}]"
    )


if __name__ == "__main__":
    main()
