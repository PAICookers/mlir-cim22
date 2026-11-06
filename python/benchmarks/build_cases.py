"""Build deterministic CIM simulator fixtures for deployment comparisons."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cim22.compiler import _compile_linalg
from cim22.runtime import MODEL_SUFFIX

type Shape = tuple[int, ...]
type Kind = Literal["conv", "depthwise", "grouped", "matvec", "gemm"]


@dataclass(frozen=True, slots=True)
class Case:
    """Shapes at the model boundary and in the Linalg weight operand.

    Attributes:
        kind: Linalg operation used by the fixture.
        input_shape: Runtime activation shape.
        weight_shape: Shape of the embedded, constant weights.
        output_shape: Runtime result shape.
        int8: Use signed INT8 operands and INT32 output instead of BF16.
        stride: Common spatial stride for a convolution.
        dilation: Common spatial dilation for a convolution.
        padding: Symmetric zero padding on both spatial axes.
    """

    kind: Kind
    input_shape: Shape
    weight_shape: Shape
    output_shape: Shape
    int8: bool = False
    stride: int = 1
    dilation: int = 1
    padding: int = 0


CASES = {
    "depthwise-3x3": Case(
        "depthwise", (1, 4, 8, 8), (4, 3, 3), (1, 4, 8, 8), padding=1
    ),
    "pointwise-small": Case("conv", (1, 2, 8, 8), (16, 2, 1, 1), (1, 16, 8, 8)),
    "pointwise-tail": Case(
        "conv", (1, 32, 3, 11), (16, 32, 1, 1), (1, 16, 3, 11)
    ),
    "grouped-dilated": Case(
        "grouped",
        (2, 2, 4, 7, 9),
        (2, 5, 4, 3, 3),
        (2, 2, 5, 7, 9),
        dilation=2,
        padding=2,
    ),
    "bf16-matvec-aligned": Case("matvec", (256,), (512, 256), (512,)),
    "bf16-matvec-tail": Case("matvec", (257,), (130, 257), (130,)),
}
for mode in ("bf16", "int8"):
    int8 = mode == "int8"
    CASES[f"{mode}-conv-strided"] = Case(
        "conv",
        (1, 8, 11, 13),
        (20, 8, 3, 3),
        (1, 20, 6, 7),
        int8=int8,
        stride=2,
        padding=1,
    )
    for k, n, columns in (
        (64, 16, 1),
        (64, 16, 31),
        (64, 16, 32),
        (65, 17, 33),
        (65, 17, 65),
        (256, 320, 3),
    ):
        CASES[f"{mode}-gemm-{k}x{n}x{columns}"] = Case(
            "gemm", (k, columns), (n, k), (n, columns), int8=int8
        )

# Isolate column blocking first, then weight-aware scheduling at fixed width.
VARIANTS = {
    "column-1": (1, False),
    "column-32": (32, False),
    "column-auto": ("auto", False),
    "resident-order": (32, True),
}


def _tensor(shape: Shape, element: str) -> str:
    return "tensor<" + "x".join((*map(str, shape), element)) + ">"


def _linalg(case: Case, weights: NDArray[np.float32 | np.int8]) -> str:
    operand = "i8" if case.int8 else "f32"
    x = _tensor(case.input_shape, operand)
    w = _tensor(case.weight_shape, operand)
    y = _tensor(case.output_shape, "i32" if case.int8 else "f32")
    zero = "0" if case.int8 else "0.0"
    pad = ""
    activation, activation_type = "%x", x
    if case.padding:
        *leading, height, width = case.input_shape
        amount = case.padding
        padded = _tensor(
            (*leading, height + 2 * amount, width + 2 * amount), operand
        )
        edges = ", ".join(map(str, (0,) * len(leading) + (amount, amount)))
        indices = ", ".join(
            f"%pad{i}: index" for i in range(len(case.input_shape))
        )
        pad = f"""
  %padded = tensor.pad %x low[{edges}] high[{edges}] {{
    ^bb0({indices}):
      %fill = arith.constant {zero} : {operand}
      tensor.yield %fill : {operand}
  }} : {x} to {padded}
"""
        activation, activation_type = "%padded", padded
    if case.kind in ("conv", "depthwise", "grouped"):
        conv = {
            "conv": "conv_2d_nchw_fchw",
            "depthwise": "depthwise_conv_2d_nchw_chw",
            "grouped": "conv_2d_ngchw_gfchw",
        }[case.kind]
        marker = "cim.onnx.conv_integer, " if case.int8 else ""
        attrs = (
            f"{{{marker}strides = dense<{case.stride}> : tensor<2xi64>, "
            f"dilations = dense<{case.dilation}> : tensor<2xi64>}}"
        )
        op = f"linalg.{conv} {attrs} ins({activation}, %w : {activation_type}, {w})"
    else:
        contraction = "matmul" if case.kind == "gemm" else "matvec"
        marker = " {cim.int8}" if case.int8 else ""
        op = f"linalg.{contraction}{marker} ins(%w, %x : {w}, {x})"
    return f"""
func.func @main(%x: {x}) -> {y} {{
  %w = arith.constant dense<{weights.tolist()}> : {w}
  %z = arith.constant dense<{zero}> : {y}
  {pad}
  %y = {op} outs(%z : {y}) -> {y}
  return %y : {y}
}}
"""


def _int8_reference(
    case: Case, inputs: NDArray[np.int8], weights: NDArray[np.int8]
) -> NDArray[np.int32]:
    """Evaluate signed INT8 fixtures with independent INT32 accumulation."""
    data = inputs.astype(np.int32)
    kernel = weights.astype(np.int32)
    if case.kind == "gemm":
        return kernel @ data
    amount = case.padding
    data = np.pad(data, ((0, 0), (0, 0), (amount, amount), (amount, amount)))
    patches = np.lib.stride_tricks.sliding_window_view(
        data, case.weight_shape[-2:], axis=(-2, -1)
    )[:, :, :: case.stride, :: case.stride]
    return np.einsum("bcijhw,fchw->bfij", patches, kernel)


def build_case(name: str, variants: list[str], output_dir: Path) -> None:
    """Generate a common sample and compile its requested simulator variants.

    Args:
        name: Key from CASES.
        variants: Ordered keys from VARIANTS. Artifacts must not already exist.
        output_dir: Root under which the named fixture and artifacts are saved.
    """
    case = CASES[name]
    folder = output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = [folder / f"{variant}{MODEL_SUFFIX}" for variant in variants]
    for path in artifacts:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Benchmark artifact already exists: {path}")

    # Restart the seed for each case so building a subset gives the same bytes.
    rng = np.random.default_rng(20261104)
    weights = rng.integers(-4, 5, case.weight_shape, dtype=np.int8)
    inputs = rng.integers(-8, 9, case.input_shape, dtype=np.int8)
    expected = None
    if case.int8:
        expected = _int8_reference(case, inputs, weights)
    else:
        weights = weights.astype(np.float32) / 8
        inputs = inputs.astype(np.float32) / 8
    source = _linalg(case, weights)
    (folder / "model.linalg.mlir").write_text(source, encoding="utf-8")
    np.savez(folder / "sample.npz", inputs)
    with TemporaryDirectory(dir=folder) as scratch:
        for variant, path in zip(variants, artifacts, strict=True):
            columns, schedule = VARIANTS[variant]
            candidate = Path(scratch) / path.name
            with _compile_linalg(
                source,
                artifact_path=candidate,
                runtime="simulator",
                max_kernel_columns=columns,
                weight_residency_schedule=schedule,
            ) as session:
                result = session(inputs)[0]
                if session.dispatch_count == 0:
                    raise RuntimeError(f"{name}/{variant} did not dispatch CIM")
                if expected is not None:
                    np.testing.assert_array_equal(result, expected)
            # Publish only after execution and never replace another result.
            if os.name == "nt":
                candidate.rename(path)
            else:
                os.link(candidate, path)
            print(path, flush=True)


def main() -> None:
    """Build selected cases into a developer-supplied output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for name in dict.fromkeys(args.case or CASES):
        build_case(
            name, list(dict.fromkeys(args.variant or VARIANTS)), args.output_dir
        )


if __name__ == "__main__":
    main()
