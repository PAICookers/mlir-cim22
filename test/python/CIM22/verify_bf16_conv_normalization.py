"""Execute original and normalized convolution IR against a scalar oracle.

Small integer-valued BF16 operands keep every reference sum exactly
representable. This tests indexing/layout semantics, not CIM hardware accuracy.
"""

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import subprocess

import numpy as np


@dataclass
class Case:
    name: str
    layout: str
    batch: int = 2
    groups: int = 1
    channels: int = 2
    filters: int = 3
    image: tuple = (5, 7)
    kernel: tuple = (2, 3)
    stride: tuple = (1, 1)
    dilation: tuple = (1, 1)
    pads: tuple = (0, 0, 0, 0)
    cast_weight: bool = False


CASES = [
    Case("pointwise", "nchw_fchw", batch=1, image=(3, 4), kernel=(1, 1),
         cast_weight=True),
    Case("pointwise_stride", "nchw_fchw", batch=1, kernel=(1, 1),
         stride=(2, 3)),
    Case("spatial_nchw", "nchw_fchw", stride=(2, 1), dilation=(2, 2)),
    Case("nhwc_hwcf", "nhwc_hwcf"),
    Case("nhwc_fhwc", "nhwc_fhwc", kernel=(3, 2), stride=(1, 2),
         dilation=(1, 2)),
    Case("grouped_fgchw", "ngchw_fgchw", groups=2),
    Case("grouped_gfchw", "ngchw_gfchw", groups=3, filters=2,
         kernel=(3, 1), stride=(2, 1)),
    Case("grouped_nhwgc", "nhwgc_gfhwc", groups=2, channels=3, filters=2,
         kernel=(1, 3), stride=(1, 2), dilation=(1, 2)),
    Case("depthwise_nchw", "depthwise_nchw_chw", groups=3, channels=1,
         filters=1, kernel=(3, 3), stride=(2, 1), pads=(1, 0, 0, 2)),
    Case("depthwise_nhwc", "depthwise_nhwc_hwc", groups=3, channels=1,
         filters=1, image=(6, 6), kernel=(3, 2), stride=(1, 2),
         dilation=(2, 2)),
    Case("depthwise_multiplier", "depthwise_nhwc_hwcm", groups=2,
         channels=1, filters=3, image=(4, 5)),
    Case("cross_k", "nchw_fchw", batch=1, channels=8, filters=2,
         image=(3, 3), kernel=(3, 3)),
]


def data_and_reference(case):
    """Cross-correlate each group directly, without im2col or matrix products."""
    b, g, c, f = case.batch, case.groups, case.channels, case.filters
    h, w = case.image
    kh, kw = case.kernel
    sh, sw = case.stride
    dh, dw = case.dilation
    top, left, bottom, right = case.pads
    x = ((np.arange(b * g * c * h * w) * 7 + 3) % 11 - 5).reshape(b, g, c, h, w)
    weights = ((np.arange(g * f * c * kh * kw) * 3 + 1) % 5 - 2).reshape(g, f, c, kh, kw)
    # Bound even intermediate sums to BF16's exact integer range.
    if c * kh * kw * 10 > 256:
        x %= 2
        weights %= 2
    oh = (h + top + bottom - dh * (kh - 1) - 1) // sh + 1
    ow = (w + left + right - dw * (kw - 1) - 1) // sw + 1
    y = np.zeros((b, g, f, oh, ow), dtype=np.int64)
    for n, group, oc, oy, ox in np.ndindex(y.shape):
        for ic, ky, kx in product(range(c), range(kh), range(kw)):
            iy, ix = oy * sh + ky * dh - top, ox * sw + kx * dw - left
            if 0 <= iy < h and 0 <= ix < w:
                y[n, group, oc, oy, ox] += (
                    x[n, group, ic, iy, ix] * weights[group, oc, ic, ky, kx]
                )
    return x, weights, y


def to_layout(case, x, weight, y):
    layout = case.layout
    if layout == "nchw_fchw":
        return x[:, 0], weight[0], y[:, 0]
    if layout in ("nhwc_hwcf", "nhwc_fhwc"):
        axes = (2, 3, 1, 0) if layout == "nhwc_hwcf" else (0, 2, 3, 1)
        return x[:, 0].transpose(0, 2, 3, 1), weight[0].transpose(axes), y[:, 0].transpose(0, 2, 3, 1)
    if layout == "ngchw_fgchw":
        return x, weight.transpose(1, 0, 2, 3, 4), y
    if layout == "ngchw_gfchw":
        return x, weight, y
    if layout == "nhwgc_gfhwc":
        return x.transpose(0, 3, 4, 1, 2), weight.transpose(0, 1, 3, 4, 2), y.transpose(0, 3, 4, 1, 2)
    if layout == "depthwise_nchw_chw":
        return x[:, :, 0], weight[:, 0, 0], y[:, :, 0]
    if layout == "depthwise_nhwc_hwc":
        return x[:, :, 0].transpose(0, 2, 3, 1), weight[:, 0, 0].transpose(1, 2, 0), y[:, :, 0].transpose(0, 2, 3, 1)
    if layout == "depthwise_nhwc_hwcm":
        return x[:, :, 0].transpose(0, 2, 3, 1), weight[:, :, 0].transpose(2, 3, 0, 1), y.transpose(0, 3, 4, 1, 2)
    raise ValueError(layout)


def tensor_type(shape, dtype="bf16"):
    return "tensor<" + "x".join([*(str(n) for n in shape), dtype]) + ">"


def dense(value):
    if value.ndim == 0:
        return str(float(value))
    return "[" + ", ".join(dense(v) for v in value) + "]"


def make_case(case):
    x, weight, expected = to_layout(case, *data_and_reference(case))
    x_type, w_type, y_type = map(lambda a: tensor_type(a.shape), (x, weight, expected))
    lines = [f"func.func @{case.name}() -> i1 {{",
             f"  %input = arith.constant dense<{dense(x)}> : {x_type}"]
    source = "%input"
    if any(case.pads):
        # This case is NCHW; tensor.pad belongs to the Host boundary.
        low = [0, 0, *case.pads[:2]]
        high = [0, 0, *case.pads[2:]]
        padded_shape = [s + lo + hi for s, lo, hi in zip(x.shape, low, high)]
        padded_type = tensor_type(padded_shape)
        lines += [f"  %padded = tensor.pad %input low{low} high{high} {{",
                  "  ^bb0(%a: index, %b: index, %c: index, %d: index):",
                  "    %pad_zero = arith.constant 0.0 : bf16",
                  "    tensor.yield %pad_zero : bf16",
                  f"  }} : {x_type} to {padded_type}"]
        source, x_type = "%padded", padded_type
    if case.cast_weight:
        # Nonzero weights round back to their integer BF16 values.
        f32_weight = weight + np.sign(weight) / 1024.0
        f32_type = tensor_type(weight.shape, "f32")
        lines += [f"  %w32 = arith.constant dense<{dense(f32_weight)}> : {f32_type}",
                  f"  %weight = arith.truncf %w32 : {f32_type} to {w_type}"]
    else:
        lines += [f"  %weight = arith.constant dense<{dense(weight)}> : {w_type}"]
    op = "linalg.conv_2d_" + case.layout
    if case.layout.startswith("depthwise_"):
        op = "linalg.depthwise_conv_2d_" + case.layout[len("depthwise_"):]
    lines += ["  %z = arith.constant 0.0 : bf16",
              f"  %empty = tensor.empty() : {y_type}",
              f"  %zero = linalg.fill ins(%z : bf16) outs(%empty : {y_type}) -> {y_type}",
              f"  %out = {op} {{strides = dense<{list(case.stride)}> : tensor<2xi64>,",
              f"    dilations = dense<{list(case.dilation)}> : tensor<2xi64>}}",
              f"    ins({source}, %weight : {x_type}, {w_type})",
              f"    outs(%zero : {y_type}) -> {y_type}",
              f"  %expected = arith.constant dense<{dense(expected)}> : {y_type}"]
    dims = ", ".join(f"d{i}" for i in range(expected.ndim))
    identity = f"affine_map<({dims}) -> ({dims})>"
    scalar = f"affine_map<({dims}) -> ()>"
    iterators = ", ".join('"reduction"' for _ in expected.shape)
    lines += ["  %seed = arith.constant dense<true> : tensor<i1>",
              f"  %ok = linalg.generic {{indexing_maps = [{identity}, {identity}, {scalar}],",
              f"    iterator_types = [{iterators}]}}",
              f"    ins(%out, %expected : {y_type}, {y_type}) outs(%seed : tensor<i1>) {{",
              "    ^bb0(%actual: bf16, %reference: bf16, %acc: i1):",
              "      %same = arith.cmpf oeq, %actual, %reference : bf16",
              "      %next = arith.andi %same, %acc : i1",
              "      linalg.yield %next : i1",
              "  } -> tensor<i1>",
              "  %ok_scalar = tensor.extract %ok[] : tensor<i1>",
              "  return %ok_scalar : i1", "}"]
    return "\n".join(lines)


def make_module():
    lines = [make_case(case) for case in CASES]
    lines += ["func.func @main() -> i32 {", "  %ok0 = arith.constant true"]
    for i, case in enumerate(CASES):
        lines += [f"  %case{i} = func.call @{case.name}() : () -> i1",
                  f"  %ok{i + 1} = arith.andi %ok{i}, %case{i} : i1"]
    lines += ["  %zero = arith.constant 0 : i32", "  %one = arith.constant 1 : i32",
              f"  %status = arith.select %ok{len(CASES)}, %zero, %one : i32",
              "  return %status : i32", "}"]
    return "\n".join(lines) + "\n"


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-utils", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.work_dir / "source.mlir"
    source.write_text(make_module(), encoding="utf-8")
    normalized = args.work_dir / "normalized.mlir"
    run([args.opt, str(source), "--normalize-cim-conv", "-o", str(normalized)])
    text = normalized.read_text(encoding="utf-8")
    if "linalg.conv_2d_" in text or "linalg.depthwise_conv_2d_" in text:
        raise AssertionError("a supported convolution was left unnormalized")
    expected_matmuls = sum(case.groups for case in CASES)
    if text.count("linalg.matmul ") != expected_matmuls:
        raise AssertionError("unexpected number of independent group MatMuls")
    formed = args.work_dir / "formed.mlir"
    run([args.opt, str(normalized), "--partition-cim-program", "--form-cim-program",
         "-o", str(formed)])
    text = formed.read_text(encoding="utf-8")
    expected_vmms = 0
    for case in CASES:
        _, _, expected = data_and_reference(case)
        k = case.channels * case.kernel[0] * case.kernel[1]
        expected_vmms += (case.groups * ((case.filters + 15) // 16)
                          * ((k + 63) // 64) * case.batch
                          * expected.shape[-2] * expected.shape[-1])
    if "linalg.matmul " in text or text.count("cim.vmm ") != expected_vmms:
        raise AssertionError("convolution MatMuls did not form the expected VMM tiles")
    pipeline = (
        "builtin.module(convert-elementwise-to-linalg,"
        "one-shot-bufferize{bufferize-function-boundaries},"
        "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
        "convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,"
        "finalize-memref-to-llvm,"
        "convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    for path in (source, normalized):
        lowered = path.with_suffix(".llvm.mlir")
        run([args.opt, str(path), "--pass-pipeline=" + pipeline, "-o", str(lowered)])
        output = run([args.runner, str(lowered), "-e", "main",
                      "-entry-point-result=i32", "-shared-libs=" + args.runner_utils])
        if output.strip() != "0":
            raise AssertionError(f"{path.name} disagrees with the scalar oracle: {output}")
    print(f"PASS {len(CASES)} convolution cases: original and normalized CPU results match")


if __name__ == "__main__":
    main()
