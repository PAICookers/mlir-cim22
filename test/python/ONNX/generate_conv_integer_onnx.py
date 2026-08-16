#!/usr/bin/env python3
"""Generate the accepted opset-10 ConvInteger corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def make_model(
    channels: int,
    height: int,
    width: int,
    filters: int,
    seed: int,
    *,
    kernel: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] | None = (1, 1, 1, 1),
    include_kernel_shape: bool = True,
    explicit_zero_points: bool,
) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    weight = rng.integers(-7, 8, size=(filters, channels, *kernel), dtype=np.int8)
    initializers = [numpy_helper.from_array(weight, name="weight")]
    node_inputs = ["input", "weight"]
    if explicit_zero_points:
        initializers.extend(
            [
                numpy_helper.from_array(np.array(0, dtype=np.int8), name="x_zp"),
                numpy_helper.from_array(np.zeros(filters, dtype=np.int8), name="w_zp"),
            ]
        )
        node_inputs.extend(["x_zp", "w_zp"])

    attributes: dict[str, object] = {
        "strides": strides,
        "dilations": [1, 1],
        "group": 1,
    }
    if pads is not None:
        attributes["pads"] = pads
    if include_kernel_shape:
        attributes["kernel_shape"] = kernel
    node = helper.make_node("ConvInteger", node_inputs, ["output"], **attributes)
    pad_top, pad_left, pad_bottom, pad_right = pads or (0, 0, 0, 0)
    output_height = (height + pad_top + pad_bottom - kernel[0]) // strides[0] + 1
    output_width = (width + pad_left + pad_right - kernel[1]) // strides[1] + 1
    graph = helper.make_graph(
        [node],
        "conv_integer",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.INT8, [1, channels, height, width]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.INT32,
                [1, filters, output_height, output_width],
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="mlir-cim22-test",
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model.ir_version = 10
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    return model


def write_model(path: Path, model: onnx.ModelProto) -> None:
    path.write_bytes(model.SerializeToString(deterministic=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.emit_dir.mkdir(parents=True, exist_ok=True)

    write_model(
        args.emit_dir / "canonical.onnx",
        make_model(8, 8, 8, 19, 2201, explicit_zero_points=False),
    )
    write_model(
        args.emit_dir / "odd.onnx",
        make_model(1, 3, 3, 1, 2202, explicit_zero_points=True),
    )
    write_model(
        args.emit_dir / "pointwise.onnx",
        make_model(
            3,
            5,
            7,
            5,
            2203,
            kernel=(1, 1),
            strides=(2, 2),
            pads=None,
            include_kernel_shape=False,
            explicit_zero_points=False,
        ),
    )
    write_model(
        args.emit_dir / "rectangular.onnx",
        make_model(
            2,
            5,
            6,
            17,
            2204,
            kernel=(2, 3),
            strides=(2, 2),
            pads=(1, 0, 0, 1),
            explicit_zero_points=False,
        ),
    )
    print(
        "PASS ConvInteger ONNX generation: canonical, odd-work, pointwise, "
        "and rectangular"
    )


if __name__ == "__main__":
    main()
