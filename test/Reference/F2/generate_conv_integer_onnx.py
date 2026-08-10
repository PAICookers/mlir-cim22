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
    explicit_zero_points: bool,
) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    weight = rng.integers(
        -7, 8, size=(filters, channels, 3, 3), dtype=np.int8
    )
    initializers = [numpy_helper.from_array(weight, name="weight")]
    node_inputs = ["input", "weight"]
    if explicit_zero_points:
        initializers.extend(
            [
                numpy_helper.from_array(np.array(0, dtype=np.int8), name="x_zp"),
                numpy_helper.from_array(
                    np.zeros(filters, dtype=np.int8), name="w_zp"
                ),
            ]
        )
        node_inputs.extend(["x_zp", "w_zp"])

    node = helper.make_node(
        "ConvInteger",
        node_inputs,
        ["output"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
    )
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
                "output", TensorProto.INT32, [1, filters, height, width]
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
    print("PASS ConvInteger ONNX generation: canonical and odd-work")


if __name__ == "__main__":
    main()
