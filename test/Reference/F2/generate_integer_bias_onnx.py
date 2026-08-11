#!/usr/bin/env python3
"""Generate the accepted integer bias epilogue corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def make_linear() -> onnx.ModelProto:
    rng = np.random.default_rng(2501)
    weight = rng.integers(-7, 8, size=(65, 17), dtype=np.int8)
    weight[:, 0] = 1
    bias = (np.arange(17, dtype=np.int64) * 7919 - 50000).astype(np.int32)
    bias[0] = np.iinfo(np.int32).max
    graph = helper.make_graph(
        [
            helper.make_node("MatMulInteger", ["input", "weight"], ["core"]),
            helper.make_node("Add", ["core", "bias"], ["output"]),
        ],
        "matmul_integer_bias",
        [helper.make_tensor_value_info("input", TensorProto.INT8, [2, 65])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [2, 17])],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    return helper.make_model(
        graph,
        producer_name="mlir-cim22-test",
        opset_imports=[helper.make_opsetid("", 10)],
        ir_version=10,
    )


def make_conv() -> onnx.ModelProto:
    rng = np.random.default_rng(2502)
    weight = rng.integers(-7, 8, size=(17, 2, 2, 3), dtype=np.int8)
    bias = (np.arange(17, dtype=np.int64) * 104729 - 700000).astype(np.int32)
    bias[::3] *= -1
    graph = helper.make_graph(
        [
            helper.make_node(
                "ConvInteger",
                ["input", "weight"],
                ["core"],
                pads=[1, 0, 0, 1],
                strides=[2, 2],
                dilations=[1, 1],
                group=1,
                kernel_shape=[2, 3],
            ),
            helper.make_node("Add", ["core", "bias"], ["output"]),
        ],
        "conv_integer_bias",
        [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 2, 5, 6])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 17, 3, 3])],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias.reshape(1, 17, 1, 1), name="bias"),
        ],
    )
    return helper.make_model(
        graph,
        producer_name="mlir-cim22-test",
        opset_imports=[helper.make_opsetid("", 10)],
        ir_version=10,
    )


def write_model(path: Path, model: onnx.ModelProto) -> None:
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    path.write_bytes(model.SerializeToString(deterministic=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.emit_dir.mkdir(parents=True, exist_ok=True)
    write_model(args.emit_dir / "linear-bias.onnx", make_linear())
    write_model(args.emit_dir / "conv-bias.onnx", make_conv())
    print("PASS integer bias ONNX generation: linear and conv")


if __name__ == "__main__":
    main()
