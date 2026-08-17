"""Generate the accepted integer bias epilogue corpus."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def make_linear(
    *, foldable: bool = False, exact_k: bool = False
) -> onnx.ModelProto:
    reduction = 64 if exact_k else 65
    batch = 1 if exact_k else 2
    rng = np.random.default_rng(2601 if foldable else 2501)
    weight = rng.integers(-7, 8, size=(reduction, 17), dtype=np.int8)
    weight[:, 0] = 1
    if foldable:
        bias = ((np.arange(17, dtype=np.int32) * 29 + 3) % 256 - 128).astype(
            np.int32
        )
        bias[:2] = [-128, 127]
    else:
        bias = (np.arange(17, dtype=np.int64) * 7919 - 50000).astype(np.int32)
        bias[0] = np.iinfo(np.int32).max
    graph = helper.make_graph(
        [
            helper.make_node("MatMulInteger", ["input", "weight"], ["core"]),
            helper.make_node("Add", ["core", "bias"], ["output"]),
        ],
        "matmul_integer_bias",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.INT8, [batch, reduction]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.INT32, [batch, 17]
            )
        ],
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


def make_conv(
    *, foldable: bool = False, exact_k: bool = False
) -> onnx.ModelProto:
    rng = np.random.default_rng(2602 if foldable else 2502)
    if exact_k:
        weight_shape = (17, 1, 8, 8)
        input_shape = [1, 1, 8, 8]
        output_shape = [1, 17, 1, 1]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
    else:
        weight_shape = (17, 2, 2, 3)
        input_shape = [1, 2, 5, 6]
        output_shape = [1, 17, 3, 3]
        pads = [1, 0, 0, 1]
        strides = [2, 2]
    weight = rng.integers(-7, 8, size=weight_shape, dtype=np.int8)
    if foldable:
        bias = ((np.arange(17, dtype=np.int32) * 31 + 5) % 256 - 128).astype(
            np.int32
        )
        bias[:2] = [-128, 127]
    else:
        bias = (np.arange(17, dtype=np.int64) * 104729 - 700000).astype(
            np.int32
        )
        bias[::3] *= -1
    graph = helper.make_graph(
        [
            helper.make_node(
                "ConvInteger",
                ["input", "weight"],
                ["core"],
                pads=pads,
                strides=strides,
                dilations=[1, 1],
                group=1,
                kernel_shape=list(weight_shape[2:]),
            ),
            helper.make_node("Add", ["core", "bias"], ["output"]),
        ],
        "conv_integer_bias",
        [helper.make_tensor_value_info("input", TensorProto.INT8, input_shape)],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.INT32, output_shape
            )
        ],
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
    write_model(
        args.emit_dir / "linear-bias-fold.onnx", make_linear(foldable=True)
    )
    write_model(args.emit_dir / "conv-bias-fold.onnx", make_conv(foldable=True))
    write_model(
        args.emit_dir / "linear-bias-exact.onnx",
        make_linear(foldable=True, exact_k=True),
    )
    write_model(
        args.emit_dir / "conv-bias-exact.onnx",
        make_conv(foldable=True, exact_k=True),
    )
    print("PASS integer bias ONNX generation: linear and conv fold policies")


if __name__ == "__main__":
    main()
