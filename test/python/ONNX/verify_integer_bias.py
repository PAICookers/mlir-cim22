"""Verify integer bias broadcasting, INT32 wrap, and unchanged CIM work."""

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


def wrap_i32(values: np.ndarray) -> np.ndarray:
    return ((values + 2**31) % 2**32 - 2**31).astype(np.int32)


def extend_bias_operands(
    weight: np.ndarray, input_value: np.ndarray, bias: np.ndarray, exact_k: bool
) -> tuple[np.ndarray, np.ndarray]:
    bias_column = bias.astype(np.int8).reshape(-1, 1)
    one_row = np.ones((1, input_value.shape[1]), dtype=np.int8)
    if exact_k:
        return (
            np.concatenate(
                (weight[:, :-1], bias_column, weight[:, -1:]), axis=1
            ),
            np.concatenate(
                (input_value[:-1], one_row, input_value[-1:]), axis=0
            ),
        )
    return (
        np.concatenate((weight, bias_column), axis=1),
        np.concatenate((input_value, one_row), axis=0),
    )


def verify_linear(
    model_path: Path, *, foldable: bool = False, exact_k: bool = False
) -> tuple[int, int]:
    model = onnx.load_model(model_path)
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    weight = initializers["weight"]
    bias = initializers["bias"]
    batch, reduction = (
        model.graph.input[0].type.tensor_type.shape.dim[index].dim_value
        for index in range(2)
    )
    if foldable:
        input_value = (
            (np.arange(batch * reduction, dtype=np.int32) % 23 - 11)
            .astype(np.int8)
            .reshape(batch, reduction)
        )
    else:
        input_value = np.stack(
            [
                np.ones(reduction, dtype=np.int8),
                -np.ones(reduction, dtype=np.int8),
            ]
        )
    unwrapped_lowered = (
        input_value.astype(np.int64) @ weight.astype(np.int64) + bias
    )
    lowered_result = wrap_i32(unwrapped_lowered)
    source_result = ReferenceEvaluator(model).run(None, {"input": input_value})[
        0
    ]
    np.testing.assert_array_equal(source_result, lowered_result)
    if foldable:
        folded_weight, folded_input = extend_bias_operands(
            weight.T, input_value.T, bias, exact_k
        )
        folded_result = wrap_i32(
            folded_weight.astype(np.int64) @ folded_input.astype(np.int64)
        ).T
        np.testing.assert_array_equal(folded_result, source_result)
    else:
        assert unwrapped_lowered[0, 0] == 2**31 + 64
        assert lowered_result[0, 0] == -(2**31) + 64
    reduction_tiles = ceil((reduction + int(foldable)) / 64)
    return batch * ceil(
        weight.shape[1] / 16
    ) * reduction_tiles, batch * reduction_tiles


def verify_conv(
    model_path: Path, *, foldable: bool = False, exact_k: bool = False
) -> tuple[int, int]:
    model = onnx.load_model(model_path)
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    node = model.graph.node[0]
    attributes = {
        value.name: helper.get_attribute_value(value)
        for value in node.attribute
    }
    weight = initializers["weight"]
    bias = initializers["bias"]
    input_shape = tuple(
        dimension.dim_value
        for dimension in model.graph.input[0].type.tensor_type.shape.dim
    )
    input_value = (
        (np.arange(np.prod(input_shape), dtype=np.int32) % 17 - 8)
        .astype(np.int8)
        .reshape(input_shape)
    )
    pad_top, pad_left, pad_bottom, pad_right = attributes["pads"]
    stride_height, stride_width = attributes["strides"]
    padded = np.pad(
        input_value,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )
    filters, channels, kernel_height, kernel_width = weight.shape
    output_height = (padded.shape[2] - kernel_height) // stride_height + 1
    output_width = (padded.shape[3] - kernel_width) // stride_width + 1
    reduction = channels * kernel_height * kernel_width
    spatial = output_height * output_width
    patches = np.empty((reduction, spatial), dtype=np.int8)
    for row in range(output_height):
        for column in range(output_width):
            input_row = row * stride_height
            input_column = column * stride_width
            patches[:, row * output_width + column] = padded[
                0,
                :,
                input_row : input_row + kernel_height,
                input_column : input_column + kernel_width,
            ].reshape(-1)
    core = (
        weight.reshape(filters, reduction).astype(np.int64)
        @ patches.astype(np.int64)
    ).reshape(1, filters, output_height, output_width)
    lowered_result = wrap_i32(core + bias.astype(np.int64))
    source_result = ReferenceEvaluator(model).run(None, {"input": input_value})[
        0
    ]
    np.testing.assert_array_equal(source_result, lowered_result)
    if foldable:
        folded_weight, folded_input = extend_bias_operands(
            weight.reshape(filters, reduction),
            patches,
            bias.reshape(-1),
            exact_k,
        )
        folded_result = wrap_i32(
            folded_weight.astype(np.int64) @ folded_input.astype(np.int64)
        ).reshape(1, filters, output_height, output_width)
        np.testing.assert_array_equal(folded_result, source_result)
    reduction_tiles = ceil((reduction + int(foldable)) / 64)
    return spatial * ceil(
        filters / 16
    ) * reduction_tiles, spatial * reduction_tiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    assert verify_linear(args.model_dir / "linear-bias.onnx") == (8, 4)
    assert verify_conv(args.model_dir / "conv-bias.onnx") == (18, 9)
    assert verify_linear(
        args.model_dir / "linear-bias-fold.onnx", foldable=True
    ) == (
        8,
        4,
    )
    assert verify_conv(
        args.model_dir / "conv-bias-fold.onnx", foldable=True
    ) == (18, 9)
    assert verify_linear(
        args.model_dir / "linear-bias-exact.onnx", foldable=True, exact_k=True
    ) == (4, 2)
    assert verify_conv(
        args.model_dir / "conv-bias-exact.onnx", foldable=True, exact_k=True
    ) == (4, 2)
    print(
        "PASS integer bias verification: tail=8/4,18/9 exact=4/2,4/2 wrap=int32"
    )


if __name__ == "__main__":
    main()
