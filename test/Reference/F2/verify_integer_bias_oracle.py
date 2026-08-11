#!/usr/bin/env python3
"""Verify integer bias broadcasting, INT32 wrap, and unchanged CIM work."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


def wrap_i32(values: np.ndarray) -> np.ndarray:
    return ((values + 2**31) % 2**32 - 2**31).astype(np.int32)


def verify_linear(model_path: Path) -> int:
    model = onnx.load_model(model_path)
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    weight = initializers["weight"]
    bias = initializers["bias"]
    input_value = np.stack([np.ones(65, dtype=np.int8), -np.ones(65, dtype=np.int8)])
    unwrapped = input_value.astype(np.int64) @ weight.astype(np.int64) + bias
    expected = wrap_i32(unwrapped)
    evaluator = ReferenceEvaluator(model)
    actual = evaluator.run(None, {"input": input_value})[0]
    repeated = evaluator.run(None, {"input": input_value})[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(repeated, expected)
    assert unwrapped[0, 0] == 2**31 + 64
    assert expected[0, 0] == -(2**31) + 64
    return 2 * ceil(17 / 16) * ceil(65 / 64)


def verify_conv(model_path: Path) -> int:
    model = onnx.load_model(model_path)
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    node = model.graph.node[0]
    attributes = {
        value.name: helper.get_attribute_value(value) for value in node.attribute
    }
    weight = initializers["weight"]
    bias = initializers["bias"]
    input_value = (np.arange(60, dtype=np.int32) % 17 - 8).astype(np.int8)
    input_value = input_value.reshape(1, 2, 5, 6)
    pad_top, pad_left, pad_bottom, pad_right = attributes["pads"]
    stride_height, stride_width = attributes["strides"]
    padded = np.pad(
        input_value,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )
    patches = np.empty((12, 9), dtype=np.int64)
    for row in range(3):
        for column in range(3):
            input_row = row * stride_height
            input_column = column * stride_width
            patches[:, row * 3 + column] = padded[
                0, :, input_row : input_row + 2, input_column : input_column + 3
            ].reshape(-1)
    core = (weight.reshape(17, 12).astype(np.int64) @ patches).reshape(1, 17, 3, 3)
    expected = wrap_i32(core + bias.astype(np.int64))
    evaluator = ReferenceEvaluator(model)
    actual = evaluator.run(None, {"input": input_value})[0]
    repeated = evaluator.run(None, {"input": input_value})[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(repeated, expected)
    return 3 * 3 * ceil(17 / 16) * ceil(12 / 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    linear_works = verify_linear(args.model_dir / "linear-bias.onnx")
    conv_works = verify_conv(args.model_dir / "conv-bias.onnx")
    assert linear_works == 8
    assert conv_works == 18
    print("PASS integer bias oracle: linear_vmm=8 conv_vmm=18 wrap=int32")


if __name__ == "__main__":
    main()
