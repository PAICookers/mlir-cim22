#!/usr/bin/env python3
"""Verify ConvInteger padding, patch order, work count, and INT32 results."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


def patch_matrix(input_value: np.ndarray) -> np.ndarray:
    _, channels, height, width = input_value.shape
    padded = np.pad(input_value, ((0, 0), (0, 0), (1, 1), (1, 1)))
    patches = np.empty((channels * 9, height * width), dtype=np.int32)
    for row in range(height):
        for column in range(width):
            patches[:, row * width + column] = padded[
                0, :, row : row + 3, column : column + 3
            ].reshape(-1)
    return patches


def verify(model_path: Path) -> int:
    model = onnx.load_model(model_path)
    checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    weight = numpy_helper.to_array(model.graph.initializer[0])
    filters, channels, _, _ = weight.shape
    output_shape = [
        dimension.dim_value
        for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ]
    _, _, height, width = output_shape
    input_value = (
        np.arange(channels * height * width, dtype=np.int32) % 17 - 8
    ).astype(np.int8).reshape(1, channels, height, width)

    patches = patch_matrix(input_value)
    expected = (
        weight.reshape(filters, channels * 9).astype(np.int32) @ patches
    ).reshape(1, filters, height, width)
    evaluator = ReferenceEvaluator(model)
    actual = evaluator.run(None, {"input": input_value})[0]
    repeated = evaluator.run(None, {"input": input_value})[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(repeated, expected)
    assert actual.dtype == np.int32
    return height * width * ceil(filters / 16) * ceil((channels * 9) / 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    canonical_works = verify(args.model_dir / "canonical.onnx")
    odd_works = verify(args.model_dir / "odd.onnx")
    assert canonical_works == 256
    assert odd_works == 9
    print("PASS ConvInteger oracle: canonical_vmm=256 odd_vmm=9 dtype=int32")


if __name__ == "__main__":
    main()
