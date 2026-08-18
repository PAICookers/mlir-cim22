"""Verify ConvInteger padding, patch order, work count, and INT32 results."""

import argparse
from math import ceil
from pathlib import Path

import int8_lowering as lowering
import numpy as np
import onnx
from onnx import checker, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


def verify(model_path: Path) -> int:
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
    weight = initializers[node.input[1]]
    filters, channels, kernel_height, kernel_width = weight.shape
    kernel = (kernel_height, kernel_width)
    strides = tuple(attributes.get("strides", [1, 1]))
    pads = tuple(attributes.get("pads", [0, 0, 0, 0]))
    output_shape = [
        dimension.dim_value
        for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ]
    _, _, output_height, output_width = output_shape
    input_shape = [
        dimension.dim_value
        for dimension in model.graph.input[0].type.tensor_type.shape.dim
    ]
    _, _, input_height, input_width = input_shape
    input_value = (
        (
            np.arange(channels * input_height * input_width, dtype=np.int32)
            % 17
            - 8
        )
        .astype(np.int8)
        .reshape(1, channels, input_height, input_width)
    )

    patches = lowering.conv_to_matrix(
        input_value, kernel, strides, pads
    ).astype(np.int32)
    patch_size = channels * kernel_height * kernel_width
    lowered_result = (
        weight.reshape(filters, patch_size).astype(np.int32) @ patches
    ).reshape(1, filters, output_height, output_width)
    source_result = ReferenceEvaluator(model).run(None, {"input": input_value})[
        0
    ]
    np.testing.assert_array_equal(source_result, lowered_result)
    assert source_result.dtype == np.int32
    return (
        output_height
        * output_width
        * ceil(filters / 16)
        * ceil(patch_size / 64)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    canonical_works = verify(args.model_dir / "canonical.onnx")
    odd_works = verify(args.model_dir / "odd.onnx")
    pointwise_works = verify(args.model_dir / "pointwise.onnx")
    rectangular_works = verify(args.model_dir / "rectangular.onnx")
    assert canonical_works == 256
    assert odd_works == 9
    assert pointwise_works == 12
    assert rectangular_works == 18
    print(
        "PASS ConvInteger verification: canonical_vmm=256 odd_vmm=9 "
        "pointwise_vmm=12 rectangular_vmm=18 dtype=int32"
    )


if __name__ == "__main__":
    main()
