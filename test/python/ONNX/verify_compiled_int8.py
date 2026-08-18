"""Compare compiled CIM22 INT8 static weights with ONNX source results."""

import argparse
from pathlib import Path

import int8_lowering as lowering
import int8_reference as reference
import numpy as np
import onnx
import verify_static_weights as static_weights
from onnx import checker, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


class VerificationError(ValueError):
    pass


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _source_and_matrices(
    model_path: Path, folded_bias: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return source golden, canonical [K,M]/[N,K], and output shape."""
    model = onnx.load_model(model_path)
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True)
    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    nodes = list(model.graph.node)
    compute = nodes[0] if nodes else None
    if compute is None or compute.op_type not in {
        "MatMulInteger",
        "ConvInteger",
    }:
        raise VerificationError(
            "model must start with MatMulInteger or ConvInteger"
        )
    input_shape = tuple(
        dimension.dim_value
        for dimension in model.graph.input[0].type.tensor_type.shape.dim
    )
    if not input_shape or any(dimension <= 0 for dimension in input_shape):
        raise VerificationError(
            "model input must have static positive dimensions"
        )
    input_value = (
        (
            np.arange(np.prod(input_shape), dtype=np.int32)
            % (23 if compute.op_type == "MatMulInteger" else 17)
            - (11 if compute.op_type == "MatMulInteger" else 8)
        )
        .astype(np.int8)
        .reshape(input_shape)
    )
    golden = ReferenceEvaluator(model).run(
        None, {compute.input[0]: input_value}
    )[0]
    weight = initializers.get(compute.input[1])
    if weight is None or weight.dtype != np.int8:
        raise VerificationError(
            "compute weight must be an embedded INT8 initializer"
        )
    if compute.op_type == "MatMulInteger":
        if (
            weight.ndim != 2
            or input_value.ndim != 2
            or input_value.shape[1] != weight.shape[0]
        ):
            raise VerificationError("unsupported MatMulInteger shape")
        input_matrix = input_value.T
        weight_matrix = weight.T
    else:
        if weight.ndim != 4 or input_value.ndim != 4:
            raise VerificationError("unsupported ConvInteger shape")
        attributes = _attributes(compute)
        pads = tuple(attributes.get("pads", [0, 0, 0, 0]))
        strides = tuple(attributes.get("strides", [1, 1]))
        input_matrix = lowering.conv_to_matrix(
            input_value, tuple(weight.shape[2:]), strides, pads
        )
        weight_matrix = weight.reshape(weight.shape[0], -1)
    if folded_bias:
        if len(nodes) != 2 or nodes[1].op_type != "Add":
            raise VerificationError("--folded-bias requires one trailing Add")
        bias_name = next(
            (name for name in nodes[1].input if name in initializers), None
        )
        if bias_name is None:
            raise VerificationError("folded Add must use an embedded bias")
        weight_matrix, input_matrix = lowering.fold_bias(
            weight_matrix,
            input_matrix,
            initializers[bias_name],
            weight_matrix.shape[1] == 64,
        )
    elif len(nodes) != 1:
        raise VerificationError(
            "unfolded source model must contain one compute node"
        )
    return golden, input_matrix, weight_matrix, tuple(golden.shape)


def _run_static_weights(
    records: list[static_weights.Record],
    input_matrix: np.ndarray,
    weight_matrix: np.ndarray,
) -> np.ndarray:
    """Execute static command records through the native-tile reference model."""
    k, m = input_matrix.shape
    n = weight_matrix.shape[0]
    k_tiles = (k + 63) // 64
    n_tiles = (n + 15) // 16
    expected = {
        (m_tile, n_tile, k_tile)
        for m_tile in range(m)
        for n_tile in range(n_tiles)
        for k_tile in range(k_tiles)
    }
    actual = {
        (record.work[0], record.work[1], record.work[2]) for record in records
    }
    if len(actual) != len(records) or actual != expected:
        raise VerificationError(
            "static commands do not cover each logical tile exactly once"
        )
    indices = np.asarray(
        [record.work[:3] for record in records], dtype=np.int64
    )
    if (
        np.any(indices < 0)
        or np.any(indices[:, 0] >= m)
        or np.any(indices[:, 1] >= n_tiles)
        or np.any(indices[:, 2] >= k_tiles)
    ):
        raise VerificationError("static command has an out-of-range tile index")
    padded_input = np.pad(input_matrix, ((0, k_tiles * 64 - k), (0, 0)))
    inputs = padded_input.reshape(k_tiles, 64, m).transpose(2, 0, 1)[
        indices[:, 0], indices[:, 2]
    ]
    weights = np.stack(
        [
            reference.decode_int8_weight_words(record.words.view(np.uint32))
            for record in records
        ]
    )
    partials = reference.simulate_int8_tiles(inputs, weights)
    result = np.zeros((m, n_tiles, 16), dtype=np.int64)
    np.add.at(result, (indices[:, 0], indices[:, 1]), partials)
    return lowering.wrap_i32(result.reshape(m, n_tiles * 16)[:, :n])


def verify(
    static_path: Path, model_path: Path, folded_bias: bool
) -> tuple[int, tuple[int, ...]]:
    golden, input_matrix, weight_matrix, output_shape = _source_and_matrices(
        model_path, folded_bias
    )
    with static_path.open() as stream:
        _, records = static_weights.parse_records(stream)
    matrix_result = _run_static_weights(records, input_matrix, weight_matrix)
    if len(output_shape) == 4:
        result = matrix_result.T.reshape(output_shape)
    else:
        result = matrix_result
    np.testing.assert_array_equal(result, golden)
    return len(records), output_shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify compiler static INT8 weights against ONNX source results"
    )
    parser.add_argument("static_weight", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--folded-bias", action="store_true")
    args = parser.parse_args()
    records, shape = verify(args.static_weight, args.model, args.folded_bias)
    print(f"PASS software-only compiled-int8 records={records} shape={shape}")


if __name__ == "__main__":
    main()
