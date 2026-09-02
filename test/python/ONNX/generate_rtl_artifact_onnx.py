"""Generate the deterministic multi-core model used by RTL artifact tests."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def make_model(batch: int = 8, heterogeneous: bool = False) -> onnx.ModelProto:
    """Create one dual-Macro INT8 transaction with multicast-safe tiles."""
    features = 65 if heterogeneous else 64
    columns = 17 if heterogeneous else 16
    k = np.arange(features, dtype=np.int16)[:, None]
    n = np.arange(columns, dtype=np.int16)[None, :]
    if heterogeneous:
        weight = ((3 * k + 5 * n + 1) % 15 - 7).astype(np.int8)
    else:
        weight = np.broadcast_to(((3 * k + 1) % 15 - 7), (64, 16)).astype(np.int8)
    weight[:, 0] = 1
    graph = helper.make_graph(
        [
            helper.make_node("MatMulInteger", ["input", "weight"], ["output"]),
        ],
        "rtl_artifact_multicore",
        [helper.make_tensor_value_info("input", TensorProto.INT8, [batch, features])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [batch, columns])],
        [
            numpy_helper.from_array(weight, name="weight"),
        ],
    )
    return helper.make_model(
        graph,
        producer_name="mlir-cim22-rtl-artifact-test",
        opset_imports=[helper.make_opsetid("", 10)],
        ir_version=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heterogeneous", action="store_true")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    model = make_model(args.batch, args.heterogeneous)
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model.SerializeToString(deterministic=True))
    print("PASS RTL artifact ONNX generation: dual Macro multicast-safe INT8")


if __name__ == "__main__":
    main()
