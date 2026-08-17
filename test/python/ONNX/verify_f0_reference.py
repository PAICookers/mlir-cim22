"""Generate the F0 source result with ONNX ReferenceEvaluator."""

import hashlib
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

EXPECTED_HASH = (
    "5fd6f6d5dfa733419c465996772022b2ac92e6547057769e29ac2c8562d8e697"
)
I21_MIN = -1048576
I21_MAX = 1048575


def partial_bounds(weights: np.ndarray) -> tuple[int, int]:
    lower = 0
    upper = 0
    for weight in weights.astype(np.int64, copy=False):
        lower += -128 * weight if weight >= 0 else 127 * weight
        upper += 127 * weight if weight >= 0 else -128 * weight
    return lower, upper


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_f0_reference.py <model.onnx>")
    model_path = sys.argv[1]
    with open(model_path, "rb") as model_file:
        assert hashlib.sha256(model_file.read()).hexdigest() == EXPECTED_HASH

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    graph = model.graph
    assert (
        len(graph.node)
        == len(graph.input)
        == len(graph.initializer)
        == len(graph.output)
        == 1
    )
    assert graph.node[0].op_type == "MatMulInteger"
    assert len(graph.node[0].input) == 2
    assert len(graph.node[0].output) == 1
    weights = numpy_helper.to_array(graph.initializer[0])
    assert weights.shape == (512, 1024) and weights.dtype == np.int8

    partial_ranges = [
        partial_bounds(weights[k : k + 64, n])
        for n in range(1024)
        for k in range(0, 512, 64)
    ]
    partial_lower = min(bound[0] for bound in partial_ranges)
    partial_upper = max(bound[1] for bound in partial_ranges)
    assert (partial_lower, partial_upper) == (-656251, 655469)
    assert I21_MIN <= partial_lower <= partial_upper <= I21_MAX
    full_ranges = [partial_bounds(weights[:, n]) for n in range(1024)]
    full_lower = min(bound[0] for bound in full_ranges)
    full_upper = max(bound[1] for bound in full_ranges)
    assert (full_lower, full_upper) == (-4483020, 4484565)

    input_values = (
        (np.arange(32 * 512, dtype=np.int32) * 37 + 11) % 256 - 128
    ).astype(np.int8)
    input_values = input_values.reshape(32, 512)
    source_result = ReferenceEvaluator(model).run(
        None, {graph.input[0].name: input_values}
    )[0]
    assert source_result.shape == (32, 1024)
    assert source_result.dtype == np.int32
    print(
        "PASS: shape=(32, 1024) dtype=int32 partial=[-656251,655469] full=[-4483020,4484565]"
    )


if __name__ == "__main__":
    main()
