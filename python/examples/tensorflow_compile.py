"""Accept TF signed-INT8 import, CIM execution and native artifact reload.

Install the tensorflow extra and run with a native C linker available. This is
a deterministic small-model test, not a trained-model or hardware benchmark.
"""

import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from cim22 import Session
from cim22.runtime import MODEL_SUFFIX
from cim22.tensorflow import compile


def main() -> None:
    """Exercise signed inputs, padding, multiple tiles and a Host bias add."""
    weights = ((np.arange(65 * 17).reshape(65, 17) % 7) - 3).astype(np.int8)
    bias = np.arange(17, dtype=np.int32) - 8

    @tf.function(input_signature=[tf.TensorSpec([2, 65], tf.int8)])
    def model(x: tf.Tensor) -> tf.Tensor:
        # Tout makes the computation explicitly i8 x i8 -> i32. Merely storing
        # an FP32 layer's checkpoint in INT8 would not exercise this path.
        return tf.raw_ops.BatchMatMulV3(
            x=x, y=tf.constant(weights), Tout=tf.int32
        ) + tf.constant(bias)

    value = ((np.arange(130).reshape(2, 65) % 256) - 128).astype(np.int8)
    value[1] = 127
    reference = model(value).numpy()
    np.testing.assert_array_equal(
        reference, value.astype(np.int32) @ weights.astype(np.int32) + bias
    )
    with tempfile.TemporaryDirectory(prefix="cim22-tensorflow-") as folder:
        artifact = Path(folder) / f"model{MODEL_SUFFIX}"
        with compile(
            model, artifact_path=artifact, runtime="software"
        ) as session:
            (actual,) = session(value)
            assert session.num_transactions > 0
            assert session.dispatch_count > 0
            np.testing.assert_array_equal(actual, reference)
            dispatches = session.dispatch_count
        with Session(artifact) as restored:
            np.testing.assert_array_equal(restored(value)[0], actual)
        print(f"TensorFlow INT8 -> native {MODEL_SUFFIX} -> CIM software: PASS")
        print(f"Saved artifact reload: PASS; CIM dispatches: {dispatches}")

    error = np.max(np.abs(actual.astype(np.int64) - reference))
    print(f"TensorFlow INT32, first lane: {reference[:, 0].tolist()}")
    print(f"CIM software, first lane:   {actual[:, 0].tolist()}")
    print(f"Maximum absolute error: {error}")
    print("Exact TensorFlow / NumPy / CIM agreement: PASS")


if __name__ == "__main__":
    main()
