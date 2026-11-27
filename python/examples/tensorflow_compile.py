"""Compare an INT8 TensorFlow Dense/ReLU model with NumPy and CIM22."""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from cim22.apps.framework_example import (
    cli,
    dense_data,
    finish,
    group_result,
    header,
)


def run(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compile once and compare each group with TensorFlow and NumPy."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    header("TensorFlow", runtime, "Dense(64→16) + ReLU")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    from cim22.tensorflow import compile

    tf.get_logger().setLevel("ERROR")
    weights, bias, inputs = dense_data()

    @tf.function(input_signature=[tf.TensorSpec([4, 64], tf.int8)])
    def model(x: tf.Tensor) -> tf.Tensor:
        product = tf.raw_ops.BatchMatMulV3(
            x=x, y=tf.constant(weights), Tout=tf.int32
        )
        return tf.nn.relu(product + tf.constant(bias))

    output.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "runtime": runtime,
        "artifact_dir": output,
        "debug_dir": output / "ir",
    }
    if usb_config is not None:
        options["usb_config"] = usb_config
    cases = []
    started = time.perf_counter()
    with compile(model, options=options) as compiled:
        compile_seconds = time.perf_counter() - started
        for index, value in enumerate(inputs, start=1):
            x = tf.constant(value)
            reference = model(x).numpy()
            expected = np.maximum(
                value.astype(np.int32) @ weights.astype(np.int32) + bias, 0
            )
            actual = compiled(x).numpy()
            cases.append(
                group_result(
                    index,
                    reference,
                    actual,
                    compiled.session.dispatch_count,
                    numpy_reference=expected,
                )
            )

    result = {
        "framework": "tensorflow",
        "runtime": runtime,
        "compile_seconds": compile_seconds,
        "dispatch_count": cases[-1]["dispatch_count"],
        "cases": cases,
        "artifact_dir": str(output.resolve()),
        "ir_dir": str((output / "ir").resolve()),
    }
    return finish(result, output, result_json)


def main(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Run the example with the selected runtime and output paths."""
    return run(
        output,
        runtime=runtime,
        usb_config=usb_config,
        result_json=result_json,
    )


if __name__ == "__main__":
    cli(main, __doc__, Path("outputs/tensorflow"))
