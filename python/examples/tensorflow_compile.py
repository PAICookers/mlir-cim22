"""Compile a tiny, reproducible INT8 network and compare its outputs.

Run with the tensorflow extra and a native C linker. Model and IR files remain
in outputs/tensorflow/<run timestamp>/ relative to the working directory.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np


def main(output: Path) -> None:
    """Run Linear(8, 8) -> clipped ReLU -> Linear(8, 4) in software."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    from cim22.tensorflow import compile

    tf.get_logger().setLevel("ERROR")
    rng = np.random.default_rng(42)
    w1 = rng.integers(-3, 4, (8, 8), dtype=np.int8)
    w2 = rng.integers(-3, 4, (8, 4), dtype=np.int8)

    @tf.function(input_signature=[tf.TensorSpec([1, 8], tf.int8)])
    def model(x: tf.Tensor) -> tf.Tensor:
        # Each Linear computes i8 x i8 -> i32. Clip before casting the hidden
        # activation back to i8, so the second layer also uses integer VMM.
        hidden = tf.raw_ops.BatchMatMulV3(x=x, y=tf.constant(w1), Tout=tf.int32)
        hidden = tf.cast(tf.clip_by_value(hidden, 0, 127), tf.int8)
        return tf.raw_ops.BatchMatMulV3(
            x=hidden, y=tf.constant(w2), Tout=tf.int32
        )

    value = rng.integers(-4, 5, (1, 8), dtype=np.int8)
    inputs = tf.constant(value)
    reference = model(inputs).numpy()
    print("Compiling: INT8 Linear(8,8) -> clipped ReLU -> Linear(8,4); seed=42")
    with compile(
        model,
        options={
            "runtime": "software",
            "artifact_dir": output,
            "debug_dir": output / "ir",
        },
    ) as compiled:
        actual = compiled(inputs).numpy()
        print(f"CIM software calls: {compiled.session.dispatch_count}")
    print(f"TensorFlow: {reference.tolist()}")
    print(f"CIM:        {actual.tolist()}")
    print(
        f"Max absolute error: {np.max(np.abs(actual.astype(np.int64) - reference))}"
    )
    print(f"Model, debug IR and compile.log: {output.resolve()}")


if __name__ == "__main__":
    output = Path("outputs/tensorflow") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    output.mkdir(parents=True)
    # Capture native TensorFlow stderr as well as Python diagnostics, while
    # leaving the concise results on stdout. Restore stderr even on failure.
    with (output / "compile.log").open("w", encoding="utf-8") as log:
        saved_stderr = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            main(output)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
