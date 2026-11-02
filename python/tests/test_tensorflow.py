"""TensorFlow capture must execute native code, preserving its flat signature."""

from pathlib import Path

import numpy as np
import pytest

from cim22 import Session
from cim22.runtime import MODEL_SUFFIX

tf = pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


def test_host_outputs_match_tensorflow() -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2, 3], tf.float32)])
    def model(x):
        return tf.nn.relu(x + 2.0), tf.reduce_sum(x)

    value = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    expected = model(value)
    with compile(model, runtime="software") as session:
        actual = session(value)
        assert session.num_transactions == 0
        assert session.dispatch_count == 0
        for output, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(output, reference.numpy())


@pytest.mark.parametrize("convolution", [False, True], ids=["dense", "conv2d"])
def test_frozen_weights_dispatch_and_reload(
    tmp_path: Path, convolution: bool
) -> None:
    from cim22.tensorflow import compile

    shape = (1, 1, 2, 64) if convolution else (2, 64)
    layer_type = (
        tf.keras.layers.Conv2D if convolution else tf.keras.layers.Dense
    )
    layer = layer_type(
        16,
        **({"kernel_size": 1} if convolution else {}),
        kernel_initializer=tf.keras.initializers.Constant(0.01),
        bias_initializer="ones",
        activation="relu",
    )
    layer.build(shape)

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32)])
    def model(x):
        return layer(x)

    artifact = tmp_path / f"model{MODEL_SUFFIX}"
    debug = tmp_path / "ir"
    value = np.full(shape, 0.01, dtype=np.float32)
    value.reshape(2, 64)[1] *= -1
    reference = model(value).numpy()
    # Supplier-model goldens, independent of the TensorFlow adapter. These are
    # deliberately not FP32 matmul expectations (see the acceptance example).
    expected = np.broadcast_to(
        np.array([0x7E520000, 0], dtype=np.uint32).view(np.float32)[:, None],
        (2, 16),
    ).reshape(*shape[:-1], 16)
    with compile(
        model, artifact_path=artifact, debug_dir=debug, runtime="software"
    ) as session:
        assert session.input_shapes == (shape,)
        assert session.num_transactions > 0
        (actual,) = session(value)
        assert session.dispatch_count > 0
        np.testing.assert_array_equal(actual, expected)
        assert np.max(np.abs(actual.astype(np.float64) - reference)) > 1
        # Export freezes the weights; the native model cannot call TensorFlow.
        layer.kernel.assign(tf.zeros_like(layer.kernel))
        np.testing.assert_array_equal(session(value)[0], expected)
    assert (debug / "model.linalg.mlir").is_file()
    with Session(artifact) as loaded:
        np.testing.assert_array_equal(loaded(value)[0], expected)
    artifact.unlink()


@pytest.mark.parametrize(
    "spec",
    [
        tf.TensorSpec([None, 64], tf.float32),
        tf.TensorSpec([2, 64], tf.float64),
        tf.TensorSpec([0, 64], tf.float32),
    ],
)
def test_rejects_nonstatic_fp32_signature(spec) -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[spec])
    def model(x):
        return x + x

    with pytest.raises(ValueError, match="static.*FP32"):
        compile(model, runtime="software")


def test_requires_explicit_function_signature() -> None:
    from cim22.tensorflow import compile

    def model(x):
        return x + x

    with pytest.raises(TypeError, match="tf.function"):
        compile(model, runtime="software")
    with pytest.raises(ValueError, match="static.*FP32"):
        compile(tf.function(model), runtime="software")


def test_rejects_unsupported_output() -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
    def model(x):
        return tf.cast(x, tf.float64)

    with pytest.raises(ValueError, match="outputs.*FP32"):
        compile(model, runtime="software")


def test_int8_matmul_tiling_and_reload(tmp_path: Path) -> None:
    """Real signed INT8 computation, including padding, Host sums and bias."""
    from cim22.tensorflow import compile

    weights = ((np.arange(65 * 17).reshape(65, 17) % 7) - 3).astype(np.int8)
    bias = np.arange(17, dtype=np.int32) - 8

    @tf.function(input_signature=[tf.TensorSpec([2, 65], tf.int8)])
    def model(x):
        return tf.raw_ops.BatchMatMulV3(
            x=x, y=tf.constant(weights), Tout=tf.int32
        ) + tf.constant(bias)

    value = ((np.arange(130).reshape(2, 65) % 256) - 128).astype(np.int8)
    value[1] = 127
    expected = value.astype(np.int32) @ weights.astype(np.int32) + bias
    np.testing.assert_array_equal(model(value).numpy(), expected)
    artifact = tmp_path / f"integer{MODEL_SUFFIX}"
    with compile(
        model,
        artifact_path=artifact,
        debug_dir=tmp_path / "ir",
        runtime="software",
    ) as session:
        assert session.input_dtypes == (np.dtype(np.int8),)
        assert session.output_dtypes == (np.dtype(np.int32),)
        assert session.num_transactions > 0
        np.testing.assert_array_equal(session(value)[0], expected)
        assert session.dispatch_count > 0
        with pytest.raises(TypeError, match="int8"):
            session(value.astype(np.float32))
    with Session(artifact) as loaded:
        np.testing.assert_array_equal(loaded(value)[0], expected)
    artifact.unlink()


def test_int8_partial_overflow_stays_on_host() -> None:
    """All -128 weights can overflow a signed-i21 tile; do not offload them."""
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([1, 64], tf.int8)])
    def model(x):
        return tf.raw_ops.BatchMatMulV3(
            x=x,
            y=tf.constant(-128, shape=[64, 16], dtype=tf.int8),
            Tout=tf.int32,
        )

    value = np.full((1, 64), -128, dtype=np.int8)
    with compile(model, runtime="software") as session:
        np.testing.assert_array_equal(session(value)[0], model(value).numpy())
        assert session.num_transactions == 0
        assert session.dispatch_count == 0


def test_mixed_integer_host_outputs() -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def model(x):
        return x + 1, tf.cast(x, tf.int8)

    value = np.array([-129, 128], dtype=np.int32)
    with compile(model, runtime="software") as session:
        assert session.output_dtypes == (np.dtype(np.int32), np.dtype(np.int8))
        for actual, expected in zip(session(value), model(value), strict=True):
            np.testing.assert_array_equal(actual, expected.numpy())
