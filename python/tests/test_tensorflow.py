"""TensorFlow capture must execute native code, preserving its flat signature."""

import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from cim22 import Session, UsbConfig
from cim22.runtime import MODEL_SUFFIX

tf: ModuleType


@pytest.fixture(scope="module", autouse=True)
def tensorflow_environment() -> None:
    """Load the optional framework when these tests run, not during collection."""
    global tf
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


@pytest.mark.skipif(sys.platform != "linux", reason="ELF LLVM collision")
@pytest.mark.parametrize("first", ["tensorflow", "backend"])
def test_torch_compile_after_tensorflow_is_rejected(first: str) -> None:
    """Reject the dangerous FX import, then keep TF compilation usable."""
    script = """
import sys
import numpy as np
import torch
from torch._dynamo.exc import BackendCompilerFailed
if sys.argv[1] == "backend":
    import cim22.backend
import tensorflow as tf
assert "torch_mlir.fx" not in sys.modules
model = torch.compile(
    lambda x: x + 1, backend="cim22", fullgraph=True, dynamic=False,
    options={"runtime": "simulator"},
)
with torch.inference_mode():
    try:
        model(torch.ones(2))
    except BackendCompilerFailed as error:
        message = str(error)
        for text in ("TensorFlow", "LLVM", "abort", "fresh Python process",
                     "without TensorFlow", "artifact_dir", "cim22.Session"):
            assert text in message, message
        assert isinstance(error.inner_exception, RuntimeError)
    else:
        raise AssertionError("unsafe FX import was not rejected")
assert "torch_mlir.fx" not in sys.modules
assert "torch_mlir._mlir_libs._mlir" not in sys.modules
from cim22.tensorflow import compile
@tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
def host(x):
    return x + 1
with compile(host, options={"runtime": "simulator"}) as compiled:
    np.testing.assert_array_equal(compiled(tf.ones(2)), [2, 2])
assert "torch_mlir.fx" not in sys.modules
print("safe rejection and TensorFlow recovery: PASS")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=os.environ
        | {"TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "safe rejection and TensorFlow recovery: PASS" in result.stdout


@pytest.mark.parametrize("first", ["tensorflow", "torch-mlir", "native"])
def test_compile_in_fresh_process(first: str) -> None:
    """Exercise independent LLVM copies, native compilation and artifact reload."""
    prelude = {
        "tensorflow": "import tensorflow",
        "torch-mlir": "import torch_mlir.fx",
        "native": """
import torch
model = torch.compile(
    lambda x: x + 1, backend="cim22", fullgraph=True, dynamic=False,
    options={"runtime": "simulator"},
)
with torch.inference_mode():
    torch.testing.assert_close(model(torch.ones(2)), torch.full((2,), 2.0))
""",
    }[first]
    script = """
import sys
flags = getattr(sys, "getdlopenflags", lambda: 0)()
"""
    script += prelude
    script += """
from cim22.tensorflow import compile
assert getattr(sys, "getdlopenflags", lambda: 0)() == flags
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from cim22 import Session
from cim22.runtime import MODEL_SUFFIX
weights = ((np.arange(65 * 17).reshape(65, 17) % 7) - 3).astype(np.int8)
bias = np.arange(17, dtype=np.int32) - 8
@tf.function(input_signature=[tf.TensorSpec([2, 65], tf.int8)])
def infer(x):
    return tf.raw_ops.BatchMatMulV3(
        x=x, y=tf.constant(weights), Tout=tf.int32
    ) + tf.constant(bias)
value = ((np.arange(130).reshape(2, 65) % 256) - 128).astype(np.int8)
value[1] = 127
expected = value.astype(np.int32) @ weights.astype(np.int32) + bias
np.testing.assert_array_equal(infer(value).numpy(), expected)
with tempfile.TemporaryDirectory() as folder:
    with compile(infer, options={"runtime": "simulator", "artifact_dir": folder}) as compiled:
        actual = compiled(tf.constant(value))
        assert isinstance(actual, tf.Tensor)
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert compiled.session.dispatch_count == 1
    artifact, = Path(folder).glob(f"graph-*{MODEL_SUFFIX}")
    with Session(artifact) as session:
        np.testing.assert_array_equal(session(value)[0], expected)
print("INT8 compile, 2 dispatches and reload: PASS")
assert getattr(sys, "getdlopenflags", lambda: 0)() == flags
"""
    if first == "tensorflow":
        script += (
            'assert not any(n.startswith("torch_mlir") for n in sys.modules)\n'
        )
    if first == "native":
        script += """
model_after_tensorflow = torch.compile(
    lambda x: x * 2, backend="cim22", fullgraph=True, dynamic=False,
    options={"runtime": "simulator"},
)
with torch.inference_mode():
    torch.testing.assert_close(
        model_after_tensorflow(torch.ones(2)), torch.full((2,), 2.0)
    )
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=os.environ
        | {"TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INT8 compile, 2 dispatches and reload: PASS" in result.stdout


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"runtime": "invalid"}, "runtime must be"),
        ({"bf16_compensation": "invalid"}, "bf16_compensation must be"),
        (
            {"runtime": "simulator", "usb_config": UsbConfig(1, 2, 0x81, 1)},
            "usb_config requires runtime='usb'",
        ),
    ],
)
def test_options_rejected_like_pytorch(
    options: dict[str, object], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both frontends reject the same options before any model conversion."""
    import torch

    from cim22.backend import backend
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
    def model(x):
        return x + 1

    def unexpected_export(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid options reached TensorFlow export")

    monkeypatch.setattr("tf2onnx.convert.from_function", unexpected_export)
    with torch.inference_mode(), pytest.raises(ValueError, match=message):
        backend(None, [], options=options)
    with pytest.raises(ValueError, match=message):
        compile(model, options=options)


@pytest.mark.parametrize("keyword", ["runtime", "artifact_path", "debug_dir"])
def test_legacy_keywords_are_removed(keyword: str) -> None:
    from cim22.tensorflow import compile

    with pytest.raises(TypeError, match="unexpected keyword"):
        compile(None, **{keyword: "unused"})


@pytest.mark.parametrize(
    ("failure", "saved"),
    [(False, False), (True, False), (True, True)],
    ids=["configured", "failure", "saved-failure"],
)
def test_default_usb_configuration_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
    saved: bool,
) -> None:
    """Configure only the parent Session; device errors must release it."""
    import cim22.tensorflow as frontend
    from cim22.compiler import BackendOptions

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config = UsbConfig(1, 2, 0x81, 1)
    sessions: list[Session] = []
    run = frontend._run

    def check_runtime(command: list[str], stage: str) -> None:
        assert command[command.index("--runtime") + 1] == "usb"
        run(command, stage)

    def configure(session: Session, received: UsbConfig) -> None:
        assert received is config
        sessions.append(session)
        if failure:
            raise RuntimeError("device unavailable")

    monkeypatch.setattr(frontend, "_run", check_runtime)
    monkeypatch.setattr(Session, "configure_usb", configure)

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
    def model(x):
        return x + 1

    options: BackendOptions = {"usb_config": config}
    artifacts = tmp_path / "models"
    if saved:
        options["artifact_dir"] = artifacts
    if failure:
        with pytest.raises(RuntimeError, match="device unavailable"):
            frontend.compile(model, options=options)
    else:
        with frontend.compile(model, options=options) as compiled:
            np.testing.assert_array_equal(compiled(tf.ones(2)).numpy(), [2, 2])
    assert len(sessions) == 1
    assert sessions[0]._handle is None
    assert not list(tmp_path.glob(".cim22-tensorflow-*"))
    if saved:
        (artifact,) = artifacts.glob(f"graph-*{MODEL_SUFFIX}")
        with Session(artifact) as session:
            np.testing.assert_array_equal(
                session(np.ones(2, dtype=np.float32))[0], [2, 2]
            )


def test_host_outputs_match_tensorflow() -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2, 3], tf.float32)])
    def model(x):
        return tf.nn.relu(x + 2.0), tf.reduce_sum(x)

    value = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    expected = model(value)
    with compile(
        model, options={"runtime": "simulator", "unused": True}
    ) as compiled:
        actual = compiled(tf.constant(value))
        assert isinstance(actual, tuple)
        assert compiled.session.num_transactions == 0
        assert compiled.session.dispatch_count == 0
        for output, reference in zip(actual, expected, strict=True):
            assert isinstance(output, tf.Tensor)
            np.testing.assert_array_equal(output.numpy(), reference.numpy())


def test_callable_preserves_nested_outputs_and_tensor_boundaries() -> None:
    """Use native results with TF containers, scalar inputs and repeated calls."""
    from cim22.tensorflow import compile

    @tf.function(
        input_signature=[
            tf.TensorSpec([2], tf.int32),
            tf.TensorSpec([], tf.int32),
        ]
    )
    def model(x, offset):
        shifted = x + offset
        return {"z": [shifted, tf.reduce_sum(shifted)], "a": (x, x)}

    value = tf.constant([1, 2], dtype=tf.int32)
    offset = tf.constant(3, dtype=tf.int32)
    expected = model(value, offset)
    with compile(model, options={"runtime": "simulator"}) as compiled:
        first = compiled(value, offset)
        tf.nest.assert_same_structure(first, expected)
        for actual, reference in zip(
            tf.nest.flatten(first), tf.nest.flatten(expected), strict=True
        ):
            assert isinstance(actual, tf.Tensor)
            assert actual.dtype == reference.dtype
            assert actual.shape == reference.shape
            np.testing.assert_array_equal(actual.numpy(), reference.numpy())
        variable = tf.Variable([5, 6], dtype=tf.int32, trainable=False)
        second = compiled(variable, offset)
        np.testing.assert_array_equal(second["z"][0], [8, 9])
        np.testing.assert_array_equal(first["z"][0], [4, 5])
        with pytest.raises(TypeError, match="tf.Tensor"):
            compiled(value.numpy(), offset)
        with pytest.raises(TypeError, match="int32"):
            compiled(tf.cast(value, tf.int64), offset)
        with pytest.raises(ValueError, match="shape"):
            compiled(tf.zeros([3], dtype=tf.int32), offset)
        with pytest.raises(ValueError, match="Input count"):
            compiled(value)
        with pytest.raises(ValueError, match="requires eager"):
            tf.function(lambda x, y: compiled(x, y))(value, offset)
    np.testing.assert_array_equal(first["z"][0], [4, 5])
    with pytest.raises(RuntimeError, match="closed"):
        compiled(value, offset)


@pytest.mark.parametrize("output", [None, ()])
def test_rejects_missing_tensor_outputs(output: object) -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
    def model(x):
        return output

    with pytest.raises(ValueError, match="outputs must be dense tensors"):
        compile(model, options={"runtime": "simulator"})


@pytest.mark.parametrize(
    ("convolution", "uncompensated"),
    [(False, False), (True, False), (False, True)],
    ids=["dense", "conv2d", "dense-uncompensated"],
)
def test_frozen_weights_dispatch_and_reload(
    tmp_path: Path, convolution: bool, uncompensated: bool
) -> None:
    from cim22.tensorflow import compile

    shape = (1, 1, 2, 64) if convolution else (2, 64)
    layer_type = (
        tf.keras.layers.Conv2D if convolution else tf.keras.layers.Dense
    )
    layer = layer_type(
        16,
        **({"kernel_size": 1} if convolution else {}),
        kernel_initializer=tf.keras.initializers.Constant(
            0.01 if uncompensated else 0.5
        ),
        bias_initializer="ones",
        activation="relu",
    )
    layer.build(shape)

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32)])
    def model(x):
        return layer(x)

    artifacts = tmp_path / "models"
    debug = tmp_path / "ir"
    value = np.full(shape, 0.01 if uncompensated else 1.0, dtype=np.float32)
    value.reshape(2, 64)[1] *= -1
    from cim22.compiler import BackendOptions

    options: BackendOptions = {
        "artifact_dir": artifacts,
        "debug_dir": debug,
        "runtime": "simulator",
    }
    if uncompensated:
        options["bf16_compensation"] = "none"
        # Raw supplier-model golden retained for the uncompensated baseline.
        rows = np.array([0x7E520000, 0], dtype=np.uint32).view(np.float32)
    else:
        # ReLU(64 * (+/-1) * 0.5 + 1) is exactly representable.
        rows = np.array([33, 0], dtype=np.float32)
    expected = np.broadcast_to(rows[:, None], (2, 16)).reshape(*shape[:-1], 16)
    with compile(model, options=options) as compiled:
        assert compiled.session.input_shapes == (shape,)
        assert compiled.session.num_transactions > 0
        actual = compiled(tf.constant(value)).numpy()
        assert compiled.session.dispatch_count > 0
        np.testing.assert_array_equal(actual, expected)
        # Export freezes the weights; the native model cannot call TensorFlow.
        layer.kernel.assign(tf.zeros_like(layer.kernel))
        np.testing.assert_array_equal(compiled(tf.constant(value)), expected)
    (artifact,) = artifacts.glob(f"graph-*{MODEL_SUFFIX}")
    assert (debug / artifact.stem / "model.linalg.mlir").is_file()
    with Session(artifact) as loaded:
        np.testing.assert_array_equal(loaded(value)[0], expected)
    artifact.unlink()


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [([None, 64], "float32"), ([2, 64], "float64"), ([0, 64], "float32")],
)
def test_rejects_nonstatic_fp32_signature(
    shape: list[int | None], dtype: str
) -> None:
    from cim22.tensorflow import compile

    spec = tf.TensorSpec(shape, getattr(tf, dtype))

    @tf.function(input_signature=[spec])
    def model(x):
        return x + x

    with pytest.raises(ValueError, match="static.*FP32"):
        compile(model, options={"runtime": "simulator"})


def test_requires_explicit_function_signature() -> None:
    from cim22.tensorflow import compile

    def model(x):
        return x + x

    with pytest.raises(TypeError, match="tf.function"):
        compile(model, options={"runtime": "simulator"})
    with pytest.raises(ValueError, match="static.*FP32"):
        compile(tf.function(model), options={"runtime": "simulator"})


def test_rejects_unsupported_output() -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
    def model(x):
        return tf.cast(x, tf.float64)

    with pytest.raises(ValueError, match="outputs.*FP32"):
        compile(model, options={"runtime": "simulator"})


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
    artifacts = tmp_path / "models"
    with compile(
        model,
        options={
            "artifact_dir": artifacts,
            "debug_dir": tmp_path / "ir",
            "runtime": "simulator",
        },
    ) as compiled:
        assert compiled.session.input_dtypes == (np.dtype(np.int8),)
        assert compiled.session.output_dtypes == (np.dtype(np.int32),)
        assert compiled.session.num_transactions > 0
        actual = compiled(tf.constant(value))
        assert isinstance(actual, tf.Tensor)
        assert actual.dtype == tf.int32
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert compiled.session.dispatch_count > 0
        with pytest.raises(TypeError, match="int8"):
            compiled(tf.constant(value, dtype=tf.float32))
    (artifact,) = artifacts.glob(f"graph-*{MODEL_SUFFIX}")
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
    with compile(model, options={"runtime": "simulator"}) as compiled:
        np.testing.assert_array_equal(
            compiled(tf.constant(value)), model(value).numpy()
        )
        assert compiled.session.num_transactions == 0
        assert compiled.session.dispatch_count == 0


def test_mixed_integer_host_outputs() -> None:
    from cim22.tensorflow import compile

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def model(x):
        return x + 1, tf.cast(x, tf.int8)

    value = np.array([-129, 128], dtype=np.int32)
    with compile(model, options={"runtime": "simulator"}) as compiled:
        assert compiled.session.output_dtypes == (
            np.dtype(np.int32),
            np.dtype(np.int8),
        )
        for actual, expected in zip(
            compiled(tf.constant(value)), model(value), strict=True
        ):
            assert isinstance(actual, tf.Tensor)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_worker_temp_lifetime_and_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A returned Session owns its temporary model; failed children leave none."""
    from cim22.tensorflow import compile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
    def model(x):
        return x + 1

    with compile(model, options={"runtime": "simulator"}) as compiled:
        (folder,) = tmp_path.glob(".cim22-tensorflow-*")
        np.testing.assert_array_equal(compiled(tf.ones(2)).numpy(), [2, 2])
        assert (folder / f"model{MODEL_SUFFIX}").is_file()
    assert not folder.exists()

    artifacts = tmp_path / "models"
    debug = tmp_path / "ir"
    options = {
        "runtime": "simulator",
        "artifact_dir": str(artifacts),
        "debug_dir": str(debug),
    }
    with compile(model, options=options):
        pass
    (first,) = artifacts.glob(f"graph-*{MODEL_SUFFIX}")
    previous = first.read_bytes()
    with compile(model, options=options) as compiled:
        np.testing.assert_array_equal(compiled(tf.ones(2)).numpy(), [2, 2])
    saved = list(artifacts.glob(f"graph-*{MODEL_SUFFIX}"))
    assert len(saved) == 2
    assert first.read_bytes() == previous
    for artifact in saved:
        assert (debug / artifact.stem / "model.linalg.mlir").is_file()
        with Session(artifact) as session:
            np.testing.assert_array_equal(
                session(np.ones(2, dtype=np.float32))[0], [2, 2]
            )

    monkeypatch.setenv("CIM22_OPT", str(tmp_path / "missing-optimizer"))
    with pytest.raises(RuntimeError, match="TensorFlow compilation failed"):
        compile(model, options={"runtime": "simulator"})
    assert not list(tmp_path.glob(".cim22-tensorflow-*"))


def test_example_keeps_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The small example leaves a reloadable model and IR in its output folder."""
    monkeypatch.chdir(tmp_path)
    example = (
        Path(__file__).resolve().parents[1] / "examples/tensorflow_compile.py"
    )
    monkeypatch.setattr(sys, "argv", [str(example)])
    runpy.run_path(str(example), run_name="__main__")
    runs = list((tmp_path / "outputs/tensorflow").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "compile.log").is_file()
    (artifact,) = runs[0].glob(f"graph-*{MODEL_SUFFIX}")
    assert (runs[0] / "ir" / artifact.stem / "model.linalg.mlir").is_file()
    rng = np.random.default_rng(42)
    w1 = rng.integers(-3, 4, (16, 32), dtype=np.int8)
    w2 = rng.integers(-3, 4, (32, 16), dtype=np.int8)
    value = rng.integers(-4, 5, (1, 4, 16), dtype=np.int8)
    hidden = np.clip(value.astype(np.int32) @ w1.astype(np.int32), 0, 127)
    expected = hidden @ w2.astype(np.int32)
    with Session(artifact) as loaded:
        np.testing.assert_array_equal(loaded(value)[0], expected)
        assert loaded.dispatch_count == 0
