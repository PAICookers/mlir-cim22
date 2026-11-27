"""Compare an INT8 JAX Dense/ReLU model with NumPy, ONNX Runtime and CIM22."""

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cim22.apps.framework_example import (
    cli,
    dense_data,
    finish,
    group_result,
    header,
)

if TYPE_CHECKING:
    from cim22.runtime import UsbConfig


def run(
    output: Path,
    *,
    runtime: Literal["simulator", "usb"] = "simulator",
    usb_config: "UsbConfig | None" = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compile once and compare each group with all three software references."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    header("JAX", runtime, "Dense(64→16) + ReLU")

    import jax
    import jax.numpy as jnp
    import onnxruntime as ort

    from cim22.compiler import BackendOptions
    from cim22.jax import compile

    weights, bias, inputs = dense_data()
    jax_weights, jax_bias = jnp.asarray(weights), jnp.asarray(bias)

    def model(x: jax.Array) -> jax.Array:
        product = jnp.matmul(x, jax_weights, preferred_element_type=jnp.int32)
        return jax.nn.relu(product + jax_bias)

    ir_root = output / "ir"
    ir_root.mkdir(parents=True, exist_ok=True)
    # Select the ONNX and packet files from this compilation on repeated runs.
    run_ir = Path(tempfile.mkdtemp(prefix="run-", dir=ir_root))
    options: BackendOptions = {
        "runtime": runtime,
        "artifact_dir": output,
        "debug_dir": run_ir,
    }
    if usb_config is not None:
        options["usb_config"] = usb_config
    signature = (jax.ShapeDtypeStruct((4, 64), jnp.int8),)
    cases = []
    started = time.perf_counter()
    with compile(model, signature, options=options) as compiled:
        compile_seconds = time.perf_counter() - started
        (source,) = run_ir.glob("*/model.onnx")
        ort_session = ort.InferenceSession(
            str(source), providers=["CPUExecutionProvider"]
        )
        (input_spec,) = ort_session.get_inputs()
        # MatMul is the only contraction; check it was mapped to device IR.
        packet = source.with_name("model.packet.mlir")
        device_ir = (
            packet.read_text(encoding="utf-8")
            .partition("module @__cim_device ")[2]
            .partition("\n  }")[0]
        )
        dispatch_ops = sum(
            line.lstrip().startswith("cim.dispatch ")
            for line in device_ir.splitlines()
        )
        for index, value in enumerate(inputs, start=1):
            x = jnp.asarray(value)
            reference = np.asarray(model(x))
            expected = np.maximum(
                value.astype(np.int32) @ weights.astype(np.int32) + bias, 0
            )
            (exported,) = ort_session.run(None, {input_spec.name: value})
            actual = np.asarray(compiled(x))
            row = group_result(
                index,
                reference,
                actual,
                compiled.session.dispatch_count,
                numpy_reference=expected,
                onnx_runtime=exported,
            )
            row["onnx_max_abs_error"] = (
                int(np.abs(exported.astype(np.int64) - reference).max())
                if exported.shape == reference.shape
                and exported.dtype == np.int32
                else None
            )
            row["passed"] = row["passed"] and dispatch_ops > 0
            cases.append(row)

    result = {
        "framework": "jax",
        "runtime": runtime,
        "compile_seconds": compile_seconds,
        "dispatch_count": cases[-1]["dispatch_count"],
        "cases": cases,
        "artifact_dir": str(output.resolve()),
        "ir_dir": str(source.parent.resolve()),
        "onnx_path": str(source.resolve()),
        "packet_ir": str(packet.resolve()),
        "dispatch_ops": dispatch_ops,
    }
    return finish(result, output, result_json)


def main(
    output: Path,
    *,
    runtime: Literal["simulator", "usb"] = "simulator",
    usb_config: "UsbConfig | None" = None,
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
    cli(main, __doc__, Path("outputs/jax"))
