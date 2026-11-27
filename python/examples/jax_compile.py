"""Compare a fixed JAX Dense/ReLU model with ONNX Runtime and CIM22."""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from cim22.runtime import UsbConfig

logger = logging.getLogger(__name__)


def run(
    output: Path,
    *,
    runtime: Literal["simulator", "usb"] = "simulator",
    usb_config: "UsbConfig | None" = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compile once and run both oracles against the selected native target."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    import jax
    import jax.numpy as jnp
    import onnxruntime as ort

    from cim22.compiler import BackendOptions
    from cim22.jax import compile

    weights = jnp.full((64, 16), 0.5, dtype=jnp.float32)
    bias = jnp.ones(16, dtype=jnp.float32)

    def model(x: jax.Array) -> jax.Array:
        return jax.nn.relu(x @ weights + bias)

    value = np.ones((2, 64), dtype=np.float32)
    value[1] = -1
    inputs = jnp.asarray(value)
    reference = np.asarray(model(inputs))
    # These dyadic values exercise both ReLU branches and are exact in BF16.
    expected = np.broadcast_to(np.array([[33], [0]], dtype=np.float32), (2, 16))
    np.testing.assert_array_equal(reference, expected, strict=True)

    ir_root = output / "ir"
    ir_root.mkdir(parents=True, exist_ok=True)
    # An explicit --output may be shared or reused across independent runs.
    run_ir = Path(tempfile.mkdtemp(prefix="run-", dir=ir_root))
    options: BackendOptions = {
        "runtime": runtime,
        "artifact_dir": output,
        "debug_dir": run_ir,
    }
    if usb_config is not None:
        options["usb_config"] = usb_config
    logger.info("Compiling FP32 relu(x @ W + b) for %s", runtime)
    signature = (jax.ShapeDtypeStruct(value.shape, value.dtype),)
    started = time.perf_counter()
    with compile(model, signature, options=options) as compiled:
        compile_seconds = time.perf_counter() - started
        (source,) = run_ir.glob("*/model.onnx")
        ort_session = ort.InferenceSession(
            str(source), providers=["CPUExecutionProvider"]
        )
        (input_spec,) = ort_session.get_inputs()
        (exported,) = ort_session.run(None, {input_spec.name: value})
        np.testing.assert_array_equal(exported, reference, strict=True)

        # MatMul is the only contraction in this graph. Its mapping must exist
        # in packet IR even when a Host implementation yields identical values.
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
        if not dispatch_ops:
            raise RuntimeError(
                f"Matrix multiply has no cim.dispatch in {packet}"
            )
        actual = np.asarray(compiled(inputs))
        dispatch_count = compiled.session.dispatch_count
        if not dispatch_count:
            raise RuntimeError(f"{runtime} executed no CIM dispatch")
        np.testing.assert_array_equal(actual, reference, strict=True)

    result: dict[str, Any] = {
        "framework": "jax",
        "runtime": runtime,
        "compile_seconds": compile_seconds,
        "dispatch_count": dispatch_count,
        "cases": [
            {
                "name": "fp32_dense_relu",
                "reference": reference.tolist(),
                "onnx_runtime": exported.tolist(),
                "actual": actual.tolist(),
                "onnx_max_abs_error": float(
                    np.max(np.abs(exported - reference))
                ),
                "max_abs_error": float(np.max(np.abs(actual - reference))),
                "dispatch_count": dispatch_count,
            }
        ],
        "artifact_dir": str(output.resolve()),
        "ir_dir": str(source.parent.resolve()),
        "onnx_path": str(source.resolve()),
        "packet_ir": str(packet.resolve()),
        "dispatch_ops": dispatch_ops,
    }
    destination = result_json or output / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "JAX, ONNX Runtime and %s agree; %d CIM dispatches; results in %s",
        runtime,
        dispatch_count,
        destination,
    )
    return result


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", choices=("simulator", "usb"), default="simulator"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--usb-config", type=Path)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    output = args.output or Path("outputs/jax") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    output.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    # The Web worker already sends stdout/stderr to compile.log.
    with (output / "compile.log").open("a", encoding="utf-8") as log:
        handler = logging.StreamHandler(
            sys.stdout
            if os.path.samestat(
                os.fstat(sys.stdout.fileno()), os.fstat(log.fileno())
            )
            else log
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        if handler.stream is log:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        try:
            usb_config = None
            if args.usb_config:
                from cim22 import CIMTimingConfig, UsbConfig

                fields = json.loads(args.usb_config.read_text(encoding="utf-8"))
                if not isinstance(fields, dict):
                    raise ValueError("USB configuration must be a JSON object")
                if "timing" in fields:
                    fields["timing"] = CIMTimingConfig(**fields["timing"])
                usb_config = UsbConfig(**fields)
            main(
                output,
                runtime=args.runtime,
                usb_config=usb_config,
                result_json=args.result_json,
            )
        except Exception:
            logger.exception("JAX example failed")
            raise
