"""Compile a deterministic two-layer INT8 TensorFlow model."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _usb_config(path: Path | None) -> object | None:
    if path is None:
        return None
    from cim22 import CIMTimingConfig, UsbConfig

    fields = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fields, dict):
        raise ValueError("USB configuration must be a JSON object")
    if "timing" in fields:
        timing = fields["timing"]
        if not isinstance(timing, dict):
            raise ValueError("timing must be a JSON object")
        fields["timing"] = CIMTimingConfig(**timing)
    return UsbConfig(**fields)


def run(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Run the fixed INT8 TensorFlow example and persist a JSON result."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    logger.info("Starting TensorFlow example (runtime=%s)", runtime)
    logger.info(
        "Model: rank-2 INT8 MatMul -> clipped ReLU -> MatMul; "
        "inputs: 1 random case (seed=42)"
    )
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    from cim22.tensorflow import compile

    output.mkdir(parents=True, exist_ok=True)
    tf.get_logger().setLevel("ERROR")
    rng = np.random.default_rng(42)
    w1 = rng.integers(-3, 4, (16, 32), dtype=np.int8)
    w2 = rng.integers(-3, 4, (32, 16), dtype=np.int8)

    @tf.function(input_signature=[tf.TensorSpec([4, 16], tf.int8)])
    def model(x: tf.Tensor) -> tf.Tensor:
        hidden = tf.raw_ops.BatchMatMulV3(x=x, y=tf.constant(w1), Tout=tf.int32)
        hidden = tf.cast(tf.clip_by_value(hidden, 0, 127), tf.int8)
        return tf.raw_ops.BatchMatMulV3(
            x=hidden, y=tf.constant(w2), Tout=tf.int32
        )

    value = rng.integers(-4, 5, (4, 16), dtype=np.int8)
    inputs = tf.constant(value)
    reference = model(inputs).numpy()
    options: dict[str, object] = {
        "runtime": runtime,
        "artifact_dir": output,
        "debug_dir": output / "ir",
    }
    if usb_config is not None:
        options["usb_config"] = usb_config
    logger.info("Compiling and running inference")
    started = time.perf_counter()
    with compile(model, options=options) as compiled:
        actual = compiled(inputs).numpy()
        dispatch_count = compiled.session.dispatch_count
    compile_seconds = time.perf_counter() - started
    error = int(np.max(np.abs(actual.astype(np.int64) - reference)))
    np.testing.assert_array_equal(actual, reference)
    if dispatch_count <= 0:
        raise RuntimeError("TensorFlow inference produced no CIM dispatch")
    logger.info(
        "Output matches reference: case=int8_linear_clip_linear, "
        "max_abs_error=%d",
        error,
    )
    result: dict[str, Any] = {
        "framework": "tensorflow",
        "runtime": runtime,
        "compile_seconds": compile_seconds,
        "dispatch_count": dispatch_count,
        "cases": [
            {
                "name": "int8_linear_clip_linear",
                "reference": reference.tolist(),
                "actual": actual.tolist(),
                "max_abs_error": error,
                "dispatch_count": dispatch_count,
            }
        ],
        "artifact_dir": str(output.resolve()),
        "ir_dir": str((output / "ir").resolve()),
    }
    destination = result_json or output / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Completed inference (cases=1, outputs match reference)")
    logger.info("CIM dispatch count: %d", dispatch_count)
    logger.info("Model directory: %s", output)
    logger.info("IR directory: %s", output / "ir")
    logger.info("Results saved to %s", destination)
    return result


def main(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compatibility entry point used by the example and Web worker."""
    return run(
        output,
        runtime=runtime,
        usb_config=usb_config,
        result_json=result_json,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", choices=("simulator", "usb"), default="simulator"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--usb-config", type=Path)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    args = _parse_args()
    output = args.output or Path(
        "outputs/tensorflow"
    ) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "compile.log").open("a", encoding="utf-8") as log:
        saved_stderr = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            main(
                output,
                runtime=args.runtime,
                usb_config=_usb_config(args.usb_config),
                result_json=args.result_json,
            )
        except Exception:
            logger.error(
                "Example failed; see traceback and %s", output / "compile.log"
            )
            raise
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
