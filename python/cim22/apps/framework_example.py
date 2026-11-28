"""Data, result checks and CLI output shared by the three framework examples."""

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cim22.runtime import CIMTimingConfig, UsbConfig
from cim22.web.storage import save_json

logger = logging.getLogger(__name__)


def make_inputs(
    shape: tuple[int, ...], rng: np.random.Generator
) -> tuple[npt.NDArray[np.int8], ...]:
    """Return zero, repeating -4..4, and seeded random INT8 groups."""
    return (
        np.zeros(shape, dtype=np.int8),
        (np.arange(math.prod(shape)) % 9 - 4).astype(np.int8).reshape(shape),
        rng.integers(-4, 5, shape, dtype=np.int8),
    )


def dense_data() -> tuple[
    npt.NDArray[np.int8],
    npt.NDArray[np.int32],
    tuple[npt.NDArray[np.int8], ...],
]:
    """Generate identical parameters and inputs for both Dense frontends."""
    rng = np.random.default_rng(42)
    weights = rng.integers(-3, 4, (64, 16), dtype=np.int8)
    bias = rng.integers(-3, 4, 16, dtype=np.int32)
    return weights, bias, make_inputs((4, 64), rng)


def header(framework: str, runtime: str, model: str) -> None:
    logger.info(
        "Framework: %s | Backend: %s | Model: %s", framework, runtime, model
    )


def group_result(
    index: int,
    reference: np.ndarray,
    actual: np.ndarray,
    dispatch_count: int,
    *,
    expected_dtype: type[np.int32] | type[np.float32] = np.int32,
    **oracles: np.ndarray,
) -> dict[str, Any]:
    """Record exact output equality separately from dispatch evidence."""
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    outputs = (actual, *oracles.values())
    correct = reference.dtype == expected_dtype and all(
        value.dtype == expected_dtype and np.array_equal(value, reference)
        for value in outputs
    )
    max_abs_error = None
    if (
        actual.dtype == reference.dtype == expected_dtype
        and actual.shape == reference.shape
    ):
        wide_dtype = np.float64 if expected_dtype == np.float32 else np.int64
        max_abs_error = (
            np.abs(actual.astype(wide_dtype) - reference).max().item()
        )
    row = {
        "name": f"Group {index}",
        "reference": reference.tolist(),
        "actual": actual.tolist(),
        "actual_dtype": str(actual.dtype),
        "max_abs_error": max_abs_error,
        "dispatch_count": dispatch_count,
        "correct": correct,
        "passed": correct and dispatch_count > 0,
        **{name: array.tolist() for name, array in oracles.items()},
    }
    logger.info(
        "%s | %s | CIM dispatches: %d",
        row["name"],
        "Correct" if correct else "Incorrect",
        dispatch_count,
    )
    return row


class _VerificationFailed(RuntimeError):
    """All groups finished and the final result contains a failed check."""


def finish(
    result: dict[str, Any], output: Path, result_json: Path | None
) -> dict[str, Any]:
    """Persist all groups, including failures, before returning the verdict."""
    cases = result["cases"]
    passed = sum(row["passed"] for row in cases)
    result["passed_groups"] = passed
    destination = result_json or output / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(destination, result)
    logger.info("%d/%d passed", passed, len(cases))
    if passed != len(cases):
        raise _VerificationFailed(f"Only {passed}/{len(cases)} groups passed")
    return result


def cli(
    main: Callable[..., object],
    description: str | None,
    default_dir: str | Path,
    *,
    timestamped: bool = True,
) -> None:
    """Run one example and send its five-line report to CLI and Web logs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--runtime", choices=("simulator", "usb"), default="simulator"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--usb-config", type=Path)
    args = parser.parse_args()
    output = args.output or Path(default_dir)
    if args.output is None and timestamped:
        output /= datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    config = None
    if args.usb_config:
        fields = json.loads(args.usb_config.read_text(encoding="utf-8"))
        if not isinstance(fields, dict):
            raise ValueError("USB configuration must be a JSON object")
        if "timing" in fields:
            if not isinstance(fields["timing"], dict):
                raise ValueError("timing must be a JSON object")
            fields["timing"] = CIMTimingConfig(**fields["timing"])
        config = UsbConfig(**fields)

    # Web already directs stdout to compile.log. Avoid a second copy, and
    # retain third-party native diagnostics separately from the group report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with (
        (output / "compile.log").open("a", encoding="utf-8") as log,
        tempfile.TemporaryFile() as diagnostics,
    ):
        handlers = [logging.StreamHandler(sys.stdout)]
        if not os.path.samestat(
            os.fstat(sys.stdout.fileno()), os.fstat(log.fileno())
        ):
            handlers.append(logging.StreamHandler(log))
        old_level, old_propagate = logger.level, logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in handlers:
            logger.addHandler(handler)
        saved = os.dup(2)
        try:
            os.dup2(diagnostics.fileno(), 2)
            main(
                output,
                runtime=args.runtime,
                usb_config=config,
                result_json=args.result_json,
            )
        except _VerificationFailed:
            raise SystemExit(1) from None
        except Exception:
            diagnostics.seek(0)
            failure = output / "diagnostics.log"
            failure.write_text(
                diagnostics.read().decode("utf-8", errors="replace")
                + traceback.format_exc(),
                encoding="utf-8",
            )
            logger.error("Example failed. See %s", failure)
            raise SystemExit(1) from None
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            for handler in handlers:
                logger.removeHandler(handler)
            logger.setLevel(old_level)
            logger.propagate = old_propagate
