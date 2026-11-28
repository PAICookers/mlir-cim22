"""Compile a PyTorch Conv2d and ReLU to CIM22 and compare with eager PyTorch."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cim22.apps.framework_example import (
    cli,
    finish,
    group_result,
    header,
    make_inputs,
)
from cim22.backend import compilation_report

MODEL = "Conv2d(3→4, 3×3) + ReLU"


class SimpleModel(nn.Module):
    """A channel-first convolution followed by a rectified linear activation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3, 4, kernel_size=3, padding=1, dtype=torch.float32
        )
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(inputs))


def run(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compare three input groups with eager execution of the same model."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    output.mkdir(parents=True, exist_ok=True)
    header("PyTorch", runtime, MODEL)
    rng = np.random.default_rng(42)
    model = SimpleModel().eval()
    # Small integer-valued FP32 parameters keep these BF16 comparisons exact.
    with torch.no_grad():
        weights = rng.integers(
            -2, 3, size=model.conv.weight.shape, dtype=np.int8
        )
        bias = rng.integers(-3, 4, size=4, dtype=np.int32)
        model.conv.weight.copy_(torch.from_numpy(weights))
        model.conv.bias.copy_(torch.from_numpy(bias))
    inputs = make_inputs((1, 3, 16, 16), rng)
    options: dict[str, object] = {
        "runtime": runtime,
        "artifact_dir": output,
        "debug_dir": output / "ir",
    }
    if usb_config is not None:
        options["usb_config"] = usb_config
    started = time.perf_counter()
    compiled = torch.compile(
        model, backend="cim22", fullgraph=True, dynamic=False, options=options
    )
    rows = []
    with torch.inference_mode():
        for index, array in enumerate(inputs, start=1):
            value = torch.from_numpy(array).float()
            reference = model(value)
            actual = compiled(value)
            report = compilation_report(output)
            rows.append(
                group_result(
                    index,
                    reference.numpy(),
                    actual.numpy(),
                    report.dispatch_count if report else 0,
                    expected_dtype=np.float32,
                )
            )
    result = {
        "framework": "pytorch",
        "runtime": runtime,
        "model": MODEL,
        "compute_dtype": "bfloat16",
        "output_kind": "relu_activation",
        "compile_seconds": time.perf_counter() - started,
        "dispatch_count": rows[-1]["dispatch_count"],
        "cases": rows,
        "artifact_dir": str(output.resolve()),
        "ir_dir": str((output / "ir").resolve()),
    }
    return finish(result, output, result_json)


def main(
    output: Path | None = None,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Run the example with optional persistent artifact and report paths."""
    return run(
        output or Path("outputs/torch"),
        runtime=runtime,
        usb_config=usb_config,
        result_json=result_json,
    )


if __name__ == "__main__":
    cli(main, __doc__, Path("outputs/torch"), timestamped=False)
