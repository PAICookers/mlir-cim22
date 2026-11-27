"""Compare an INT8 PyTorch convolution and classifier with CIM22 inference."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cim22.apps.framework_example import (
    cli,
    finish,
    group_result,
    header,
    make_inputs,
)
from cim22.backend import compilation_report

MODEL = "Conv2d(3→4, 3×3) + Clipped ReLU + Linear(1024→8)"


class SimpleModel(nn.Module):
    """Use two INT8 matrix multiplies with INT32 biases and accumulation."""

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__()

        def parameter(
            shape: tuple[int, ...], dtype: type[np.int8] | type[np.int32]
        ) -> nn.Parameter:
            values = rng.integers(-3, 4, shape, dtype=dtype)
            return nn.Parameter(torch.from_numpy(values), requires_grad=False)

        self.conv_weight = parameter((27, 4), np.int8)
        self.conv_bias = parameter((4,), np.int32)
        self.linear_weight = parameter((1024, 8), np.int8)
        self.linear_bias = parameter((8,), np.int32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map each padded 3×3 window to one row in the first matrix multiply."""
        padded = F.pad(inputs, (1, 1, 1, 1))
        windows = torch.stack(
            [
                padded[:, :, dy : dy + 16, dx : dx + 16]
                for dy in range(3)
                for dx in range(3)
            ],
            dim=-1,
        )
        patches = windows.permute(0, 2, 3, 1, 4).reshape(256, 27)
        hidden = torch._int_mm(patches.contiguous(), self.conv_weight)
        hidden = (hidden + self.conv_bias).clamp(0, 127).to(torch.int8)
        # The classifier consumes the same channel-first order as conv2d.
        features = hidden.reshape(1, 16, 16, 4).permute(0, 3, 1, 2).flatten(1)
        return (
            torch._int_mm(features.contiguous(), self.linear_weight)
            + self.linear_bias
        )

    def reference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Check patch ordering independently using native INT32 CPU conv2d."""
        kernel = self.conv_weight.T.reshape(4, 3, 3, 3).int()
        hidden = F.conv2d(inputs.int(), kernel, self.conv_bias, padding=1)
        return F.linear(
            hidden.clamp(0, 127).flatten(1),
            self.linear_weight.T.int(),
            self.linear_bias,
        )


def run(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compare each of three input groups with an independent PyTorch model."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    output.mkdir(parents=True, exist_ok=True)
    header("PyTorch", runtime, MODEL)
    rng = np.random.default_rng(42)
    model = SimpleModel(rng).eval()
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
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options=options,
    )
    rows = []
    with torch.inference_mode():
        for index, array in enumerate(inputs, start=1):
            value = torch.from_numpy(array)
            reference = model.reference(value)
            actual = compiled(value)
            report = compilation_report(output)
            rows.append(
                group_result(
                    index,
                    reference.numpy(),
                    actual.numpy(),
                    report.dispatch_count if report else 0,
                )
            )
    result = {
        "framework": "pytorch",
        "runtime": runtime,
        "model": MODEL,
        "compute_dtype": "int8",
        "output_kind": "final_linear_activation",
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
