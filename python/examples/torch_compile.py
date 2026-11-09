"""Compile a deterministic BF16 Conv2d/ReLU/Linear model."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


class SimpleModel(nn.Module):
    """Run BF16 convolution and linear layers with an FP32 public boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1).bfloat16()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(4 * 8 * 8, 8).bfloat16()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the final linear activation as FP32 for Session ABI support."""
        hidden = inputs.to(torch.bfloat16)
        hidden = self.relu(self.conv(hidden))
        return self.linear(hidden.flatten(1)).float()


def _usb_config(path: Path | None) -> object | None:
    if path is None:
        return None
    from cim22 import CIMTimingConfig, UsbConfig

    fields = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fields, dict):
        raise ValueError("USB 配置必须是 JSON 对象")
    if "timing" in fields:
        timing = fields["timing"]
        if not isinstance(timing, dict):
            raise ValueError("timing 必须是 JSON 对象")
        fields["timing"] = CIMTimingConfig(**timing)
    return UsbConfig(**fields)


def _cases() -> tuple[tuple[str, torch.Tensor], ...]:
    ramp = torch.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 3, 8, 8)
    checker = torch.tensor([[-0.25, 0.5] * 4] * 8).repeat(1, 3, 1, 1)
    channel_bias = torch.cat(
        (
            torch.full((1, 1, 8, 8), 0.1),
            torch.full((1, 1, 8, 8), 0.2),
            torch.full((1, 1, 8, 8), 0.3),
        ),
        dim=1,
    )
    return (
        ("constant_positive", torch.full((1, 3, 8, 8), 0.25)),
        ("constant_negative", torch.full((1, 3, 8, 8), -0.25)),
        ("ramp", ramp),
        ("checkerboard", checker),
        ("channel_bias", channel_bias),
    )


def run(
    output: Path,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Run deterministic BF16 internal cases through PyTorch and CIM."""
    if runtime not in {"simulator", "usb"}:
        raise ValueError("runtime must be simulator or usb")
    output.mkdir(parents=True, exist_ok=True)
    model = SimpleModel().cpu().eval()
    with torch.no_grad():
        model.conv.weight.fill_(0.125)
        model.conv.bias.fill_(0.05)
        model.linear.weight.fill_(0.0625)
        model.linear.bias.fill_(0.1)
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
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case_name, inputs in _cases():
            reference = model(inputs)
            actual = compiled(inputs)
            torch.testing.assert_close(
                actual,
                reference,
                rtol=0,
                atol=0,
                msg=lambda message, name=case_name: f"{name}: {message}",
            )
            difference = (actual - reference).abs().max().item()
            rows.append(
                {
                    "name": case_name,
                    "reference": reference.tolist(),
                    "actual": actual.tolist(),
                    "max_abs_error": float(difference),
                }
            )
    compile_seconds = time.perf_counter() - started
    # shortcut: torch.compile hides dispatch_count; expose it when the backend
    # gains a stable public session hook.
    dispatch_count = None
    result: dict[str, Any] = {
        "framework": "pytorch",
        "runtime": runtime,
        "model": "BF16 Conv2d(3,4,3x3) -> ReLU -> Linear(256,8)",
        "compute_dtype": "bfloat16",
        "output_kind": "final_linear_activation",
        "compile_seconds": compile_seconds,
        "dispatch_count": dispatch_count,
        "cases": rows,
        "artifact_dir": str(output.resolve()),
        "ir_dir": str((output / "ir").resolve()),
    }
    destination = result_json or output / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"CIM22 torch.compile {runtime} BF16 Conv2d/ReLU/Linear checks passed.")
    print(f"Cases: {len(rows)}; dispatches: {dispatch_count}")
    print(f"Model, debug IR and result.json: {output.resolve()}")
    return result


def main(
    output: Path | None = None,
    *,
    runtime: str = "simulator",
    usb_config: object | None = None,
    result_json: Path | None = None,
) -> dict[str, Any]:
    """Compatibility entry point used by the example and Web worker."""
    return run(
        output or Path("outputs/torch"),
        runtime=runtime,
        usb_config=usb_config,
        result_json=result_json,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("simulator", "usb"), default="simulator")
    parser.add_argument("--output", type=Path, default=Path("outputs/torch"))
    parser.add_argument("--usb-config", type=Path)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "compile.log").open("w", encoding="utf-8") as log:
        import os

        saved_stderr = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            main(
                args.output,
                runtime=args.runtime,
                usb_config=_usb_config(args.usb_config),
                result_json=args.result_json,
            )
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
