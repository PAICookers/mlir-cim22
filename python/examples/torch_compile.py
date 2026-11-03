"""Compile an original model with the default corrected CIM BF16 pipeline."""

import torch
from torch import nn


class SimpleModel(nn.Module):
    """Apply a learned affine transform followed by ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 16)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map contiguous CPU FP32 inputs from 64 features to 16."""
        return self.relu(self.linear(inputs))


def main() -> None:
    """Check exact normal inputs and automatic signed-input preparation."""
    model = SimpleModel().cpu().eval()
    with torch.no_grad():
        model.linear.weight.fill_(0.5)
        model.linear.bias.fill_(1)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "simulator"},
    )
    # Powers of two make the expected corrected values independently exact.
    cases = (
        ("negative_relu", -1.0, -1.0, 0.0),
        ("positive", 1.0, 1.0, 33.0),
        ("double_input", 2.0, 2.0, 65.0),
        ("quadruple_input", 4.0, 4.0, 129.0),
        ("mixed_sign", -1.0, 2.0, 63.5),
    )
    prepared = []
    for name, first, other, expected in cases:
        value = torch.full((1, 64), other)
        value[0, 0] = first
        prepared.append((name, value, expected))
    with torch.inference_mode():
        for case_name, inputs, expected_value in prepared:
            expected = torch.full((1, 16), expected_value)
            torch.testing.assert_close(
                compiled(inputs),
                expected,
                rtol=0,
                atol=0,
                msg=lambda message, name=case_name: f"{name}: {message}",
            )
    print("CIM22 torch.compile sign-split BF16 checks passed.")


if __name__ == "__main__":
    main()
