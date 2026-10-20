"""Compile a small model and check the adopted CIM22 software reference."""

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
    """Check fixed-weight results, repeated calls and signed prealignment."""
    model = SimpleModel().cpu().eval()
    with torch.no_grad():
        model.linear.weight.fill_(0.01)
        model.linear.bias.fill_(1)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    # Fixed supplier-model BF16 goldens; positive outputs absorb bias 1 at
    # FP32 precision. Each input row is [first_value, other_value, ...].
    # The mixed case aligns to [-117, 82, ...], giving INT21=414018 -> 0x7f4a;
    # the old magnitude-first shift would instead produce 0x7f4e.
    cases = (
        ("negative_relu", -0.01, -0.01, 0x0000),
        ("positive", 0.01, 0.01, 0x7E52),
        ("double_input", 0.02, 0.02, 0x7ED2),
        ("quadruple_input", 0.04, 0.04, 0x7F52),
        ("mixed_sign_exponents", -0.01, 0.04, 0x7F4A),
    )
    with torch.inference_mode():
        for case_name, first_value, other_value, expected_bits in cases:
            inputs = torch.full((1, 64), other_value)
            inputs[0, 0] = first_value
            expected = (
                torch.full((1, 16), expected_bits, dtype=torch.int16)
                .view(torch.bfloat16)
                .float()
            )
            torch.testing.assert_close(
                compiled(inputs),
                expected,
                rtol=0,
                atol=0,
                msg=lambda message, name=case_name: f"{name}: {message}",
            )
    print("CIM22 torch.compile supplier-model checks passed.")


if __name__ == "__main__":
    main()
