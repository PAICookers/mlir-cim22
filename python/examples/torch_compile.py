"""Compile a small custom model and check software inference results."""

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
    """Compile the model and check fixed-weight results across three calls."""
    model = SimpleModel().cpu().eval()
    # Unit weights make the expected BF16 software results exact.
    with torch.no_grad():
        model.linear.weight.fill_(1)
        model.linear.bias.fill_(1)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    with torch.inference_mode():
        for input_value, expected_value in (
            (-1.0, 0.0),
            (1.0, 65.0),
            (2.0, 129.0),
        ):
            inputs = torch.full((1, 64), input_value)
            expected = torch.full((1, 16), expected_value)
            torch.testing.assert_close(
                compiled(inputs), expected, rtol=0, atol=0
            )
    print("CIM22 torch.compile software execution passed (0, 65, 129).")


if __name__ == "__main__":
    main()
