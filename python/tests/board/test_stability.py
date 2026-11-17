"""Repeated pointwise convolution with distinct, exactly representable inputs."""

import numpy as np
import pytest
import torch

from .conftest import BoardCheck

pytestmark = [pytest.mark.board, pytest.mark.integration]


class Pointwise(torch.nn.Module):
    """Expose nonnegative inputs to the compiler before the convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1)
        with torch.no_grad():
            self.conv.weight.zero_()
            for lane in range(16):
                scale = 2.0 ** (lane % 5 - 2)
                self.conv.weight[lane, (lane * 3) % 16] = 0.5 * scale
                self.conv.weight[lane, (lane * 3 + 1) % 16] = -0.25 * scale
                self.conv.bias[lane] = (lane % 7 - 3) * 0.125

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv((value >= 0).to(value.dtype))


def test_pointwise_repeated(
    board_check: BoardCheck, board_repeats: int
) -> None:
    """Check every call while reusing one Session across changing inputs."""
    model = Pointwise().eval()
    rng = np.random.default_rng(220407)
    values = rng.integers(-2, 3, size=(6, 1, 16, 4, 80)).astype(np.float32)
    weight = model.conv.weight.detach().numpy()[:, :, 0, 0]
    bias = model.conv.bias.detach().numpy()[None, :, None, None]
    expected = np.einsum("snchw,oc->snohw", values >= 0, weight) + bias
    assert not np.array_equal(expected[0], expected[1])
    order = [index % len(values) for index in range(board_repeats)]
    board_check(
        model,
        [torch.from_numpy(values[index]) for index in order],
        [expected[index] for index in order],
    )
