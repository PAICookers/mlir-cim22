"""Signed INT8 USB inference against an independent wide-integer dot product."""

import numpy as np
import pytest
import torch

from .conftest import BoardCheck

pytestmark = [pytest.mark.board, pytest.mark.integration]


class Int8Matmul(torch.nn.Module):
    """Freeze distinct signed lanes for typed i8 x i8 -> i32 capture."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Multiply contiguous CPU INT8 inputs by the fixed matrix."""
        return torch._int_mm(value, self.weight)


@pytest.mark.parametrize(
    ("reduction", "columns", "wide"),
    [(64, 16, False), (65, 17, False), (128, 48, False), (128, 16, True)],
    ids=["tile", "k-n-tails", "multi-core", "wide-host-sum"],
)
def test_int8_matmul(
    board_check: BoardCheck, reduction: int, columns: int, wide: bool
) -> None:
    """Exercise signed endpoints, zero, repeated input and Host K reduction."""
    weight = (
        (torch.arange(reduction * columns).reshape(reduction, columns) * 7 % 15)
        - 7
    ).to(torch.int8)
    value = (torch.arange(2 * reduction).reshape(2, reduction) % 256 - 128).to(
        torch.int8
    )
    if wide:
        weight.fill_(127)
        value.fill_(127)
    inputs = [
        value,
        torch.full_like(value, -128),
        torch.zeros_like(value),
        value,
    ]
    reference = [
        (x.numpy().astype(np.int64) @ weight.numpy().astype(np.int64)).astype(
            np.int32
        )
        for x in inputs
    ]
    board_check(Int8Matmul(weight).eval(), inputs, reference)
