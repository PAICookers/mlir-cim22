"""FP32 public ABI with internal BF16 CIM, compensation and Host reduction.

Exact dyadic cases have an independent mathematical reference. General values
check USB/simulator consistency, not eager accuracy.
"""

import numpy as np
import pytest
import torch

from .conftest import BoardCheck

pytestmark = [pytest.mark.board, pytest.mark.integration]


def test_signed_bias_and_repeat(board_check: BoardCheck) -> None:
    """Check four sign branches, bias once, zeros and input-cache replacement."""
    model = torch.nn.Linear(64, 16).eval()
    with torch.no_grad():
        model.weight[:, ::2] = 0.5
        model.weight[:, 1::2] = -0.25
        model.bias.fill_(1)
    value = torch.ones(1, 64)
    value[:, 1::2] = -2
    inputs = [value, -value, torch.zeros_like(value), value]
    expected = [np.full((1, 16), x, np.float32) for x in (33, -31, 1, 33)]
    board_check(model, inputs, expected)


def test_disjoint_support(board_check: BoardCheck) -> None:
    """A zero dot product must survive the chip converter's nonzero zero code."""
    model = torch.nn.Linear(64, 16, bias=False).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[:, ::2] = 0.5
    value = torch.zeros(1, 64)
    value[:, 1::2] = 1
    board_check(model, [value], [np.zeros((1, 16), np.float32)])


def test_tails_lanes_and_k_sum(board_check: BoardCheck) -> None:
    """Distinct powers of two expose lane routing, padding and K64 boundaries."""
    model = torch.nn.Linear(65, 17, bias=False).eval()
    scale = torch.pow(2.0, torch.arange(17) - 8)
    with torch.no_grad():
        model.weight.copy_(scale[:, None].expand(17, 65))
    value = torch.ones(2, 65)
    value[:, -1] = 2
    value[1].neg_()
    expected = torch.stack((66 * scale, -66 * scale)).numpy()
    board_check(
        model,
        [value, -value, value],
        [expected, -expected, expected],
    )


def test_exponents_and_fractions(board_check: BoardCheck) -> None:
    """Dyadic mixed exponents/fractions remain exactly representable here."""
    model = torch.nn.Linear(64, 16, bias=False).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[:, 0] = 1.5
        model.weight[:, 1] = 0.25
    value = torch.zeros(1, 64)
    value[:, 0] = 1
    value[:, 1] = 0.125
    board_check(model, [value], [np.full((1, 16), 1.53125, np.float32)])


def test_general_numeric_differential(board_check: BoardCheck) -> None:
    """Non-dyadic mixed signs/exponents compare the two production runners."""
    generator = torch.Generator().manual_seed(22)
    model = torch.nn.Linear(65, 17, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.randn(17, 65, generator=generator) * 0.3)
    value = torch.randn(2, 65, generator=generator) * torch.pow(
        2.0, torch.arange(65) % 7 - 3
    )
    board_check(model, [value, -value, value], None)
