"""BF16 control-scale cases for the public FP32 Session path.

The dimensions mirror the recovered runtime-scale suite.  Values are finite
dyadics so the expected result is independently calculable while every case
still exercises the current sign-split compensation path.
"""

from collections.abc import Callable

import pytest
import torch

from .conftest import BoardCheck

pytestmark = [pytest.mark.board, pytest.mark.integration]

type Case = Callable[[BoardCheck], None]


def _matmul_case(
    board_check: BoardCheck,
    *,
    k: int,
    n: int,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    model = torch.nn.Linear(k, n, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(weight)
    values = [inputs, -inputs]
    opposite = torch.where(expected == 0, torch.zeros_like(expected), -expected)
    references = [expected.numpy(), opposite.numpy()]
    board_check(model, values, references)


def two_macro(board_check: BoardCheck) -> None:
    """N=32 places two 16-lane Macro tiles on one input row."""
    inputs = torch.ones(1, 64)
    weight = torch.full((32, 64), 0.5)
    _matmul_case(
        board_check,
        k=64,
        n=32,
        inputs=inputs,
        weight=weight,
        expected=torch.full((1, 32), 32.0),
    )


def multi_core_batch(board_check: BoardCheck) -> None:
    """M=3 and N=48 require multiple input rows and cores."""
    inputs = torch.tensor([[1.0] * 64, [-1.0] * 64, [0.0] * 64])
    weight = torch.full((48, 64), 0.5)
    expected = torch.tensor([[32.0] * 48, [-32.0] * 48, [0.0] * 48])
    _matmul_case(
        board_check, k=64, n=48, inputs=inputs, weight=weight, expected=expected
    )


def k_reduce(board_check: BoardCheck) -> None:
    """K=128 requires two CIM partials followed by Host reduction."""
    inputs = torch.ones(1, 128)
    weight = torch.full((16, 128), 0.5)
    _matmul_case(
        board_check,
        k=128,
        n=16,
        inputs=inputs,
        weight=weight,
        expected=torch.full((1, 16), 64.0),
    )


def tail(board_check: BoardCheck) -> None:
    """K=65 and N=17 exercise both padding and output-lane tails."""
    inputs = torch.ones(2, 65)
    inputs[:, -1] = 2
    inputs[1].neg_()
    weight = torch.full((17, 65), 0.5)
    expected = torch.tensor([[33.0] * 17, [-33.0] * 17])
    _matmul_case(
        board_check, k=65, n=17, inputs=inputs, weight=weight, expected=expected
    )


def combined(board_check: BoardCheck) -> None:
    """M=3, N=48 and K=128 combine core, Macro and K-reduction pressure."""
    inputs = torch.tensor([[1.0] * 128, [-1.0] * 128, [0.0] * 128])
    weight = torch.full((48, 128), 0.5)
    expected = torch.tensor([[64.0] * 48, [-64.0] * 48, [0.0] * 48])
    _matmul_case(
        board_check,
        k=128,
        n=48,
        inputs=inputs,
        weight=weight,
        expected=expected,
    )


def capacity_reuse(board_check: BoardCheck) -> None:
    """N=672 emits 42 tiles, exceeding the 40 physical Macro slots."""
    inputs = torch.ones(1, 64)
    lane = torch.arange(672)
    # Reused Macro slots must get different encoded weights, including exponents.
    scale = torch.pow(2.0, (lane // 16) % 7 - 3)
    weight = torch.zeros(672, 64)
    weight[lane, (lane + lane // 16) % 64] = scale
    _matmul_case(
        board_check,
        k=64,
        n=672,
        inputs=inputs,
        weight=weight,
        expected=scale.unsqueeze(0),
    )


def reduction_order(board_check: BoardCheck) -> None:
    """Three K tiles distinguish one final BF16 round from premature rounds."""
    inputs = torch.ones(2, 129)
    inputs[1].neg_()
    weight = torch.zeros((17, 129))
    weight[:, 0] = 1.0
    weight[:, 64] = 1.0 / 256.0
    weight[:, 128] = 1.0 / 256.0
    expected = torch.tensor([[1.0078125] * 17, [-1.0078125] * 17])
    _matmul_case(
        board_check,
        k=129,
        n=17,
        inputs=inputs,
        weight=weight,
        expected=expected,
    )


SCALE_CASES: tuple[tuple[str, Case], ...] = (
    ("two-macro", two_macro),
    ("multi-core-batch", multi_core_batch),
    ("k-reduce", k_reduce),
    ("tail", tail),
    ("combined", combined),
    ("capacity-reuse", capacity_reuse),
    ("reduction-order", reduction_order),
)


@pytest.mark.parametrize(
    "name,case", SCALE_CASES, ids=[item[0] for item in SCALE_CASES]
)
def test_bf16_scale_case(
    board_check: BoardCheck, name: str, case: Case
) -> None:
    """Run one recovered control-scale geometry through simulator then USB."""
    del name
    case(board_check)
