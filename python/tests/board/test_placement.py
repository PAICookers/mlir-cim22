"""Exercise physical placement, Session reuse and Host-to-CIM data flow."""

import re
from collections import defaultdict
from itertools import pairwise

import numpy as np
import pytest
import torch

from .._generated import cim22_model_pb2 as proto
from .conftest import BoardCheck, BoardExpectation
from .test_int8 import Int8Matmul

pytestmark = [pytest.mark.board, pytest.mark.integration]


def _single_macro_int8(artifact: proto.ModelArtifact, packet: str) -> None:
    (kernel,) = artifact.kernels
    (group,) = kernel.transaction.groups
    (work,) = group.works
    assert (work.core_slot, work.macro_slot) == (0, 0)
    assert {weight.data_type for weight in kernel.transaction.weights} == {
        proto.INT8
    }


def _int8_slot_reuse(artifact: proto.ModelArtifact, packet: str) -> None:
    slots: dict[tuple[int, int], list[tuple[int, tuple[int, ...]]]] = (
        defaultdict(list)
    )
    assert len(artifact.kernels) == 2
    for index, kernel in enumerate(artifact.kernels):
        for weight in kernel.transaction.weights:
            assert weight.data_type == proto.INT8
            slots[(weight.core_slot, weight.macro_slot)].append(
                (index, tuple(weight.words))
            )
    assert len(slots) == 40
    assert any(
        first_index != second_index and first_words != second_words
        for visits in slots.values()
        for (first_index, first_words), (
            second_index,
            second_words,
        ) in pairwise(visits)
    )


def _int8_board_case(
    board_check: BoardCheck, columns: int, expectation: BoardExpectation
) -> None:
    k = torch.arange(64)[:, None]
    n = torch.arange(columns)[None, :]
    weight = (k * (n % 5 + 1) + n * 3).remainder(13).sub(6).to(torch.int8)
    x = (torch.arange(64)[None, :] % 17 - 8).to(torch.int8)
    inputs = [x, x.flip(1).contiguous()]
    expected = [
        (
            value.numpy().astype(np.int64) @ weight.numpy().astype(np.int64)
        ).astype(np.int32)
        for value in inputs
    ]
    assert not np.array_equal(*expected)
    if columns > 640:
        assert not np.array_equal(expected[0][:, :16], expected[0][:, -16:])
    board_check(
        Int8Matmul(weight).eval(), inputs, expected, expectation=expectation
    )


def test_int8_single_macro_residency_and_reopen(
    board_check: BoardCheck,
) -> None:
    """Reuse the same decoded model and physical slot after reopening."""
    _int8_board_case(
        board_check,
        16,
        BoardExpectation(
            _single_macro_int8, 1, reopen=True, verify_weights="residency"
        ),
    )


def test_int8_multiple_transactions(board_check: BoardCheck) -> None:
    """Cross array capacity and require different weights on a reused slot."""
    _int8_board_case(
        board_check,
        656,
        BoardExpectation(_int8_slot_reuse, 2, verify_weights="overwrite"),
    )


def _single_macro_bf16(artifact: proto.ModelArtifact, packet: str) -> None:
    (kernel,) = artifact.kernels
    (weight,) = kernel.transaction.weights
    assert weight.data_type == proto.BF16


def test_bf16_single_transaction_uncompensated(board_check: BoardCheck) -> None:
    """Check the chip's raw exponent convention without compensation branches."""
    layer = torch.nn.Linear(64, 16, bias=False).eval()
    weight = np.zeros((16, 64), np.float32)
    weight[np.arange(16), np.arange(16)] = 2**-32
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
    mantissas = (
        1 + torch.arange(64, dtype=torch.float32)[None, :].remainder(4) / 4
    )
    x = mantissas * 2**-32
    values = [x, x.flip(1).contiguous()]
    board_check(
        layer,
        values,
        # Each lane has one product: 64 * (64 * mantissa), with raw
        # exponents 95 + 95. Normalization adds 1 + (12 - 7), yielding 196.
        [
            np.ldexp(np.ldexp(value.numpy()[:, :16], 32), 196 - 127)
            for value in values
        ],
        expectation=BoardExpectation(
            _single_macro_bf16, 1, compensation="none"
        ),
    )


class HostHandoff(torch.nn.Module):
    """Feed a sliced, reduced first CIM output directly to the next CIM layer."""

    def __init__(self, first: np.ndarray, second: np.ndarray) -> None:
        super().__init__()
        self.first = torch.nn.Linear(65, 17, bias=False)
        self.second = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.first.weight.copy_(torch.from_numpy(first))
            self.second.weight.copy_(torch.from_numpy(second))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value)[:, 1:])


def _host_handoff(artifact: proto.ModelArtifact, packet: str) -> None:
    assert len(artifact.kernels) == 8
    host = packet.split("module @__cim_host", 1)[1].split(
        "module @__cim_device", 1
    )[0]
    main = host.split("func.func @main(", 1)[1].split("func.func private", 1)[0]
    graph_calls = 0
    transferred: set[str] = set()
    found_slice = False
    for line in main.splitlines():
        result, equals, operation = line.partition("=")
        if not equals:
            continue
        (value, *_) = re.findall(r"%[\w.]+", result)
        inputs = set(re.findall(r"%[\w.]+", operation))
        if "call @__cim_resident_linear" in operation:
            graph_calls += 1
            if graph_calls == 1:
                transferred.add(value)
            else:
                assert found_slice and inputs <= transferred
        elif (
            operation.lstrip().startswith("tensor.extract_slice")
            and "[0, 1] [1, 16]" in operation
        ):
            assert graph_calls == 1 and inputs & transferred
            found_slice = True
            transferred = {value}
        elif inputs & transferred:
            transferred.add(value)
    assert graph_calls == 2 and found_slice
    assert re.search(r"arith.addf[^\n]*tensor<16xf32>", host)
    assert all(
        weight.data_type == proto.BF16
        for kernel in artifact.kernels
        for weight in kernel.transaction.weights
    )


def test_bf16_host_handoff(board_check: BoardCheck) -> None:
    """Validate a K-tail contribution and the nonzero-offset Host slice."""
    first = np.zeros((17, 65), np.float32)
    first[np.arange(17), np.arange(17)] = 0.5
    first[:, 64] = (np.arange(17) % 3) * np.float32(0.125)
    second = np.zeros((16, 16), np.float32)
    second[np.arange(16), (np.arange(16) + 3) % 16] = -0.5
    x = torch.arange(65, dtype=torch.float32)[None, :].remainder(7) - 3
    values = [x, x.flip(1).contiguous()]
    expected = [(value.numpy() @ first.T)[:, 1:] @ second.T for value in values]
    assert not np.array_equal(*expected)
    assert not np.array_equal(
        expected[0], (x.numpy() @ first.T)[:, :-1] @ second.T
    )
    assert not np.array_equal(
        expected[0], (x.numpy()[:, :64] @ first[:, :64].T)[:, 1:] @ second.T
    )
    board_check(
        HostHandoff(first, second).eval(),
        values,
        expected,
        expectation=BoardExpectation(_host_handoff, 8),
    )
