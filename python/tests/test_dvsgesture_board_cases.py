"""Offline production-simulator and independent-oracle checks for DVS cases."""

import numpy as np
import pytest
import torch

from .board import test_dvsgesture as dvs
from .board.conftest import Array, BoardCheck, assert_output
from .test_board_cases import simulator_check as simulator_check


@pytest.mark.parametrize(
    "name,case", dvs.DVS_CASES, ids=[name for name, _ in dvs.DVS_CASES]
)
def test_dvs_independent_oracle(name: str, case: dvs.Case) -> None:
    """Validate stimulus/oracle separately from the compiler and its simulator."""
    del name

    def check(
        model: torch.nn.Module,
        inputs: list[torch.Tensor],
        expected: list[Array] | None,
    ) -> None:
        with torch.inference_mode():
            for index, value in enumerate(inputs):
                actual = model(value).numpy()
                assert np.isfinite(actual).all()
                if expected is not None:
                    assert_output(actual, expected[index])

    case(check)


@pytest.mark.integration
@pytest.mark.parametrize(
    "name,case", dvs.DVS_CASES, ids=[name for name, _ in dvs.DVS_CASES]
)
def test_dvs_production_simulator(
    simulator_check: BoardCheck, name: str, case: dvs.Case
) -> None:
    """Compile the board builders fullgraph with sign-split and actual CIM work."""
    del name
    case(simulator_check)
