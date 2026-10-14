"""Integration with the pinned wheels, LLVM Host, and candidate CIM runner.

Run with Python 3.12 and CIM22_OPT/CIM22_RUNTIME_LIBRARY set. No replacement
kernel or eager fallback is used; unit weights make expected numbers exact
under the candidate model.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import cim22.backend as adapter
from cim22.compiler import compile_exported

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


@pytest.fixture(autouse=True)
def reset_dynamo() -> Iterator[None]:
    """Do not let compilation guards or cached graphs leak between tests."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class ResetSequence(torch.nn.Module):
    """Shared Linear with temporal recurrence and a plain Python reset state."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(65, 17)
        self.v: float | torch.Tensor = 0.0
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(1.0)

    def reset(self) -> None:
        self.v = 0.0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        current = self.linear(value)
        if isinstance(self.v, float):
            self.v = torch.full_like(current[0].data, self.v)
        spikes = []
        for frame in current:
            self.v = (self.v + frame) * 0.5
            spike = (self.v >= 48.0).to(current.dtype)
            self.v = self.v * (1.0 - spike)
            spikes.append(spike)
        return torch.stack(spikes)


def test_shared_batch_temporal_reset_and_cache(tmp_path: Path) -> None:
    model = ResetSequence().eval()
    value = torch.ones(2, 3, 65)
    value[:, 1] *= 2
    value[:, 2] *= 0.5
    # Keep both inputs ordinary tensors, as when prepared by a DataLoader.
    # Inference-created tensors have different Dynamo dispatch-key guards.
    other_value = value * 2
    with (
        torch.inference_mode(),
        patch.object(
            adapter, "compile_exported", wraps=compile_exported
        ) as compiler,
    ):
        model(value)
        model.reset()
        compiled = torch.compile(
            model,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={"artifact_dir": str(tmp_path)},
        )
        first = compiled(value)
        assert torch.is_inference(first)
        expected = torch.ones(2, 3, 17)
        expected[0, 0] = 0
        expected[:, 2] = 0
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        model.reset()
        second = compiled(other_value)
        expected_second = torch.ones_like(second)
        expected_second[0, 2] = 0
        torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
        model.reset()
        third = compiled(value)
        torch.testing.assert_close(first, third, rtol=0, atol=0)
        assert compiler.call_count == 1
        products = list(tmp_path.iterdir())
        assert len(products) == 1
        assert products[0].suffix == ".so"
        with products[0].open("rb") as stream:
            assert stream.read(4) == b"\x7fELF"
        assert first.data_ptr() != second.data_ptr()
        assert isinstance(model.v, torch.Tensor)
        model.linear.weight.add_(1)
        model.reset()
        with pytest.raises(ValueError, match="changed; recompile"):
            compiled(value)


def test_host_only_multiple_outputs() -> None:
    class HostOnly(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"value": x * 2.0 + 3.0, "sum": x.sum()}

    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with torch.inference_mode():
        compiled = torch.compile(
            HostOnly(), backend="cim22", fullgraph=True, dynamic=False
        )
        actual = compiled(value)
        torch.testing.assert_close(actual["value"], value * 2.0 + 3.0)
        torch.testing.assert_close(actual["sum"], value.sum())
