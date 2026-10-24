"""Integration with the selected wheels, LLVM Host, and supplier CIM model.

Run with an installed torch-mlir Release wheel and Python 3.12. No replacement
kernel or eager fallback is used. Fixed subnormal input encodings keep the
supplier's raw-exponent reconstruction finite with exact scalar expectations.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

import cim22.backend as adapter
from cim22 import Session
from cim22.compiler import _compile
from cim22.runtime import MODEL_SUFFIX

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
            self.linear.weight.fill_(0.5)
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
    # Raw BF16 0x0002 with weight 0.5 yields 64+1 across the K=64/1 tiles.
    value = torch.full((2, 3, 65), 2.0**-132)
    value[:, 1] *= 2
    value[:, 2] *= 0.5
    # Keep both inputs ordinary tensors, as when prepared by a DataLoader.
    # Inference-created tensors have different Dynamo dispatch-key guards.
    other_value = value * 2
    with (
        torch.inference_mode(),
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
        patch("torch.export.export", wraps=torch.export.export) as exporter,
    ):
        model(value)
        model.reset()
        compiled = torch.compile(
            model,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={"artifact_dir": str(tmp_path), "runtime": "software"},
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
        assert exporter.call_count == 1
        products = list(tmp_path.glob(f"*{MODEL_SUFFIX}"))
        assert len(products) == 1
        assert products[0].suffix == MODEL_SUFFIX
        with products[0].open("rb") as stream:
            assert stream.read(2 if MODEL_SUFFIX == ".dll" else 4) == (
                b"MZ" if MODEL_SUFFIX == ".dll" else b"\x7fELF"
            )
        assert first.data_ptr() != second.data_ptr()
        assert isinstance(model.v, torch.Tensor)


def test_host_shape_specialization_and_cache() -> None:
    def host_only(value: torch.Tensor) -> torch.Tensor:
        return value * value + value

    scalar = torch.tensor(4.0)
    values = [scalar, torch.ones(2, 3), scalar]
    with (
        torch.inference_mode(),
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
    ):
        compiled = torch.compile(
            host_only,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={"runtime": "software"},
        )
        for value in values:
            torch.testing.assert_close(compiled(value), host_only(value))
        assert compiler.call_count == 2


def test_host_relu_deallocation() -> None:
    """Host selects require Arith's promised buffer-deallocation interface."""
    values = torch.tensor([[-2.0, 0.0, 3.0]])
    with torch.inference_mode():
        compiled = torch.compile(
            torch.relu,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={"runtime": "software"},
        )
        torch.testing.assert_close(compiled(values), torch.relu(values))


def test_layer_norm_decomposition() -> None:
    """AOT decomposes composite operators while preserving affine parameters."""
    model = torch.nn.LayerNorm(4).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        model.bias.copy_(torch.tensor([1.0, -1.0, 0.0, 0.5]))
    values = torch.tensor([[1.0, 2.0, 4.0, 8.0], [3.0, 3.0, 3.0, 3.0]])
    with torch.inference_mode():
        compiled = torch.compile(
            model,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={"runtime": "software"},
        )
        torch.testing.assert_close(compiled(values), model(values))


def test_runtime_weights_stay_on_host(tmp_path: Path) -> None:
    def linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, weight)

    value = torch.ones(1, 64)
    weight = torch.ones(16, 64)
    compiled = torch.compile(
        linear,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software", "artifact_dir": tmp_path},
    )
    with (
        torch.inference_mode(),
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
    ):
        for scale in (1, 2):
            weight.fill_(scale)
            torch.testing.assert_close(
                compiled(value, weight), linear(value, weight)
            )
        assert compiler.call_count == 1
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    with Session(artifact) as session:
        assert session.num_transactions == 0


def test_backend_requires_inference_context() -> None:
    compiled = torch.compile(
        lambda value: value * 2,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    with (
        torch.enable_grad(),
        pytest.raises(BackendCompilerFailed, match="no_grad"),
    ):
        compiled(torch.ones(2, 3))


@pytest.mark.parametrize(
    "operation", ["add_", "mul_", "relu_", "copy_", "view", "slice"]
)
def test_inplace_input_writeback(operation: str) -> None:
    class Mutating(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
            if operation == "add_":
                value.add_(2)
            elif operation == "mul_":
                value.mul_(2)
            elif operation == "relu_":
                value.relu_()
            elif operation == "copy_":
                value.copy_(value * 2 + 1)
            elif operation == "view":
                value.view(-1).add_(2)
            else:
                value[:, 1].add_(2)
            return {"result": value * 3, "input": value, "view": value.view(-1)}

    value = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    expected_input = value.clone()
    model = Mutating()
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    with (
        torch.inference_mode(),
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
    ):
        for _ in range(2):
            expected = model(expected_input)
            actual = compiled(value)
            torch.testing.assert_close(value, expected_input)
            torch.testing.assert_close(actual["result"], expected["result"])
            assert actual["input"] is value
            assert (
                actual["view"].untyped_storage().data_ptr()
                == value.untyped_storage().data_ptr()
            )
        assert compiler.call_count == 1


@pytest.mark.parametrize("state_kind", ["buffer", "parameter"])
def test_mutable_state_and_cim_weights(
    inference_context: AbstractContextManager[None],
    state_kind: str,
    tmp_path: Path,
) -> None:
    class Accumulator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 16)
            if state_kind == "buffer":
                self.register_buffer("state", torch.zeros(1, 16))
            else:
                self.state = torch.nn.Parameter(torch.zeros(1, 16))
            with torch.no_grad():
                self.linear.weight.fill_(1)
                self.linear.bias.fill_(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.state.add_(self.linear(x))
            return self.state * 2

    model = Accumulator().eval()
    # Raw BF16 0x0001 and unit weights reconstruct a 64-item total of 64.
    value = torch.full((1, 64), 2.0**-133)
    original = value.clone()
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software", "artifact_dir": tmp_path},
    )
    with (
        inference_context,
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
    ):
        first = compiled(value)
        torch.testing.assert_close(first, torch.full((1, 16), 130.0))
        torch.testing.assert_close(model.state, torch.full((1, 16), 65.0))
        second = compiled(value)
        torch.testing.assert_close(second, torch.full((1, 16), 260.0))
        torch.testing.assert_close(model.state, torch.full((1, 16), 130.0))
        torch.testing.assert_close(first, torch.full((1, 16), 130.0))
        model.state.zero_()
        torch.testing.assert_close(compiled(value), first)
        assert compiler.call_count == 1
    torch.testing.assert_close(value, original)
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    with Session(artifact) as session:
        assert session.num_transactions > 0


def test_duplicate_input_writeback(
    inference_context: AbstractContextManager[None],
) -> None:
    def mutating(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first.add_(2)
        return second * 3

    value = torch.ones(2, 2)
    expected_value = value.clone()
    compiled = torch.compile(
        mutating,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    with (
        inference_context,
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
    ):
        for _ in range(2):
            expected = mutating(expected_value, expected_value)
            actual = compiled(value, value)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(value, expected_value)
        assert compiler.call_count == 1


@pytest.mark.parametrize("mutate", [False, True], ids=["view", "data-mutation"])
def test_parameter_output_alias(
    inference_context: AbstractContextManager[None], mutate: bool
) -> None:
    class ParameterOutput(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if mutate:
                self.weight.data.add_(value)
            return self.weight * 3, self.weight, self.weight.view(-1)

    model = ParameterOutput()
    eager = ParameterOutput()
    value = torch.ones(2, 2)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    with (
        inference_context,
        patch.object(adapter, "_compile", wraps=_compile) as compiler,
    ):
        for _ in range(2):
            expected = eager(value)
            actual = compiled(value)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(model.weight, eager.weight)
            assert actual[1] is model.weight
            assert (
                actual[2].untyped_storage().data_ptr()
                == model.weight.data_ptr()
            )
        assert compiler.call_count == 1


def test_failed_compilation_can_retry() -> None:
    model = torch.nn.Linear(64, 16).eval()
    value = torch.full((1, 64), 2.0**-133)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(1)
        with (
            patch.object(
                adapter, "_compile", side_effect=RuntimeError("retry")
            ),
            pytest.raises(BackendCompilerFailed, match="retry"),
        ):
            compiled(value)
        model.weight.add_(1)
        torch.testing.assert_close(compiled(value), torch.full((1, 16), 129.0))
