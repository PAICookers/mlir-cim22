"""Dynamo inference, internal model metadata and atomic diagnostic publication."""

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import torch

from cim22 import Session
from cim22.compiler import _compile

from ._generated import cim22_model_pb2 as proto

if TYPE_CHECKING:
    from .conftest import EmbeddedModel


type LinearOutputs = dict[str, tuple[torch.Tensor, list[torch.Tensor]]]
type CompiledGraph = tuple[Callable[[torch.Tensor], LinearOutputs], Path]


pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


class StructuredLinear(torch.nn.Module):
    """Unit-weight Linear plus scalar Host results in nested containers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(65, 17)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(1.0)

    def forward(self, x: torch.Tensor) -> LinearOutputs:
        # Destination-style bufferization must not reuse caller-owned storage.
        y = self.linear(x + x)
        return {"result": (y, [x.sum(), y.sum()])}


@pytest.fixture(scope="module")
def compiled_graph(
    native_toolchain: None, tmp_path_factory: pytest.TempPathFactory
) -> CompiledGraph:
    """Share one immutable Dynamo specialization and its diagnostic artifact."""
    folder = tmp_path_factory.mktemp("artifacts")
    compiled = torch.compile(
        StructuredLinear().eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software", "artifact_dir": folder},
    )
    with torch.inference_mode():
        compiled(torch.ones(2, 3, 65))
    (artifact,) = folder.glob("*.so")
    return compiled, artifact


def test_inference_ownership_and_error_recovery(
    compiled_graph: CompiledGraph,
) -> None:
    compiled, _ = compiled_graph
    value = torch.ones(2, 3, 65)
    value[:, 1] *= 2
    value[:, 2] *= 4
    original = value.clone()
    expected = torch.tensor([131.0, 261.0, 521.0])[None, :, None].expand(
        2, 3, 17
    )
    with torch.inference_mode():
        first = compiled(value)["result"]
        assert isinstance(first, tuple) and isinstance(first[1], list)
        torch.testing.assert_close(first[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(first[1][0], value.sum(), rtol=0, atol=0)
        torch.testing.assert_close(first[1][1], expected.sum(), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="BF16"):
            compiled(torch.full_like(value, float("nan")))
        second = compiled(value * 2)["result"][0]
        third = compiled(value)["result"][0]
        torch.testing.assert_close(second, expected * 2 - 1, rtol=0, atol=0)
        torch.testing.assert_close(first[0], third, rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert first[0].data_ptr() != second.data_ptr()


def test_atomic_publication(
    compiled_graph: CompiledGraph, tmp_path: Path
) -> None:
    _, artifact = compiled_graph
    original = hashlib.sha256(artifact.read_bytes()).digest()
    graph = torch.fx.symbolic_trace(torch.nn.Identity())
    with (
        patch("subprocess.run", side_effect=AssertionError("recompiled")),
        pytest.raises(FileExistsError),
    ):
        _compile(
            graph, torch.ones(2), runtime="software", artifact_path=artifact
        )
    assert hashlib.sha256(artifact.read_bytes()).digest() == original
    incomplete = tmp_path / "incomplete.so"
    with (
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, stderr="failed"),
        ),
        pytest.raises(RuntimeError, match="CIM lowering failed"),
    ):
        _compile(
            graph, torch.ones(2), runtime="software", artifact_path=incomplete
        )
    assert not incomplete.exists()


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ((), ValueError, "Input count"),
        ((np.zeros((2, 3, 65), dtype=np.float64),), TypeError, "FP32"),
        ((np.zeros((2, 3, 64), dtype=np.float32),), ValueError, "shape"),
        (
            (np.zeros((2, 3, 130), dtype=np.float32)[..., ::2],),
            ValueError,
            "contiguous",
        ),
        (
            (
                np.ndarray(
                    (2, 3, 65),
                    dtype=np.float32,
                    buffer=bytearray(1561),
                    offset=1,
                ),
            ),
            ValueError,
            "aligned",
        ),
    ],
    ids=["count", "dtype", "shape", "layout", "alignment"],
)
def test_session_rejects_invalid_inputs(
    compiled_graph: CompiledGraph,
    inputs: tuple[npt.NDArray[np.float32 | np.float64], ...],
    error: type[Exception],
    message: str,
) -> None:
    _, artifact = compiled_graph
    with Session(artifact) as session:
        with pytest.raises(error, match=message):
            session(*inputs)
        assert session.dispatch_count == 0


def test_embedded_protobuf_contract(
    compiled_graph: CompiledGraph,
    embedded_model: "EmbeddedModel",
) -> None:
    _, path = compiled_graph
    model = embedded_model(path)
    assert model.data_format_version == 1
    assert model.runtime_backend == proto.SOFTWARE
    assert list(model.inputs[0].dimensions) == [2, 3, 65]
    assert [list(value.dimensions) for value in model.outputs] == [
        [2, 3, 17],
        [],
        [],
    ]
    assert model.kernels
