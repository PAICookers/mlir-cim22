"""Native object publication must propagate open, write and rename failures."""

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from cim22.compiler import _emit_object
from cim22.runtime import MODEL_SUFFIX, Session

from ._generated import cim22_model_pb2 as proto

if TYPE_CHECKING:
    from .conftest import EmbeddedModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_runtime"),
]


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    """A minimal split program isolates the native compiler/file boundary."""
    path = tmp_path / "model.packet.mlir"
    path.write_text(
        """
module attributes {cim.split_program} {
  module @host attributes {cim.pipeline_part = "host"} {
    func.func @main(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %y = arith.addf %x, %x : tensor<2x3xf32>
      return %y : tensor<2x3xf32>
    }
  }
  module @device attributes {cim.pipeline_part = "device"} {}
}
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("destination", ["missing", "directory"])
def test_object_destination_failure(
    packet: Path, tmp_path: Path, destination: str, native_runtime: Path
) -> None:
    output = tmp_path / destination / "model.o"
    if destination == "directory":
        output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="cannot .* object file"):
        _emit_object(packet, output, native_runtime)
    assert not list(tmp_path.rglob("*.tmp-*"))
    if destination == "missing":
        assert not output.exists()
    else:
        assert output.is_dir()


@pytest.mark.skipif(
    sys.platform == "win32", reason="requires POSIX RLIMIT_FSIZE"
)
def test_object_write_failure_preserves_previous_output(
    packet: Path, tmp_path: Path, native_runtime: Path
) -> None:
    output = tmp_path / "model.o"
    output.write_bytes(b"previous object")
    # Limit only a child process: ignore SIGXFSZ so the file write returns EFBIG
    # and the native API has to propagate the error instead of being killed.
    script = """
import resource
import signal
import sys
from pathlib import Path
from cim22.compiler import _emit_object
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))
try:
    _emit_object(Path(sys.argv[1]), Path(sys.argv[2]),
                 Path(sys.argv[3]))
except RuntimeError as error:
    assert 'cannot write object file' in str(error), str(error)
else:
    raise AssertionError('native object write failure was swallowed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(packet),
            str(output),
            str(native_runtime),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes() == b"previous object"
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.parametrize(
    "visibility", ["", "private "], ids=["public", "private"]
)
def test_runtime_entry_preserves_input(packet: Path, visibility: str) -> None:
    packet.write_text(
        packet.read_text().replace(
            "func.func @main", f"func.func {visibility}@main"
        )
    )
    value = np.ones((2, 3), dtype=np.float32)
    with Session(packet) as session:
        np.testing.assert_array_equal(session(value)[0], np.full_like(value, 2))
    np.testing.assert_array_equal(value, np.ones_like(value))


def test_packet_parser_rejects_invalid_return(packet: Path) -> None:
    packet.write_text(
        packet.read_text().replace(
            ") -> tensor<2x3xf32>", ") -> tensor<1x3xf32>"
        )
    )
    with pytest.raises(RuntimeError, match="return"):
        Session(packet)


@pytest.mark.usefixtures("native_toolchain")
def test_multiple_kernels_retain_distinct_weight_resources(
    tmp_path: Path, embedded_model: "EmbeddedModel"
) -> None:
    """Per-kernel extraction retains every referenced weight and packet."""

    class TwoWeights(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(64, 32, bias=False)
            self.second = torch.nn.Linear(64, 16, bias=False)
            with torch.no_grad():
                self.first.weight.fill_(1.0)
                self.second.weight.fill_(2.0)

        def forward(
            self, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.first(value), self.second(value)

    compiled = torch.compile(
        TwoWeights().eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software", "artifact_dir": tmp_path},
    )
    # Supplier reconstruction of 64 raw BF16 0x0001 inputs gives 64/128
    # for the two unit/double-weight kernels, rather than IEEE matmul values.
    value = torch.full((1, 64), 2.0**-133)
    expected = (torch.full((1, 32), 64.0), torch.full((1, 16), 128.0))
    with torch.inference_mode():
        for actual, reference in zip(compiled(value), expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    model = embedded_model(artifact)
    assert len(model.kernels) == 2
    assert sorted(len(k.transaction.weights) for k in model.kernels) == [1, 2]
    assert (
        len({w.resource for k in model.kernels for w in k.transaction.weights})
        == 3
    )
    for kernel in model.kernels:
        transaction = kernel.transaction
        assert len(transaction.inputs) == len(transaction.weights)
        assert len(transaction.readbacks) == len(transaction.weights)
        assert sum(p.kind == proto.WEIGHT for p in transaction.packets) == len(
            transaction.weights
        )
        assert transaction.flits
    with Session(artifact) as session:
        assert session.num_transactions == 2
        for actual, reference in zip(
            session(value.numpy()), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, reference.numpy())
