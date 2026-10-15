"""Native object publication must propagate open, write and rename failures."""

import marshal
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from torch.utils import _pytree

from cim22._generated import cim22_model_pb2 as proto
from cim22._schema import encode_outputs
from cim22.compiler import _emit_object
from cim22.runtime import Session

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
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
    packet: Path, tmp_path: Path, destination: str
) -> None:
    output = tmp_path / destination / "model.o"
    if destination == "directory":
        output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="cannot .* object file"):
        _emit_object(
            packet,
            output,
            encode_outputs(_pytree.tree_structure(0)),
            Path(os.environ["CIM22_RUNTIME_LIBRARY"]),
        )
    assert not list(tmp_path.rglob("*.tmp-*"))
    if destination == "missing":
        assert not output.exists()
    else:
        assert output.is_dir()


def test_object_write_failure_preserves_previous_output(
    packet: Path, tmp_path: Path
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
from cim22.runtime import Session
from cim22._generated import cim22_model_pb2 as proto
from cim22._schema import encode_outputs
from torch.utils import _pytree
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))
try:
    _emit_object(Path(sys.argv[1]), Path(sys.argv[2]),
                 encode_outputs(_pytree.tree_structure(0)), Path(sys.argv[3]))
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
            os.environ["CIM22_RUNTIME_LIBRARY"],
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes() == b"previous object"
    assert not list(tmp_path.glob("*.tmp-*"))


def deep_schema() -> bytes:
    """Fits standalone OutputSchema, but exceeds the enclosing model depth."""
    schema = proto.OutputSchema(version=1)
    node = schema.root
    for _ in range(48):
        node = node.list.children.add()
    node.dictionary.entries.add(key="x").value.tensor_leaf = 0
    return schema.SerializeToString()


@pytest.mark.parametrize(
    "schema",
    [
        b"",
        marshal.dumps((1, (0,)), 4),
        b"\x08\x01\x12\x02\x08\x01",
        deep_schema(),
    ],
    ids=["missing", "marshal", "ordinal", "model-depth"],
)
def test_invalid_schema_preserves_output(
    packet: Path, tmp_path: Path, schema: bytes
) -> None:
    output = tmp_path / "model.o"
    output.write_bytes(b"previous object")
    with pytest.raises(RuntimeError, match="Protobuf model.*recompile"):
        _emit_object(
            packet, output, schema, Path(os.environ["CIM22_RUNTIME_LIBRARY"])
        )
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
