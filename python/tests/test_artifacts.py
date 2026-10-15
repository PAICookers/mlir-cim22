"""Linux shared models: native execution without compiler or JSON sidecars."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from cim22 import compile_exported, load_compiled

if TYPE_CHECKING:
    from .conftest import EmbeddedModel, ModelLibrary


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

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, list[torch.Tensor]]]:
        # Destination-style bufferization must not reuse caller-owned storage.
        y = self.linear(x + x)
        return {"result": (y, [x.sum(), y.sum()])}


# Load only the relocated model/runtime DSOs, with frontend execution disabled.
# No compiler module, MLIR/compiler DSO or JIT may be loaded. Candidate
# arithmetic runs through native CIM dispatch.
RELOAD_SCRIPT = """
import importlib.abc
import sys
from pathlib import Path
from unittest.mock import patch

class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'cim22.compiler' or fullname == 'torch_mlir'
                or fullname.startswith('torch_mlir.')):
            raise AssertionError(f'reload imported compiler module: {fullname}')

sys.meta_path.insert(0, NoFrontend())
import torch
from cim22 import load_compiled

value = torch.ones(2, 3, 65)
value[:, 1] *= 2
value[:, 2] *= 4
original = value.clone()
expected = torch.tensor([131., 261., 521.])[None, :, None].expand(2, 3, 17)
with patch('subprocess.run', side_effect=AssertionError('frontend invoked')), \
     patch('json.loads', side_effect=AssertionError('runtime JSON read')):
    with load_compiled(Path(sys.argv[1])) as compiled:
        assert compiled.session.num_transactions > 0
        first = compiled(value)['result']
        assert isinstance(first, tuple) and isinstance(first[1], list)
        torch.testing.assert_close(first[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(first[1][0], value.sum(), rtol=0, atol=0)
        torch.testing.assert_close(first[1][1], expected.sum(), rtol=0, atol=0)
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert compiled.session.dispatch_count > 0
        with load_compiled(Path(sys.argv[1])) as another:
            torch.testing.assert_close(another(value)['result'][0], expected)
        try:
            compiled(torch.full_like(value, float('nan')))
        except RuntimeError as error:
            assert 'BF16' in str(error)
        else:
            raise AssertionError('runner error was swallowed')
        second = compiled(value * 2)['result'][0]
        torch.testing.assert_close(second, expected * 2 - 1, rtol=0, atol=0)
        third = compiled(value)['result'][0]
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        torch.testing.assert_close(first[0], third, rtol=0, atol=0)
        assert first[0].data_ptr() != second.data_ptr()
        for invalid in (value.double(), value[:, :2], value.transpose(0, 1)):
            try:
                compiled(invalid)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError('invalid static input was accepted')
    torch.testing.assert_close(first[0], expected, rtol=0, atol=0)
    try:
        compiled(value)
    except RuntimeError as error:
        assert 'closed' in str(error)
    else:
        raise AssertionError('closed callable was accepted')
assert 'torch_mlir' not in sys.modules
assert 'cim22.compiler' not in sys.modules
maps = Path('/proc/self/maps').read_text()
assert not any(name in maps for name in ('libMLIR', 'libLLVM', 'libcim22_compiler'))
"""


@pytest.fixture(scope="module")
def compiled_artifact(
    native_toolchain: None, tmp_path_factory: pytest.TempPathFactory
) -> tuple[torch.export.ExportedProgram, Path]:
    """Compile once; publication and reload tests share the immutable artifact."""
    artifact = tmp_path_factory.mktemp("artifacts") / "model.so"
    program = torch.export.export(
        StructuredLinear().eval(), (torch.ones(2, 3, 65),), strict=True
    )
    with compile_exported(program, artifact_path=artifact) as compiled:
        assert compiled.session.num_transactions > 0, (
            "Expected real CIM offload"
        )
    return program, artifact


def test_atomic_publication(
    compiled_artifact: tuple[torch.export.ExportedProgram, Path], tmp_path: Path
) -> None:
    program, artifact = compiled_artifact
    original = hashlib.sha256(artifact.read_bytes()).digest()
    with (
        patch("subprocess.run", side_effect=AssertionError("recompiled")),
        pytest.raises(FileExistsError),
    ):
        compile_exported(program, artifact_path=artifact)
    assert hashlib.sha256(artifact.read_bytes()).digest() == original
    incomplete = tmp_path / "incomplete.so"
    with (
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, stderr="failed"),
        ),
        pytest.raises(RuntimeError, match="import failed"),
    ):
        compile_exported(program, artifact_path=incomplete)
    assert not incomplete.exists()


def test_embedded_protobuf_contract(
    compiled_artifact: tuple[torch.export.ExportedProgram, Path],
    embedded_model: "EmbeddedModel",
    model_library: "ModelLibrary",
) -> None:
    _, path = compiled_artifact
    model = embedded_model(path)
    assert model.data_format_version == 3
    assert model.numeric_contract_version == 2
    assert list(model.inputs[0].dimensions) == [2, 3, 65]
    assert [list(value.dimensions) for value in model.outputs] == [
        [2, 3, 17],
        [],
        [],
    ]
    entries = model.output_schema.root.dictionary.entries
    assert [entry.key for entry in entries] == ["result"]
    assert entries[0].value.tuple.children[0].tensor_leaf == 0
    assert entries[0].value.tuple.children[1].list.children[1].tensor_leaf == 2
    assert model.kernels
    assert len({kernel.kernel_name for kernel in model.kernels}) == len(
        model.kernels
    )
    for kernel in model.kernels:
        transaction = kernel.transaction
        assert transaction.weights and transaction.flits
        assert all(
            len(weight.values) == 1024 and len(weight.words) == 256
            for weight in transaction.weights
        )
    # Native-generated transactions can be parsed/re-serialized by Python and
    # loaded back by C++, with future descriptive metadata safely ignored.
    with load_compiled(
        model_library(model.SerializeToString() + b"\xa0\x06\x01", 1)
    ) as loaded:
        assert loaded.session.num_transactions == len(model.kernels)


def test_host_scalar_model(
    tmp_path: Path, embedded_model: "EmbeddedModel"
) -> None:
    class HostOnly(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, object]:
            return {"z": (x + x, []), "a": [x * x], "": {}}

    value = torch.tensor(4.0)
    program = torch.export.export(HostOnly(), (value,), strict=True)
    shared = tmp_path / "scalar.so"
    with compile_exported(program, artifact_path=shared) as compiled:
        assert compiled.session.num_transactions == 0
        result = compiled(value)
        assert list(result) == ["z", "a", ""]
        assert result["z"][1] == [] and result[""] == {}
        torch.testing.assert_close(result["z"][0], value * 2)
        torch.testing.assert_close(result["a"][0], value * value)
    model = embedded_model(shared)
    assert list(model.inputs[0].dimensions) == []
    assert [
        entry.key for entry in model.output_schema.root.dictionary.entries
    ] == ["z", "a", ""]


def test_installed_wheel_reload_without_frontend(
    compiled_artifact: tuple[torch.export.ExportedProgram, Path], tmp_path: Path
) -> None:
    """A clean deployment env contains the wheel and runtime dependencies only."""
    project = Path(__file__).resolve().parents[1]
    deployment = tmp_path / "venv"
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(deployment)
    environment.pop("PYTHONPATH", None)
    commands = [
        [
            "uv",
            "sync",
            "--project",
            str(project),
            "--frozen",
            "--no-dev",
            "--no-install-project",
        ],
        [
            "uv",
            "build",
            "--project",
            str(project),
            "--wheel",
            "--out-dir",
            str(tmp_path / "wheels"),
        ],
    ]
    for command in commands:
        subprocess.run(
            command, env=environment, check=True, capture_output=True
        )
    wheel = next((tmp_path / "wheels").glob("*.whl"))
    python = deployment / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(wheel),
        ],
        env=environment,
        check=True,
        capture_output=True,
    )
    _, artifact = compiled_artifact
    shared = tmp_path / "model.so"
    shutil.copyfile(artifact, shared)
    runtime = tmp_path / "libcim22_runtime.so"
    shutil.copyfile(Path(os.environ["CIM22_RUNTIME_LIBRARY"]), runtime)
    environment["CIM22_OPT"] = "/nonexistent/optimizer"
    environment["CIM22_COMPILER_LIBRARY"] = "/nonexistent/compiler.so"
    environment["CIM22_RUNTIME_LIBRARY"] = str(runtime)
    script = (
        """
import importlib.util
import sys
from pathlib import Path
assert importlib.util.find_spec('torch_mlir') is None
import cim22
assert Path(cim22.__file__).is_relative_to(Path(sys.prefix))
"""
        + RELOAD_SCRIPT
    )
    result = subprocess.run(
        [str(python), "-c", script, str(shared)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
