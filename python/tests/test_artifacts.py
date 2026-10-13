"""Linux shared models: native execution without compiler or JSON sidecars."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cim22 import compile_exported, load_compiled

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
        y = self.linear(x)
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
expected = torch.tensor([66., 131., 261.])[None, :, None].expand(2, 3, 17)
with patch('subprocess.run', side_effect=AssertionError('frontend invoked')), \
     patch('json.loads', side_effect=AssertionError('runtime JSON read')):
    with load_compiled(Path(sys.argv[1])) as compiled:
        assert compiled.session.num_transactions > 0
        first = compiled(value)['result']
        assert isinstance(first, tuple) and isinstance(first[1], list)
        torch.testing.assert_close(first[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(first[1][0], value.sum(), rtol=0, atol=0)
        torch.testing.assert_close(first[1][1], expected.sum(), rtol=0, atol=0)
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
assert 'libMLIR.so' not in maps and 'libcim22_compiler.so' not in maps
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


def test_relocated_reload_without_frontend(
    compiled_artifact: tuple[torch.export.ExportedProgram, Path], tmp_path: Path
) -> None:
    _, artifact = compiled_artifact
    shared = tmp_path / "model.so"
    shutil.copyfile(artifact, shared)
    runtime = tmp_path / "libcim22_runtime.so"
    shutil.copyfile(Path(os.environ["CIM22_RUNTIME_LIBRARY"]), runtime)
    environment = os.environ.copy()
    environment["CIM22_OPT"] = "/nonexistent/optimizer"
    environment["CIM22_COMPILER_LIBRARY"] = "/nonexistent/compiler.so"
    environment["CIM22_RUNTIME_LIBRARY"] = str(runtime)
    result = subprocess.run(
        [sys.executable, "-c", RELOAD_SCRIPT, str(shared)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.parametrize(
    ("version", "data_version", "message"),
    [(99, 2, "ABI"), (1, 1, "recompile"), (1, 2, "data")],
    ids=["incompatible-abi", "old-arithmetic", "truncated-data"],
)
def test_invalid_library(
    tmp_path: Path, version: int, data_version: int, message: str
) -> None:
    source = tmp_path / "invalid.c"
    source.write_text(
        """
#include <stdint.h>
struct Model { uint64_t version; const void *data; uint64_t size;
               void (*invoke)(void **); };
static const unsigned char data[8] = {DATA_VERSION};
static void invoke(void **args) {}
static const struct Model model = {VERSION, data, sizeof(data), invoke};
const struct Model *cim22_model_v1(void) { return &model; }
""".replace("DATA_VERSION", str(data_version)).replace("VERSION", str(version)),
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.so"
    subprocess.run(
        [
            os.environ.get("CIM22_CC", "cc"),
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(invalid),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(RuntimeError, match=message):
        load_compiled(invalid)
