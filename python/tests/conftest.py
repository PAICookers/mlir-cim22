"""Shared preflight for tests that actually compile and load Linux models."""

import ctypes
import importlib.util
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from cim22._generated import cim22_model_pb2 as proto

type ModelLibrary = Callable[[bytes, int], Path]
type EmbeddedModel = Callable[[Path], proto.ModelArtifact]


@pytest.fixture(scope="session")
def native_runtime() -> Path:
    """Require only execution dependencies for metadata/loader tests."""
    runtime = os.environ.get("CIM22_RUNTIME_LIBRARY")
    if runtime is None or not Path(runtime).is_file():
        pytest.fail("Set CIM22_RUNTIME_LIBRARY to libcim22_runtime.so")
    if shutil.which(os.environ.get("CIM22_CC", "cc")) is None:
        pytest.fail("Install a C linker or set CIM22_CC")
    return Path(runtime)


@pytest.fixture(scope="session")
def native_toolchain(native_runtime: Path) -> None:
    """Require frontend and compiler dependencies only for compilation tests."""
    if importlib.util.find_spec("torch_mlir") is None:
        pytest.fail(
            "Install the frontend with uv sync --locked --extra compiler"
        )
    if shutil.which(os.environ.get("CIM22_OPT", "mlir-cim22-opt")) is None:
        pytest.fail("Set CIM22_OPT to the built mlir-cim22-opt executable")
    compiler = Path(
        os.environ.get(
            "CIM22_COMPILER_LIBRARY",
            str(native_runtime.with_name("libcim22_compiler.so")),
        )
    )
    if not compiler.is_file():
        pytest.fail("Build cim22_compiler or set CIM22_COMPILER_LIBRARY")


class Descriptor(ctypes.Structure):
    """ctypes mirror of the versioned native model descriptor."""

    _fields_ = [
        ("abi_version", ctypes.c_uint64),
        ("data", ctypes.c_void_p),
        ("data_size", ctypes.c_uint64),
        ("invoke", ctypes.c_void_p),
    ]


@pytest.fixture
def embedded_model() -> EmbeddedModel:
    """Inspect embedded bytes while the model library remains loaded."""

    def read(path: Path) -> proto.ModelArtifact:
        """Read the same descriptor bytes as the native loader, with the DSO alive."""
        library = ctypes.CDLL(str(path))
        getter = library.cim22_model_v1
        getter.restype = ctypes.POINTER(Descriptor)
        descriptor = getter().contents
        assert descriptor.abi_version == 1
        return proto.ModelArtifact.FromString(
            ctypes.string_at(descriptor.data, descriptor.data_size)
        )

    return read


@pytest.fixture
def model_library(tmp_path: Path, native_runtime: Path) -> ModelLibrary:
    """Build one isolated inert model library for each metadata test."""

    def build(data: bytes, abi_version: int = 1) -> Path:
        """An inert trusted DSO fixture isolates metadata validation from codegen."""
        source = tmp_path / "fixture.c"
        source.write_text(
            """
#include <stdint.h>
#include <stdlib.h>
struct Model { uint64_t version; const void *data; uint64_t size;
                   void (*invoke)(void **); };
static const unsigned char data[] = {BYTES};
static void invoke(void **args) { abort(); }
static const struct Model model = {ABI_VERSION, data, SIZE, invoke};
const struct Model *cim22_model_v1(void) { return &model; }
    """.replace("BYTES", ",".join(map(str, data)) or "0")
            .replace("SIZE", str(len(data)))
            .replace("ABI_VERSION", str(abi_version)),
            encoding="utf-8",
        )
        shared = tmp_path / "fixture.so"
        subprocess.run(
            [
                os.environ.get("CIM22_CC", "cc"),
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(shared),
            ],
            check=True,
            capture_output=True,
        )
        return shared

    return build
