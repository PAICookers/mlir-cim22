"""Shared preflight for tests that actually compile and load native models."""

import _ctypes
import ctypes
import importlib.util
import os
import shutil
import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib.resources import files
from pathlib import Path

import pytest
import torch

from cim22.runtime import compiler_library, runtime_library

from ._generated import cim22_model_pb2 as proto

type ModelLibrary = Callable[[bytes, int], Path]
type EmbeddedModel = Callable[[Path], proto.ModelArtifact]


@pytest.fixture(
    params=[torch.no_grad, torch.inference_mode], ids=["no-grad", "inference"]
)
def inference_context(
    request: pytest.FixtureRequest,
) -> AbstractContextManager[None]:
    """Exercise both PyTorch inference paths for mutation and state tests."""
    return request.param()


@pytest.fixture(scope="session")
def native_runtime() -> Path:
    """Require only execution dependencies for metadata/loader tests."""
    if (
        shutil.which(
            os.environ.get("CIM22_CC", "cl" if os.name == "nt" else "cc")
        )
        is None
    ):
        pytest.fail("Install a C linker or set CIM22_CC")
    return runtime_library()


@pytest.fixture(scope="session")
def native_toolchain(native_runtime: Path) -> None:
    """Require frontend and compiler dependencies only for compilation tests."""
    if importlib.util.find_spec("torch_mlir") is None:
        pytest.fail(
            "Install the torch-mlir Release wheel before running compiler tests"
        )
    optimizer = os.environ.get("CIM22_OPT") or str(
        files("cim22")
        / "bin"
        / ("mlir-cim22-opt.exe" if os.name == "nt" else "mlir-cim22-opt")
    )
    if shutil.which(optimizer) is None:
        pytest.fail("Set CIM22_OPT to the built mlir-cim22-opt executable")
    compiler_library(native_runtime)


@pytest.fixture
def host_artifact() -> proto.ModelArtifact:
    model = proto.ModelArtifact(
        schema_version=1,
        backend=proto.SOFTWARE,
    )
    model.outputs.add(data_type=proto.FP32)  # Rank-zero is valid.
    return model


@pytest.fixture
def transaction_artifact(
    host_artifact: proto.ModelArtifact,
) -> proto.ModelArtifact:
    model = proto.ModelArtifact()
    model.CopyFrom(host_artifact)
    transaction = model.kernels.add(kernel_name="kernel").transaction
    transaction.target_profile = "cim22-4x5-v1"
    group = transaction.groups.add(group_id=0)
    group.works.add(work_id=0, core_slot=0, macro_slot=0, route=[0] * 6)
    transaction.weights.add(
        group_id=0,
        work_id=0,
        core_slot=0,
        macro_slot=0,
        resource="weight",
        route=[0] * 6,
        words=[0] * 256,
        values=bytes(1024),
        exponents=bytes(16),
        data_type=proto.BF16,
    )
    transaction.inputs.add(
        group_id=0,
        work_id=0,
        macro_slot=0,
        input_slot=0,
        input_cache_address=0,
        data_type=proto.BF16,
    )
    transaction.readbacks.add(
        group_id=0,
        work_id=0,
        macro_slot=0,
        output_cache_address=7,
        route=[0] * 6,
        test_core=[0] * 3,
        data_type=proto.BF16,
    )
    for kind in (
        proto.WEIGHT,
        proto.WEIGHT_EXPONENT,
        proto.INPUT_CACHE_WRITE,
        proto.RETURN_ROUTE,
        proto.CONTROL,
        proto.OUTPUT_CACHE_READ,
    ):
        transaction.packets.add(
            kind=kind,
            group_id=0,
            work_id=0,
            macro_slot=0,
            cache_address=7 if kind == proto.OUTPUT_CACHE_READ else -1,
            route=[0] * 6,
            test_core=[0] * 3,
            data_type=proto.BF16,
        )
    transaction.flits.extend([0, 2**64 - 1])
    return model


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
        try:
            return proto.ModelArtifact.FromString(
                ctypes.string_at(descriptor.data, descriptor.data_size)
            )
        finally:
            if os.name == "nt":
                _ctypes.FreeLibrary(library._handle)

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
        shared = tmp_path / ("fixture.dll" if os.name == "nt" else "fixture.so")
        compiler = os.environ.get("CIM22_CC", "cl" if os.name == "nt" else "cc")
        command = (
            [
                compiler,
                "/nologo",
                "/MD",
                "/LD",
                str(source),
                "/link",
                f"/OUT:{shared}",
                "/EXPORT:cim22_model_v1",
            ]
            if os.name == "nt"
            else [compiler, "-shared", "-fPIC", str(source), "-o", str(shared)]
        )
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
        return shared

    return build
