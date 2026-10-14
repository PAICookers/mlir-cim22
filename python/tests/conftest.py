"""Shared preflight for tests that actually compile and load Linux models."""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def native_toolchain() -> None:
    """Fail explicitly when requested integration dependencies are missing."""
    if importlib.util.find_spec("torch_mlir") is None:
        pytest.fail(
            "Install the frontend with uv sync --locked --extra compiler"
        )
    optimizer = os.environ.get("CIM22_OPT", "mlir-cim22-opt")
    if shutil.which(optimizer) is None:
        pytest.fail("Set CIM22_OPT to the built mlir-cim22-opt executable")
    runtime = os.environ.get("CIM22_RUNTIME_LIBRARY")
    if runtime is None or not Path(runtime).is_file():
        pytest.fail("Set CIM22_RUNTIME_LIBRARY to libcim22_runtime.so")
    compiler = Path(
        os.environ.get(
            "CIM22_COMPILER_LIBRARY",
            str(Path(runtime).with_name("libcim22_compiler.so")),
        )
    )
    if not compiler.is_file():
        pytest.fail("Build cim22_compiler or set CIM22_COMPILER_LIBRARY")
    if shutil.which(os.environ.get("CIM22_CC", "cc")) is None:
        pytest.fail("Install a C linker or set CIM22_CC")
