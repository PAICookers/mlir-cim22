"""Wheel-based torch-mlir capture followed by the native CIM22 compiler."""

import ctypes
import os
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path
from typing import Literal

import torch
from torch._dynamo.utils import dynamo_timed
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from .runtime import (
    MODEL_SUFFIX,
    Session,
    _load_library,
    compiler_library,
    runtime_library,
)

type RuntimeBackend = Literal["software", "usb"]


def _emit_object(
    packet: Path,
    object_file: Path,
    library: Path,
    runtime: RuntimeBackend = "usb",
) -> None:
    native = _load_library(compiler_library(library))
    emit = native.cim22_compile_object
    emit.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    emit.restype = ctypes.c_int
    error = ctypes.create_string_buffer(16384)
    if emit(
        os.fsencode(packet),
        os.fsencode(object_file),
        runtime.encode(),
        error,
        len(error),
    ):
        raise RuntimeError(error.value.decode(errors="replace"))


def _run(command: list[str], stage: str) -> None:
    result = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"CIM22 {stage} failed:\n{result.stdout or ''}{result.stderr or ''}"
        )


def _compile(
    graph: torch.fx.GraphModule,
    *example_inputs: torch.Tensor,
    artifact_path: Path | None = None,
    debug_dir: Path | None = None,
    runtime: RuntimeBackend = "usb",
) -> Session:
    """Compile a functional ATen graph; return its internal flat runtime.

    The loaded Session owns the code and any files needed until it closes.
    Optional artifact/debug paths retain diagnostics for this specialization.
    """
    library = runtime_library()
    executable = os.environ.get("CIM22_OPT") or str(
        files("cim22")
        / "bin"
        / ("mlir-cim22-opt.exe" if os.name == "nt" else "mlir-cim22-opt")
    )
    linker = os.environ.get("CIM22_CC") or shutil.which(
        "cl" if os.name == "nt" else "cc"
    )
    if not Path(executable).is_file() or not linker:
        raise ValueError(
            "CIM22 compilation needs mlir-cim22-opt and a C linker "
            "(Windows: cl in an x64 MSVC developer shell; Linux: cc)"
        )
    if artifact_path is not None:
        artifact_path = artifact_path.absolute()
        if artifact_path.suffix != MODEL_SUFFIX:
            raise ValueError(f"artifact_path must end in {MODEL_SUFFIX}")
        if artifact_path.exists() or artifact_path.is_symlink():
            raise FileExistsError(f"Artifact already exists: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        folder = Path(
            cleanup.enter_context(
                tempfile.TemporaryDirectory(
                    prefix=".cim22-",
                    dir=artifact_path.parent
                    if artifact_path is not None
                    else None,
                )
            )
        )
        linalg = folder / "model.linalg.mlir"
        packet = folder / "model.packet.mlir"
        object_file = folder / ("model.obj" if os.name == "nt" else "model.o")
        shared = folder / f"model{MODEL_SUFFIX}"
        with dynamo_timed("cim22.torch_mlir"):
            module = fx.export_and_import(
                graph,
                *example_inputs,
                output_type=OutputType.LINALG_ON_TENSORS,
                # AOTAutograd already applied the torch-mlir decompositions.
                decomposition_table={},
            )
        with dynamo_timed("cim22.serialize"):
            linalg.write_text(str(module), encoding="utf-8")
        with dynamo_timed("cim22.lowering"):
            _run(
                [
                    executable,
                    str(linalg),
                    "--cim-bf16-pipeline=stage=packet max-kernel-columns=1",
                    "-o",
                    str(packet),
                ],
                "CIM lowering",
            )
        with dynamo_timed("cim22.object"):
            _emit_object(packet, object_file, library, runtime)
        # Bind model-local function names locally; different loaded models may
        # all have an entry called main. The common runtime remains external.
        with dynamo_timed("cim22.link"):
            _run(
                [
                    linker,
                    "/nologo",
                    "/LD",
                    "/MD",
                    str(object_file),
                    str(library.with_suffix(".lib")),
                    "/link",
                    f"/OUT:{shared}",
                    "/EXPORT:cim22_model_v1",
                    f"/IMPLIB:{folder / 'model.lib'}",
                    "msvcrt.lib",
                ]
                if os.name == "nt"
                else [
                    linker,
                    "-shared",
                    "-Wl,-z,defs",
                    "-Wl,-Bsymbolic-functions",
                    "-Wl,-rpath,$ORIGIN",
                    str(object_file),
                    str(library),
                    "-lm",
                    "-o",
                    str(shared),
                ],
                "shared-library link",
            )
        with dynamo_timed("cim22.load"):
            session = Session(shared, library=library)
        try:
            if debug_dir is not None:
                debug_dir.mkdir(parents=True)
                for path in (linalg, packet, object_file):
                    shutil.copyfile(path, debug_dir / path.name)
            if artifact_path is not None:
                # Atomic publication on the same filesystem, without replacing
                # an existing artifact if a concurrent compilation won the race.
                os.link(shared, artifact_path)
            if os.name == "nt":
                # Retain only the locked DLL until native Session destruction.
                for path in folder.iterdir():
                    if path != shared:
                        path.unlink()
                session._resources.callback(cleanup.pop_all().close)
        except BaseException:
            session.close()
            raise
    return session
