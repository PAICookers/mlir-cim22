"""Wheel-based torch-mlir capture followed by the native CIM22 compiler."""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from cim22._schema import encode_outputs
from cim22.model import CompiledProgram, load_compiled
from cim22.runtime import compiler_library, runtime_library


def _emit_object(
    packet: Path, object_file: Path, schema: bytes, library: Path
) -> None:
    native = ctypes.CDLL(str(compiler_library(library)))
    emit = native.cim22_compile_object
    emit.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    emit.restype = ctypes.c_int
    error = ctypes.create_string_buffer(16384)
    if emit(
        os.fsencode(packet),
        os.fsencode(object_file),
        schema,
        len(schema),
        error,
        len(error),
    ):
        raise RuntimeError(error.value.decode(errors="replace"))


def _run(command: list[str], stage: str) -> None:
    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"CIM22 {stage} failed:\n{result.stderr}")


def compile_exported(
    program: torch.export.ExportedProgram,
    *,
    optimizer: Path | None = None,
    library: Path | None = None,
    artifact_path: Path | None = None,
    debug_dir: Path | None = None,
) -> CompiledProgram:
    """Compile a reset-state, static inference program to a Linux shared library.

    Args:
        program: Exported model with flat positional tensor inputs and frozen
            weights. Export after resetting any model state.
        optimizer: mlir-cim22-opt executable; defaults to CIM22_OPT or PATH.
        library: Native execution library, with compiler library beside it.
        artifact_path: Optional new .so file to retain for independent loading.
            Otherwise the library is temporary and lives through the session.
        debug_dir: Optional new directory retaining the intermediate pt2, MLIR
            and object files. Inference never reads this directory.

    Returns:
        A callable using compiled CPU code and the candidate BF16 software runner.
        Each invocation starts from the exported state and retains recurrence
        within that invocation. Outputs have independent ownership.
    """
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("CIM22 compilation requires Linux and Python 3.12")
    if not isinstance(program, torch.export.ExportedProgram):
        raise TypeError("Expected a torch.export.ExportedProgram")
    if program.range_constraints:
        raise ValueError("CIM22 requires static shapes")
    example_args, example_kwargs = program.example_inputs
    if example_kwargs or not all(
        isinstance(v, torch.Tensor) for v in example_args
    ):
        raise ValueError("CIM22 requires flat positional tensor inputs")
    schema = encode_outputs(program.call_spec.out_spec)
    library = runtime_library(library)
    compiler_library(library)
    executable = str(optimizer) if optimizer else os.environ.get("CIM22_OPT")
    executable = executable or shutil.which("mlir-cim22-opt")
    linker = os.environ.get("CIM22_CC") or shutil.which("cc")
    if not executable or not linker:
        raise ValueError(
            "CIM22 compilation needs mlir-cim22-opt and a C linker (cc)"
        )
    if artifact_path is not None:
        artifact_path = artifact_path.absolute()
        if artifact_path.suffix != ".so":
            raise ValueError("artifact_path must end in .so")
        if artifact_path.exists() or artifact_path.is_symlink():
            raise FileExistsError(f"Artifact already exists: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None and (debug_dir.exists() or debug_dir.is_symlink()):
        raise FileExistsError(f"Debug directory already exists: {debug_dir}")
    with tempfile.TemporaryDirectory(
        prefix=".cim22-",
        dir=artifact_path.parent if artifact_path is not None else None,
    ) as temporary:
        folder = Path(temporary)
        exported = folder / "model.pt2"
        linalg = folder / "model.linalg.mlir"
        packet = folder / "model.packet.mlir"
        object_file = folder / "model.o"
        shared = folder / "model.so"
        torch.export.save(program, exported)
        _run(
            [
                sys.executable,
                str(Path(__file__).with_name("_import.py")),
                str(exported),
                str(linalg),
            ],
            "torch-mlir import",
        )
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
        _emit_object(packet, object_file, schema, library)
        # Bind model-local function names locally; different loaded models may
        # all have an entry called main. The common runtime remains external.
        _run(
            [
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
        compiled = load_compiled(shared, library=library)
        try:
            if debug_dir is not None:
                debug_dir.mkdir(parents=True)
                for path in (exported, linalg, packet, object_file):
                    shutil.copyfile(path, debug_dir / path.name)
            if artifact_path is not None:
                # Atomic publication on the same filesystem, without replacing
                # an existing artifact if a concurrent compilation won the race.
                os.link(shared, artifact_path)
        except BaseException:
            compiled.close()
            raise
    return compiled
