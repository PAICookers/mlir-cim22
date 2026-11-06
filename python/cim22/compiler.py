"""Wheel-based torch-mlir capture followed by the native CIM22 compiler."""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypedDict

import torch
from torch._dynamo.utils import dynamo_timed

from .runtime import (
    MODEL_SUFFIX,
    Session,
    UsbConfig,
    _load_library,
    compiler_library,
    runtime_library,
)

type RuntimeBackend = Literal["simulator", "usb"]
type BF16Compensation = Literal["sign-split", "legacy", "none"]


class BackendOptions(TypedDict, total=False):
    """Application options shared by the PyTorch and TensorFlow frontends."""

    runtime: RuntimeBackend | Literal["reference"]
    bf16_compensation: BF16Compensation
    usb_config: UsbConfig
    artifact_dir: str | Path
    debug_dir: str | Path


def _validate_options(
    options: BackendOptions, *, allow_reference: bool = False
) -> None:
    """Reject unsupported runtime choices before capture or compilation."""
    runtime = options.get("runtime", "usb")
    allowed = (
        ("reference", "simulator", "usb")
        if allow_reference
        else ("simulator", "usb")
    )
    if runtime not in allowed:
        raise ValueError(f"CIM22 runtime must be one of {allowed}")
    if options.get("usb_config") is not None and runtime != "usb":
        raise ValueError("CIM22 usb_config requires runtime='usb'")
    if runtime != "reference" and options.get(
        "bf16_compensation", "sign-split"
    ) not in ("sign-split", "legacy", "none"):
        raise ValueError(
            "CIM22 bf16_compensation must be 'sign-split', 'legacy' or 'none'"
        )


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
    bf16_compensation: BF16Compensation = "sign-split",
) -> Session:
    """Compile a functional ATen graph; return its internal flat runtime.

    The loaded Session owns the code and any files needed until it closes.
    Optional artifact/debug paths retain diagnostics for this specialization.
    """
    # Check before importing FX: LLVM's duplicate registration aborts Python.
    if (
        sys.platform == "linux"
        and "tensorflow" in sys.modules
        and "torch_mlir.fx" not in sys.modules
    ):
        raise RuntimeError(
            "CIM22 cannot first load torch-mlir FX after TensorFlow on Linux: "
            "their bundled LLVM libraries can conflict and abort the process. "
            "Compile the PyTorch model in a fresh Python process without "
            "TensorFlow; save it with artifact_dir, then load the native "
            "model with cim22.Session for inference."
        )
    from torch_mlir import fx
    from torch_mlir.compiler_utils import OutputType

    with dynamo_timed("cim22.torch_mlir"):
        module = fx.export_and_import(
            graph,
            *example_inputs,
            output_type=OutputType.LINALG_ON_TENSORS,
            # AOTAutograd already applied the torch-mlir decompositions.
            decomposition_table={},
        )
    return _compile_linalg(
        str(module),
        artifact_path=artifact_path,
        debug_dir=debug_dir,
        runtime=runtime,
        bf16_compensation=bf16_compensation,
    )


def _compile_linalg(
    linalg_ir: str,
    *,
    artifact_path: Path | None = None,
    debug_dir: Path | None = None,
    runtime: RuntimeBackend = "usb",
    bf16_compensation: BF16Compensation = "sign-split",
    max_kernel_columns: int | Literal["auto"] | None = None,
    weight_residency_schedule: bool = True,
) -> Session:
    """Lower Linalg IR, link the native model, and load it into a Session.

    Args:
        linalg_ir: Textual Linalg module with a static inference entry.
        artifact_path: New path ending in the platform's model suffix. If
            omitted, the model library lives in the Session's temporary area.
        debug_dir: Optional new directory for Linalg IR, packet IR and the
            native object file.
        runtime: Backend encoded into the artifact. USB needs a configured
            bridge before execution.
        bf16_compensation: Numeric mode for native BF16 kernels.
        max_kernel_columns: None or "auto" selects a width by wave count and
            static Work budget. A positive integer fixes the block width
            limit. Use 32 for fixed blocking or 1 for a per-column baseline.
        weight_residency_schedule: Group independent FC/Linear transactions
            that reuse weights when legal. False preserves the original order
            and leaves runtime residency enabled.

    Returns:
        The loaded Session, which owns any temporary files.

    Raises:
        ValueError: The artifact suffix or local tool configuration is invalid.
        FileExistsError: The destination artifact already exists.
        RuntimeError: Lowering, object emission, linking, or loading failed.
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
    pipeline = f"--cim-bf16-pipeline=stage=packet bf16-compensation={bf16_compensation}"
    if max_kernel_columns is not None:
        pipeline += f" max-kernel-columns={max_kernel_columns}"
    if not weight_residency_schedule:
        pipeline += " weight-residency-schedule=false"
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
        with dynamo_timed("cim22.serialize"):
            linalg.write_text(linalg_ir, encoding="utf-8")
        with dynamo_timed("cim22.lowering"):
            _run(
                [
                    executable,
                    str(linalg),
                    pipeline,
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
                if os.name == "nt":
                    # A saved DLL must not keep the Session's private DLL locked.
                    publication = folder / "artifact.dll"
                    shutil.copyfile(shared, publication)
                    # Windows rename is atomic and refuses an existing target.
                    os.rename(publication, artifact_path)
                else:
                    # Atomic publication without replacing a concurrent winner.
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
