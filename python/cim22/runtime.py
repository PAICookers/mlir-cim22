"""Ownership-checked Python interface to the native CIM22 software runtime."""

import ctypes
import os
import threading
import weakref
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]


def runtime_library(library: Path | None = None) -> Path:
    """Resolve the installed execution library without loading compiler code."""
    if library is None:
        configured = os.environ.get("CIM22_RUNTIME_LIBRARY")
        library = (
            Path(configured)
            if configured
            else Path(__file__).parent / "lib" / "libcim22_runtime.so"
        )
    if not library.is_file():
        raise FileNotFoundError(
            "CIM22 runtime library not found; set CIM22_RUNTIME_LIBRARY "
            "to the built libcim22_runtime.so"
        )
    return library.resolve(strict=True)


def compiler_library(library: Path) -> Path:
    """The compilation adapter is required at build time, never for .so loading."""
    configured = os.environ.get("CIM22_COMPILER_LIBRARY")
    path = (
        Path(configured)
        if configured
        else library.with_name("libcim22_compiler.so")
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Build cim22_compiler or set CIM22_COMPILER_LIBRARY"
        )
    return path.resolve(strict=True)


class Session:
    """Load a compiled Linux model, or a raw packet for compiler diagnostics.

    Inputs must be contiguous CPU FP32 arrays with the compiled static shapes.
    Every call allocates independent output arrays owned by Python. Loading and
    inference never invoke PyTorch or a substitute for the native CIM runner.
    """

    def __init__(
        self, artifact: Path, library: Path | None = None, entry: str = "main"
    ) -> None:
        library = runtime_library(library)
        self._library = ctypes.CDLL(str(library))
        self._lock = threading.RLock()
        self._handle: int | None = None
        lib = self._library
        signatures = {
            "destroy": ([ctypes.c_void_p], None),
            "num_inputs": ([ctypes.c_void_p], ctypes.c_int64),
            "num_outputs": ([ctypes.c_void_p], ctypes.c_int64),
            "rank": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_int64),
            "dim": (
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64],
                ctypes.c_int64,
            ),
            "num_transactions": ([ctypes.c_void_p], ctypes.c_int64),
            "dispatch_count": ([ctypes.c_void_p], ctypes.c_uint64),
            "error": ([ctypes.c_void_p], ctypes.c_char_p),
            "output_schema": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
                ctypes.c_void_p,
            ),
            "invoke": (
                [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int64,
                ],
                ctypes.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, f"cim22_session_{name}")
            function.argtypes = args
            function.restype = result
        error = ctypes.create_string_buffer(16384)
        arguments = [os.fsencode(artifact.resolve(strict=True))]
        if artifact.suffix == ".so":
            loader = lib.cim22_session_load_shared
            loader.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
        else:
            self._compiler = ctypes.CDLL(str(compiler_library(library)))
            loader = self._compiler.cim22_session_load_packet
            loader.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            arguments.append(entry.encode())
        loader.restype = ctypes.c_void_p
        self._handle = loader(*arguments, error, len(error))
        if not self._handle:
            raise RuntimeError(error.value.decode(errors="replace"))
        self._finalizer = weakref.finalize(
            self, lib.cim22_session_destroy, self._handle
        )
        num_inputs = lib.cim22_session_num_inputs(self._handle)
        num_outputs = lib.cim22_session_num_outputs(self._handle)
        shapes = tuple(
            tuple(
                lib.cim22_session_dim(self._handle, i, d)
                for d in range(lib.cim22_session_rank(self._handle, i))
            )
            for i in range(num_inputs + num_outputs)
        )
        self.input_shapes = shapes[:num_inputs]
        self.output_shapes = shapes[num_inputs:]
        self.num_transactions = lib.cim22_session_num_transactions(self._handle)
        size = ctypes.c_size_t()
        schema = lib.cim22_session_output_schema(
            self._handle, ctypes.byref(size)
        )
        self.output_schema = ctypes.string_at(schema, size.value)

    @property
    def dispatch_count(self) -> int:
        """Number of real software-runner dispatches in the latest invocation."""
        with self._lock:
            self._require_open()
            return self._library.cim22_session_dispatch_count(self._handle)

    def _require_open(self) -> None:
        if self._handle is None:
            raise RuntimeError("CIM22 session is closed")

    def __call__(self, *inputs: FloatArray) -> tuple[FloatArray, ...]:
        """Execute synchronously, raising on shape, type or native runner errors."""
        with self._lock:
            self._require_open()
            if len(inputs) != len(self.input_shapes):
                raise ValueError("Input count differs from compiled signature")
            for index, (array, shape) in enumerate(
                zip(inputs, self.input_shapes, strict=True)
            ):
                if (
                    not isinstance(array, np.ndarray)
                    or array.dtype != np.float32
                ):
                    raise TypeError(f"Input {index} must be a NumPy FP32 array")
                if array.shape != shape or not array.flags.c_contiguous:
                    raise ValueError(
                        f"Input {index} requires contiguous shape {shape}"
                    )
            outputs = tuple(
                np.empty(shape, dtype=np.float32)
                for shape in self.output_shapes
            )
            arrays = (*inputs, *outputs)
            pointers = (ctypes.c_void_p * len(arrays))(
                *(array.ctypes.data for array in arrays)
            )
            counts = (ctypes.c_int64 * len(arrays))(*(a.size for a in arrays))
            if self._library.cim22_session_invoke(
                self._handle, pointers, counts, len(arrays)
            ):
                message = self._library.cim22_session_error(self._handle)
                raise RuntimeError(message.decode(errors="replace"))
            return outputs

    def close(self) -> None:
        """Release native code/resources; existing output buffers remain valid."""
        with self._lock:
            if self._handle is not None:
                self._finalizer()
                self._handle = None

    def __enter__(self) -> "Session":
        self._require_open()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
