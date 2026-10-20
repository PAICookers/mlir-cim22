"""Ownership-checked Python interface to the native CIM22 runtimes."""

import ctypes
import os
import threading
import weakref
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class UsbConfig:
    """Linux USB bridge selection and transfer settings."""

    vendor_id: int
    product_id: int
    bulk_in_endpoint: int
    bulk_out_endpoint: int
    bus_number: int = 0xFF
    device_address: int = 0xFF
    interface_number: int = 0
    alternate_setting: int = 0
    physical_link: int = 2
    logical_channel: int = 3
    word_byte_order: int = 0
    transfer_timeout_ms: int = 5000
    receive_poll_timeout_ms: int = 200
    response_timeout_ms: int = 60000

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("vendor_id", 1, 0xFFFF),
            ("product_id", 1, 0xFFFF),
            ("bulk_in_endpoint", 0x81, 0x8F),
            ("bulk_out_endpoint", 1, 0x0F),
            ("bus_number", 0, 0xFF),
            ("device_address", 0, 0xFF),
            ("interface_number", 0, 0xFF),
            ("alternate_setting", 0, 0xFF),
            ("physical_link", 0, 3),
            ("logical_channel", 0, 7),
            ("word_byte_order", 0, 1),
            ("transfer_timeout_ms", 1, 0xFFFFFFFF),
            ("receive_poll_timeout_ms", 1, 0xFFFFFFFF),
            ("response_timeout_ms", 1, 0xFFFFFFFF),
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        if (self.bus_number == 0xFF) != (self.device_address == 0xFF):
            raise ValueError("Specify both bus_number and device_address")


def runtime_library(library: Path | None = None) -> Path:
    """Resolve the installed execution library without loading compiler code."""
    if library is None:
        configured = os.environ.get("CIM22_RUNTIME_LIBRARY")
        library = (
            Path(configured)
            if configured
            else Path(str(files("cim22") / "lib" / "libcim22_runtime.so"))
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

    def configure_usb(self, config: UsbConfig) -> None:
        """Open the configured Linux USB device for a USB artifact."""
        if not isinstance(config, UsbConfig):
            raise TypeError("config must be a UsbConfig")
        with self._lock:
            self._require_open()
            native = self._library.cim22_session_configure_usb
            native.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_UsbConfig),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            native.restype = ctypes.c_int
            native_config = _UsbConfig.from_config(config)
            error = ctypes.create_string_buffer(16384)
            if native(
                self._handle, ctypes.byref(native_config), error, len(error)
            ):
                raise RuntimeError(error.value.decode(errors="replace"))

    def capture_usb(
        self, transaction: int, inputs: NDArray[np.uint16]
    ) -> NDArray[np.uint64]:
        """Capture raw output flits for one transaction, without Host execution.

        Inputs are contiguous uint16 BF16 bit patterns shaped (bindings, 64),
        in dynamic-binding order. Returns (readbacks, 7) uint64 flits: header,
        then six body flits. This diagnostic preserves the response unchanged
        for protocol inspection; use normal invocation for tensor outputs.
        """
        with self._lock:
            self._require_open()
            if type(transaction) is not int:
                raise TypeError("transaction must be an integer")
            if not 0 <= transaction < self.num_transactions:
                raise ValueError("Invalid transaction index")
            if not isinstance(inputs, np.ndarray) or inputs.dtype != np.uint16:
                raise TypeError(
                    "Capture inputs must be uint16 BF16 bit patterns"
                )
            if (
                inputs.ndim != 2
                or inputs.shape[1] != 64
                or not inputs.flags.c_contiguous
                or not inputs.flags.aligned
            ):
                raise ValueError(
                    "Capture inputs require aligned, contiguous (bindings, 64)"
                )
            count = self._library.cim22_session_num_readbacks
            count.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            count.restype = ctypes.c_int64
            outputs = np.empty(
                (count(self._handle, transaction), 7), dtype=np.uint64
            )
            capture = self._library.cim22_session_capture_usb
            capture.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            capture.restype = ctypes.c_int
            if capture(
                self._handle,
                transaction,
                inputs.ctypes.data,
                inputs.size,
                outputs.ctypes.data,
                outputs.size,
            ):
                message = self._library.cim22_session_error(self._handle)
                raise RuntimeError(message.decode(errors="replace"))
            return outputs

    @property
    def dispatch_count(self) -> int:
        """Number of native runner dispatches in the latest invocation."""
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
                if (
                    array.shape != shape
                    or not array.flags.c_contiguous
                    or not array.flags.aligned
                ):
                    raise ValueError(
                        f"Input {index} requires aligned, contiguous shape {shape}"
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


class _UsbConfig(ctypes.Structure):
    _fields_ = [
        ("vendor_id", ctypes.c_uint16),
        ("product_id", ctypes.c_uint16),
        ("bus_number", ctypes.c_uint8),
        ("device_address", ctypes.c_uint8),
        ("interface_number", ctypes.c_uint8),
        ("alternate_setting", ctypes.c_uint8),
        ("bulk_in_endpoint", ctypes.c_uint8),
        ("bulk_out_endpoint", ctypes.c_uint8),
        ("physical_link", ctypes.c_uint8),
        ("logical_channel", ctypes.c_uint8),
        ("word_byte_order", ctypes.c_uint8),
        ("transfer_timeout_ms", ctypes.c_uint32),
        ("receive_poll_timeout_ms", ctypes.c_uint32),
        ("response_timeout_ms", ctypes.c_uint32),
    ]

    @classmethod
    def from_config(cls, config: UsbConfig) -> "_UsbConfig":
        return cls(
            config.vendor_id,
            config.product_id,
            config.bus_number,
            config.device_address,
            config.interface_number,
            config.alternate_setting,
            config.bulk_in_endpoint,
            config.bulk_out_endpoint,
            config.physical_link,
            config.logical_channel,
            config.word_byte_order,
            config.transfer_timeout_ms,
            config.receive_poll_timeout_ms,
            config.response_timeout_ms,
        )
