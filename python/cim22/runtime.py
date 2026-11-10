"""Ownership-checked Python interface to the native CIM22 runtimes."""

import ctypes
import os
import threading
import weakref
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from .pll import PLLClockSetting, select_pll_clock

type TensorArray = NDArray[np.float32 | np.int8 | np.int32]

_BOUNDARY_DTYPES = {
    1: np.dtype(np.int8),
    3: np.dtype(np.float32),
    4: np.dtype(np.int32),
}

MODEL_SUFFIX = ".dll" if os.name == "nt" else ".so"


def _load_library(path: Path) -> ctypes.CDLL:
    """Resolve adjacent DLL dependencies on Windows without changing PATH."""
    with (
        os.add_dll_directory(str(path.parent))
        if os.name == "nt"
        else nullcontext()
    ):
        # Publish the declared project ABI so dependent native components share
        # Error type tags; the Linux export map hides LLVM registries/functions.
        return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _release_session(
    library: ctypes.CDLL, handle: int, resources: ExitStack
) -> None:
    # Windows cannot remove a model DLL until its native Session unloads it.
    library.cim22_session_destroy(handle)
    resources.close()


@dataclass(frozen=True, slots=True)
class PLLConfig:
    """Serial PLL commands for the configuration, work and readback phases.

    The native runtime checks PLL limits before opening USB. On Linux, a
    missing serial device uses the manually configured clock. A present
    device must echo the command exactly. An echoed word can be reused
    across phases and transactions within the open USB session. Configure
    the clock manually on Windows.

    Attributes:
        port: Linux serial device path.
        config_hex: Six-digit hexadecimal word for the configuration phase.
        work_hex: Six-digit hexadecimal word for the WorkOnce phase.
        readback_hex: Six-digit hexadecimal word for the readback phase.
    """

    port: str
    config_hex: str
    work_hex: str
    readback_hex: str

    def __post_init__(self) -> None:
        if not self.port or "\0" in self.port:
            raise ValueError("port must be non-empty and contain no NUL")
        for name, value in (
            ("config_hex", self.config_hex),
            ("work_hex", self.work_hex),
            ("readback_hex", self.readback_hex),
        ):
            if len(value) != 6 or any(
                char not in "0123456789abcdefABCDEF" for char in value
            ):
                raise ValueError(f"{name} must be a six-digit HEX string")


@dataclass(frozen=True, slots=True)
class CIMTimingConfig:
    """Nonnegative USB board delays in milliseconds.

    Attributes:
        startup_delay_ms: Wait after opening the bridge.
        cache_write_delay_ms: Wait after each input Cache write.
        weight_write_delay_ms: Wait after each actual weight write.
        work_delay_ms: Wait after the full WorkOnce batch for computation and
            output Cache writeback to finish.
        readback_select_delay_ms: Wait after selecting the Macros in a batch
            and before requesting their output Cache rows.
    """

    startup_delay_ms: int = 20
    cache_write_delay_ms: int = 1
    weight_write_delay_ms: int = 2
    work_delay_ms: int = 10
    readback_select_delay_ms: int = 40

    def __post_init__(self) -> None:
        for name, value in (
            ("startup_delay_ms", self.startup_delay_ms),
            ("cache_write_delay_ms", self.cache_write_delay_ms),
            ("weight_write_delay_ms", self.weight_write_delay_ms),
            ("work_delay_ms", self.work_delay_ms),
            ("readback_select_delay_ms", self.readback_select_delay_ms),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"{name} must be in [0, 4294967295]")


@dataclass(frozen=True, slots=True)
class UsbConfig:
    """Platform USB bridge selection, transport limits and board timing.

    Attributes:
        vendor_id: USB vendor ID.
        product_id: USB product ID.
        bulk_in_endpoint: Bulk receive endpoint address, including IN bit.
        bulk_out_endpoint: Bulk transmit endpoint address.
        bus_number: Linux bus selector. 0xFF means unspecified. Supply it
            together with device_address on Linux.
        device_address: Device selector. 0xFF means unspecified.
        interface_number: USB interface to claim.
        alternate_setting: Alternate setting on the selected interface.
        physical_link: Bridge physical-link index (0–3).
        logical_channel: Bridge transmit-channel index (0–7).
        word_byte_order: Record word order, 0 for big-endian and 1 for
            little-endian.
        transfer_timeout_ms: Timeout for a USB write transfer.
        receive_poll_timeout_ms: Poll timeout for the USB receive loop.
        response_timeout_ms: Deadline for collecting one Macro readback batch.
        pll: Optional serial PLL words, one per phase.
        clock_frequency_mhz: Optional table target in MHz for all three phases.
            Mutually exclusive with pll.
        pll_port: Optional Linux serial path for clock_frequency_mhz. Without
            it the caller sets the clock manually.
        timing: Board startup and operation pacing.
        clock_setting: Resolved table entry, if requested. It records nominal
            table settings. Readback timing still uses the configured delays.
    """

    vendor_id: int
    product_id: int
    bulk_in_endpoint: int
    bulk_out_endpoint: int
    bus_number: int = 0xFF
    device_address: int = 0xFF
    interface_number: int = 0
    alternate_setting: int = 0
    physical_link: int = 0
    logical_channel: int = 3
    word_byte_order: Literal[0, 1] = 0
    transfer_timeout_ms: int = 5000
    receive_poll_timeout_ms: int = 200
    response_timeout_ms: int = 7000
    pll: PLLConfig | None = None
    clock_frequency_mhz: float | None = None
    pll_port: str | None = None
    timing: CIMTimingConfig = field(default_factory=CIMTimingConfig)
    clock_setting: PLLClockSetting | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # ctypes wraps integers, so check wire ranges before conversion.
        for name, value, lower, upper in (
            ("vendor_id", self.vendor_id, 1, 0xFFFF),
            ("product_id", self.product_id, 1, 0xFFFF),
            ("bulk_in_endpoint", self.bulk_in_endpoint, 0x81, 0x8F),
            ("bulk_out_endpoint", self.bulk_out_endpoint, 1, 0x0F),
            ("bus_number", self.bus_number, 0, 0xFF),
            ("device_address", self.device_address, 0, 0xFF),
            ("interface_number", self.interface_number, 0, 0xFF),
            ("alternate_setting", self.alternate_setting, 0, 0xFF),
            ("physical_link", self.physical_link, 0, 3),
            ("logical_channel", self.logical_channel, 0, 7),
            ("word_byte_order", self.word_byte_order, 0, 1),
            ("transfer_timeout_ms", self.transfer_timeout_ms, 1, 0xFFFFFFFF),
            (
                "receive_poll_timeout_ms",
                self.receive_poll_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            ("response_timeout_ms", self.response_timeout_ms, 1, 0xFFFFFFFF),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        if os.name == "nt":
            if self.bus_number != 0xFF:
                raise ValueError("bus_number is not supported by Windows CyAPI")
        elif (self.bus_number == 0xFF) != (self.device_address == 0xFF):
            raise ValueError("Specify both bus_number and device_address")
        if self.pll_port is not None:
            if not self.pll_port or "\0" in self.pll_port:
                raise ValueError(
                    "pll_port must be non-empty and contain no NUL"
                )
            if self.clock_frequency_mhz is None:
                raise ValueError("pll_port requires clock_frequency_mhz")
        if self.clock_frequency_mhz is not None:
            if self.pll is not None:
                raise ValueError("Specify either pll or clock_frequency_mhz")
            object.__setattr__(
                self,
                "clock_setting",
                select_pll_clock(self.clock_frequency_mhz),
            )
        if not isinstance(self.timing, CIMTimingConfig):
            raise TypeError("timing must be a CIMTimingConfig")


def runtime_library(library: Path | None = None) -> Path:
    """Resolve the native execution library.

    Args:
        library: Explicit library path. None uses CIM22_RUNTIME_LIBRARY or
            the packaged library.

    Returns:
        The resolved absolute path.

    Raises:
        FileNotFoundError: The selected library is missing.
    """
    if library is None:
        configured = os.environ.get("CIM22_RUNTIME_LIBRARY")
        library = (
            Path(configured)
            if configured
            else Path(
                str(
                    files("cim22")
                    / "lib"
                    / (
                        "cim22_runtime.dll"
                        if os.name == "nt"
                        else "libcim22_runtime.so"
                    )
                )
            )
        )
    if not library.is_file():
        raise FileNotFoundError(
            "CIM22 runtime library not found; set CIM22_RUNTIME_LIBRARY "
            "to the built native runtime library"
        )
    return library.resolve(strict=True)


def compiler_library(library: Path) -> Path:
    """Resolve the compiler adapter used for compilation and raw packet loads.

    Args:
        library: Runtime library path. The compiler is adjacent unless
            CIM22_COMPILER_LIBRARY selects a different path.

    Returns:
        The resolved absolute path.

    Raises:
        FileNotFoundError: The selected compiler adapter is missing.
    """
    configured = os.environ.get("CIM22_COMPILER_LIBRARY")
    path = (
        Path(configured)
        if configured
        else library.with_name(
            "cim22_compiler.dll"
            if os.name == "nt"
            else "libcim22_compiler.so"
        )
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Build cim22_compiler or set CIM22_COMPILER_LIBRARY"
        )
    return path.resolve(strict=True)


class Session:
    """Load a compiled native model, or a raw packet for compiler diagnostics.

    Inputs must match the compiled static shapes and FP32/INT8/INT32 dtypes.
    All arrays must be contiguous and aligned in CPU memory.
    Every call allocates independent output arrays owned by Python. Loading and
    inference use the native CIM runner. Use the context manager or close()
    to release the loaded model and any USB resources.

    Attributes:
        input_shapes: Input shapes in flat signature order.
        input_dtypes: NumPy dtypes in flat input order.
        output_shapes: Output shapes in flat result order.
        output_dtypes: NumPy dtypes in flat result order.
        num_transactions: Number of static CIM transactions in the model.
    """

    def __init__(
        self, artifact: Path, library: Path | None = None, entry: str = "main"
    ) -> None:
        """Open a native model or a diagnostic packet artifact.

        Args:
            artifact: Compiled .so/.dll or textual packet IR.
            library: Optional override for the native execution library.
            entry: Function name for raw packet IR. A native model supplies
                its own entry.

        Raises:
            FileNotFoundError: The artifact or a required library is missing.
            RuntimeError: Native loading rejected the artifact.
        """
        library = runtime_library(library)
        self._library = _load_library(library)
        self._lock = threading.RLock()
        self._handle: int | None = None
        lib = self._library
        signatures = {
            "destroy": ([ctypes.c_void_p], None),
            "num_inputs": ([ctypes.c_void_p], ctypes.c_int64),
            "num_outputs": ([ctypes.c_void_p], ctypes.c_int64),
            "rank": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_int64),
            "dtype": ([ctypes.c_void_p, ctypes.c_int64], ctypes.c_int64),
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
        if artifact.suffix == MODEL_SUFFIX:
            loader = lib.cim22_session_load_shared
            loader.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
        else:
            self._compiler = _load_library(compiler_library(library))
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
        self._resources = ExitStack()
        self._finalizer = weakref.finalize(
            self, _release_session, lib, self._handle, self._resources
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
        dtypes = tuple(
            _BOUNDARY_DTYPES[lib.cim22_session_dtype(self._handle, i)]
            for i in range(num_inputs + num_outputs)
        )
        self.input_dtypes = dtypes[:num_inputs]
        self.output_dtypes = dtypes[num_inputs:]
        self.num_transactions = lib.cim22_session_num_transactions(self._handle)

    def configure_usb(self, config: UsbConfig) -> None:
        """Open or replace the bridge for a loaded USB model.

        Args:
            config: Bridge identity, transport settings, timing and PLL
                settings. Reopening starts a fresh device state.

        Raises:
            RuntimeError: The Session is closed, the runtime rejects USB
                configuration, or bridge setup fails.
        """
        with self._lock:
            self._require_open()
            native_config = _UsbConfig.from_config(config)
            pll = config.pll
            if config.clock_setting is not None and config.pll_port is not None:
                word = config.clock_setting.serial_hex
                pll = PLLConfig(config.pll_port, word, word, word)
            try:
                native = self._library.cim22_session_configure_usb_with_hardware
            except AttributeError as error:
                raise RuntimeError(
                    "runtime library lacks CIM22 hardware configuration ABI"
                ) from error
            native_pll = (
                _PLLConfig.from_config(pll) if pll is not None else None
            )
            native_hardware = _HardwareConfig.from_config(config.timing)
            native.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_UsbConfig),
                ctypes.POINTER(_PLLConfig),
                ctypes.POINTER(_HardwareConfig),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            native.restype = ctypes.c_int
            arguments = [
                self._handle,
                ctypes.byref(native_config),
                ctypes.byref(native_pll) if native_pll is not None else None,
                ctypes.byref(native_hardware),
            ]
            error = ctypes.create_string_buffer(16384)
            if native(*arguments, error, len(error)):
                raise RuntimeError(error.value.decode(errors="replace"))

    def capture_usb(
        self, transaction: int, inputs: NDArray[np.uint16]
    ) -> NDArray[np.uint64]:
        """Capture raw output flits for one transaction, without Host execution.

        Capture writes all weights and invalidates cached weight residency.

        Args:
            transaction: Zero-based index into the artifact's transactions.
            inputs: Aligned contiguous uint16 BF16 bit patterns of shape
                (bindings, 64), in dynamic-binding order.

        Returns:
            uint64 array shaped (readbacks, 7) in binding order. Each row
            contains the raw header and six raw body flits.

        Raises:
            TypeError: The index or input dtype is invalid.
            ValueError: The index is out of range or input layout is invalid.
            RuntimeError: The Session is closed or the native capture fails.
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

    def __call__(self, *inputs: TensorArray) -> tuple[TensorArray, ...]:
        """Execute one synchronous inference.

        Args:
            *inputs: Aligned, C-contiguous NumPy arrays matching the static
                input shapes, dtypes and flat signature order.

        Returns:
            Independent NumPy arrays in the model's flat result order.

        Raises:
            TypeError: An input is not a NumPy array of the required dtype.
            ValueError: The input count, shape, alignment or layout is invalid.
            RuntimeError: The Session is closed or native execution fails.
        """
        with self._lock:
            self._require_open()
            if len(inputs) != len(self.input_shapes):
                raise ValueError("Input count differs from compiled signature")
            for index, (array, shape, dtype) in enumerate(
                zip(inputs, self.input_shapes, self.input_dtypes, strict=True)
            ):
                if not isinstance(array, np.ndarray) or array.dtype != dtype:
                    raise TypeError(
                        f"Input {index} must be a NumPy {dtype.name} array"
                    )
                if (
                    array.shape != shape
                    or not array.flags.c_contiguous
                    or not array.flags.aligned
                ):
                    raise ValueError(
                        f"Input {index} requires aligned, contiguous shape {shape}"
                    )
            outputs = tuple(
                np.empty(shape, dtype=dtype)
                for shape, dtype in zip(
                    self.output_shapes, self.output_dtypes, strict=True
                )
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
                # Cleanup can raise after native destruction; never reuse it.
                self._handle = None
                self._finalizer()

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


def _initialize_session(
    session: Session,
    usb_config: UsbConfig | None = None,
    resources: ExitStack | None = None,
) -> Session:
    """Adopt owned resources and configure a loaded Session; close on failure."""
    try:
        if resources is not None:
            session._resources.callback(resources.close)
        if usb_config is not None:
            session.configure_usb(usb_config)
    except BaseException:
        session.close()
        raise
    return session


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
    def from_config(cls, config: UsbConfig) -> Self:
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


class _PLLConfig(ctypes.Structure):
    _fields_ = [
        ("port", ctypes.c_char_p),
        ("configuration_word", ctypes.c_uint32),
        ("work_word", ctypes.c_uint32),
        ("readback_word", ctypes.c_uint32),
    ]

    @classmethod
    def from_config(cls, config: PLLConfig) -> Self:
        return cls(
            os.fsencode(config.port),
            int(config.config_hex, 16),
            int(config.work_hex, 16),
            int(config.readback_hex, 16),
        )


class _HardwareConfig(ctypes.Structure):
    _fields_ = [
        ("startup_delay_ms", ctypes.c_uint32),
        ("cache_write_delay_ms", ctypes.c_uint32),
        ("weight_write_delay_ms", ctypes.c_uint32),
        ("work_delay_ms", ctypes.c_uint32),
        ("readback_select_delay_ms", ctypes.c_uint32),
    ]

    @classmethod
    def from_config(cls, config: CIMTimingConfig) -> Self:
        return cls(
            config.startup_delay_ms,
            config.cache_write_delay_ms,
            config.weight_write_delay_ms,
            config.work_delay_ms,
            config.readback_select_delay_ms,
        )
