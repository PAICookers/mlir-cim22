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

from ._pll_serial import ClockControl, ClockResources, PLLSerial
from .pll import PLLClockSetting, select_pll_clock

type TensorArray = NDArray[np.float32 | np.int8 | np.int32]

_BOUNDARY_DTYPES = {
    1: np.dtype(np.int8),
    3: np.dtype(np.float32),
    4: np.dtype(np.int32),
}

MODEL_SUFFIX = ".dll" if os.name == "nt" else ".so"
_DEFAULT_BOARD_CLOCK_MHZ = 128


def _load_library(path: Path) -> ctypes.CDLL:
    """Resolve adjacent DLL dependencies on Windows without changing PATH."""
    with (
        os.add_dll_directory(str(path.parent))
        if os.name == "nt"
        else nullcontext()
    ):
        # Keep the runtime out of the process-wide lookup scope. Model DSOs
        # resolve it through their explicit DT_NEEDED/import-library edge.
        return ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_LOCAL", 0))


def _release_session(
    library: ctypes.CDLL,
    handle: int,
    resources: ExitStack,
    clock_lease: list[ClockResources] | None = None,
) -> None:
    # Windows cannot remove a model DLL until its native Session unloads it.
    with clock_lease[0].lock if clock_lease else nullcontext():
        library.cim22_session_destroy(handle)
        resources.close()


def _release_clock_lease(lease: list[ClockResources]) -> None:
    # The mutable holder lets attach transfer ownership without capturing Session.
    clock = lease[0]
    with clock.lock:
        clock.owners -= 1
        if clock.owners == 0:
            clock.close()


@dataclass(frozen=True, slots=True)
class PLLConfig:
    """Serial PLL commands for the configuration, work and readback phases.

    Python checks PLL limits before opening USB and uses pyserial for
    Linux and Windows 9600/8N1 serial ports. An explicitly selected port
    must exist and be accessible. The first successful configuration in
    an open USB session requires an exact echo; later changed commands are
    sent without waiting for echo. An unchanged current command is reused.
    Reopening the session requires a new initial echo.

    Attributes:
        port: Linux serial device path or Windows COM name (for example COM10).
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
            word = int(value, 16)
            nf = ((word >> 18) & 63) + 1
            nr = ((word >> 14) & 15) + 1
            if 24 * nf < 250 * nr:
                raise ValueError("PLL VCO must be at least 250 MHz")


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
    cache_write_delay_ms: int = 0
    weight_write_delay_ms: int = 0
    work_delay_ms: int = 0
    readback_select_delay_ms: int = 2

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
        receive_ready_timeout_ms: Deadline for the receive loop to start.
        response_timeout_ms: Deadline for collecting one Macro readback batch.
        linux_async_in_timeout_ms: Linux libcyusb IN transfer timeout.
        linux_sync_out_timeout_ms: Linux libcyusb OUT transfer timeout.
        linux_event_poll_timeout_ms: Linux libusb event poll interval.
        windows_async_in_timeout_ms: Windows CyAPI IN endpoint timeout.
        windows_sync_out_timeout_ms: Windows CyAPI OUT endpoint timeout.
        windows_event_wait_timeout_ms: Windows event wait interval.
        pll: Optional serial PLL words, one per phase.
        clock_frequency_mhz: Frequency in MHz for configuration, work and
            readback. None selects 128 MHz when pll is omitted. Unlisted
            values use the nearest available table entry with a warning.
        pll_port: Serial port for the table-selected clock. Without it the
            caller sets the clock manually or supplies a PLLConfig.
        timing: Board startup and operation pacing.
        clock_setting: Resolved nominal table entry. None when the caller
            supplies a PLLConfig.
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
    receive_ready_timeout_ms: int = 5000
    response_timeout_ms: int = 7000
    linux_async_in_timeout_ms: int = 5000
    linux_sync_out_timeout_ms: int = 5000
    linux_event_poll_timeout_ms: int = 100
    windows_async_in_timeout_ms: int = 5000
    windows_sync_out_timeout_ms: int = 5000
    windows_event_wait_timeout_ms: int = 100
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
            (
                "receive_ready_timeout_ms",
                self.receive_ready_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            ("response_timeout_ms", self.response_timeout_ms, 1, 0xFFFFFFFF),
            (
                "linux_async_in_timeout_ms",
                self.linux_async_in_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            (
                "linux_sync_out_timeout_ms",
                self.linux_sync_out_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            (
                "linux_event_poll_timeout_ms",
                self.linux_event_poll_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            (
                "windows_async_in_timeout_ms",
                self.windows_async_in_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            (
                "windows_sync_out_timeout_ms",
                self.windows_sync_out_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
            (
                "windows_event_wait_timeout_ms",
                self.windows_event_wait_timeout_ms,
                1,
                0xFFFFFFFF,
            ),
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
        if self.pll_port is not None and (
            not self.pll_port or "\0" in self.pll_port
        ):
            raise ValueError("pll_port must be non-empty and contain no NUL")
        if self.pll is not None:
            if (
                self.clock_frequency_mhz is not None
                or self.pll_port is not None
            ):
                raise ValueError(
                    "Specify either pll or clock_frequency_mhz/pll_port"
                )
        else:
            frequency = (
                _DEFAULT_BOARD_CLOCK_MHZ
                if self.clock_frequency_mhz is None
                else self.clock_frequency_mhz
            )
            object.__setattr__(
                self,
                "clock_setting",
                select_pll_clock(frequency),
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


def compile_tool(tool: Path | None = None) -> Path:
    """Resolve the packet-to-object compiler executable."""
    if tool is None:
        configured = os.environ.get("CIM22_COMPILE_TOOL")
        tool = (
            Path(configured)
            if configured
            else Path(
                str(
                    files("cim22")
                    / "bin"
                    / (
                        "mlir-cim22-compile.exe"
                        if os.name == "nt"
                        else "mlir-cim22-compile"
                    )
                )
            )
        )
    if not tool.is_file():
        raise FileNotFoundError(
            "CIM22 compile tool not found; set CIM22_COMPILE_TOOL "
            "to mlir-cim22-compile"
        )
    return tool.resolve(strict=True)


class Session:
    """Load a compiled native model.

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

    def __init__(self, artifact: Path, library: Path | None = None) -> None:
        """Open a compiled native model.

        Args:
            artifact: Compiled .so/.dll model.
            library: Optional override for the native execution library.

        Raises:
            FileNotFoundError: The artifact or a required library is missing.
            RuntimeError: Native loading rejected the artifact.
        """
        library = runtime_library(library)
        self._library = _load_library(library)
        self._lock = threading.RLock()
        self._handle: int | None = None
        self._invocation_count = 0
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
        if artifact.suffix != MODEL_SUFFIX:
            raise ValueError(
                f"Session requires a compiled {MODEL_SUFFIX} model, got {artifact}"
            )
        loader = lib.cim22_session_load_shared
        loader.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        arguments = [os.fsencode(artifact.resolve(strict=True))]
        loader.restype = ctypes.c_void_p
        self._handle = loader(*arguments, error, len(error))
        if not self._handle:
            raise RuntimeError(error.value.decode(errors="replace"))
        self._resources = ExitStack()
        self._clock_resources = ClockResources()
        self._clock_lease = [self._clock_resources]
        self._resources.callback(_release_clock_lease, self._clock_lease)
        self._finalizer = weakref.finalize(
            self,
            _release_session,
            lib,
            self._handle,
            self._resources,
            self._clock_lease,
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
        with self._lock, self._clock_resources.lock:
            self._require_open()
            if self._clock_resources.owners > 1:
                raise RuntimeError(
                    "Cannot reconfigure a shared USB connection; close attached Sessions first"
                )
            native_config = _UsbConfig.from_config(config)
            pll = config.pll
            if config.clock_setting is not None and config.pll_port is not None:
                word = config.clock_setting.serial_hex
                pll = PLLConfig(config.pll_port, word, word, word)
            try:
                native = self._library.cim22_session_configure_usb_with_clock
            except AttributeError as error:
                raise RuntimeError(
                    "runtime library lacks CIM22 clock callback ABI; reinstall mlir-cim22"
                ) from error
            controller = (
                PLLSerial(
                    pll.port,
                    tuple(
                        bytes.fromhex(word)
                        for word in (
                            pll.config_hex,
                            pll.work_hex,
                            pll.readback_hex,
                        )
                    ),
                )
                if pll is not None
                else None
            )
            native_hardware = _HardwareConfig.from_config(config.timing)
            native.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_UsbConfig),
                ctypes.POINTER(_HardwareConfig),
                ctypes.POINTER(ClockControl),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            native.restype = ctypes.c_int
            error = ctypes.create_string_buffer(16384)
            try:
                status = native(
                    self._handle,
                    ctypes.byref(native_config),
                    ctypes.byref(native_hardware),
                    ctypes.byref(controller.native)
                    if controller is not None
                    else None,
                    error,
                    len(error),
                )
                if status:
                    raise RuntimeError(error.value.decode(errors="replace"))
            except BaseException as failure:
                self._clock_resources.fail(failure)
                if controller is not None:
                    try:
                        controller.close()
                    except Exception as cleanup:
                        failure.add_note(
                            f"PLL serial cleanup failed: {cleanup}"
                        )
                raise
            previous = self._clock_resources.controller
            self._clock_resources.controller = controller
            self._clock_resources.failed = False
            self._clock_resources.configured = True
            if previous is not None:
                try:
                    previous.close()
                except Exception as failure:
                    self._clock_resources.fail(failure)
                    raise

    def share_usb(self, source: "Session") -> None:
        """Attach an unconfigured USB model to another Session's connection.

        Both models share transport, physical weight residency, fault and PLL
        state. Closing either model leaves the other usable; the last model
        releases the connection. Reconfiguration requires a single owner.
        """
        if not isinstance(source, Session):
            raise TypeError("source must be a configured Session")
        if source is self:
            raise ValueError(
                "Cannot share a USB connection with the same Session"
            )
        first, second = sorted((self, source), key=id)
        with first._lock, second._lock, source._clock_resources.lock:
            self._require_open()
            source._require_open()
            source._clock_resources.require_ready()
            if self._library._handle != source._library._handle:
                raise ValueError(
                    "USB Sessions must use the same native runtime"
                )
            if self._clock_resources.configured:
                raise RuntimeError("Target Session is already configured")
            if not source._clock_resources.configured:
                raise RuntimeError("Source USB Session is not configured")
            try:
                native = self._library.cim22_session_share_usb
            except AttributeError as error:
                raise RuntimeError(
                    "runtime lacks shared USB support; reinstall mlir-cim22"
                ) from error
            native.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            native.restype = ctypes.c_int
            error = ctypes.create_string_buffer(16384)
            if native(self._handle, source._handle, error, len(error)):
                raise RuntimeError(error.value.decode(errors="replace"))
            _release_clock_lease(self._clock_lease)
            self._clock_resources = source._clock_resources
            self._clock_lease[0] = self._clock_resources
            self._clock_resources.owners += 1

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
        with self._lock, self._clock_resources.lock:
            self._require_open()
            self._clock_resources.require_ready()
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
                failure = RuntimeError(message.decode(errors="replace"))
                if self._clock_resources.configured:
                    self._clock_resources.fail(failure)
                raise failure
            return outputs

    @property
    def dispatch_count(self) -> int:
        """Number of native runner dispatches in the latest invocation."""
        with self._lock, self._clock_resources.lock:
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
        with self._lock, self._clock_resources.lock:
            self._require_open()
            self._clock_resources.require_ready()
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
                failure = RuntimeError(message.decode(errors="replace"))
                if self._clock_resources.configured:
                    self._clock_resources.fail(failure)
                raise failure
            self._invocation_count += 1
            return outputs

    def close(self) -> None:
        """Release native code/resources; existing output buffers remain valid."""
        with self._lock, self._clock_resources.lock:
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
    *,
    usb_session: Session | None = None,
) -> Session:
    """Adopt owned resources and configure a loaded Session; close on failure."""
    try:
        if resources is not None:
            session._resources.callback(resources.close)
        if usb_session is not None:
            session.share_usb(usb_session)
        elif usb_config is not None:
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
        ("receive_ready_timeout_ms", ctypes.c_uint32),
        ("response_timeout_ms", ctypes.c_uint32),
        ("linux_async_in_timeout_ms", ctypes.c_uint32),
        ("linux_sync_out_timeout_ms", ctypes.c_uint32),
        ("linux_event_poll_timeout_ms", ctypes.c_uint32),
        ("windows_async_in_timeout_ms", ctypes.c_uint32),
        ("windows_sync_out_timeout_ms", ctypes.c_uint32),
        ("windows_event_wait_timeout_ms", ctypes.c_uint32),
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
            config.receive_ready_timeout_ms,
            config.response_timeout_ms,
            config.linux_async_in_timeout_ms,
            config.linux_sync_out_timeout_ms,
            config.linux_event_poll_timeout_ms,
            config.windows_async_in_timeout_ms,
            config.windows_sync_out_timeout_ms,
            config.windows_event_wait_timeout_ms,
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
