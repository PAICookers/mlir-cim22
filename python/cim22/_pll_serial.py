"""Session-owned pyserial PLL exchange and its synchronous C callback."""

import ctypes
import time

import serial

_BytePointer = ctypes.POINTER(ctypes.c_uint8)
_ClockCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint32,
    _BytePointer,
    ctypes.c_void_p,
    ctypes.c_size_t,
)


class ClockControl(ctypes.Structure):
    """Mirror CIM22ClockControl; the owner keeps the callback alive."""

    _fields_ = [("callback", _ClockCallback), ("context", ctypes.c_void_p)]


class PLLSerial:
    """Configure phase words, confirming only the session's first command."""

    def __init__(self, port: str, commands: tuple[bytes, bytes, bytes]) -> None:
        self.port = port
        self.commands = commands
        self._serial: serial.Serial | None = None
        self._current: bytes | None = None
        self._current_verified = False
        self._echo_confirmed = False
        self._closed = False
        self.callback = _ClockCallback(self._configure_callback)
        self.native = ClockControl(self.callback, None)

    def close(self) -> None:
        """Invalidate configuration and release the serial handle once."""
        self._closed = True
        self._current = None
        self._current_verified = self._echo_confirmed = False
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _open(self) -> serial.Serial:
        if self._closed:
            raise RuntimeError("PLL controller is closed; reconfigure USB")
        if self._serial is None:
            port = serial.Serial(
                port=None,
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.2,
                write_timeout=0.2,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            # Set control lines before opening to avoid intentional RTS/DTR pulses.
            port.rts = port.dtr = False
            port.port = self.port
            self._serial = port
            port.open()
            port.reset_input_buffer()
        return self._serial

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("PLL serial command timed out")
        return remaining

    def configure(self, phase: int) -> bool:
        """Send the phase command and report current-word echo evidence."""
        if not 0 <= phase < len(self.commands):
            raise ValueError("invalid CIM clock phase")
        port = self._open()
        command = self.commands[phase]
        if command == self._current:
            return self._current_verified
        deadline = time.monotonic() + 0.2
        offset = 0
        while offset < len(command):
            port.write_timeout = self._remaining(deadline)
            written = port.write(command[offset:])
            offset += written
            if written == 0:
                time.sleep(min(0.001, self._remaining(deadline)))
        verified = not self._echo_confirmed
        if verified:
            window = b""
            while True:
                port.timeout = self._remaining(deadline)
                # Read one byte so a complete echo never waits for a larger read.
                window = (window + port.read(1))[-3:]
                if window == command:
                    break
        else:
            while port.out_waiting:
                time.sleep(min(0.001, self._remaining(deadline)))
            # shortcut: driver queues omit adapter buffers; replace this 4 ms
            # wire-time guard when measured hardware completion is available.
            if self._remaining(deadline) < 0.004:
                raise TimeoutError("PLL serial transmit timed out")
            time.sleep(0.004)
        self._remaining(deadline)
        self._current = command
        self._current_verified = verified
        self._echo_confirmed |= verified
        return verified

    def _configure_callback(
        self,
        _context: int | None,
        phase: int,
        verified: _BytePointer,
        error: int | None,
        capacity: int,
    ) -> int:
        verified[0] = 0
        try:
            verified[0] = self.configure(phase)
            return 0
        except BaseException as failure:
            # No Python exception may escape a ctypes callback into C++.
            message = f"PLL serial {self.port}: {failure}"
            try:
                self.close()
            except BaseException as cleanup:
                message += f"; close failed: {cleanup}"
            if error and capacity:
                data = message.encode("utf-8", errors="replace")[: capacity - 1]
                ctypes.memmove(error, data + b"\0", len(data) + 1)
            return 1


class ClockResources:
    """Keep a native callback alive through destruction without owning Session."""

    def __init__(self) -> None:
        self.controller: PLLSerial | None = None
        self.failed = False
        self.configured = False

    def close(self) -> None:
        if self.controller is not None:
            self.controller.close()

    def fail(self, failure: BaseException) -> None:
        self.failed = True
        try:
            self.close()
        except Exception as cleanup:
            # Preserve the operation error while reporting cleanup failure.
            failure.add_note(f"PLL serial cleanup failed: {cleanup}")

    def require_ready(self) -> None:
        if self.failed:
            raise RuntimeError("USB clock state is faulted; reconfigure USB")
