"""Shared serial state machine, real ctypes boundary and Linux PTY checks."""

import ctypes
import gc
import os
import threading
import weakref
from unittest.mock import Mock

import pytest
import serial

from cim22._pll_serial import ClockControl, PLLSerial

COMMANDS = tuple(bytes.fromhex(word) for word in ("2828A4", "2C28A4", "3028A4"))


class FakeSerial:
    """Stateful port double: only the first command produces a fragmented echo."""

    def __init__(self, **settings: object) -> None:
        self.settings = settings
        self.is_open = False
        self.rts = self.dtr = True
        self.out_waiting = 0
        self.reply = bytearray()
        self.writes: list[bytes] = []
        self.reads = 0
        self.closes = 0
        self.reset_input_buffer = Mock()
        self.flush = Mock(side_effect=AssertionError("unbounded flush"))

    def open(self) -> None:
        assert not self.rts and not self.dtr
        self.is_open = True

    def write(self, data: bytes) -> int:
        if not self.writes:
            self.reply.extend(b"noise" + data)
        self.writes.append(data)
        return len(data)

    def read(self, count: int) -> bytes:
        self.reads += 1
        data = bytes(self.reply[:count])
        del self.reply[:count]
        return data

    def close(self) -> None:
        self.is_open = False
        self.closes += 1


@pytest.fixture
def port(monkeypatch: pytest.MonkeyPatch) -> FakeSerial:
    port = FakeSerial()
    factory = Mock(return_value=port)
    monkeypatch.setattr("cim22._pll_serial.serial.Serial", factory)
    return port


def call(
    controller: PLLSerial, phase: int, capacity: int = 256
) -> tuple[int, int, bytes]:
    verified = ctypes.c_uint8(9)
    error = ctypes.create_string_buffer(capacity)
    status = controller.native.callback(
        None, phase, ctypes.byref(verified), error, capacity
    )
    return status, verified.value, error.value


def test_first_echo_then_unchecked_changes_and_same_word_reuse(
    port: FakeSerial,
) -> None:
    controller = PLLSerial("COM10", COMMANDS)
    assert not port.is_open
    assert call(controller, 0) == (0, 1, b"")
    reads = port.reads
    assert call(controller, 0) == (0, 1, b"")
    assert call(controller, 1) == (0, 0, b"")
    assert call(controller, 2) == (0, 0, b"")
    assert call(controller, 0) == (0, 0, b"")
    assert port.writes == [*COMMANDS, COMMANDS[0]]
    assert port.reads == reads
    port.flush.assert_not_called()
    controller.close()
    controller.close()
    assert port.closes == 1


def test_identical_phase_words_send_once(port: FakeSerial) -> None:
    controller = PLLSerial("pll", (COMMANDS[0],) * 3)
    for phase in (0, 1, 2, 0, 1, 2):
        assert call(controller, phase) == (0, 1, b"")
    assert port.writes == [COMMANDS[0]]


def test_serial_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(side_effect=FakeSerial)
    monkeypatch.setattr("cim22._pll_serial.serial.Serial", factory)
    controller = PLLSerial("COM10", COMMANDS)
    assert call(controller, 0)[0] == 0
    settings = factory.call_args.kwargs
    assert settings == dict(
        port=None,
        baudrate=9600,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=0.2,
        write_timeout=0.2,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    controller.close()


@pytest.mark.parametrize("reply", [b"", b"wrong"])
def test_bad_or_missing_first_echo_closes_without_caching(
    port: FakeSerial, monkeypatch: pytest.MonkeyPatch, reply: bytes
) -> None:
    port.write = Mock(
        side_effect=lambda data: port.reply.extend(reply) or len(data)
    )
    ticks = iter([0, 0.01, 0.02, 0.3])
    monkeypatch.setattr(
        "cim22._pll_serial.time.monotonic", lambda: next(ticks, 0.3)
    )
    controller = PLLSerial("pll", COMMANDS)
    status, verified, error = call(controller, 0)
    assert status == 1 and verified == 0 and b"timed out" in error
    assert controller._current is None
    assert port.closes == 1
    assert b"reconfigure" in call(controller, 0)[2]


@pytest.mark.parametrize(
    "failure",
    [
        serial.SerialException("missing port"),
        PermissionError("permission denied"),
    ],
)
def test_explicit_port_errors_are_fatal(
    port: FakeSerial, failure: Exception
) -> None:
    port.open = Mock(side_effect=failure)
    status, verified, error = call(PLLSerial("pll", COMMANDS), 0)
    assert status == 1 and verified == 0
    assert str(failure).encode() in error
    assert port.closes == 1 and not port.writes


def test_partial_writes_preserve_byte_order(port: FakeSerial) -> None:
    chunks: list[bytes] = []

    def write(data: bytes) -> int:
        chunks.append(data[:1])
        if len(chunks) == 3:
            port.reply.extend(b"".join(chunks))
        return 1

    port.write = write
    controller = PLLSerial("pll", COMMANDS)
    assert call(controller, 0) == (0, 1, b"")
    assert b"".join(chunks) == COMMANDS[0]


@pytest.mark.parametrize("failure_point", ["write", "queue"])
def test_transmit_deadline_closes_port(
    port: FakeSerial, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    controller = PLLSerial("pll", COMMANDS)
    assert call(controller, 0)[0] == 0
    if failure_point == "write":
        port.write = Mock(return_value=0)
    else:
        port.out_waiting = 3
    ticks = iter([0, 0.01, 0.3])
    monkeypatch.setattr(
        "cim22._pll_serial.time.monotonic", lambda: next(ticks, 0.3)
    )
    assert b"timed out" in call(controller, 1)[2]
    assert controller._current is None and port.closes == 1


def test_error_buffer_truncation_and_base_exception_boundary(
    port: FakeSerial,
) -> None:
    port.open = Mock(side_effect=KeyboardInterrupt("interrupted"))
    status, verified, error = call(PLLSerial("pll", COMMANDS), 0, capacity=8)
    assert status == 1 and verified == 0 and error == b"PLL ser"


def test_new_controller_requires_new_echo(port: FakeSerial) -> None:
    first = PLLSerial("pll", COMMANDS)
    assert call(first, 0)[0] == 0
    first.close()
    port.writes.clear()
    assert call(PLLSerial("pll", COMMANDS), 0) == (0, 1, b"")
    assert port.reads > 3


def test_callback_lifetime_owned_by_resource_holder(port: FakeSerial) -> None:
    from cim22._pll_serial import ClockResources

    owner = ClockResources()
    owner.controller = PLLSerial("pll", COMMANDS)
    ref = weakref.ref(owner.controller)
    gc.collect()
    assert ref() is not None
    assert call(owner.controller, 0)[0] == 0
    owner.fail(RuntimeError("USB failed"))
    assert port.closes == 1
    with pytest.raises(RuntimeError, match="reconfigure"):
        owner.require_ready()


@pytest.mark.integration
@pytest.mark.parametrize("failed_phase", [None, 0, 1, 2])
def test_real_ctypes_callback_through_native_usb_stages(
    port: FakeSerial, failed_phase: int | None
) -> None:
    fixture = os.environ.get("CIM22_CLOCK_TEST_LIBRARY")
    if fixture is None:
        pytest.skip("requires the test-only native clock callback fixture")
    controller = PLLSerial("pll", COMMANDS)
    original = controller.configure
    phases: list[int] = []

    def configure(phase: int) -> bool:
        phases.append(phase)
        if phase == failed_phase:
            raise RuntimeError("injected serial failure")
        return original(phase)

    controller.configure = configure
    library = ctypes.CDLL(fixture)
    function = library.cim22_test_clock
    function.argtypes = [
        ctypes.POINTER(ClockControl),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    verified = (ctypes.c_uint8 * 3)()
    writes = ctypes.c_uint32()
    error = ctypes.create_string_buffer(256)
    status = function(
        ctypes.byref(controller.native),
        verified,
        ctypes.byref(writes),
        error,
        len(error),
    )
    if failed_phase is None:
        assert status == 0 and writes.value == 4
        assert phases == [0, 1, 2]
        assert list(verified) == [1, 0, 0]
    else:
        assert status != 0 and writes.value == failed_phase
        assert phases == list(range(failed_phase + 1))
        assert b"injected serial failure" in error.value


@pytest.mark.skipif(os.name == "nt", reason="POSIX pseudo-terminal")
def test_real_pyserial_pty_three_phases() -> None:
    import pty
    import select

    master, slave = pty.openpty()
    received: list[bytes] = []
    errors: list[BaseException] = []

    def board() -> None:
        try:
            for index, command in enumerate(COMMANDS):
                data = b""
                while len(data) < 3:
                    if not select.select([master], [], [], 2)[0]:
                        raise TimeoutError("test PTY did not receive command")
                    data += os.read(master, 3 - len(data))
                received.append(data)
                if index == 0:
                    os.write(master, b"noise" + command[:1])
                    os.write(master, command[1:])
        except BaseException as failure:
            errors.append(failure)

    thread = threading.Thread(target=board)
    thread.start()
    controller = PLLSerial(os.ttyname(slave), COMMANDS)
    try:
        assert [call(controller, phase)[:2] for phase in range(3)] == [
            (0, 1),
            (0, 0),
            (0, 0),
        ]
    finally:
        controller.close()
        thread.join(timeout=3)
        os.close(slave)
        os.close(master)
    assert not thread.is_alive() and not errors
    assert received == list(COMMANDS)
