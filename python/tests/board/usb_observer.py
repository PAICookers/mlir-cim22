"""Connect the test-only runtime observer to a local Vivado Lab capture."""

import csv
import ctypes
import json
import os
import signal
import subprocess
from contextlib import contextmanager, suppress
from pathlib import Path
from time import monotonic, sleep

from cim22 import Session, UsbConfig
from cim22._pll_serial import ClockControl
from cim22.runtime import _HardwareConfig, _UsbConfig

_Gate = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
)
_SIGNALS = (
    "in_counter[31:0]",
    "out_counter[31:0]",
    "gpif_write_counter[31:0]",
    "available_read_from_usb",
    "request_write_to_usb",
    "left_down_request_in_chip_0_io_OBUF",
    "left_down_acknowledge_out_chip_0_io_IBUF",
    "left_down_request_out_chip_1_io_IBUF",
    "left_down_acknowledge_in_chip_1_io_OBUF",
)


class IlaCapture:
    """Keep one ILA process alive and exchange numbered capture requests."""

    def __init__(self, folder: Path, process: subprocess.Popen) -> None:
        self.folder = folder
        self.process = process
        self.index = 0

    @classmethod
    @contextmanager
    def start(cls, configuration: Path, folder: Path):
        config = json.loads(configuration.read_text(encoding="utf-8-sig"))
        command = [
            config["vivado_lab"],
            "-mode",
            "batch",
            "-log",
            str(folder / "vivado.log"),
            "-journal",
            str(folder / "vivado.jou"),
            "-source",
            str(Path(__file__).with_name("capture.tcl")),
            "-tclargs",
            str(folder),
            config["hw_server"],
            config["cs_server"],
            config["target"],
            config["device"],
            config["ltx"],
            config["ila_uuid"],
        ]
        folder.mkdir()
        with (folder / "console.log").open("wb") as console:
            process = subprocess.Popen(
                command,
                cwd=folder,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            try:
                yield cls(folder, process)
            finally:
                (folder / "stop").touch()
                try:
                    status = process.wait(timeout=30)
                except subprocess.TimeoutExpired as error:
                    # The launcher can have a separate Vivado process.
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/pid", str(process.pid), "/t", "/f"],
                            stdout=console,
                            stderr=subprocess.STDOUT,
                            timeout=10,
                            check=False,
                        )
                    else:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                    raise RuntimeError(
                        f"ILA capture timed out during close, see {console.name}"
                    ) from error
                if status:
                    raise RuntimeError(
                        f"ILA capture failed, see {console.name}"
                    )

    def sample(self) -> tuple[int, tuple[int, ...]]:
        index = self.index
        self.index += 1
        path = self.folder / f"sample-{index}.csv"
        (self.folder / f"request-{index}").touch(exist_ok=False)
        deadline = monotonic() + (90 if index == 0 else 30)
        while not path.is_file():
            if self.process.poll() is not None or monotonic() >= deadline:
                raise RuntimeError(f"ILA did not produce {path}")
            sleep(0.01)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            next(rows)  # Vivado radix row.
            records = list(rows)
            samples = {
                tuple(int(row[name], 16) for name in _SIGNALS)
                for row in records
            }
        if len(records) != 1024 or len(samples) != 1:
            raise RuntimeError(f"Expected stable ILA signals in {path}")
        (values,) = samples
        if (
            values[3]
            or values[4]
            or values[5] != values[6]
            or values[7] != values[8]
        ):
            raise RuntimeError(
                f"FPGA FIFO or handshake busy in {path}: {values}"
            )
        return index, values


class UsbObservation:
    """Scope captured traffic and hardware evidence to one USB Session."""

    def __init__(self, capture: IlaCapture, folder: Path) -> None:
        self.capture = capture
        self.folder = folder
        self.folder.mkdir()
        self.gates = []
        self.checked_tx = 0
        self.callback = _Gate(self._before_work)

    def _check(self, tx: int, rx: int) -> tuple[int, tuple[int, ...]]:
        index, actual = self.capture.sample()
        counters = [
            (actual[i] - self.baseline[i]) % 2**32 * 4 for i in range(3)
        ]
        if tx % 12 or rx % 12 or counters != [tx, rx, rx]:
            raise RuntimeError(
                f"FPGA byte counts {counters} differ from completed TX/RX {[tx, rx]}"
            )
        with (
            (self.folder / "tx.bin").open("rb") as sent,
            (self.folder / "planned.bin").open("rb") as planned,
        ):
            sent.seek(self.checked_tx)
            planned.seek(self.checked_tx)
            size = tx - self.checked_tx
            actual_bytes = sent.read(size)
            if len(actual_bytes) != size or actual_bytes != planned.read(size):
                raise RuntimeError(
                    "Completed USB OUT differs from the encoded plan"
                )
        self.checked_tx = tx
        return index, actual

    def _before_work(self, context: int, tx: int, rx: int) -> int:
        event = {"tx": tx, "rx": rx}
        self.gates.append(event)
        try:
            event["sample"], event["ila"] = self._check(tx, rx)
        except BaseException as error:
            # A ctypes callback must always return a defined gate decision.
            event["error"] = str(error)
            return 1
        return 0

    def configure(self, session: Session, config: UsbConfig) -> None:
        library = session._library
        try:
            native = library.cim22_board_configure_usb
        except AttributeError as error:
            raise RuntimeError(
                "Load a runtime built with CIM22_BOARD_OBSERVER=ON"
            ) from error
        native.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_UsbConfig),
            ctypes.POINTER(_HardwareConfig),
            ctypes.POINTER(ClockControl),
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            _Gate,
            ctypes.c_void_p,
        ]
        native.restype = ctypes.c_int
        self.initial_sample, self.baseline = self.capture.sample()

        class ObservedLibrary:
            def __getattr__(proxy, name: str):
                return getattr(library, name)

        def configure(*args):
            return native(*args, str(self.folder).encode(), self.callback, None)

        proxy = ObservedLibrary()
        proxy.cim22_session_configure_usb_with_clock = configure
        session._library = proxy
        try:
            session.configure_usb(config)
        finally:
            session._library = library

    def tx_position(self) -> int:
        return (self.folder / "tx.bin").stat().st_size

    def finish(self, session: Session) -> None:
        native = session._library.cim22_board_close_usb
        native.argtypes = [ctypes.c_void_p]
        native.restype = ctypes.c_int
        report = {"initial_sample": self.initial_sample, "gates": self.gates}
        try:
            faulted = native(session._handle)
            tx = self.tx_position()
            rx = (self.folder / "rx.bin").stat().st_size
            report.update(tx=tx, rx=rx, faulted=bool(faulted))
            report["final_sample"], report["final_ila"] = self._check(tx, rx)
            if faulted or (self.folder / "planned.bin").stat().st_size != tx:
                raise RuntimeError(
                    "USB faults or incomplete writes at session close"
                )
        finally:
            (self.folder / "observed.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )

    def weights_between(self, begin: int, end: int) -> list[tuple[int, bytes]]:
        """Read complete weight records from the actual OUT stream."""
        result = []
        with (
            (self.folder / "weights.csv").open(newline="") as index,
            (self.folder / "tx.bin").open("rb") as sent,
        ):
            for row in csv.DictReader(index):
                start, stop = int(row["begin"]), int(row["end"])
                if begin <= start < end:
                    assert stop <= end and stop - start == 257 * 12
                    sent.seek(start)
                    data = sent.read(stop - start)
                    assert len(data) == stop - start
                    result.append((int(row["slot"]), data))
        return result
