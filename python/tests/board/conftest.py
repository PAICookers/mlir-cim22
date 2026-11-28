"""Board configuration, evidence retention and production Session ownership."""

import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Literal, Protocol

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from cim22 import CIMTimingConfig, PLLConfig, Session, UsbConfig
from cim22.compiler import BackendOptions, BF16Compensation
from cim22.runtime import MODEL_SUFFIX, compile_tool, runtime_library

from .._generated import cim22_model_pb2 as proto
from ..conftest import EmbeddedModel
from .usb_observer import IlaCapture, UsbObservation

type Array = NDArray[np.float32] | NDArray[np.int32]


@dataclass(frozen=True)
class BoardExpectation:
    """Optional checks for a specific compiler placement and invocation."""

    inspect: Callable[[proto.ModelArtifact, str], None]
    dispatches: int
    compensation: BF16Compensation = "sign-split"
    reopen: bool = False
    verify_weights: Literal["residency", "overwrite"] | None = None


class BoardCheck(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        inputs: list[torch.Tensor],
        expected: list[Array] | None,
        *,
        expectation: BoardExpectation | None = None,
    ) -> None: ...


def assert_output(actual: Array, expected: Array) -> None:
    """Compare final ABI values exactly, including floating-point bit patterns."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(
        actual.view(np.uint32), expected.view(np.uint32)
    )


@pytest.fixture(scope="session")
def board_settings(
    pytestconfig: pytest.Config,
) -> tuple[UsbConfig, Path, dict[str, object]]:
    """Validate all local settings before allowing device initialization."""
    name = pytestconfig.getoption("board_config")
    if not name:
        raise pytest.UsageError(
            "--run-board requires --board-config=<JSON file>"
        )
    try:
        fields = json.loads(Path(name).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise pytest.UsageError(
            f"Cannot read board config {name}: {error}"
        ) from error
    if not isinstance(fields, dict):
        raise pytest.UsageError(
            "board config must be a JSON object of UsbConfig fields"
        )
    metadata_fields = dict(fields)
    if "clock_setting" in fields:
        raise pytest.UsageError(
            "clock_setting is runtime-derived; use clock_frequency_mhz/pll_port "
            "or pll"
        )
    if "pll" in fields:
        if not isinstance(fields["pll"], dict):
            raise pytest.UsageError("pll must be a JSON object")
        fields["pll"] = PLLConfig(**fields["pll"])
    if "pll" not in fields and not fields.get("pll_port"):
        raise pytest.UsageError("real board tests require pll_port or pll")
    if "timing" in fields:
        if not isinstance(fields["timing"], dict):
            raise pytest.UsageError("timing must be a JSON object")
        fields["timing"] = CIMTimingConfig(**fields["timing"])
    config = UsbConfig(**fields)
    metadata_fields["timing"] = asdict(config.timing)
    if config.pll is not None:
        clock_metadata = {
            "mode": "serial",
            "requested_frequency_mhz": None,
            "port": config.pll.port,
            "phase_commands": {
                "configuration": config.pll.config_hex,
                "work": config.pll.work_hex,
                "readback": config.pll.readback_hex,
            },
            "verification": "not confirmed",
        }
    else:
        setting = config.clock_setting
        assert setting is not None
        clock_metadata = {
            "mode": "serial",
            "requested_frequency_mhz": (
                config.clock_frequency_mhz
                if config.clock_frequency_mhz is not None
                else setting.target_frequency_mhz
            ),
            "selected_frequency_mhz": setting.actual_frequency_mhz,
            "table_target_frequency_mhz": setting.target_frequency_mhz,
            "table_status": setting.status,
            "port": config.pll_port,
            "phase_commands": {
                "configuration": setting.serial_hex,
                "work": setting.serial_hex,
                "readback": setting.serial_hex,
            },
            "verification": "not confirmed",
        }
    parent = Path(pytestconfig.getoption("board_output"))
    output = parent / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        [
            "git",
            "diff",
            "HEAD",
            "--",
            "python",
            "lib",
            "include",
            "runtime",
            "schema",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    native_files = [runtime_library(), compile_tool()]
    metadata = {
        "revision": revision,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "usb_config": metadata_fields,
        "clock": clock_metadata,
        "command": sys.argv,
        "python": sys.version,
        "torch": torch.__version__,
        "test_sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        "native_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in native_files
        },
        "ila_config": pytestconfig.getoption("board_ila_config"),
    }
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nBoard evidence: {output.resolve()}")
    return config, output, clock_metadata


@pytest.fixture
def board_repeats(pytestconfig: pytest.Config) -> int:
    """Number of calls in the repeated workload, including the cold call."""
    return pytestconfig.getoption("board_repeats")


@pytest.fixture(scope="session")
def board_ila(
    pytestconfig: pytest.Config,
) -> Iterator[IlaCapture | None]:
    name = pytestconfig.getoption("board_ila_config")
    if name is None:
        if pytestconfig.getoption("run_board"):
            raise pytest.UsageError(
                "Physical board tests require --board-ila-config"
            )
        yield None
    else:
        parent = Path(pytestconfig.getoption("board_output")).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        folder = parent / datetime.now().strftime("ila-%Y%m%d-%H%M%S-%f")
        with IlaCapture.start(Path(name).resolve(), folder) as capture:
            yield capture


@pytest.fixture
def board_check(
    request: pytest.FixtureRequest,
    board_settings: tuple[UsbConfig, Path, dict[str, object]],
    native_toolchain: None,
    embedded_model: EmbeddedModel,
    monkeypatch: pytest.MonkeyPatch,
    board_ila: IlaCapture | None,
) -> Iterator[BoardCheck]:
    """Check software first, then execute identical inputs through real USB.

    Instrument only Session ownership; compilation, configuration and inference
    use the unchanged production backend. Every acquired Session is closed.
    """
    config, output, clock_metadata = board_settings
    folder = output / request.node.name
    folder.mkdir()
    backend = importlib.import_module("cim22.backend")
    initialize = backend._initialize_session
    compile_graph = backend._compile
    sessions: list[Session] = []
    staged: tuple[Path, Path] | None = None
    active_expectation: BoardExpectation | None = None
    observers: dict[Session, UsbObservation] = {}

    def configure_session(
        session: Session, usb_config: UsbConfig, label: str
    ) -> None:
        if board_ila is None:
            session.configure_usb(usb_config)
        else:
            observer = UsbObservation(board_ila, (folder / label).resolve())
            observer.configure(session, usb_config)
            observers[session] = observer

    def close_session(session: Session) -> None:
        sessions.remove(session)
        observation = observers.pop(session, None)
        try:
            if observation is not None:
                observation.finish(session)
        finally:
            session.close()

    def capture_outputs(
        *args: object, artifact_path: Path, debug_dir: Path, **kwargs: object
    ) -> Session:
        nonlocal staged
        staged = (artifact_path, debug_dir / "model.packet.mlir")
        return compile_graph(
            *args, artifact_path=artifact_path, debug_dir=debug_dir, **kwargs
        )

    def retain(
        session: Session,
        usb_config: UsbConfig | None = None,
        resources: ExitStack | None = None,
        *,
        usb_session: Session | None = None,
    ) -> Session:
        sessions.append(session)
        assert session.num_transactions > 0, (
            "model was retained entirely on Host"
        )
        if usb_config is not None or active_expectation is not None:
            assert staged is not None
            artifact, packet = staged
            metadata = embedded_model(artifact)
            if active_expectation is not None:
                active_expectation.inspect(
                    metadata, packet.read_text(encoding="utf-8")
                )
        if usb_config is not None:
            assert metadata.backend == proto.USB
            (software_packet,) = (folder / "simulator").glob(
                "ir/*/model.packet.mlir"
            )
            usb_packet = packet
            assert usb_packet.read_bytes() == software_packet.read_bytes(), (
                "software and USB must execute the same lowered packet IR"
            )
            if board_ila is not None:
                initialize(session, None, resources)
                configure_session(session, usb_config, "wire")
                return session
        return initialize(
            session, usb_config, resources, usb_session=usb_session
        )

    monkeypatch.setattr(backend, "_compile", capture_outputs)
    monkeypatch.setattr(backend, "_initialize_session", retain)
    torch.compiler.reset()

    def check_model(
        model: torch.nn.Module,
        inputs: list[torch.Tensor],
        expected: list[Array] | None,
        *,
        expectation: BoardExpectation | None = None,
    ) -> None:
        nonlocal active_expectation
        active_expectation = expectation
        if not inputs or (
            expected is not None and len(inputs) != len(expected)
        ):
            raise ValueError("Each test input requires one reference output")
        references: list[Array] = []
        compensation = expectation.compensation if expectation else "sign-split"
        weight_check = expectation.verify_weights if expectation else None
        weights: list[list[tuple[int, bytes]]] = []
        report: dict[str, object] = {
            "strategy": compensation,
            "status": "started",
            "clock": dict(clock_metadata),
        }
        try:
            for runtime, backend_kind in (
                ("simulator", proto.SOFTWARE),
                ("usb", proto.USB),
            ):
                options: BackendOptions = {
                    "runtime": runtime,
                    "bf16_compensation": compensation,
                    "artifact_dir": folder / runtime,
                    "debug_dir": folder / runtime / "ir",
                }
                if runtime == "usb":
                    options["usb_config"] = config
                compiled = torch.compile(
                    model,
                    backend="cim22",
                    fullgraph=True,
                    dynamic=False,
                    options=options,
                )
                counts: list[int] = []
                elapsed_ms: list[float] = []
                result = {"dispatch_counts": counts, "call_ms": elapsed_ms}
                report[runtime] = result
                previous_tx = 0
                with torch.inference_mode():
                    for index, value in enumerate(inputs):
                        if runtime == "simulator":
                            np.save(
                                folder / f"input-{index}.npy", value.numpy()
                            )
                            if expected is not None:
                                np.save(
                                    folder / f"expected-{index}.npy",
                                    expected[index],
                                )
                        started = perf_counter_ns()
                        actual = compiled(value)
                        elapsed_ms.append((perf_counter_ns() - started) / 1e6)
                        actual = actual.numpy().copy()
                        np.save(folder / f"{runtime}-{index}.npy", actual)
                        counts.append(sessions[-1].dispatch_count)
                        if weight_check and (
                            observer := observers.get(sessions[-1])
                        ):
                            current_tx = observer.tx_position()
                            weights.append(
                                observer.weights_between(
                                    previous_tx, current_tx
                                )
                            )
                            previous_tx = current_tx
                        assert counts[-1] > 0, (
                            "no CIM dispatch during inference"
                        )
                        if expectation is not None:
                            assert counts[-1] == expectation.dispatches
                        if runtime == "simulator":
                            if expected is not None:
                                assert_output(actual, expected[index])
                            if actual.dtype == np.float32:
                                assert np.isfinite(actual).all()
                            references.append(actual)
                        else:
                            assert_output(actual, references[index])
                (artifact,) = (folder / runtime).glob(f"*{MODEL_SUFFIX}")
                assert embedded_model(artifact).backend == backend_kind
                close_session(sessions[-1])
                if (
                    runtime == "usb"
                    and expectation is not None
                    and expectation.reopen
                ):
                    reopened = Session(artifact)
                    sessions.append(reopened)
                    try:
                        configure_session(reopened, config, "wire-reopened")
                        (actual,) = reopened(inputs[0].numpy())
                        assert reopened.dispatch_count == expectation.dispatches
                        assert_output(actual, references[0])
                        np.save(folder / "usb-reopened.npy", actual)
                        result["reopened_dispatch_count"] = (
                            reopened.dispatch_count
                        )
                        if observer := observers.get(reopened):
                            weights.append(
                                observer.weights_between(
                                    0, observer.tx_position()
                                )
                            )
                    finally:
                        close_session(reopened)
                if len(elapsed_ms) > 1:
                    warm = elapsed_ms[1:]
                    result["warm_call_ms"] = {
                        "count": len(warm),
                        "median": median(warm),
                        "min": min(warm),
                        "max": max(warm),
                    }
            if weights:
                report["weight_write_slots"] = [
                    [slot for slot, _ in call] for call in weights
                ]
                if weight_check == "residency":
                    cold, warm, reloaded = weights
                    assert (
                        len(cold) == 1
                        and cold[0][0] == 0
                        and not warm
                        and cold == reloaded
                    )
                else:
                    seen: dict[int, bytes] = {}
                    changed = set()
                    for slot, data in weights[0]:
                        if slot in seen and data != seen[slot]:
                            changed.add(slot)
                        seen[slot] = data
                    assert changed, (
                        "No changed weight payload was written to a reused slot"
                    )
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=str(error))
            if "PLL" in str(error):
                report["clock"]["verification"] = "failed"
            else:
                report["clock"]["verification"] = "not confirmed"
            raise
        finally:
            (folder / "result.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )

    try:
        yield check_model
    finally:
        try:
            with ExitStack() as cleanup:
                for session in sessions:
                    cleanup.callback(close_session, session)
        finally:
            torch.compiler.reset()
