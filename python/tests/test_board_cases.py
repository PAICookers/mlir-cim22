"""Offline checks for board stimuli, numerical assertions and hardware opt-in."""

import importlib
import json
import os
import signal
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cim22 import Session, UsbConfig
from cim22.runtime import MODEL_SUFFIX

from .board import (
    test_bf16,
    test_bf16_scale,
    test_int8,
    test_placement,
    test_stability,
    usb_observer,
)
from .board.conftest import Array, BoardCheck, assert_output
from .board.conftest import board_check as board_check
from .board.conftest import board_ila as board_ila

pytest_plugins = ["pytester"]


def test_capture_timeout_reaps_its_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "ila.json"
    config.write_text(
        json.dumps(
            dict.fromkeys(
                (
                    "vivado_lab",
                    "hw_server",
                    "cs_server",
                    "target",
                    "device",
                    "ltx",
                    "ila_uuid",
                ),
                "test-value",
            )
        ),
        encoding="utf-8",
    )
    process = Mock(pid=123)
    process.wait.side_effect = [subprocess.TimeoutExpired("Vivado", 30), 1]
    launch = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", launch)
    if os.name == "nt":
        stop = Mock()
        monkeypatch.setattr(subprocess, "run", stop)
    else:
        stop = Mock()
        monkeypatch.setattr(os, "killpg", stop)
    folder = tmp_path / "capture"
    with (
        pytest.raises(RuntimeError, match="timed out during close"),
        usb_observer.IlaCapture.start(config, folder),
    ):
        pass
    assert (folder / "stop").exists()
    assert process.wait.call_count == 2
    if os.name == "nt":
        assert stop.call_args.args[0] == ["taskkill", "/pid", "123", "/t", "/f"]
    else:
        assert launch.call_args.kwargs["start_new_session"] is True
        stop.assert_called_once_with(123, signal.SIGKILL)


@pytest.fixture
def board_settings(
    tmp_path: Path,
) -> tuple[UsbConfig, Path, dict[str, object]]:
    """Keep harness substitution tests independent of all hardware settings."""
    config = UsbConfig(
        0x04B4,
        0xF1,
        0x81,
        1,
        clock_frequency_mhz=24,
        pll_port="/dev/pll",
    )
    return (
        config,
        tmp_path,
        {
            "mode": "serial",
            "requested_frequency_mhz": 24,
            "selected_frequency_mhz": 24,
            "table_target_frequency_mhz": 24,
            "table_status": "available",
            "port": "/dev/pll",
            "phase_commands": {
                "configuration": "2828A4",
                "work": "2828A4",
                "readback": "2828A4",
            },
            "verification": "not confirmed",
        },
    )


@pytest.fixture
def device_substitute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[list[Session]]:
    """Replace only physical USB execution with the already compiled simulator.

    This exercises both real backend artifacts, preflight ordering, assertions,
    evidence and cleanup; it supplies no evidence about hardware or transport.
    """
    redirected: dict[Session, Session] = {}
    opened: list[Session] = []
    invoke = Session.__call__
    count = Session.dispatch_count.fget

    def configure(session: Session, config: UsbConfig) -> None:
        opened.append(session)
        (artifact,) = tmp_path.glob(f"*/simulator/*{MODEL_SUFFIX}")
        redirected[session] = Session(artifact)

    def run(session: Session, *inputs: Array) -> tuple[Array, ...]:
        return invoke(redirected.get(session, session), *inputs)

    monkeypatch.setattr(Session, "configure_usb", configure)
    monkeypatch.setattr(Session, "__call__", run)
    monkeypatch.setattr(
        Session,
        "dispatch_count",
        property(lambda session: count(redirected.get(session, session))),
    )
    try:
        yield opened
    finally:
        for session in redirected.values():
            session.close()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["int8", "bf16", "bad-golden", "bad-packet"])
def test_harness_with_device_substitute(
    device_substitute: list[Session],
    board_check: BoardCheck,
    tmp_path: Path,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full orchestration must work and must block USB after bad preflight."""
    if kind == "bad-golden":
        model = test_int8.Int8Matmul(torch.ones(64, 16, dtype=torch.int8))
        with pytest.raises(AssertionError):
            board_check(
                model,
                [torch.ones(1, 64, dtype=torch.int8)],
                [np.zeros((1, 16), np.int32)],
            )
        assert not list(tmp_path.glob("*/usb"))
        assert not device_substitute
    elif kind == "bad-packet":
        compiler = importlib.import_module("cim22.compiler")
        emit = compiler._emit_object

        def alter_packet(
            packet: Path, object_file: Path, library: Path, runtime: str
        ) -> None:
            if runtime == "usb":
                packet.write_text(packet.read_text() + "\n// altered packet\n")
            emit(packet, object_file, library, runtime)

        monkeypatch.setattr(compiler, "_emit_object", alter_packet)
        with pytest.raises(RuntimeError, match="same lowered packet IR"):
            test_int8.test_int8_matmul(board_check, 64, 16, False)
        assert not device_substitute
    elif kind == "int8":
        test_int8.test_int8_matmul(board_check, 65, 17, False)
    else:
        test_bf16.test_signed_bias_and_repeat(board_check)
    (report,) = tmp_path.glob("*/result.json")
    result = json.loads(report.read_text())
    assert result["status"] == (
        "failed" if kind.startswith("bad-") else "passed"
    )
    assert list(tmp_path.glob("*/simulator-0.npy"))


@pytest.fixture
def simulator_check(
    tmp_path: Path, native_toolchain: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[BoardCheck]:
    """Run the exact board case builders without configuring or opening USB."""
    sessions: list[Session] = []
    backend = importlib.import_module("cim22.backend")

    def retain(
        session: Session,
        usb_config: None = None,
        *,
        usb_session: None = None,
    ) -> Session:
        assert usb_config is None
        assert usb_session is None
        assert session.num_transactions > 0
        sessions.append(session)
        return session

    monkeypatch.setattr(backend, "_initialize_session", retain)
    torch.compiler.reset()

    def check(
        model: torch.nn.Module,
        inputs: list[torch.Tensor],
        expected: list[Array] | None,
    ) -> None:
        compiled = torch.compile(
            model,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={
                "runtime": "simulator",
                "bf16_compensation": "sign-split",
                "debug_dir": tmp_path,
            },
        )
        outputs = []
        with torch.inference_mode():
            for index, value in enumerate(inputs):
                actual = compiled(value).numpy().copy()
                assert sessions[-1].dispatch_count > 0
                if expected is not None:
                    assert_output(actual, expected[index])
                elif actual.dtype == np.float32:
                    assert np.isfinite(actual).all()
                outputs.append(actual)
        if len(inputs) > 1 and torch.equal(inputs[-1], inputs[0]):
            assert_output(outputs[-1], outputs[0])

    try:
        yield check
    finally:
        for session in sessions:
            session.close()
        torch.compiler.reset()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case", "args"),
    [
        (test_int8.test_int8_matmul, (64, 16, False)),
        (test_int8.test_int8_matmul, (65, 17, False)),
        (test_int8.test_int8_matmul, (128, 48, False)),
        (test_int8.test_int8_matmul, (128, 16, True)),
        (test_bf16.test_signed_bias_and_repeat, ()),
        (test_bf16.test_disjoint_support, ()),
        (test_bf16.test_tails_lanes_and_k_sum, ()),
        (test_bf16.test_exponents_and_fractions, ()),
        (test_bf16.test_general_numeric_differential, ()),
        *[(case, ()) for _, case in test_bf16_scale.SCALE_CASES],
        (test_stability.test_pointwise_repeated, (6,)),
    ],
    ids=[
        "int8-tile",
        "int8-tails",
        "int8-core",
        "int8-wide-sum",
        "bf16-signs",
        "bf16-support",
        "bf16-tails",
        "bf16-exponents",
        "bf16-general",
        *[name for name, _ in test_bf16_scale.SCALE_CASES],
        "pointwise-repeated",
    ],
)
def test_case_software_reference(
    simulator_check: BoardCheck, case: Callable, args: tuple
) -> None:
    """Prove independent expected values before a user enables real hardware."""
    case(simulator_check, *args)


def test_numeric_corruption_is_rejected() -> None:
    """A one-bit payload error, signed zero or shape/dtype error must fail."""
    expected = np.array([[1.0, 0.0]], np.float32)
    changed = expected.copy()
    changed.view(np.uint32)[0, 0] ^= 1
    for actual in (
        changed,
        -expected,
        expected.reshape(2),
        expected.astype(np.int32),
    ):
        with pytest.raises(AssertionError):
            assert_output(actual, expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case", "sessions"),
    [
        (test_placement.test_int8_single_macro_residency_and_reopen, 2),
        (test_placement.test_int8_multiple_transactions, 1),
        (test_placement.test_bf16_single_transaction_uncompensated, 1),
        (test_placement.test_bf16_host_handoff, 1),
    ],
    ids=[
        "int8-single-and-reopen",
        "int8-multiple",
        "bf16-single",
        "host-handoff",
    ],
)
def test_placement_preflight(
    board_check: BoardCheck,
    device_substitute: list[Session],
    case: Callable[[BoardCheck], None],
    sessions: int,
) -> None:
    """Check both artifacts and Session orchestration before physical USB."""
    case(board_check)
    assert len(device_substitute) == sessions


@pytest.mark.parametrize(
    "enabled", [False, True], ids=["default-skip", "stop-on-failure"]
)
def test_board_opt_in_and_failure_stop(
    pytester: pytest.Pytester, enabled: bool
) -> None:
    """Exercise real pytest ordering without importing a hardware runner."""
    python_root = Path(__file__).resolve().parents[1]
    pytester.makeconftest(
        f"import sys\nsys.path.insert(0, {str(python_root)!r})\n"
        "from tests.conftest import (pytest_addoption, pytest_configure, "
        "pytest_collection_modifyitems, pytest_runtest_setup, pytest_runtest_makereport)\n"
    )
    pytester.makeini("[pytest]\nmarkers = board: physical hardware\n")
    pytester.makepyfile(
        "import pytest\nfrom pathlib import Path\npytestmark = pytest.mark.board\n"
        "def test_first():\n    assert False, 'intentional failure'\n"
        "def test_second():\n    Path('unexpected-device-access').touch()\n"
    )
    result = pytester.runpytest_subprocess(
        "-q", *(["--run-board"] if enabled else [])
    )
    result.assert_outcomes(
        failed=1 if enabled else 0, skipped=1 if enabled else 2
    )
    assert not (pytester.path / "unexpected-device-access").exists()
