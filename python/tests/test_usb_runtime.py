"""USB artifact selection and boundary checks without a physical device."""

import ctypes
import os
import threading
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

from cim22 import CIMTimingConfig, PLLConfig, Session, UsbConfig
from cim22._pll_serial import ClockControl, ClockResources
from cim22.runtime import (
    MODEL_SUFFIX,
    _HardwareConfig,
    _initialize_session,
    _release_session,
    _UsbConfig,
)

from ._generated import cim22_model_pb2 as proto


def test_public_runtime_exports() -> None:
    import cim22

    assert set(cim22.__all__) == {
        "CIMTimingConfig",
        "PLLConfig",
        "Session",
        "UsbConfig",
    }


if TYPE_CHECKING:
    from .conftest import EmbeddedModel, ModelLibrary


@pytest.fixture
def usb_config() -> UsbConfig:
    return UsbConfig(0x1234, 0x5678, 0x81, 0x01)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("vendor_id", 0, ValueError),
        ("product_id", 0x10000, ValueError),
        ("bulk_in_endpoint", 0x91, ValueError),
        ("bulk_in_endpoint", 1, ValueError),
        ("bulk_out_endpoint", 0x81, ValueError),
        ("bulk_out_endpoint", 0, ValueError),
        ("interface_number", -1, ValueError),
        ("alternate_setting", 256, ValueError),
        ("physical_link", 4, ValueError),
        ("logical_channel", 8, ValueError),
        ("word_byte_order", 2, ValueError),
        ("transfer_timeout_ms", 0, ValueError),
        ("receive_poll_timeout_ms", 2**32, ValueError),
        ("response_timeout_ms", 1.5, TypeError),
        ("vendor_id", True, TypeError),
        ("bus_number", 1, ValueError),
    ],
)
def test_usb_config_rejects_truncation(
    usb_config: UsbConfig, field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(
        error,
        match=field if os.name == "nt" or field != "bus_number" else "both",
    ):
        replace(usb_config, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("startup_delay_ms", -1, ValueError),
        ("work_delay_ms", 2**32, ValueError),
        ("readback_select_delay_ms", 1.5, TypeError),
    ],
)
def test_timing_config_rejects_invalid_values(
    field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match=field):
        replace(CIMTimingConfig(), **{field: value})


@pytest.fixture
def pll_config() -> PLLConfig:
    return PLLConfig("/dev/serial/by-id/pll", "2828A4", "2c28a4", "3028A4")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", ""),
        ("port", "pll\0other"),
        ("config_hex", "2828A"),
        ("config_hex", "02828A4"),
        ("work_hex", "0x2828A4"),
        ("work_hex", "28 8A4"),
        ("readback_hex", "2828AG"),
    ],
)
def test_pll_config_rejects_invalid_commands(
    pll_config: PLLConfig,
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(pll_config, **{field: value})


def test_usb_defaults_preserve_legacy_usb_abi(
    usb_config: UsbConfig,
) -> None:
    assert usb_config.physical_link == 0
    assert usb_config.logical_channel == 3
    assert usb_config.response_timeout_ms == 7000
    assert usb_config.timing.work_delay_ms == 10
    assert usb_config.timing.readback_select_delay_ms == 40
    assert usb_config.pll is None
    # Existing native callers pass this unversioned structure by pointer.
    assert ctypes.sizeof(_UsbConfig) == 28
    assert _UsbConfig.transfer_timeout_ms.offset == 16
    assert _UsbConfig.response_timeout_ms.offset == 24


def _make_mock_usb_session() -> tuple[Session, Mock, Mock]:
    session = Session.__new__(Session)
    session._lock = threading.RLock()
    session._handle = 123
    session._clock_resources = ClockResources()
    legacy = Mock(return_value=0)
    hardware = Mock(return_value=0)
    session._library = SimpleNamespace(
        cim22_session_configure_usb=legacy,
        cim22_session_configure_usb_with_clock=hardware,
    )
    return session, legacy, hardware


@pytest.fixture
def mock_usb_session() -> tuple[Session, Mock, Mock]:
    return _make_mock_usb_session()


@pytest.mark.parametrize("failure", [False, True])
def test_shared_session_initialization_owns_cleanup(
    usb_config: UsbConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
) -> None:
    """Session destruction must precede cleanup of its loaded model files."""
    session, _, _ = mock_usb_session
    events: list[str] = []
    session._resources = ExitStack()
    owned = ExitStack()
    owned.callback(events.append, "cleanup")

    def release() -> None:
        events.append("destroy")
        session._resources.close()

    def configure(config: UsbConfig) -> None:
        assert config is usb_config
        events.append("configure")
        if failure:
            raise RuntimeError("device unavailable")

    session._finalizer = release
    monkeypatch.setattr(session, "configure_usb", configure)
    if failure:
        with pytest.raises(RuntimeError, match="device unavailable"):
            _initialize_session(session, usb_config, owned)
    else:
        assert _initialize_session(session, usb_config, owned) is session
        assert events == ["configure"]
        session.close()
    assert session._handle is None
    assert events == ["configure", "destroy", "cleanup"]
    session.close()
    assert events == ["configure", "destroy", "cleanup"]


def test_cleanup_failure_leaves_session_closed(
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    """A failed file cleanup must not permit reuse of a destroyed native handle."""
    session, _, _ = mock_usb_session
    destroy = Mock()
    session._library.cim22_session_destroy = destroy
    session._resources = ExitStack()

    def fail() -> None:
        raise RuntimeError("cleanup failed")

    session._resources.callback(fail)
    session._finalizer = lambda: _release_session(
        session._library, 123, session._resources
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.close()
    destroy.assert_called_once_with(123)
    assert session._handle is None
    with pytest.raises(RuntimeError, match="closed"):
        session(np.ones(2, dtype=np.float32))
    session.close()
    destroy.assert_called_once_with(123)


def test_manual_clock_uses_unified_hardware_entry(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, legacy, with_hardware = mock_usb_session
    session.configure_usb(usb_config)
    legacy.assert_not_called()
    with_hardware.assert_called_once()
    native = ctypes.cast(
        with_hardware.call_args.args[1], ctypes.POINTER(_UsbConfig)
    )
    assert native.contents.physical_link == 0
    assert native.contents.response_timeout_ms == 7000


def test_hardware_timing_reaches_native_configuration() -> None:
    session = Session.__new__(Session)
    session._lock = threading.RLock()
    session._handle = 123
    session._clock_resources = ClockResources()
    configure = Mock(return_value=0)
    session._library = SimpleNamespace(
        cim22_session_configure_usb_with_clock=configure,
    )
    config = UsbConfig(
        0x1234,
        0x5678,
        0x81,
        0x01,
        timing=CIMTimingConfig(
            startup_delay_ms=3,
            cache_write_delay_ms=4,
            weight_write_delay_ms=5,
            work_delay_ms=6,
            readback_select_delay_ms=7,
        ),
    )
    session.configure_usb(config)
    configure.assert_called_once()
    native = ctypes.cast(
        configure.call_args.args[2], ctypes.POINTER(_HardwareConfig)
    )
    assert [
        native.contents.startup_delay_ms,
        native.contents.cache_write_delay_ms,
        native.contents.weight_write_delay_ms,
        native.contents.work_delay_ms,
        native.contents.readback_select_delay_ms,
    ] == [3, 4, 5, 6, 7]


def test_optional_pll_commands_reach_separate_native_config(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    session, legacy, with_hardware = mock_usb_session
    session.configure_usb(replace(usb_config, pll=pll_config))
    legacy.assert_not_called()
    with_hardware.assert_called_once()
    native = ctypes.cast(
        with_hardware.call_args.args[3], ctypes.POINTER(ClockControl)
    )
    assert native.contents.callback
    assert session._clock_resources.controller.port == pll_config.port
    assert session._clock_resources.controller.commands == (
        bytes.fromhex("2828A4"),
        bytes.fromhex("2C28A4"),
        bytes.fromhex("3028A4"),
    )


def test_present_pll_error_is_not_hidden(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    session, _, with_hardware = mock_usb_session

    def fail(
        _handle: int,
        _usb: object,
        _pll: object,
        _hardware: object,
        error: ctypes.Array,
        _capacity: int,
    ) -> int:
        error.value = b"PLL echo timeout"
        return 1

    with_hardware.side_effect = fail
    with pytest.raises(RuntimeError, match="PLL echo timeout"):
        session.configure_usb(replace(usb_config, pll=pll_config))


@pytest.mark.integration
def test_usb_session_checks(
    model_library: "ModelLibrary",
    usb_config: UsbConfig,
    transaction_artifact: proto.ModelArtifact,
) -> None:
    model = transaction_artifact
    model.backend = proto.USB
    compiled = Session(model_library(model.SerializeToString(), 1))
    with compiled:
        assert compiled.num_transactions == 1
        capture = compiled.capture_usb
        inputs = np.zeros((1, 64), dtype=np.uint16)
        with pytest.raises(RuntimeError, match="configured USB"):
            capture(0, inputs)
        with pytest.raises(TypeError, match="uint16"):
            capture(0, inputs.astype(np.float32))
        with pytest.raises(ValueError, match="contiguous"):
            capture(0, inputs[:, ::2])
        unaligned = np.ndarray(
            inputs.shape,
            dtype=np.uint16,
            buffer=bytearray(inputs.nbytes + 1),
            offset=1,
        )
        assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
        with pytest.raises(ValueError, match="aligned"):
            capture(0, unaligned)
        with pytest.raises(ValueError, match="transaction index"):
            capture(1, inputs)
    with pytest.raises(RuntimeError, match="closed"):
        compiled.configure_usb(usb_config)
    with pytest.raises(RuntimeError, match="closed"):
        compiled.capture_usb(0, inputs)


@pytest.mark.integration
@pytest.mark.parametrize("field", ["config_hex", "work_hex", "readback_hex"])
def test_python_rejects_pll_limits_before_opening_usb(
    model_library: "ModelLibrary",
    transaction_artifact: proto.ModelArtifact,
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    field: str,
) -> None:
    transaction_artifact.backend = proto.USB
    with pytest.raises(ValueError, match="PLL VCO must be at least 250 MHz"):
        replace(pll_config, **{field: "000000"})


@pytest.mark.integration
def test_software_session_rejects_usb(
    model_library: "ModelLibrary",
    usb_config: UsbConfig,
    host_artifact: proto.ModelArtifact,
) -> None:
    model = host_artifact
    model.ClearField("backend")
    with (
        Session(model_library(model.SerializeToString(), 1)) as compiled,
        pytest.raises(RuntimeError, match="backend is not USB"),
    ):
        compiled.configure_usb(usb_config)


class UsbInt8Matmul(torch.nn.Module):
    """Exercise INT8 USB routing and configuration failure cleanup."""

    def __init__(self, columns: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(64, columns, dtype=torch.int8), requires_grad=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch._int_mm(value, self.weight)


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize(
    "options", [{}, {"runtime": "usb"}], ids=["default", "explicit"]
)
def test_compiled_usb_backend(
    tmp_path: Path, embedded_model: "EmbeddedModel", options: dict[str, str]
) -> None:
    value = torch.ones(1, 64, dtype=torch.int8)
    compiled = torch.compile(
        UsbInt8Matmul(48).eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"artifact_dir": tmp_path, **options},
    )
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="USB runtime is not configured"),
    ):
        compiled(value)
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    model = embedded_model(artifact)
    assert model.kernels
    assert model.schema_version == 1
    assert model.backend == proto.USB
    assert all(
        binding.HasField("input_cache_address")
        for kernel in model.kernels
        for binding in kernel.transaction.inputs
    )
    # The supplier return field xy6(-15, 1)=0x3f is stored as -31.
    # Cover both macros and a non-ingress core independently of the forward route.
    works = [
        (work.core_slot, work.macro_slot)
        for kernel in model.kernels
        for group in kernel.transaction.groups
        for work in group.works
    ]
    assert works == [(0, 0), (0, 1), (1, 0)]
    for kernel in model.kernels:
        assert kernel.transaction.readbacks
        assert all(
            list(binding.test_core) == [-31, 0, 0]
            for binding in kernel.transaction.readbacks
        )
        routes = [
            packet
            for packet in kernel.transaction.packets
            if packet.kind == proto.RETURN_ROUTE
        ]
        assert len(routes) == len(kernel.transaction.readbacks)
        assert all(list(packet.test_core) == [-31, 0, 0] for packet in routes)


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
def test_backend_usb_configuration_failure_closes_session(
    usb_config: UsbConfig,
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    value = torch.ones(1, 64, dtype=torch.int8)
    compiled = torch.compile(
        UsbInt8Matmul(16).eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={
            "usb_config": usb_config,
            "artifact_dir": tmp_path / "models",
            "debug_dir": tmp_path / "ir",
        },
    )
    # Exercise configuration propagation/cleanup without opening physical USB.
    with (
        torch.inference_mode(),
        patch.object(
            Session,
            "configure_usb",
            autospec=True,
            side_effect=[RuntimeError("device unavailable"), None],
        ) as configure,
    ):
        with pytest.raises(BackendCompilerFailed, match="device unavailable"):
            compiled(value)
        # A mock successful configuration cannot open hardware. Reaching the
        # native dispatch proves retry passed publication and device setup.
        with pytest.raises(RuntimeError, match="USB runtime is not configured"):
            compiled(value)
    assert configure.call_count == 2
    sessions = [call.args[0] for call in configure.call_args_list]
    assert all(call.args[1] is usb_config for call in configure.call_args_list)
    assert len(list((tmp_path / "models").glob(f"*{MODEL_SUFFIX}"))) == 2
    assert len(list((tmp_path / "ir").iterdir())) == 2
    with pytest.raises(RuntimeError, match="closed"):
        sessions[0](value.numpy())
    sessions[1].close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"runtime": "invalid"}, "runtime must be"),
        (
            {"runtime": "simulator", "usb_config": UsbConfig(1, 2, 0x81, 1)},
            "requires runtime='usb'",
        ),
    ],
    ids=["runtime", "software-usb-config"],
)
def test_backend_rejects_invalid_options(
    options: dict[str, object], message: str
) -> None:
    import cim22.backend as adapter

    with pytest.raises(ValueError, match=message):
        adapter.backend(
            torch.fx.symbolic_trace(torch.nn.Identity()),
            [],
            options=options,
        )


def test_usb_address_selection(usb_config: UsbConfig) -> None:
    """CyAPI exposes addresses but no libusb bus number."""
    if os.name == "nt":
        assert replace(usb_config, device_address=3).device_address == 3
        with pytest.raises(ValueError, match="bus_number"):
            replace(usb_config, bus_number=1, device_address=3)
    else:
        with pytest.raises(ValueError, match="both"):
            replace(usb_config, device_address=3)
        assert (
            replace(usb_config, bus_number=1, device_address=3).device_address
            == 3
        )


def test_frequency_selection_reaches_all_native_pll_phases(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, legacy, with_hardware = mock_usb_session
    selected = replace(
        usb_config,
        clock_frequency_mhz=24,
        pll_port="/dev/serial/by-id/pll",
    )
    assert selected.clock_setting.serial_hex == "2828A4"
    session.configure_usb(selected)
    legacy.assert_not_called()
    controller = session._clock_resources.controller
    assert controller.port == "/dev/serial/by-id/pll"
    assert controller.commands == (bytes.fromhex("2828A4"),) * 3
    session.configure_usb(replace(selected, clock_frequency_mhz=32))
    assert controller._closed
    assert (
        session._clock_resources.controller.commands
        == (bytes.fromhex("2C20B4"),) * 3
    )


def test_maybe_available_row_is_warned_and_its_literal_bytes_are_sent(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, _, with_hardware = mock_usb_session
    with pytest.warns(RuntimeWarning, match="may be unstable"):
        selected = replace(
            usb_config, clock_frequency_mhz=284, pll_port="/missing-pll"
        )
    assert selected.clock_setting.actual_frequency_mhz == 283.2
    assert selected.clock_setting.status == "maybe available"
    session.configure_usb(selected)
    assert (
        session._clock_resources.controller.commands
        == (bytes.fromhex("E813A4"),) * 3
    )


def test_frequency_without_serial_keeps_manual_clock(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, legacy, with_hardware = mock_usb_session
    selected = replace(usb_config, clock_frequency_mhz=24)
    assert selected.clock_setting.actual_frequency_mhz == 24
    session.configure_usb(selected)
    legacy.assert_not_called()
    with_hardware.assert_called_once()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pll_port": "/dev/pll"}, "requires clock_frequency_mhz"),
        ({"clock_frequency_mhz": 24, "pll_port": ""}, "pll_port"),
        ({"clock_frequency_mhz": 24, "pll_port": "a\0b"}, "pll_port"),
    ],
)
def test_frequency_usb_configuration_validation(
    usb_config: UsbConfig,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(usb_config, **changes)


def test_raw_pll_and_frequency_cannot_conflict(
    usb_config: UsbConfig, pll_config: PLLConfig
) -> None:
    with pytest.raises(ValueError, match="either pll or clock_frequency_mhz"):
        replace(usb_config, pll=pll_config, clock_frequency_mhz=24)


def test_old_runtime_requires_reinstall(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, _, _ = mock_usb_session
    del session._library.cim22_session_configure_usb_with_clock
    with pytest.raises(RuntimeError, match="reinstall mlir-cim22"):
        session.configure_usb(usb_config)


def test_failed_reconfiguration_keeps_callback_alive_but_blocks_execution(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    session, _, configure = mock_usb_session
    session.configure_usb(replace(usb_config, pll=pll_config))
    previous = session._clock_resources.controller
    configure.return_value = 1
    with pytest.raises(RuntimeError):
        session.configure_usb(replace(usb_config, pll=pll_config))
    assert session._clock_resources.controller is previous
    assert previous._closed
    with pytest.raises(RuntimeError, match="reconfigure"):
        session()
    configure.return_value = 0
    session.configure_usb(usb_config)
    assert session._clock_resources.controller is None
    assert not session._clock_resources.failed


@pytest.mark.parametrize("capture", [False, True])
def test_native_execution_failure_closes_serial_and_requires_reconfigure(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
    capture: bool,
) -> None:
    session, _, _ = mock_usb_session
    session.configure_usb(replace(usb_config, pll=pll_config))
    controller = session._clock_resources.controller
    session.input_shapes = session.output_shapes = ()
    session.input_dtypes = session.output_dtypes = ()
    session.num_transactions = 1
    session._library.cim22_session_error = Mock(return_value=b"USB failed")
    session._library.cim22_session_invoke = Mock(return_value=1)
    session._library.cim22_session_num_readbacks = Mock(return_value=0)
    session._library.cim22_session_capture_usb = Mock(return_value=1)
    with pytest.raises(RuntimeError, match="USB failed"):
        if capture:
            session.capture_usb(0, np.zeros((1, 64), dtype=np.uint16))
        else:
            session()
    assert controller._closed
    with pytest.raises(RuntimeError, match="reconfigure"):
        session()


def test_session_gc_keeps_callback_until_native_destroy(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
) -> None:
    import gc
    import weakref

    session, _, _ = _make_mock_usb_session()
    session.configure_usb(replace(usb_config, pll=pll_config))
    controller = session._clock_resources.controller
    events: list[str] = []
    destroy = Mock(side_effect=lambda handle: events.append("destroy"))
    session._library.cim22_session_destroy = destroy
    session._resources = ExitStack()
    session._resources.callback(session._clock_resources.close)
    session._resources.callback(events.append, "cleanup")
    session._finalizer = weakref.finalize(
        session, _release_session, session._library, 123, session._resources
    )
    reference = weakref.ref(session)
    del session
    gc.collect()
    assert reference() is None
    assert controller._closed
    assert events == ["destroy", "cleanup"]
    destroy.assert_called_once_with(123)


@pytest.mark.parametrize("capture", [False, True])
def test_serial_cleanup_does_not_hide_native_execution_error(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
    capture: bool,
) -> None:
    session, _, _ = mock_usb_session
    session.configure_usb(replace(usb_config, pll=pll_config))
    controller = session._clock_resources.controller
    controller._serial = Mock(close=Mock(side_effect=OSError("close failed")))
    session.input_shapes = session.output_shapes = ()
    session.input_dtypes = session.output_dtypes = ()
    session.num_transactions = 1
    session._library.cim22_session_error = Mock(
        return_value=b"original USB error"
    )
    session._library.cim22_session_invoke = Mock(return_value=1)
    session._library.cim22_session_num_readbacks = Mock(return_value=0)
    session._library.cim22_session_capture_usb = Mock(return_value=1)
    with pytest.raises(RuntimeError, match="original USB error") as caught:
        if capture:
            session.capture_usb(0, np.zeros((1, 64), dtype=np.uint16))
        else:
            session()
    assert session._clock_resources.failed
    assert "close failed" in caught.value.__notes__[0]


@pytest.mark.parametrize("native_failure", [False, True])
def test_reconfiguration_cleanup_error_requires_explicit_recovery(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
    native_failure: bool,
) -> None:
    session, _, configure = mock_usb_session
    session.configure_usb(replace(usb_config, pll=pll_config))
    previous = session._clock_resources.controller
    previous._serial = Mock(close=Mock(side_effect=OSError("close failed")))

    def native(*args: object) -> int:
        args[4].value = b"original configuration error"
        return int(native_failure)

    configure.side_effect = native
    with pytest.raises(
        RuntimeError if native_failure else OSError,
        match="original configuration error"
        if native_failure
        else "close failed",
    ) as caught:
        session.configure_usb(replace(usb_config, pll=pll_config))
    assert session._clock_resources.failed
    if native_failure:
        assert "close failed" in caught.value.__notes__[0]
    else:
        assert session._clock_resources.controller._closed
