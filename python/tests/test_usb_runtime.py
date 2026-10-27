"""USB artifact selection and boundary checks without a physical device."""

import ctypes
import os
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

from cim22 import PLLConfig, Session, UsbConfig
from cim22.runtime import MODEL_SUFFIX, _PLLConfig, _UsbConfig

from ._generated import cim22_model_pb2 as proto


def test_public_runtime_exports() -> None:
    import cim22

    assert set(cim22.__all__) == {"PLLConfig", "Session", "UsbConfig"}


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
        ("pll", "/dev/ttyUSB0", TypeError),
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


@pytest.fixture
def pll_config() -> PLLConfig:
    return PLLConfig("/dev/serial/by-id/pll", "2828A4", "2c28a4", "3028A4")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("port", 1, TypeError),
        ("port", "", ValueError),
        ("port", "pll\0other", ValueError),
        ("config_hex", 0x2828A4, TypeError),
        ("config_hex", "2828A", ValueError),
        ("config_hex", "02828A4", ValueError),
        ("work_hex", "0x2828A4", ValueError),
        ("work_hex", "28 8A4", ValueError),
        ("readback_hex", "2828AG", ValueError),
        ("readback_hex", "000000", ValueError),
    ],
)
def test_pll_config_rejects_invalid_commands(
    pll_config: PLLConfig,
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=field):
        replace(pll_config, **{field: value})


def test_supplier_defaults_preserve_legacy_usb_abi(
    usb_config: UsbConfig,
) -> None:
    assert usb_config.physical_link == 0
    assert usb_config.logical_channel == 3
    assert usb_config.response_timeout_ms == 7000
    assert usb_config.pll is None
    # Existing native callers pass this unversioned structure by pointer.
    assert ctypes.sizeof(_UsbConfig) == 28
    assert _UsbConfig.transfer_timeout_ms.offset == 16
    assert _UsbConfig.response_timeout_ms.offset == 24


@pytest.fixture
def mock_usb_session() -> tuple[Session, Mock, Mock]:
    session = Session.__new__(Session)
    session._lock = threading.RLock()
    session._handle = 123
    legacy = Mock(return_value=0)
    with_pll = Mock(return_value=0)
    session._library = SimpleNamespace(
        cim22_session_configure_usb=legacy,
        cim22_session_configure_usb_with_pll=with_pll,
    )
    return session, legacy, with_pll


def test_manual_clock_uses_legacy_usb_entry(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, legacy, with_pll = mock_usb_session
    session.configure_usb(usb_config)
    legacy.assert_called_once()
    with_pll.assert_not_called()
    native = ctypes.cast(legacy.call_args.args[1], ctypes.POINTER(_UsbConfig))
    assert native.contents.physical_link == 0
    assert native.contents.response_timeout_ms == 7000


def test_optional_pll_commands_reach_separate_native_config(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    session, legacy, with_pll = mock_usb_session
    session.configure_usb(replace(usb_config, pll=pll_config))
    legacy.assert_not_called()
    with_pll.assert_called_once()
    native = ctypes.cast(with_pll.call_args.args[2], ctypes.POINTER(_PLLConfig))
    assert native.contents.port == os.fsencode(pll_config.port)
    assert native.contents.configuration_word == 0x2828A4
    assert native.contents.work_word == 0x2C28A4
    assert native.contents.readback_word == 0x3028A4


def test_missing_pll_device_is_delegated_to_native_manual_fallback(
    tmp_path: Path,
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    session, _, with_pll = mock_usb_session
    missing = tmp_path / "no-such-serial-device"
    assert not missing.exists()
    session.configure_usb(
        replace(usb_config, pll=replace(pll_config, port=str(missing)))
    )
    # Native open decides availability; Python must not reject an absent port.
    with_pll.assert_called_once()


def test_present_pll_error_is_not_hidden(
    usb_config: UsbConfig,
    pll_config: PLLConfig,
    mock_usb_session: tuple[Session, Mock, Mock],
) -> None:
    session, _, with_pll = mock_usb_session

    def fail(
        _handle: int,
        _usb: object,
        _pll: object,
        error: ctypes.Array,
        _capacity: int,
    ) -> int:
        error.value = b"PLL echo timeout"
        return 1

    with_pll.side_effect = fail
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


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize(
    "options", [{}, {"runtime": "usb"}], ids=["default", "explicit"]
)
def test_compiled_usb_backend(
    tmp_path: Path, embedded_model: "EmbeddedModel", options: dict[str, str]
) -> None:
    value = torch.ones(1, 64)
    compiled = torch.compile(
        torch.nn.Linear(64, 48).eval(),
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

    value = torch.ones(1, 64)
    compiled = torch.compile(
        torch.nn.Linear(64, 16).eval(),
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
        ({"unknown": True}, "Unknown CIM22"),
        (
            {"runtime": "software", "usb_config": UsbConfig(1, 2, 0x81, 1)},
            "requires runtime='usb'",
        ),
    ],
    ids=["runtime", "unknown-option", "software-usb-config"],
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
    session, legacy, with_pll = mock_usb_session
    selected = replace(
        usb_config,
        clock_frequency_mhz=24,
        pll_port="/dev/serial/by-id/pll",
    )
    assert selected.clock_setting.serial_hex == "2828A4"
    session.configure_usb(selected)
    legacy.assert_not_called()
    native = ctypes.cast(with_pll.call_args.args[2], ctypes.POINTER(_PLLConfig))
    assert native.contents.port == b"/dev/serial/by-id/pll"
    assert native.contents.configuration_word == 0x2828A4
    assert native.contents.work_word == 0x2828A4
    assert native.contents.readback_word == 0x2828A4
    changed = replace(selected, clock_frequency_mhz=32)
    session.configure_usb(changed)
    native = ctypes.cast(with_pll.call_args.args[2], ctypes.POINTER(_PLLConfig))
    assert native.contents.work_word == 0x2C20B4


def test_maybe_available_row_is_warned_and_its_literal_bytes_are_sent(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, _, with_pll = mock_usb_session
    with pytest.warns(RuntimeWarning, match="may be unstable"):
        selected = replace(
            usb_config, clock_frequency_mhz=284, pll_port="/missing-pll"
        )
    assert selected.clock_setting.actual_frequency_mhz == 283.2
    assert selected.clock_setting.status == "maybe available"
    session.configure_usb(selected)
    native = ctypes.cast(with_pll.call_args.args[2], ctypes.POINTER(_PLLConfig))
    assert native.contents.work_word == 0xE813A4


def test_frequency_without_serial_keeps_manual_clock(
    usb_config: UsbConfig, mock_usb_session: tuple[Session, Mock, Mock]
) -> None:
    session, legacy, with_pll = mock_usb_session
    selected = replace(usb_config, clock_frequency_mhz=24)
    assert selected.clock_setting.actual_frequency_mhz == 24
    session.configure_usb(selected)
    legacy.assert_called_once()
    with_pll.assert_not_called()


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"pll_port": "/dev/pll"}, ValueError, "requires clock_frequency_mhz"),
        ({"clock_frequency_mhz": 24, "pll_port": 1}, TypeError, "pll_port"),
        ({"clock_frequency_mhz": 24, "pll_port": ""}, ValueError, "pll_port"),
        (
            {"clock_frequency_mhz": 24, "pll_port": "a\0b"},
            ValueError,
            "pll_port",
        ),
        ({"clock_frequency_mhz": True}, TypeError, "clock_frequency_mhz"),
        ({"clock_frequency_mhz": 26}, ValueError, "not a listed target"),
    ],
)
def test_frequency_usb_configuration_validation(
    usb_config: UsbConfig,
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        replace(usb_config, **changes)


def test_raw_pll_and_frequency_cannot_conflict(
    usb_config: UsbConfig, pll_config: PLLConfig
) -> None:
    with pytest.raises(ValueError, match="either pll or clock_frequency_mhz"):
        replace(usb_config, pll=pll_config, clock_frequency_mhz=24)
