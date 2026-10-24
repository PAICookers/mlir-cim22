"""USB artifact selection and boundary checks without a physical device."""

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

from cim22 import Session, UsbConfig
from cim22.runtime import MODEL_SUFFIX

from ._generated import cim22_model_pb2 as proto

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
        torch.nn.Linear(64, 16).eval(),
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
