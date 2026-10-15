"""USB artifact selection and boundary checks without a physical device."""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from cim22 import UsbConfig, compile_exported, load_compiled
from cim22._generated import cim22_model_pb2 as proto

from .test_model_format import host_artifact, transaction_artifact

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
    with pytest.raises(error, match=field if field != "bus_number" else "both"):
        replace(usb_config, **{field: value})


def test_usb_config_explicit_device(usb_config: UsbConfig) -> None:
    configured = replace(usb_config, bus_number=1, device_address=2)
    assert (configured.bus_number, configured.device_address) == (1, 2)


@pytest.mark.integration
def test_usb_session_checks(
    model_library: "ModelLibrary", usb_config: UsbConfig
) -> None:
    model = transaction_artifact()
    model.runtime_backend = proto.USB
    compiled = load_compiled(model_library(model.SerializeToString(), 1))
    with compiled:
        assert compiled.session.num_transactions == 1
        capture = compiled.session.capture_usb
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
        compiled.session.configure_usb(usb_config)
    with pytest.raises(RuntimeError, match="closed"):
        compiled.session.capture_usb(0, inputs)


@pytest.mark.integration
def test_software_session_rejects_usb(
    model_library: "ModelLibrary", usb_config: UsbConfig
) -> None:
    model = host_artifact()
    model.ClearField("runtime_backend")
    with (
        load_compiled(model_library(model.SerializeToString(), 1)) as compiled,
        pytest.raises(RuntimeError, match="backend is not USB"),
    ):
        compiled.session.configure_usb(usb_config)


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize(
    "options", [{}, {"runtime": "usb"}], ids=["default", "explicit"]
)
def test_compiled_usb_backend(
    tmp_path: Path, embedded_model: "EmbeddedModel", options: dict[str, str]
) -> None:
    value = torch.ones(1, 64)
    program = torch.export.export(torch.nn.Linear(64, 16).eval(), (value,))
    artifact = tmp_path / "usb.so"
    with compile_exported(
        program, artifact_path=artifact, **options
    ) as compiled:
        assert compiled.session.num_transactions > 0
        with pytest.raises(RuntimeError, match="USB runtime is not configured"):
            compiled(value)
    model = embedded_model(artifact)
    assert model.data_format_version == 1
    assert model.runtime_backend == proto.USB
    assert all(
        binding.HasField("input_cache_address")
        for kernel in model.kernels
        for binding in kernel.transaction.inputs
    )
    with (
        load_compiled(artifact) as compiled,
        pytest.raises(RuntimeError, match="USB runtime is not configured"),
    ):
        compiled(value)
    with pytest.raises(ValueError, match="runtime must be"):
        compile_exported(program, runtime="invalid")
