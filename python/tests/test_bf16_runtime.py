"""BF16 compensation preserves signed inputs and strategy-specific artifacts."""

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from cim22 import Session
from cim22.compiler import BF16Compensation
from cim22.runtime import MODEL_SUFFIX

from ._generated import cim22_model_pb2 as proto

if TYPE_CHECKING:
    from .conftest import EmbeddedModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


@pytest.fixture(autouse=True)
def reset_dynamo() -> Iterator[None]:
    """Keep independent strategy cases out of Dynamo's shared cache budget."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def test_default_sign_split_handles_signed_inputs_and_bias(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(64, 16).eval()
    with torch.no_grad():
        model.weight[:, ::2] = 0.5
        model.weight[:, 1::2] = -0.25
        model.bias.fill_(1.0)
    value = torch.ones(1, 64)
    value[:, 1::2] = -2
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={
            "runtime": "simulator",
            "artifact_dir": tmp_path / "artifacts",
            "debug_dir": tmp_path / "debug",
        },
    )
    values = (value, -value, torch.zeros_like(value), value)
    with torch.inference_mode():
        for current in values:
            torch.testing.assert_close(
                compiled(current), model(current), rtol=0, atol=0
            )
    (packet,) = (tmp_path / "debug").glob("*/model.packet.mlir")
    assert 'cim.bf16_compensation = "sign-split"' in packet.read_text()


def test_invalid_compensation_option() -> None:
    compiled = torch.compile(
        torch.relu,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "simulator", "bf16_compensation": "unknown"},
    )
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="bf16_compensation must"),
    ):
        compiled(torch.ones(1))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"runtime": "usb", "bf16_compensation": "legacy"},
        {"runtime": "usb", "bf16_compensation": "none"},
    ],
    ids=["default-sign-split", "legacy", "none"],
)
def test_usb_compensation_requires_device_setup(
    tmp_path: Path, embedded_model: "EmbeddedModel", options: dict[str, str]
) -> None:
    compiled = torch.compile(
        torch.nn.Linear(64, 16).eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={**options, "artifact_dir": tmp_path},
    )
    value = torch.ones(1, 64)
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="USB runtime is not configured"),
    ):
        compiled(value)
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    assert embedded_model(artifact).backend == proto.USB


def test_compensation_strategies_persist_and_have_separate_specializations(
    tmp_path: Path, embedded_model: "EmbeddedModel"
) -> None:
    model = torch.nn.Linear(64, 16, bias=False).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[:, 0] = 1
        model.weight[:, 1] = -0.5
    value = torch.zeros(1, 64)
    value[:, :2] = 1
    modes: list[BF16Compensation] = ["sign-split", "legacy", "none"]
    artifact_dir = tmp_path / "artifacts"
    debug_dir = tmp_path / "debug"
    compiled = [
        torch.compile(
            model,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={
                "runtime": "simulator",
                "bf16_compensation": mode,
                "artifact_dir": artifact_dir,
                "debug_dir": debug_dir,
            },
        )
        for mode in modes
    ]
    expected = (0.5, -0.5, -np.inf)
    symbols = (
        "cim22_dispatch_sign_split",
        "cim22_dispatch_legacy",
        "cim22_dispatch",
    )
    artifacts: list[Path] = []
    with torch.inference_mode():
        for index, mode in enumerate(modes):
            before = set(artifact_dir.glob(f"*{MODEL_SUFFIX}"))
            before_packets = set(debug_dir.glob("*/model.packet.mlir"))
            reference = np.full((1, 16), expected[index], np.float32)
            np.testing.assert_array_equal(
                compiled[index](value).numpy(), reference
            )
            (artifact,) = set(artifact_dir.glob(f"*{MODEL_SUFFIX}")) - before
            (packet,) = (
                set(debug_dir.glob("*/model.packet.mlir")) - before_packets
            )
            assert f'cim.bf16_compensation = "{mode}"' in packet.read_text()
            assert symbols[index].encode() in artifact.read_bytes()
            assert embedded_model(artifact).backend == proto.SOFTWARE
            for path in (packet, artifact):
                with Session(path) as session:
                    np.testing.assert_array_equal(
                        session(value.numpy())[0], reference
                    )
                    assert session.dispatch_count > 0
            artifacts.append(artifact)
        np.testing.assert_array_equal(
            compiled[0](value).numpy(),
            np.full((1, 16), expected[0], np.float32),
        )
    assert len(list(artifact_dir.glob(f"*{MODEL_SUFFIX}"))) == 3
    # One fresh process checks all strategies without the compiler's state.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
import numpy as np
from cim22 import Session
x = np.zeros((1, 64), np.float32)
x[:, :2] = 1
for path, expected in zip(sys.argv[1:], (0.5, -0.5, -np.inf), strict=True):
    with Session(Path(path)) as session:
        np.testing.assert_array_equal(
            session(x)[0], np.full((1, 16), expected, np.float32)
        )
        assert session.dispatch_count > 0
""",
            *map(str, artifacts),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
