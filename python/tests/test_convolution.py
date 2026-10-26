"""Real wheel imports retain grouped Conv bias while dispatching CIM work."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from cim22.runtime import MODEL_SUFFIX

if TYPE_CHECKING:
    from .conftest import EmbeddedModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


@pytest.mark.parametrize(
    ("input_channels", "output_channels", "bias_scale"),
    [(2, 4, 0.0), (2, 4, 1.0), (4, 6, 1.0)],
    ids=["multiplier-zero-bias", "multiplier-bias", "grouped-bias"],
)
def test_grouped_convolution_bias_offloads(
    input_channels: int,
    output_channels: int,
    bias_scale: float,
    tmp_path: Path,
    embedded_model: "EmbeddedModel",
) -> None:
    """Exercise expanded channel bias from the pinned torch-mlir importer."""
    model = torch.nn.Conv2d(
        input_channels, output_channels, 3, stride=2, padding=1, groups=2
    ).eval()
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.copy_(
            torch.arange(1, output_channels + 1, dtype=torch.float32)
            * bias_scale
        )
    logical_input = torch.ones(2, input_channels, 5, 5)
    logical_input[:, 1::2] *= 2.0
    # With unit weights, raw BF16 0x0001/0x0002 reconstruct integer counts
    # 1/2 per tap. The eager oracle consumes those logical counts directly.
    value = logical_input * 2.0**-133
    other = value * 2.0
    original = value.clone()
    with torch.inference_mode():
        expected = model(logical_input)
        expected_other = model(logical_input * 2.0)
        compiled = torch.compile(
            model,
            backend="cim22",
            fullgraph=True,
            dynamic=False,
            options={"runtime": "software", "artifact_dir": tmp_path},
        )
        torch.testing.assert_close(compiled(value), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            compiled(other), expected_other, rtol=0, atol=0
        )
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    assert embedded_model(artifact).kernels, "Expected real CIM offload"
