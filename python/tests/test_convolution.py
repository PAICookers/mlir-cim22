"""Real wheel imports retain grouped Conv bias while dispatching CIM work."""

import pytest
import torch

from cim22 import compile_exported

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
    input_channels: int, output_channels: int, bias_scale: float
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
    value = torch.ones(2, input_channels, 5, 5)
    value[:, 1::2] *= 2.0
    other = value * 2.0
    original = value.clone()
    with torch.inference_mode():
        expected = model(value)
        expected_other = model(other)
        program = torch.export.export(model, (value,), strict=True)
        with compile_exported(program) as compiled:
            assert compiled.session.num_transactions > 0
            torch.testing.assert_close(
                compiled(value), expected, rtol=0, atol=0
            )
            assert compiled.session.dispatch_count > 0
            torch.testing.assert_close(
                compiled(other), expected_other, rtol=0, atol=0
            )
            torch.testing.assert_close(value, original, rtol=0, atol=0)
