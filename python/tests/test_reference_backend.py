"""CPU reference uses original PyTorch operators without CIM compilation."""

import copy
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import cim22.backend as adapter


@pytest.fixture(autouse=True)
def reset_dynamo() -> Iterator[None]:
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("mode", [None, "ignored"])
def test_reference_preserves_eager_precision_without_native_compile(
    mode: str | None,
) -> None:
    model = torch.nn.Linear(64, 16).eval()
    with torch.no_grad():
        model.weight.fill_(0.1)
        model.bias.fill_(0.03)
    value = torch.full((2, 64), 0.1)
    options = {"runtime": "reference"}
    if mode is not None:
        options["bf16_compensation"] = mode
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options=options,
    )
    with (
        torch.inference_mode(),
        patch.object(adapter, "_compile", side_effect=AssertionError("native")),
        patch.object(
            adapter, "aot_autograd", side_effect=AssertionError("AOT")
        ),
    ):
        torch.testing.assert_close(
            compiled(value), model(value), rtol=0, atol=0
        )
        model.weight.add_(0.1)
        torch.testing.assert_close(
            compiled(value), model(value), rtol=0, atol=0
        )


def test_reference_preserves_state_mutation_and_output_aliases() -> None:
    class Stateful(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.zeros(1))

        def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
            value.add_(1)
            self.state.add_(value.sum())
            return {
                "input": value,
                "view": value.view(-1),
                "result": value * self.state,
            }

    model = Stateful()
    eager = copy.deepcopy(model)
    value = torch.ones(2, 3)
    expected_value = value.clone()
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "reference"},
    )
    with torch.inference_mode():
        for _ in range(2):
            expected = eager(expected_value)
            actual = compiled(value)
            torch.testing.assert_close(
                actual["result"], expected["result"], rtol=0, atol=0
            )
            torch.testing.assert_close(model.state, eager.state, rtol=0, atol=0)
            assert actual["input"] is value
            assert actual["view"].untyped_storage().data_ptr() == (
                value.untyped_storage().data_ptr()
            )


@pytest.mark.parametrize("option", ["artifact_dir", "debug_dir"])
def test_reference_rejects_native_artifact_options(
    tmp_path: Path,
    option: str,
) -> None:
    compiled = torch.compile(
        torch.relu,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "reference", option: tmp_path},
    )
    with torch.inference_mode(), pytest.raises(RuntimeError, match="artifacts"):
        compiled(torch.ones(1))
    assert not list(tmp_path.iterdir())


def test_reference_rejects_non_cpu_inputs_without_device_conversion() -> None:
    graph = torch.fx.symbolic_trace(torch.nn.ReLU())
    with torch.inference_mode(), pytest.raises(ValueError, match="CPU"):
        adapter.backend(
            graph,
            [torch.empty(1, device="meta")],
            options={"runtime": "reference"},
        )


def test_reference_does_not_import_torch_mlir_in_fresh_process() -> None:
    script = """
import builtins
import os
import torch
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith('torch_mlir'):
        raise AssertionError('reference imported torch_mlir')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
os.environ['CIM22_RUNTIME_LIBRARY'] = '/missing/cim22-reference-runtime'
os.environ['CIM22_COMPILE_TOOL'] = '/missing/cim22-reference-compiler'
from cim22 import backend
compiled = torch.compile(torch.relu, backend='cim22', fullgraph=True,
                         dynamic=False, options={'runtime': 'reference'})
with torch.inference_mode():
    torch.testing.assert_close(compiled(torch.tensor([-1., 2.])),
                               torch.tensor([0., 2.]), rtol=0, atol=0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
