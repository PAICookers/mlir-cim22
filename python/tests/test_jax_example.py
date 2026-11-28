"""Exercise the integer Dense model and its ONNX export through the JAX CLI."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
def test_jax_example_cli(tmp_path: Path) -> None:
    if any(
        importlib.util.find_spec(name) is None
        for name in ("jax", "jax2onnx", "onnxruntime")
    ):
        pytest.skip("requires the optional JAX dependencies")
    script = Path(__file__).resolve().parents[1] / "examples" / "jax_compile.py"
    result_path = tmp_path / "results" / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runtime",
            "simulator",
            "--output",
            str(tmp_path),
            "--result-json",
            str(result_path),
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result_path.read_text("utf-8"))
    assert report["framework"] == "jax"
    assert report["runtime"] == "simulator"
    assert report["dispatch_ops"] > 0
    assert report["dispatch_count"] > 0
    assert report["passed_groups"] == 3
    assert Path(report["onnx_path"]).is_file()
    assert Path(report["packet_ir"]).is_file()
    lines = ["Framework: JAX | Backend: simulator | Model: Dense(64→16) + ReLU"]
    # Independent fingerprints for the zero, repeating and seed-42 inputs.
    expected_prefixes = [[0, 0, 3, 3], [16, 48, 6, 53], [0, 39, 0, 0]]
    for index, (row, prefix) in enumerate(
        zip(report["cases"], expected_prefixes, strict=True), 1
    ):
        assert row["name"] == f"Group {index}"
        actual = np.asarray(row["actual"])
        assert actual.shape == (4, 16)
        assert row["actual_dtype"] == "int32"
        assert actual[0, :4].tolist() == prefix
        for key in ("reference", "numpy_reference", "onnx_runtime"):
            np.testing.assert_array_equal(actual, row[key])
        assert row["max_abs_error"] == 0
        assert row["correct"] and row["passed"]
        count = row["dispatch_count"]
        assert count > 0
        lines.append(f"Group {index} | Correct | CIM dispatches: {count}")
    lines.append("3/3 passed")
    assert result.stdout.splitlines() == lines
    assert (tmp_path / "compile.log").read_text("utf-8").splitlines() == lines
