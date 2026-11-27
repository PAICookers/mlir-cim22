"""Exercise the installed JAX frontend through the user-facing example CLI."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
def test_jax_example_cli(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("jax2onnx")
    pytest.importorskip("onnxruntime")
    script = Path(__file__).resolve().parents[1] / "examples" / "jax_compile.py"
    result_path = tmp_path / "results" / "report.json"
    subprocess.run(
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
        check=True,
        timeout=180,
    )
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["framework"] == "jax"
    assert report["runtime"] == "simulator"
    assert report["dispatch_ops"] > 0
    assert report["dispatch_count"] > 0
    (case,) = report["cases"]
    expected = np.broadcast_to([[33], [0]], (2, 16))
    for key in ("reference", "onnx_runtime", "actual"):
        np.testing.assert_array_equal(case[key], expected)
    assert Path(report["onnx_path"]).is_file()
    assert Path(report["packet_ir"]).is_file()
    assert (tmp_path / "compile.log").is_file()
