"""Examples log their progress and fail on incorrect inference."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from cim22.runtime import MODEL_SUFFIX

requires_tensorflow = pytest.mark.skipif(
    any(
        importlib.util.find_spec(name) is None
        for name in ("tensorflow", "tf2onnx")
    ),
    reason="requires the optional TensorFlow dependencies",
)


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize(
    ("script", "framework", "count"),
    [
        ("torch_compile.py", "PyTorch", 5),
        pytest.param(
            "tensorflow_compile.py", "TensorFlow", 1, marks=requires_tensorflow
        ),
    ],
)
def test_example_logging(
    tmp_path: Path, script: str, framework: str, count: int
) -> None:
    """Both CLIs log progress and result paths without dumping tensors."""
    example = Path(__file__).resolve().parents[1] / "examples" / script
    (tmp_path / "artifacts").mkdir()
    log = tmp_path / "artifacts/compile.log"
    log.write_text("existing process output\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--runtime",
            "simulator",
            "--output",
            "artifacts",
            "--result-json",
            "custom-result.json",
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONSAFEPATH": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert all(
        re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO .+", line)
        for line in lines
    )
    messages = [line[25:] for line in lines]
    assert messages[0] == f"Starting {framework} example (runtime=simulator)"
    assert messages[2] == "Compiling and running inference"
    assert result.stdout.count("Output matches reference:") == count
    assert (
        f"Completed inference (cases={count}, outputs match reference)"
        in result.stdout
    )
    assert "Results saved to custom-result.json" in result.stdout
    assert "[[[" not in result.stdout
    evidence = json.loads((tmp_path / "custom-result.json").read_text())
    assert len(evidence["cases"]) == count
    assert all(case["max_abs_error"] == 0 for case in evidence["cases"])
    assert list((tmp_path / "artifacts").glob(f"graph-*{MODEL_SUFFIX}"))
    assert (tmp_path / "artifacts/ir").is_dir()
    assert log.read_text(encoding="utf-8").startswith(
        "existing process output\n"
    )


@requires_tensorflow
def test_tensorflow_mismatch_fails(tmp_path: Path) -> None:
    """A mismatched inference logs an error and cannot report completion."""
    example = (
        Path(__file__).resolve().parents[1] / "examples/tensorflow_compile.py"
    )
    script = """
import runpy
import sys
from types import SimpleNamespace
import cim22.tensorflow as frontend
class IncorrectModel:
    def __init__(self, model):
        self.model = model
        self.session = SimpleNamespace(dispatch_count=0)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __call__(self, inputs):
        return self.model(inputs) + 1
frontend.compile = lambda model, **kwargs: IncorrectModel(model)
sys.argv = [sys.argv[1], '--output', 'artifacts']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", script, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode != 0
    assert (
        "ERROR Example failed; see traceback and artifacts/compile.log"
        in result.stdout
    )
    assert "Completed inference" not in result.stdout
    assert not (tmp_path / "artifacts/result.json").exists()
