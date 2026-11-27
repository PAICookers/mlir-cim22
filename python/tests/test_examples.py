"""Exercise the integer examples' group output, verdicts and saved evidence."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cim22.apps.framework_example import group_result
from cim22.runtime import MODEL_SUFFIX

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
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
    ("script", "framework", "model", "shape"),
    [
        pytest.param(
            "tensorflow_compile.py",
            "TensorFlow",
            "Dense(64→16) + ReLU",
            (4, 16),
            marks=requires_tensorflow,
        ),
    ],
)
def test_example_logging(
    tmp_path: Path,
    script: str,
    framework: str,
    model: str,
    shape: tuple[int, int],
) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    log = output / "compile.log"
    log.write_text("existing process output\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / script),
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
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "custom-result.json").read_text("utf-8"))
    assert len(report["cases"]) == report["passed_groups"] == 3
    lines = [f"Framework: {framework} | Backend: simulator | Model: {model}"]
    for index, row in enumerate(report["cases"], 1):
        assert row["name"] == f"Group {index}"
        assert row["passed"] and row["correct"]
        assert row["actual_dtype"] == "int32"
        assert np.shape(row["actual"]) == shape
        assert row["actual"] == row["reference"]
        assert row["max_abs_error"] == 0
        count = row["dispatch_count"]
        assert count > 0
        lines.append(f"Group {index} | Correct | CIM dispatches: {count}")
    lines.append("3/3 passed")
    assert result.stdout.splitlines() == lines
    assert log.read_text("utf-8").splitlines() == [
        "existing process output",
        *lines,
    ]
    assert list(output.glob(f"graph-*{MODEL_SUFFIX}"))
    assert (output / "ir").is_dir()
    assert not (output / "result.json").exists()


@requires_tensorflow
def test_tensorflow_mismatch_fails(tmp_path: Path) -> None:
    """A wrong middle group cannot hide either its failure or the last result."""
    example = EXAMPLES / "tensorflow_compile.py"
    script = """
import runpy
import sys
from types import SimpleNamespace
import cim22.tensorflow as frontend
class IncorrectModel:
    def __init__(self, model):
        self.model = model
        self.session = SimpleNamespace(dispatch_count=0)
        self.group = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __call__(self, inputs):
        self.group += 1
        self.session.dispatch_count = (2, 4, 6)[self.group - 1]
        return self.model(inputs) + int(self.group == 2)
frontend.compile = lambda model, **kwargs: IncorrectModel(model)
sys.argv = [sys.argv[1], '--output', 'artifacts']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", script, str(example)],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode != 0
    assert result.stdout.splitlines() == [
        "Framework: TensorFlow | Backend: simulator | Model: Dense(64→16) + ReLU",
        "Group 1 | Correct | CIM dispatches: 2",
        "Group 2 | Incorrect | CIM dispatches: 4",
        "Group 3 | Correct | CIM dispatches: 6",
        "2/3 passed",
    ]
    report = json.loads((tmp_path / "artifacts/result.json").read_text("utf-8"))
    assert report["passed_groups"] == 2
    assert [case["max_abs_error"] for case in report["cases"]] == [0, 1, 0]


def test_torch_usb_requires_board_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runpy

    import torch

    example = Path(__file__).resolve().parents[1] / "examples/torch_compile.py"
    run = runpy.run_path(str(example))["run"]
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    with pytest.raises(RuntimeError, match="no CIM computation"):
        run(tmp_path, runtime="usb")
    assert not (tmp_path / "result.json").exists()


@pytest.mark.integration
@pytest.mark.usefixtures("native_toolchain")
def test_torch_example_logging(tmp_path: Path) -> None:
    script, framework, count = "torch_compile.py", "PyTorch", 5
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
    assert evidence["dispatch_count"] > 0
    assert all(case["dispatch_count"] > 0 for case in evidence["cases"])
    assert all(case["max_abs_error"] == 0 for case in evidence["cases"])
    assert list((tmp_path / "artifacts").glob(f"graph-*{MODEL_SUFFIX}"))
    assert (tmp_path / "artifacts/ir").is_dir()
    assert log.read_text(encoding="utf-8").startswith(
        "existing process output\n"
    )


@pytest.mark.parametrize(
    "actual", [np.zeros((1,), dtype=np.int32), np.zeros((1, 1), dtype=np.int8)]
)
def test_group_requires_matching_shape_and_dtype(actual: np.ndarray) -> None:
    row = group_result(1, np.zeros((1, 1), dtype=np.int32), actual, 1)
    assert not row["correct"] and not row["passed"]
