"""Fixed DVS CLI orchestration without opening a board or compiling a model."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("spikingjelly")

from cim22.apps import dvs
from cim22.web.storage import _storage_lock, read_json, read_rows

USB_CONFIG = {
    "vendor_id": 1204,
    "product_id": 241,
    "bulk_in_endpoint": 129,
    "bulk_out_endpoint": 1,
    "timing": {"work_delay_ms": 10},
}


@pytest.fixture
def cli_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    model = tmp_path / "model"
    (model / "weights").mkdir(parents=True)
    (model / "model_mobilenet_snn_deploy.py").write_text("# fixture\n")
    (model / "weights/dvsgesture_mobilenet_snn_qat_int8.pt").write_bytes(
        b"weights"
    )
    dataset = model / "data/test/0"
    dataset.mkdir(parents=True)
    for name in ("a.npz", "b.npz"):
        (dataset / name).write_bytes(name.encode())
    usb = tmp_path / "usb.json"
    usb.write_text(json.dumps(USB_CONFIG))
    return model, usb, tmp_path / "output"


def install_engine_stub(
    monkeypatch: pytest.MonkeyPatch, failure: str | None = None
) -> list[dict[str, Any]]:
    calls = []

    class StubEngine:
        def __init__(self, config: dict[str, Any], directory: Path) -> None:
            self.config = config
            self.dispatch_count = 2
            calls.append({"config": config, "directory": directory})

        def compile(self, phase: object) -> float:
            phase("compiling")
            if failure == "compile":
                raise RuntimeError("USB open failed")
            return 1.25

        def metadata(self) -> dict[str, Any]:
            return {
                "cim_layers": self.config["cim_layers"],
                "dispatch_count": self.dispatch_count,
            }

        def load(self, index: int) -> int:
            return index

        def infer(self, side: str, sample: int) -> tuple[int, float]:
            if failure == "infer" and side == "cim":
                raise RuntimeError("USB transfer failed")
            return 0, 0.25 if side == "original" else 0.5

    monkeypatch.setattr(dvs, "DvsEngine", StubEngine)
    monkeypatch.setattr(dvs, "validate_cim_layer_names", lambda *_: None)
    return calls


@pytest.mark.parametrize(
    ("action", "completed", "status"),
    [("compile", 0, "ready"), ("single", 1, "ready"), ("full", 2, "completed")],
)
def test_cli_uses_shared_dvs_engine_and_persists_results(
    cli_inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    completed: int,
    status: str,
) -> None:
    model, usb, output = cli_inputs
    calls = install_engine_stub(monkeypatch)
    dvs.main(
        [
            "--model-dir",
            str(model),
            "--usb-config",
            str(usb),
            "--output",
            str(output),
            "--cim-layers",
            "first",
            "second",
            "--action",
            action,
        ]
    )
    assert len(calls) == 1
    config = calls[0]["config"]
    assert config["runtime"] == "usb"
    assert config["cim_layers"] == ["first", "second"]
    assert config["usb_config"] == USB_CONFIG
    assert config["weights"] == str(
        model / "weights/dvsgesture_mobilenet_snn_qat_int8.pt"
    )
    assert config["dataset"] == str(model / "data/test")
    assert calls[0]["directory"] == output
    assert read_json(output / "config.json") == config
    state = read_json(output / "state.json")
    assert (state["status"], state["compiled"]) == (status, True)
    assert state["compile_seconds"] == 1.25
    assert len(read_rows(output)) == completed
    summary = read_json(output / "summary.json")
    assert summary["runtime"] == "usb"
    assert summary["cim_layers"] == ["first", "second"]
    assert summary["totals"]["cim"]["correct"] == completed


@pytest.mark.parametrize("failure", ["compile", "infer"])
def test_cli_board_failure_leaves_no_incomplete_result(
    cli_inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    model, usb, output = cli_inputs
    install_engine_stub(monkeypatch, failure)
    with pytest.raises(RuntimeError, match="USB"):
        dvs.main(
            [
                "--model-dir",
                str(model),
                "--usb-config",
                str(usb),
                "--output",
                str(output),
                "--action",
                "full",
            ]
        )
    state = read_json(output / "state.json")
    assert state["status"] == "interrupted"
    assert not state["compiled"]
    assert "USB" in state["error"]
    assert read_rows(output) == []
    assert not (output / "summary.json").exists()


def test_cli_replaces_existing_outputs(
    cli_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, usb, output = cli_inputs
    install_engine_stub(monkeypatch)
    output.mkdir()
    marker = output / "notes.txt"
    marker.write_bytes(b"keep this")
    (output / ".storage.lock").touch()
    (output / "state.json.tmp").write_text("{")
    args = [
        "--model-dir",
        str(model),
        "--usb-config",
        str(usb),
        "--output",
        str(output),
    ]
    dvs.main([*args, "--action", "full"])
    assert len(read_rows(output)) == 2
    artifacts = output / "artifacts"
    artifacts.mkdir()
    (artifacts / "old-model.so").touch()
    dvs.main([*args, "--action", "single"])
    assert [row["index"] for row in read_rows(output)] == [0]
    assert not (artifacts / "old-model.so").exists()
    dvs.main([*args, "--action", "compile"])
    assert not (output / "results.csv").exists()
    assert marker.read_bytes() == b"keep this"


def test_cli_refuses_existing_artifacts_from_another_application(
    cli_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, usb, output = cli_inputs
    install_engine_stub(monkeypatch)
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True)
    previous = artifacts / "my-model.so"
    previous.write_bytes(b"keep this")
    with pytest.raises(ValueError, match="conflicts with existing files"):
        dvs.main(
            [
                "--model-dir",
                str(model),
                "--usb-config",
                str(usb),
                "--output",
                str(output),
            ]
        )
    assert previous.read_bytes() == b"keep this"


def test_cli_refuses_output_used_by_another_runner(
    cli_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, usb, output = cli_inputs
    calls = install_engine_stub(monkeypatch)
    output.mkdir()
    with (
        _storage_lock(output / ".dvs-run.lock"),
        pytest.raises(RuntimeError, match="Cannot lock DVS output directory"),
    ):
        dvs.main(
            [
                "--model-dir",
                str(model),
                "--usb-config",
                str(usb),
                "--output",
                str(output),
            ]
        )
    assert not calls
    assert not (output / "state.json").exists()


def test_cli_list_and_help_need_no_board(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layers = [{"name": "first", "type": "Linear"}]
    monkeypatch.setattr(dvs, "discover_cim_layers", lambda _: layers)
    dvs.main(["--model-dir", str(tmp_path), "--list-cim-layers"])
    assert json.loads(capsys.readouterr().out) == layers
    with pytest.raises(SystemExit) as exit_info:
        dvs.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--usb-config" in help_text
    assert "--cim-layers" in help_text
    assert "{compile,single,full}" in help_text
    with pytest.raises(SystemExit) as exit_info:
        dvs.main(["--model-dir", str(tmp_path)])
    assert exit_info.value.code == 2
    assert "--usb-config is required" in capsys.readouterr().err
