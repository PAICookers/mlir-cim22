"""Web CLI path defaults and help without starting a service."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from cim22.web import __main__ as cli


@pytest.mark.parametrize("data_dir", [None, "results", "~/cim22-web-test"])
def test_workspace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_dir: str | None
) -> None:
    """Resolve paths from the startup directory without creating output."""
    monkeypatch.chdir(tmp_path)
    args = ["cim22-web", "--model-dir", "model"]
    if data_dir is not None:
        args += ["--data-dir", data_dir, "--examples-dir", "examples"]
    monkeypatch.setattr(sys, "argv", args)
    create = Mock(return_value=object())
    server = Mock()
    monkeypatch.setattr(cli, "create_app", create)
    monkeypatch.setattr(cli.uvicorn, "run", server)
    cli.main()
    expected_examples = (
        tmp_path / "examples"
        if data_dir is not None
        else Path(cli.__file__).resolve().parents[2] / "examples"
    )
    create.assert_called_once_with(
        Path(data_dir or "cim22-web-data").expanduser().resolve(),
        tmp_path / "model",
        expected_examples,
        None,
    )
    assert server.call_args.args == (create.return_value,)
    assert server.call_args.kwargs["host"] == "127.0.0.1"
    assert server.call_args.kwargs["port"] == 8000
    assert not list(tmp_path.iterdir())


def test_help_is_english(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep standard argparse help and explain the default output path."""
    monkeypatch.setattr(sys, "argv", ["cim22-web", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert help_text.isascii()
    assert "./cim22-web-data" in help_text
    assert "read-only" in help_text
