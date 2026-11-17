"""Lifecycle regressions without network or long training runs."""

import threading
import time
from pathlib import Path

import pytest

from cim22.web import mnist_manager as module
from cim22.web.storage import read_json, save_json


def blocked_download(directory: Path) -> None:
    state = read_json(directory / "state.json")
    state["status"] = "downloading"
    save_json(directory / "state.json", state)
    (directory / "train.log").write_text("download started\n")
    time.sleep(60)


def test_cancel_stops_blocked_download_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_worker_main", blocked_download)
    manager = module.MnistManager(tmp_path, threading.RLock(), lambda: None)
    try:
        manager.start(
            data_dir=None,
            model="small",
            epochs=1,
            batch_size=8,
            learning_rate=0.001,
        )
        deadline = time.monotonic() + 15
        while manager.snapshot()["task"]["status"] != "downloading":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        with pytest.raises(ValueError, match="已有 MNIST"):
            manager.ensure_idle()
        assert manager.cancel()["task"]["status"] == "cancelled"
        assert manager.process is None
        manager.ensure_idle()
    finally:
        manager.close()
    restored = module.MnistManager(tmp_path, threading.RLock(), lambda: None)
    assert restored.snapshot()["task"]["status"] == "cancelled"
    assert "train.log" in restored.snapshot()["task"]["files"]
    restored.close()


def test_worker_uses_saved_config_and_reports_failure(tmp_path, monkeypatch):
    save_json(
        tmp_path / "config.json",
        {
            "model": "small",
            "data_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.005,
            "hidden_size": 64,
        },
    )
    save_json(tmp_path / "state.json", {"status": "starting"})
    monkeypatch.setattr(module.os, "dup2", lambda *args: None)

    def fail(config, progress):
        assert config.epochs == 3 and config.learning_rate == 0.005
        progress({"stage": "downloading", "message": "下载失败测试"})
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(module, "train_mnist", fail)
    module._worker_main(tmp_path)
    state = read_json(tmp_path / "state.json")
    assert state["status"] == "failed"
    assert state["error"] == "network unavailable"
    assert "下载失败测试" in (tmp_path / "train.log").read_text()
    assert not (tmp_path / "summary.json").exists()
