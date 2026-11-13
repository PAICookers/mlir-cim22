"""Durable Web task manager for the MNIST acceptance training run."""

import json
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ..mnist_training import MnistConfig, train_mnist
from .storage import Document, read_json, save_json

MnistModel = Literal["small", "standard"]
ACTIVE = {"starting", "downloading", "loading", "training", "evaluating"}


def _worker_main(
    directory: Path,
) -> None:
    """Run training in a child process so downloads never block requests."""
    log_path = directory / "train.log"
    with log_path.open("a", encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)

        def update(event: dict[str, object]) -> None:
            state = read_json(directory / "state.json")
            stage = str(event["stage"])
            message = event["message"]
            if isinstance(message, dict):
                log.write(json.dumps(message, ensure_ascii=False) + "\n")
                state.update(message)
                state["last_message"] = json.dumps(message, ensure_ascii=False)
            else:
                log.write(str(message) + "\n")
                state["last_message"] = str(message)
            state.update(status=stage, updated_at=time.time())
            log.flush()
            save_json(directory / "state.json", state)

        try:
            saved = read_json(directory / "config.json")
            saved.pop("hidden_size")
            saved["data_dir"] = Path(saved["data_dir"])
            saved["output_dir"] = directory
            config = MnistConfig(**saved)
            train_mnist(config, update)
        except InterruptedError:
            state = read_json(directory / "state.json")
            state.update(status="cancelled", error=None, updated_at=time.time())
            save_json(directory / "state.json", state)
        except Exception as error:
            traceback.print_exc(file=log)
            state = read_json(directory / "state.json")
            state.update(
                status="failed", error=str(error), updated_at=time.time()
            )
            save_json(directory / "state.json", state)


class MnistManager:
    """Own one persisted MNIST task and its exclusive child process."""

    def __init__(
        self,
        root: Path,
        lock: threading.RLock,
        ensure_idle: Callable[[], None],
    ) -> None:
        self.root = root.resolve() / "mnist"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock
        self._ensure_idle = ensure_idle
        self.context = mp.get_context("spawn")
        self.process: mp.Process | None = None
        self.current: Path | None = None
        pointer = self.root / "current.json"
        if pointer.exists():
            self.current = self.root / read_json(pointer)["id"]
            self._recover("服务已重启，训练结果已保存，可重新开始")

    def _recover(self, message: str) -> None:
        if self.current is None:
            return
        state = read_json(self.current / "state.json")
        if state.get("status") in ACTIVE:
            state.update(
                status="interrupted", error=message, updated_at=time.time()
            )
            save_json(self.current / "state.json", state)

    def _reap(self) -> None:
        if self.process is None or self.process.is_alive():
            return
        code = self.process.exitcode
        self.process.join()
        self.process.close()
        self.process = None
        if self.current is None:
            return
        state = read_json(self.current / "state.json")
        if code and state.get("status") not in {"cancelled", "failed"}:
            state.update(
                status="failed",
                error=f"训练进程异常退出（退出码 {code}）",
                updated_at=time.time(),
            )
            save_json(self.current / "state.json", state)

    @staticmethod
    def _log_tail(path: Path, limit: int = 12_000) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - limit))
                data = stream.read()
        except OSError:
            return ""
        text = data.decode("utf-8", errors="replace")
        return ("[日志过长，仅显示最后 12 KB]\n" if size > limit else "") + text

    def _files(self, state: Document) -> dict[str, str]:
        task_id = str(state["id"])
        directory = self.current
        if directory is None:
            return {}
        names = {
            name: directory / name
            for name in (
                "model.pt",
                "config.json",
                "metrics.jsonl",
                "summary.json",
                "train.log",
            )
        }
        return {
            name: f"/api/mnist/file?task_id={task_id}&name={path.name}"
            for name, path in names.items()
            if path.is_file()
        }

    def snapshot(self) -> Document:
        with self.lock:
            self._reap()
            if self.current is None:
                return {"task": None}
            state = read_json(self.current / "state.json")
            state.update(
                log_text=self._log_tail(self.current / "train.log"),
                files=self._files(state),
            )
            return {"task": state}

    def ensure_idle(self) -> None:
        with self.lock:
            self._reap()
            if self.process is not None:
                raise ValueError("已有 MNIST 训练任务，请等待完成或取消")

    def start(
        self,
        *,
        data_dir: Path | None,
        model: MnistModel,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> Document:
        with self.lock:
            self._reap()
            self.ensure_idle()
            self._ensure_idle()
            config = MnistConfig(
                data_dir=(data_dir or self.root / "data")
                .expanduser()
                .resolve(),
                output_dir=self.root,
                model=model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            config.validate()
            task_id = uuid4().hex
            directory = self.root / task_id
            directory.mkdir()
            config = MnistConfig(
                data_dir=config.data_dir,
                output_dir=directory,
                model=config.model,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
            )
            save_json(directory / "config.json", config.json_value())
            save_json(
                directory / "state.json",
                {
                    "id": task_id,
                    "status": "starting",
                    "model": model,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "data_dir": str(config.data_dir),
                    "epoch": 0,
                    "error": None,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                },
            )
            save_json(self.root / "current.json", {"id": task_id})
            self.current = directory
            self.process = self.context.Process(
                target=_worker_main,
                args=(directory,),
                daemon=True,
            )
            try:
                self.process.start()
            except Exception as error:
                self.process.close()
                self.process = None
                state = read_json(directory / "state.json")
                state.update(
                    status="failed", error=f"无法启动训练进程：{error}"
                )
                save_json(directory / "state.json", state)
                raise ValueError(state["error"]) from error
            return self.snapshot()

    def cancel(self) -> Document:
        with self.lock:
            self._reap()
            if self.process is None:
                raise ValueError("没有正在运行的 MNIST 训练")
            state = read_json(self.current / "state.json")
            if state.get("status") not in ACTIVE:
                raise ValueError("没有正在运行的 MNIST 训练")
            # Stop downloads too, then become the sole state writer.
            self._stop()
            state = read_json(self.current / "state.json")
            if state.get("status") == "completed":
                return self.snapshot()
            state.update(status="cancelled", error=None, updated_at=time.time())
            save_json(self.current / "state.json", state)
            return self.snapshot()

    def _stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None

    def close(self) -> None:
        with self.lock:
            self._stop()
            self._recover("服务已停止，训练结果已保存")
