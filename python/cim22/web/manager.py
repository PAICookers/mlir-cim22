"""Single-worker lifecycle, restart recovery, and serialized UI commands."""

import fcntl
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

from .storage import (
    ACTIVE,
    Document,
    make_config,
    read_json,
    read_rows,
    save_json,
    totals,
)
from .worker import worker_main


class TaskManager:
    """One service owns one result directory and at most one compute process."""

    def __init__(self, root: Path, model_dir: Path) -> None:
        self.root = root.resolve()
        self.model_dir = model_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / ".lock").open("a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError("该结果目录已被另一个服务占用") from None
        self.lock = threading.RLock()
        self.context = mp.get_context("spawn")
        self.process = None
        self.commands = None
        self.cancel_event = None
        self.current: Path | None = None
        pointer = self.root / "current.json"
        if pointer.exists():
            self.current = self.root / read_json(pointer)["id"]
            self._recover("服务已重启，已保存结果可以续跑")

    def _recover(self, message: str) -> None:
        if self.current is None:
            return
        state = read_json(self.current / "state.json")
        if state["status"] in ACTIVE:
            state.update(status="interrupted", error=message)
        state.update(compiled=False, eta_seconds=None)
        save_json(self.current / "state.json", state)

    def _reap(self) -> None:
        if self.process is not None and not self.process.is_alive():
            code = self.process.exitcode
            self.process.join()
            self.process.close()
            self.process = None
            if self.commands is not None:
                self.commands.close()
                self.commands = None
            if code and self.current is not None:
                state = read_json(self.current / "state.json")
                state.update(
                    status="interrupted",
                    error=f"计算进程异常退出（退出码 {code}），已保存结果可续跑",
                )
                save_json(self.current / "state.json", state)
            self._recover(
                f"计算进程已退出（退出码 {code}），可从已保存样本续跑"
            )

    def snapshot(self) -> Document:
        with self.lock:
            self._reap()
            if self.current is None:
                return {"task": None}
            state = read_json(self.current / "state.json")
            rows = read_rows(self.current)
            state.update(
                completed=len(rows),
                totals=totals(rows),
                rows=rows,
                cancel_requested=bool(
                    self.cancel_event
                    and self.cancel_event.is_set()
                    and state["status"] in ACTIVE
                ),
            )
            return {"task": state}

    def import_model(self, weights: Path, dataset: Path) -> Document:
        with self.lock:
            self._ensure_idle()
            config = make_config(
                self.model_dir,
                weights.expanduser().resolve(),
                dataset.expanduser().resolve(),
            )
            self._stop()
            task_id = uuid4().hex
            directory = self.root / task_id
            directory.mkdir()
            save_json(directory / "config.json", config)
            save_json(
                directory / "state.json",
                {
                    "id": task_id,
                    "status": "imported",
                    "compiled": False,
                    "weights": config["weights"],
                    "dataset": config["dataset"],
                    "total": len(config["samples"]),
                    "completed": 0,
                    "input_shape": config["input_shape"],
                    "classes": config["classes"],
                    "runtime": "software",
                    "compile_seconds": None,
                    "compile_attempts": [],
                    "error": None,
                    "eta_seconds": None,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                },
            )
            save_json(self.root / "current.json", {"id": task_id})
            self.current = directory
            return self.snapshot()

    def _ensure_idle(self) -> None:
        self._reap()
        if self.current is not None:
            state = read_json(self.current / "state.json")
            if state["status"] in ACTIVE:
                raise ValueError("已有计算任务，请等待完成或取消后再操作")

    def start(self, action: str) -> Document:
        with self.lock:
            if action not in {"compile", "single", "full"}:
                raise ValueError("未知运行命令")
            self._ensure_idle()
            if self.current is None:
                raise ValueError("请先导入权重和测试集")
            state = read_json(self.current / "state.json")
            rows = read_rows(self.current)
            if action != "compile" and len(rows) == state["total"]:
                raise ValueError("所有样本已完成；重新导入可创建新测试")
            # A failed worker may publish its error just before it exits.
            # Join that process before accepting a retry into a dead pipe.
            if self.process is not None and not state["compiled"]:
                self._stop()
            if action == "compile" and self.process is not None:
                return self.snapshot()
            if self.process is None:
                state.update(status="starting", compiled=False, error=None)
                save_json(self.current / "state.json", state)
                parent, child = self.context.Pipe()
                self.cancel_event = self.context.Event()
                self.commands = parent
                self.process = self.context.Process(
                    target=worker_main,
                    args=(
                        self.current,
                        child,
                        self.cancel_event,
                        os.getpid(),
                        action,
                    ),
                    daemon=True,
                )
                try:
                    self.process.start()
                except Exception:
                    # Startup can fail before a child exists; make retry possible.
                    self.process = None
                    parent.close()
                    self.commands = None
                    self._recover("无法启动计算进程")
                    raise
                finally:
                    child.close()
            elif action != "compile":
                self.cancel_event.clear()
                state.update(status="running", error=None)
                save_json(self.current / "state.json", state)
                self.commands.send(action)
            return self.snapshot()

    def cancel(self) -> Document:
        with self.lock:
            self._reap()
            if self.process is None or self.current is None:
                raise ValueError("没有正在运行的任务")
            state = read_json(self.current / "state.json")
            if state["status"] not in ACTIVE:
                raise ValueError("没有正在运行的任务")
            self.cancel_event.set()
            return self.snapshot()

    def _stop(self) -> None:
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.commands is not None:
            self.commands.close()
            self.commands = None
        self.cancel_event = None

    def close(self) -> None:
        with self.lock:
            self._stop()
            self._recover("服务已停止，已保存结果可以续跑")
            self._lock_file.close()
