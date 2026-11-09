"""Run the fixed framework examples as durable, serial Web tasks."""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from cim22 import CIMTimingConfig, UsbConfig

from .storage import Document, read_json, save_json

Framework = Literal["tensorflow", "pytorch"]
Runtime = Literal["simulator", "usb"]
ACTIVE = {"starting", "running"}
SCRIPTS = {"tensorflow": "tensorflow_compile.py", "pytorch": "torch_compile.py"}


class ExampleManager:
    """Own one framework example process and its JSON result."""

    def __init__(
        self,
        root: Path,
        examples_dir: Path,
        lock: threading.RLock,
        ensure_dvs_idle: object,
    ) -> None:
        self.root = root.resolve() / "examples"
        self.root.mkdir(parents=True, exist_ok=True)
        self.examples_dir = examples_dir.resolve()
        self.lock = lock
        self._ensure_dvs_idle = ensure_dvs_idle
        self.process: subprocess.Popen[bytes] | None = None
        self.log = None
        self.current: Path | None = None
        pointer = self.root / "current.json"
        if pointer.exists():
            self.current = self.root / read_json(pointer)["id"]
            self._recover("服务已重启，框架验证结果可重新运行")

    def _recover(self, message: str) -> None:
        if self.current is None:
            return
        state = read_json(self.current / "state.json")
        if state.get("status") in ACTIVE:
            state.update(status="interrupted", error=message)
            save_json(self.current / "state.json", state)

    def _reap(self) -> None:
        if self.process is None or self.process.poll() is None:
            return
        code = self.process.returncode
        self.process.wait()
        if self.log is not None:
            self.log.close()
            self.log = None
        self.process = None
        if self.current is None:
            return
        state = read_json(self.current / "state.json")
        if code == 0 and (self.current / "result.json").is_file():
            state.update(
                read_json(self.current / "result.json"),
                status="completed",
                error=None,
                updated_at=time.time(),
            )
        elif state.get("status") == "cancelled":
            state.update(error=None)
        else:
            state.update(
                status="interrupted",
                error=f"示例进程失败（退出码 {code}），详情见 compile.log",
            )
        save_json(self.current / "state.json", state)

    def snapshot(self) -> Document:
        with self.lock:
            self._reap()
            if self.current is None:
                return {"task": None}
            state = read_json(self.current / "state.json")
            state.update(
                artifact_url=self._url_for(state, "artifact_dir"),
                debug_url=self._url_for(state, "ir_dir"),
                log_url=self._url_for(state, "log_path"),
                result_url=self._url_for(state, "result_json"),
            )
            return {"task": state}

    @staticmethod
    def _url_for(state: Document, key: str) -> str | None:
        value = state.get(key)
        if not value:
            return None
        path = Path(str(value))
        task_id = str(state["id"])
        if key == "artifact_dir":
            artifacts = sorted(path.glob("*.so")) + sorted(path.glob("*.dll"))
            if not artifacts:
                return None
            name = artifacts[0].name
        elif key == "ir_dir":
            return f"/api/examples/ir?task_id={task_id}"
        else:
            name = path.name
        return f"/api/examples/file?task_id={task_id}&name={name}"

    def ensure_idle(self) -> None:
        with self.lock:
            self._reap()
            if self.process is not None:
                raise ValueError("已有框架验证任务，请等待完成或取消")

    @staticmethod
    def _parse_usb_config(fields: object) -> UsbConfig:
        if not isinstance(fields, dict):
            raise ValueError("usb_config 必须是 JSON 对象")
        values = dict(fields)
        timing = values.get("timing")
        if timing is not None:
            if not isinstance(timing, dict):
                raise ValueError("usb_config.timing 必须是 JSON 对象")
            values["timing"] = CIMTimingConfig(**timing)
        return UsbConfig(**values)

    def start(self, framework: Framework, runtime: Runtime, usb_config: object | None) -> Document:
        with self.lock:
            self._reap()
            if framework not in SCRIPTS:
                raise ValueError("未知框架")
            if runtime not in {"simulator", "usb"}:
                raise ValueError("未知运行后端")
            if self.process is not None:
                raise ValueError("已有框架验证任务，请等待完成或取消")
            if runtime == "usb":
                if usb_config is None:
                    raise ValueError("usb 后端必须提供 usb_config")
                try:
                    self._parse_usb_config(usb_config)
                except (TypeError, ValueError) as error:
                    raise ValueError(str(error)) from error
            elif usb_config is not None:
                raise ValueError("simulator 不接受 usb_config")
            self._ensure_dvs_idle()
            script = self.examples_dir / SCRIPTS[framework]
            if not script.is_file():
                raise ValueError(f"找不到示例程序：{script}")
            task_id = uuid4().hex
            directory = self.root / task_id
            directory.mkdir()
            config_path = directory / "usb-config.json"
            if usb_config is not None:
                save_json(config_path, usb_config)
            state: Document = {
                "id": task_id,
                "framework": framework,
                "runtime": runtime,
                "status": "starting",
                "compile_seconds": None,
                "dispatch_count": None,
                "cases": [],
                "artifact_dir": str(directory),
                "ir_dir": str(directory / "ir"),
                "log_path": str(directory / "compile.log"),
                "result_json": str(directory / "result.json"),
                "error": None,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            save_json(directory / "state.json", state)
            save_json(self.root / "current.json", {"id": task_id})
            command = [
                sys.executable,
                str(script),
                "--runtime",
                runtime,
                "--output",
                str(directory),
                "--result-json",
                str(directory / "result.json"),
            ]
            if usb_config is not None:
                command.extend(["--usb-config", str(config_path)])
            log = (directory / "compile.log").open("ab")
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.examples_dir.parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                log.close()
                raise
            self.log = log
            self.current = directory
            state.update(status="running", updated_at=time.time())
            save_json(directory / "state.json", state)
            return self.snapshot()

    def cancel(self) -> Document:
        with self.lock:
            self._reap()
            if self.process is None or self.current is None:
                raise ValueError("没有正在运行的框架验证")
            state = read_json(self.current / "state.json")
            if state.get("status") not in ACTIVE:
                raise ValueError("没有正在运行的框架验证")
            self.process.terminate()
            state.update(status="cancelled", error=None, updated_at=time.time())
            save_json(self.current / "state.json", state)
            return self.snapshot()

    def close(self) -> None:
        with self.lock:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
            if self.log is not None:
                self.log.close()
                self.log = None
            self._recover("服务已停止，框架验证结果已保存")
