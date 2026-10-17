"""One long-lived process: compile once, then consume serial run commands."""

import ctypes
import os
import signal
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import DvsEngine

from .storage import (
    Document,
    Result,
    read_json,
    read_rows,
    save_json,
    save_rows,
    validate_config,
)


def worker_main(
    directory: Path,
    commands: Connection,
    cancel: Event,
    parent_pid: int,
    first_action: str,
) -> None:
    """Retain the Dynamo callable; die with the service, not the browser."""
    # Linux parent-death signal also covers a killed/crashed API process.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        return
    with (directory / "worker.log").open("a") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    state = read_json(directory / "state.json")

    def update(**changes: object) -> None:
        state.update(changes)
        state["updated_at"] = time.time()
        save_json(directory / "state.json", state)

    try:
        config = read_json(directory / "config.json")
        update(status="validating", error=None)
        validate_config(config)
        update(status="loading")
        from .engine import DvsEngine

        engine = DvsEngine(config, directory)
        seconds = engine.compile(lambda phase: update(status=phase))
        attempts = state.get("compile_attempts", []) + [seconds]
        update(
            status="running"
            if first_action != "compile"
            else "cancelled"
            if cancel.is_set()
            else "ready",
            compile_seconds=seconds,
            compile_attempts=attempts,
            compiled=True,
        )
        if first_action != "compile":
            run_samples(directory, config, engine, cancel, update, first_action)
        while True:
            command = commands.recv()
            if command == "close":
                return
            if command not in {"single", "full"}:
                raise ValueError("未知运行命令")
            validate_config(config)
            run_samples(directory, config, engine, cancel, update, command)
    except EOFError:
        return
    except Exception as error:
        # Process boundary: persist all failures, including third-party model
        # errors, and keep the traceback in the local worker log.
        traceback.print_exc()
        update(status="interrupted", error=str(error), compiled=False)
    finally:
        commands.close()


def run_samples(
    directory: Path,
    config: Document,
    engine: "DvsEngine",
    cancel: Event,
    update: Callable[..., None],
    command: str,
) -> None:
    """Commit complete pairs only; an interrupted pair is retried on resume."""
    rows = read_rows(directory)
    entries = config["samples"]
    for index, row in enumerate(rows):
        if (row["sample"], row["label"]) != (
            entries[index]["path"],
            entries[index]["label"],
        ):
            raise ValueError("保存结果与测试集顺序不符")
    stop = (
        min(len(rows) + 1, len(entries))
        if command == "single"
        else len(entries)
    )
    update(status="running", target=stop, error=None, eta_seconds=None)
    started = time.monotonic()
    initial = len(rows)
    while len(rows) < stop:
        if cancel.is_set():
            update(status="cancelled", eta_seconds=None)
            return
        index = len(rows)
        entry = entries[index]
        sample = engine.load(index)
        original, original_seconds = engine.infer("original", sample)
        cim, cim_seconds = engine.infer("cim", sample)
        rows.append(
            Result(
                index=index,
                sample=entry["path"],
                label=entry["label"],
                original_prediction=original,
                cim_prediction=cim,
                original_correct=original == entry["label"],
                cim_correct=cim == entry["label"],
                original_seconds=original_seconds,
                cim_seconds=cim_seconds,
            )
        )
        save_rows(directory, rows)
        eta = (
            (time.monotonic() - started)
            / (len(rows) - initial)
            * (stop - len(rows))
        )
        update(completed=len(rows), eta_seconds=eta)
    update(
        status="completed"
        if len(rows) == len(entries)
        else "cancelled"
        if cancel.is_set()
        else "ready",
        eta_seconds=None,
    )
