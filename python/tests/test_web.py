"""File recovery, worker exclusivity, and HTTP contracts without compilation."""

import multiprocessing as mp
import time
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from cim22.web.app import create_app
from cim22.web.manager import TaskManager
from cim22.web.storage import (
    Result,
    make_config,
    read_json,
    read_rows,
    save_json,
    save_rows,
    totals,
    validate_config,
)
from cim22.web.worker import run_samples

type Inputs = tuple[Path, Path, Path]


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_mobilenet_snn_deploy.py").write_text("# fixture\n")
    weights = model / "weights.pt"
    weights.write_bytes(b"fixed weights")
    dataset = tmp_path / "data"
    for name in ("0", "1", "10", "2"):
        (dataset / name).mkdir(parents=True)
        (dataset / name / "a.npz").write_bytes(name.encode())
    return model, weights, dataset


def test_fixed_label_mapping_and_changed_content(inputs: Inputs) -> None:
    config = make_config(*inputs)
    assert [(r["path"], r["label"]) for r in config["samples"]] == [
        ("0/a.npz", 0),
        ("1/a.npz", 1),
        ("10/a.npz", 2),
        ("2/a.npz", 3),
    ]
    validate_config(config)
    (inputs[2] / "2/a.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="已变化"):
        validate_config(config)


def test_unknown_classes_and_empty_dataset(inputs: Inputs) -> None:
    (inputs[2] / "11").mkdir()
    with pytest.raises(ValueError, match="未知类别"):
        make_config(*inputs)
    empty = inputs[2].parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="没有"):
        make_config(inputs[0], inputs[1], empty)


class FakeEngine:
    """Known predictions/times independent of IO and wall-clock overhead."""

    def __init__(
        self,
        cancel: Event | None = None,
        fail: tuple[str, int] | None = None,
    ) -> None:
        self.cancel = cancel
        self.fail = fail
        self.loaded = []

    def load(self, index: int) -> int:
        self.loaded.append(index)
        return index

    def infer(self, side: str, sample: int) -> tuple[int, float]:
        if (side, sample) == self.fail:
            raise RuntimeError("simulated crash")
        if self.cancel is not None and side == "cim":
            self.cancel.set()
        return (
            sample if side == "original" else 0,
            0.25 if side == "original" else 2.0,
        )


def test_cancel_resume_pair_commit_and_call_only_totals(
    tmp_path: Path,
    inputs: Inputs,
) -> None:
    config = make_config(*inputs)
    cancel = mp.get_context("spawn").Event()
    state = {}
    first = FakeEngine(cancel)
    run_samples(tmp_path, config, first, cancel, state.update, "full")
    assert state["status"] == "cancelled"
    assert len(read_rows(tmp_path)) == 1
    cancel.clear()
    remaining = FakeEngine()
    run_samples(tmp_path, config, remaining, cancel, state.update, "full")
    assert remaining.loaded == [1, 2, 3]
    assert state["status"] == "completed"
    assert totals(read_rows(tmp_path)) == {
        "original": {"correct": 4, "accuracy": 1.0, "seconds": 1.0},
        "cim": {"correct": 1, "accuracy": 0.25, "seconds": 8.0},
    }


def test_incomplete_pair_is_retried(tmp_path: Path, inputs: Inputs) -> None:
    config = make_config(*inputs)
    cancel = mp.get_context("spawn").Event()
    with pytest.raises(RuntimeError, match="simulated"):
        run_samples(
            tmp_path,
            config,
            FakeEngine(fail=("cim", 1)),
            cancel,
            lambda **kw: None,
            "full",
        )
    assert len(read_rows(tmp_path)) == 1
    engine = FakeEngine()
    run_samples(tmp_path, config, engine, cancel, lambda **kw: None, "single")
    assert engine.loaded == [1]
    assert len(read_rows(tmp_path)) == 2


def test_restart_uses_csv_even_when_status_lags(
    tmp_path: Path,
    inputs: Inputs,
) -> None:
    root = tmp_path / "results"
    manager = TaskManager(root, inputs[0])
    manager.import_model(inputs[1], inputs[2])
    directory = manager.current
    state = read_json(directory / "state.json")
    state.update(status="running", completed=0)
    save_json(directory / "state.json", state)
    row = Result(
        index=0,
        sample="0/a.npz",
        label=0,
        original_prediction=0,
        cim_prediction=1,
        original_correct=True,
        cim_correct=False,
        original_seconds=1.5,
        cim_seconds=2.5,
    )
    save_rows(directory, [row])
    manager.close()
    restarted = TaskManager(root, inputs[0])
    try:
        state = restarted.snapshot()["task"]
        assert state["status"] == "interrupted"
        assert state["compiled"] is False
        assert state["completed"] == 1
        assert state["totals"]["cim"]["seconds"] == 2.5
    finally:
        restarted.close()


def test_exclusive_service_and_active_task_guard(
    tmp_path: Path,
    inputs: Inputs,
) -> None:
    root = tmp_path / "results"
    manager = TaskManager(root, inputs[0])
    try:
        with pytest.raises(RuntimeError, match="另一个服务"):
            TaskManager(root, inputs[0])
        manager.import_model(inputs[1], inputs[2])
        state = read_json(manager.current / "state.json")
        state["status"] = "compiling"
        save_json(manager.current / "state.json", state)
        with pytest.raises(ValueError, match="已有计算"):
            manager.start("full")
        with pytest.raises(ValueError, match="已有计算"):
            manager.import_model(inputs[1], inputs[2])
    finally:
        manager.close()


def test_api_validation_and_local_origin(
    tmp_path: Path, inputs: Inputs
) -> None:
    app = create_app(tmp_path / "results", inputs[0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/state").json() == {"task": None}
        assert (
            client.post("/api/action", json={"action": "full"}).status_code
            == 409
        )
        assert (
            client.post("/api/action", json={"action": "bogus"}).status_code
            == 422
        )
        body = {"weights": str(inputs[1]), "dataset": str(inputs[2])}
        assert (
            client.post(
                "/api/import",
                json=body,
                headers={"Origin": "https://elsewhere.example"},
            ).status_code
            == 403
        )
        result = client.post("/api/import", json=body)
        assert result.status_code == 200
        assert result.json()["task"]["total"] == 4
        assert client.get("/api/results.csv").status_code == 404
        assert (
            client.get(
                "/api/state", headers={"Host": "elsewhere.example"}
            ).status_code
            == 400
        )
        assert client.post("/api/cancel", json={}).status_code == 409


def test_failed_atomic_replace_preserves_previous_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cim22.web.storage as storage

    row = Result(
        index=0,
        sample="0/a.npz",
        label=0,
        original_prediction=0,
        cim_prediction=0,
        original_correct=True,
        cim_correct=True,
        original_seconds=0.5,
        cim_seconds=1.0,
    )
    save_rows(tmp_path, [row])
    original = (tmp_path / "results.csv").read_bytes()

    def fail(*args: object) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        save_rows(tmp_path, [])
    assert (tmp_path / "results.csv").read_bytes() == original


def test_worker_death_is_explicitly_interrupted(
    tmp_path: Path,
    inputs: Inputs,
) -> None:
    manager = TaskManager(tmp_path / "results", inputs[0])
    try:
        manager.import_model(inputs[1], inputs[2])
        state = read_json(manager.current / "state.json")
        state.update(status="ready", compiled=True)
        save_json(manager.current / "state.json", state)
        process = manager.context.Process(target=time.sleep, args=(30,))
        process.start()
        manager.process = process
        process.kill()
        process.join(timeout=5)
        state = manager.snapshot()["task"]
        assert state["status"] == "interrupted"
        assert "退出码" in state["error"]
        assert not state["compiled"]
        assert manager.process is None
    finally:
        manager.close()


def test_engine_resets_before_each_call_and_excludes_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("spikingjelly")
    import torch

    from cim22.web import engine as implementation

    trace = []

    class Stateful(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = 10

        def reset(self) -> None:
            assert torch.is_inference_mode_enabled()
            self.value = 0
            trace.append("reset")

        def forward(self, sample: torch.Tensor) -> torch.Tensor:
            assert torch.is_inference_mode_enabled()
            assert self.value == 0
            self.value += 1
            trace.append("call")
            return torch.ones((32, 1, 11))

    instance = implementation.DvsEngine.__new__(implementation.DvsEngine)
    instance.original = Stateful()
    instance.cim = Stateful()
    instance.compiled = instance.cim
    instance.directory = tmp_path
    instance.artifacts = set()
    instance.trim = lambda: trace.append("trim")
    sample = torch.zeros(1)
    ticks = iter([1.0, 1.25, 5.0, 7.0, 10.0, 10.25])
    monkeypatch.setattr(
        implementation.time, "perf_counter", lambda: next(ticks)
    )
    assert instance.infer("original", sample) == (0, 0.25)
    assert instance.infer("cim", sample) == (0, 2.0)
    assert instance.infer("original", sample) == (0, 0.25)
    assert trace == ["reset", "call", "trim"] * 3


@pytest.mark.parametrize(
    "changed_source", ["backend.py", "compiler.py", "runtime.py"]
)
def test_resume_rejects_changed_python_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inputs: Inputs,
    changed_source: str,
) -> None:
    pytest.importorskip("spikingjelly")
    from cim22.web import engine

    revisions = {}
    monkeypatch.setattr(engine.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(
        engine, "runtime_library", lambda: tmp_path / "runtime.so"
    )
    monkeypatch.setattr(
        engine, "compiler_library", lambda _: tmp_path / "compiler.so"
    )
    monkeypatch.setattr(
        engine, "digest", lambda path: revisions.get(path.name, "unchanged")
    )

    def reach_model_load(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model loading reached")

    monkeypatch.setattr(engine.torch, "load", reach_model_load)
    config = make_config(*inputs)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="model loading reached"):
            engine.DvsEngine(config, tmp_path)
    revisions[changed_source] = "updated"
    with pytest.raises(ValueError, match="运行环境已变化"):
        engine.DvsEngine(config, tmp_path)
