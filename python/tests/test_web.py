"""File recovery, worker exclusivity, and HTTP contracts without compilation."""

import io
import multiprocessing as mp
import time
import zipfile
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

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
from cim22.web.usb import parse_usb_config
from cim22.web.worker import run_samples
from cim22.web.worker import worker_main as compute_worker

USB_CONFIG = {
    "vendor_id": 1204,
    "product_id": 241,
    "bulk_in_endpoint": 129,
    "bulk_out_endpoint": 1,
    "timing": {"work_delay_ms": 10},
}

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
    config = make_config(*inputs, usb_config=USB_CONFIG)
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
        make_config(*inputs, usb_config=USB_CONFIG)
    empty = inputs[2].parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="没有"):
        make_config(inputs[0], inputs[1], empty, usb_config=USB_CONFIG)


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
    config = make_config(*inputs, usb_config=USB_CONFIG)
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
    config = make_config(*inputs, usb_config=USB_CONFIG)
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
    manager.import_model(inputs[1], inputs[2], usb_config=USB_CONFIG)
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
        manager.import_model(inputs[1], inputs[2], usb_config=USB_CONFIG)
        state = read_json(manager.current / "state.json")
        state["status"] = "compiling"
        save_json(manager.current / "state.json", state)
        with pytest.raises(ValueError, match="已有计算"):
            manager.start("full")
        with pytest.raises(ValueError, match="已有计算"):
            manager.import_model(inputs[1], inputs[2], usb_config=USB_CONFIG)
    finally:
        manager.close()


def test_api_validation_and_local_origin(
    tmp_path: Path, inputs: Inputs
) -> None:
    app = create_app(
        tmp_path / "results", inputs[0], examples_dir=tmp_path / "examples"
    )
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
        body = {
            "weights": str(inputs[1]),
            "dataset": str(inputs[2]),
            "usb_config": USB_CONFIG,
        }
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
        assert client.get("/api/examples/state").json() == {"task": None}
        assert (
            client.post(
                "/api/examples/run",
                json={"framework": "tensorflow", "runtime": "usb"},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/examples/run",
                json={"framework": "tensorflow", "runtime": "simulator"},
            ).status_code
            == 409
        )


def test_example_usb_config_reuses_runtime_validation() -> None:
    config = parse_usb_config(
        {
            "vendor_id": 1204,
            "product_id": 241,
            "bulk_in_endpoint": 129,
            "bulk_out_endpoint": 1,
            "timing": {"work_delay_ms": 10},
        }
    )
    assert config.vendor_id == 1204
    assert config.timing.work_delay_ms == 10
    with pytest.raises(ValueError, match="JSON 对象"):
        parse_usb_config([])
    with pytest.raises(ValueError, match="timing"):
        parse_usb_config(
            {
                "vendor_id": 1204,
                "product_id": 241,
                "bulk_in_endpoint": 129,
                "bulk_out_endpoint": 1,
                "timing": [],
            }
        )


@pytest.mark.parametrize("kind", ["example", "mnist"])
def test_web_log_preserves_full_output(tmp_path: Path, kind: str) -> None:
    from cim22.web.example_manager import ExampleManager
    from cim22.web.mnist_manager import MnistManager

    path = tmp_path / "compile.log"
    content = "first output\r\n" + "运行日志\n" * 5000 + "last output\n"
    path.write_bytes(content.encode("utf-8"))
    manager = ExampleManager if kind == "example" else MnistManager
    assert manager._read_log(path) == content
    assert manager._read_log(tmp_path / "missing.log") == ""


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
        manager.import_model(inputs[1], inputs[2], usb_config=USB_CONFIG)
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


def _worker_with_stub_model(
    directory: Path,
    commands: Connection,
    cancel: Event,
    parent_pid: int,
    first_action: str,
) -> None:
    """Exercise the real worker lifecycle without model/compiler dependencies."""

    class StubEngine:
        def __init__(self, config: object, directory: Path) -> None:
            pass

        def compile(self, phase: Callable[[str], None]) -> float:
            return 0.0

    with patch.dict(
        "sys.modules",
        {"cim22.web.engine": SimpleNamespace(DvsEngine=StubEngine)},
    ):
        compute_worker(directory, commands, cancel, parent_pid, first_action)


def test_http_thread_retirement_keeps_compute_worker_alive(
    tmp_path: Path,
    inputs: Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anyio._backends._asyncio import _threadpool_workers

    from cim22.web import manager as implementation

    monkeypatch.setattr(implementation, "worker_main", _worker_with_stub_model)
    app = create_app(tmp_path / "results", inputs[0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/import",
            json={
                "weights": str(inputs[1]),
                "dataset": str(inputs[2]),
                "usb_config": USB_CONFIG,
            },
        )
        assert response.status_code == 200
        assert (
            client.post("/api/action", json={"action": "compile"}).status_code
            == 200
        )
        deadline = time.monotonic() + 10
        while True:
            state = client.get("/api/state").json()["task"]
            if state["status"] == "ready":
                break
            assert state["status"] != "interrupted", state["error"]
            assert time.monotonic() < deadline, state
            time.sleep(0.01)

        async def retire_request_threads() -> list[Thread]:
            # Use AnyIO's actual retirement operation without its idle timeout.
            # The service event loop remains alive throughout this test.
            workers = list(_threadpool_workers.get())
            for worker in workers:
                worker.stop()
            return workers

        process = app.state.manager.process
        for thread in client.portal.call(retire_request_threads):
            thread.join(timeout=5)
            assert not thread.is_alive()
        process.join(timeout=0.2)
        assert process.is_alive(), f"worker exited with {process.exitcode}"
        state = client.get("/api/state").json()["task"]
        assert state["status"] == "ready"
        assert state["compiled"]


def test_framework_board_run_releases_idle_dvs_worker(
    tmp_path: Path,
    inputs: Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cim22.web import manager as implementation

    monkeypatch.setattr(implementation, "worker_main", _worker_with_stub_model)
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "torch_compile.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[sys.argv.index('--result-json') + 1]).write_text('{}')\n",
        encoding="utf-8",
    )
    app = create_app(tmp_path / "results", inputs[0], examples)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert (
            client.post(
                "/api/import",
                json={
                    "weights": str(inputs[1]),
                    "dataset": str(inputs[2]),
                    "usb_config": USB_CONFIG,
                },
            ).status_code
            == 200
        )
        assert (
            client.post("/api/action", json={"action": "compile"}).status_code
            == 200
        )
        deadline = time.monotonic() + 10
        while (state := client.get("/api/state").json()["task"])[
            "status"
        ] != "ready":
            assert state["status"] != "interrupted", state["error"]
            assert time.monotonic() < deadline, state
            time.sleep(0.01)
        manager = app.state.manager
        worker = manager.process
        assert state["compiled"] and worker.is_alive()
        assert (
            client.post(
                "/api/examples/run",
                json={
                    "framework": "pytorch",
                    "runtime": "usb",
                    "usb_config": USB_CONFIG,
                },
            ).status_code
            == 200
        )
        state = client.get("/api/state").json()["task"]
        assert state["status"] == "ready"
        assert not state["compiled"]
        assert manager.process is None


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
    instance.cim_layers = ()
    monkeypatch.setattr(
        implementation,
        "session_for_artifact",
        lambda _: SimpleNamespace(dispatch_count=2),
    )
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
    "changed_source", ["backend.py", "compiler.py", "runtime.py", "usb_config"]
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
        engine,
        "compile_tool",
        lambda: tmp_path / "mlir-cim22-compile",
    )
    monkeypatch.setattr(
        engine, "digest", lambda path: revisions.get(path.name, "unchanged")
    )

    def reach_model_load(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model loading reached")

    monkeypatch.setattr(engine.torch, "load", reach_model_load)
    config = make_config(*inputs, usb_config=USB_CONFIG)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="model loading reached"):
            engine.DvsEngine(config, tmp_path)
    if changed_source == "usb_config":
        config["usb_config"] = {**USB_CONFIG, "timing": {"work_delay_ms": 20}}
    else:
        revisions[changed_source] = "updated"
    with pytest.raises(ValueError, match="运行环境已变化"):
        engine.DvsEngine(config, tmp_path)


def test_layer_defaults_overrides_and_recovery(tmp_path, inputs, monkeypatch):
    import torch

    from cim22.web import engine
    from cim22.web.engine import validate_cim_layers

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    monkeypatch.setattr(
        engine,
        "validate_cim_layer_names",
        lambda _, selected: validate_cim_layers(model, selected),
    )
    root = tmp_path / "results"
    app = create_app(root, inputs[0], default_cim_layers=["0"])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        body = {
            "weights": str(inputs[1]),
            "dataset": str(inputs[2]),
            "usb_config": USB_CONFIG,
        }
        assert client.get("/api/defaults").json()["cim_layers"] == ["0"]
        task = client.post("/api/import", json=body).json()["task"]
        assert task["cim_layers"] == ["0"]
        assert task["execution_mode"] == "hybrid"
        for names in ([], ["0", "0"], ["missing"], ["1"]):
            response = client.post(
                "/api/import", json={**body, "cim_layers": names}
            )
            assert response.status_code == 409
            assert client.get("/api/state").json()["task"]["id"] == task["id"]
        response = client.post("/api/import", json={**body, "cim_layers": None})
        assert response.json()["task"]["execution_mode"] == "full_model"
        response = client.post(
            "/api/import", json={**body, "cim_layers": ["0"]}
        )
        task = response.json()["task"]
    manager = TaskManager(root, inputs[0])
    try:
        assert manager.snapshot()["task"]["cim_layers"] == ["0"]
        assert read_json(manager.current / "config.json")["cim_layers"] == ["0"]
    finally:
        manager.close()


def test_mnist_api_configuration_and_state(
    tmp_path: Path, inputs: Inputs
) -> None:
    app = create_app(
        tmp_path / "results", inputs[0], examples_dir=tmp_path / "examples"
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/mnist/state").json() == {"task": None}
        response = client.post(
            "/api/mnist/run",
            json={"model": "standard", "epochs": 0},
        )
        assert response.status_code == 409
        assert "训练轮数" in response.json()["detail"]
        response = client.post(
            "/api/mnist/run",
            json={"model": "unknown", "epochs": 1},
        )
        assert response.status_code == 422


def test_mnist_archive_download_contains_training_outputs(
    tmp_path: Path, inputs: Inputs
) -> None:
    app = create_app(tmp_path / "results", inputs[0])
    task = tmp_path / "results/mnist/task"
    task.mkdir(parents=True)
    for name in (
        "model.pt",
        "config.json",
        "metrics.jsonl",
        "summary.json",
        "train.log",
    ):
        (task / name).write_text(name, encoding="utf-8")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/mnist/archive?task_id=task")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert response.headers["content-disposition"] == (
            'attachment; filename="training-results-task.zip"'
        )
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.namelist() == [
                "model.pt",
                "config.json",
                "metrics.jsonl",
                "summary.json",
                "train.log",
            ]
        assert (
            client.get("/api/mnist/archive?task_id=../task").status_code == 404
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"epochs": True},
        {"epochs": 1.5},
        {"learning_rate": "bad"},
        {"model": "unknown"},
    ],
)
def test_mnist_rejects_invalid_input_in_chinese(tmp_path, inputs, invalid):
    app = create_app(tmp_path / "results", inputs[0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/mnist/run", json=invalid)
        assert response.status_code == 422
        assert "训练配置无效" in response.json()["detail"]
        assert client.get("/api/mnist/state").json() == {"task": None}


def test_dvs_usb_config_is_persisted_across_restart(
    tmp_path: Path, inputs: Inputs
) -> None:
    root = tmp_path / "results"
    manager = TaskManager(root, inputs[0])
    manager.import_model(inputs[1], inputs[2], usb_config=USB_CONFIG)
    directory = manager.current
    config = read_json(directory / "config.json")
    assert config["runtime"] == "usb"
    assert config["usb_config"] == USB_CONFIG
    manager.close()
    manager = TaskManager(root, inputs[0])
    try:
        task = manager.snapshot()["task"]
        assert task["runtime"] == "usb"
        assert task["usb_config"] == USB_CONFIG
        validate_config(config)
    finally:
        manager.close()


@pytest.mark.parametrize("runtime", ["simulator", None])
def test_dvs_rejects_incompatible_task_before_recovery(
    tmp_path: Path, inputs: Inputs, runtime: str | None
) -> None:
    root = tmp_path / "results"
    manager = TaskManager(root, inputs[0])
    manager.import_model(inputs[1], inputs[2], usb_config=USB_CONFIG)
    directory = manager.current
    manager.close()
    config = read_json(directory / "config.json")
    invalid = dict(config)
    if runtime is None:
        invalid.pop("runtime")
    else:
        invalid["runtime"] = runtime
    save_json(directory / "config.json", invalid)
    state = read_json(directory / "state.json")
    state.update(status="running", runtime=runtime)
    save_json(directory / "state.json", state)
    save_rows(directory, [])
    paths = [
        root / "current.json",
        directory / "config.json",
        directory / "state.json",
        directory / "results.csv",
    ]
    contents = [path.read_bytes() for path in paths]
    failures = []
    for _ in range(2):
        with pytest.raises(ValueError, match="runtime 必须为 usb") as error:
            TaskManager(root, inputs[0])
        failures.append(error)
        assert [path.read_bytes() for path in paths] == contents
    # Startup failure must release the service lock, even with its traceback held.
    save_json(directory / "config.json", config)
    state["runtime"] = "usb"
    save_json(directory / "state.json", state)
    restarted = TaskManager(root, inputs[0])
    try:
        assert restarted.snapshot()["task"]["runtime"] == "usb"
    finally:
        restarted.close()


def test_dvs_config_requires_persisted_layer_selection(inputs: Inputs) -> None:
    config = make_config(*inputs, usb_config=USB_CONFIG)
    config.pop("cim_layers")
    with pytest.raises(ValueError, match="已变化"):
        validate_config(config)


@pytest.mark.parametrize(
    "fields",
    [
        {**USB_CONFIG, "product_id": 0},
        {**USB_CONFIG, "timing": {"work_delay_ms": -1}},
        {**USB_CONFIG, "timing": {"work_delay_ms": "10"}},
        {**USB_CONFIG, "unknown": 1},
        {**USB_CONFIG, "pll": {}},
    ],
)
def test_dvs_import_rejects_bad_board_config_before_task_creation(
    tmp_path: Path, inputs: Inputs, fields: dict[str, object]
) -> None:
    app = create_app(tmp_path / "results", inputs[0])
    with TestClient(app, base_url="http://127.0.0.1") as client:
        body = {"weights": str(inputs[1]), "dataset": str(inputs[2])}
        assert client.post("/api/import", json=body).status_code == 422
        response = client.post(
            "/api/import", json={**body, "usb_config": fields}
        )
        assert response.status_code == 409
        assert client.get("/api/state").json() == {"task": None}


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("dispatch_count", [None, 0, 3])
def test_dvs_compile_and_infer_require_usb_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
    dispatch_count: int | None,
) -> None:
    import torch

    from cim22.runtime import MODEL_SUFFIX
    from cim22.web import engine

    instance = engine.DvsEngine.__new__(engine.DvsEngine)
    instance.cim = torch.nn.Identity()
    instance.original = instance.cim
    instance.directory = tmp_path
    instance.cim_layers = ("head",) if selected else ()
    instance.usb_config = parse_usb_config(USB_CONFIG)
    instance.load = lambda _: torch.zeros(1)
    instance.trim = lambda: None
    calls = []
    session = SimpleNamespace(dispatch_count=dispatch_count or 0)
    monkeypatch.setattr(
        engine,
        "session_for_artifact",
        lambda _: session if dispatch_count is not None else None,
    )

    class Compiled(torch.nn.Module):
        layer_artifacts = {"head": ["model" + MODEL_SUFFIX]}
        layer_compile_seconds = {"head": 0.1}

        @property
        def dispatch_count(self) -> int:
            return session.dispatch_count

        def forward(self, _: torch.Tensor) -> torch.Tensor:
            return torch.ones(32, 1, 11)

    def compile_model(*args: object, **kwargs: object) -> Compiled:
        options = kwargs if selected else kwargs["options"]
        calls.append(options)
        assert options["runtime"] == "usb"
        assert options["usb_config"] is instance.usb_config
        assert instance.usb_config.timing.work_delay_ms == 10
        (options["artifact_dir"] / ("model" + MODEL_SUFFIX)).touch()
        return Compiled()

    monkeypatch.setattr(engine, "LayerOffload", compile_model)
    monkeypatch.setattr(engine.torch, "compile", compile_model)
    if not dispatch_count:
        with pytest.raises(RuntimeError, match="未执行 CIM 板卡计算"):
            instance.compile(lambda _: None)
    else:
        instance.compile(lambda _: None)
        assert instance.dispatch_count == dispatch_count
        assert instance.infer("cim", torch.zeros(1))[0] == 0
        session.dispatch_count = 0
        with pytest.raises(RuntimeError, match="未执行 CIM 板卡计算"):
            instance.infer("cim", torch.zeros(1))

        def board_error(_: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("USB transfer failed")

        instance.original = Compiled()
        instance.compiled = board_error
        cancel = mp.get_context("spawn").Event()
        config = {"samples": [{"path": "sample.npz", "label": 0}]}
        with pytest.raises(RuntimeError, match="USB transfer failed"):
            run_samples(
                tmp_path, config, instance, cancel, lambda **_: None, "single"
            )
        assert read_rows(tmp_path) == []
    assert len(calls) == 1
