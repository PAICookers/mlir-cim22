"""Loopback HTTP API and SSE snapshots; computation never runs in requests."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .example_manager import ExampleManager
from .manager import USE_DEFAULT_CIM_LAYERS, TaskManager
from .storage import Document


class ImportRequest(BaseModel):
    """Local paths selected by the single user."""

    weights: str
    dataset: str
    cim_layers: list[str] | None = None


class ActionRequest(BaseModel):
    """Allowed serial compute commands."""

    action: Literal["compile", "single", "full"]


class ExampleRunRequest(BaseModel):
    """Fixed framework example and native execution backend."""

    framework: Literal["tensorflow", "pytorch"]
    runtime: Literal["simulator", "usb"] = "simulator"
    usb_config: dict[str, object] | None = None


def create_app(
    root: Path,
    model_dir: Path,
    examples_dir: Path | None = None,
    default_cim_layers: list[str] | None = None,
) -> FastAPI:
    """Create one local service. Its lifespan owns the exclusive worker lock."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.manager = TaskManager(root, model_dir, default_cim_layers)
        selected_examples = (
            examples_dir or Path(__file__).resolve().parents[2] / "examples"
        )
        app.state.examples = ExampleManager(
            root,
            selected_examples,
            app.state.manager.lock,
            app.state.manager.ensure_dvs_idle,
        )
        app.state.manager.external_idle = app.state.examples.ensure_idle
        try:
            yield
        finally:
            app.state.examples.close()
            app.state.manager.close()

    app = FastAPI(title="CIM22 · 编译验证工作台", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"]
    )

    @app.middleware("http")
    async def local_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Reject cross-site writes to a local filesystem-capable service.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get(
                "host"
            ):
                return JSONResponse({"detail": "只接受本地页面操作"}, 403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_request(
        request: Request, error: ValueError
    ) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/api/state")
    def state() -> Document:
        return app.state.manager.snapshot()

    @app.get("/api/defaults")
    def defaults() -> Document:
        return {
            "weights": str(
                model_dir / "weights/dvsgesture_mobilenet_snn_qat_int8.pt"
            ),
            "dataset": str(model_dir / "data/test"),
            "cim_layers": default_cim_layers,
        }

    @app.get("/api/cim-layers")
    def cim_layers() -> list[dict[str, object]]:
        from .engine import discover_cim_layers

        return discover_cim_layers(model_dir)

    @app.post("/api/import")
    def import_model(body: ImportRequest) -> Document:
        try:
            selected_layers = (
                body.cim_layers
                if "cim_layers" in body.model_fields_set
                else USE_DEFAULT_CIM_LAYERS
            )
            return app.state.manager.import_model(
                Path(body.weights), Path(body.dataset), selected_layers
            )
        except OSError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/action")
    async def action(body: ActionRequest) -> Document:
        # Linux PDEATHSIG follows the spawning thread. Keep process creation
        # on the service event loop, not an expendable HTTP pool thread.
        return app.state.manager.start(body.action)

    @app.post("/api/cancel")
    def cancel() -> Document:
        return app.state.manager.cancel()

    @app.get("/api/examples/state")
    def example_state() -> Document:
        return app.state.examples.snapshot()

    @app.post("/api/examples/run")
    def example_run(body: ExampleRunRequest) -> Document:
        return app.state.examples.start(
            body.framework, body.runtime, body.usb_config
        )

    @app.post("/api/examples/cancel")
    def example_cancel() -> Document:
        return app.state.examples.cancel()

    @app.get("/api/examples/file")
    def example_file(task_id: str, name: str) -> Response:
        """Serve only files from one persisted example result directory."""
        from fastapi.responses import FileResponse

        base = app.state.examples.root / task_id
        path = (base / name).resolve()
        if base.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "示例产物不存在")
        return FileResponse(path)

    @app.get("/api/examples/ir")
    def example_ir(task_id: str) -> JSONResponse:
        directory = (app.state.examples.root / task_id / "ir").resolve()
        root = app.state.examples.root.resolve()
        if root not in directory.parents or not directory.is_dir():
            raise HTTPException(404, "示例 IR 不存在")
        files = [
            str(path.relative_to(directory))
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        ]
        return JSONResponse({"task_id": task_id, "files": files})

    @app.get("/api/results.csv")
    def results() -> Response:
        manager = app.state.manager
        with manager.lock:
            if (
                manager.current is None
                or not (manager.current / "results.csv").exists()
            ):
                raise HTTPException(404, "尚无已完成样本")
            # Read one immutable version so a simultaneous atomic replacement
            # cannot change Content-Length between stat and streaming.
            return Response(
                (manager.current / "results.csv").read_bytes(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="dvs-{manager.current.name}.csv"'
                },
            )

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            previous = None
            while not await request.is_disconnected():
                snapshot = await asyncio.to_thread(app.state.manager.snapshot)
                data = json.dumps(snapshot, ensure_ascii=False)
                if data != previous:
                    yield f"data: {data}\n\n"
                    previous = data
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/examples/events")
    async def example_events(request: Request) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            previous = None
            while not await request.is_disconnected():
                snapshot = await asyncio.to_thread(app.state.examples.snapshot)
                data = json.dumps(snapshot, ensure_ascii=False)
                if data != previous:
                    yield f"data: {data}\n\n"
                    previous = data
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    static = Path(__file__).parent / "static"
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    else:

        @app.get("/")
        def missing_build() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": "前端尚未构建。请在 web/ 执行 pnpm install 和 pnpm build。"
                },
                status_code=503,
            )

    return app
