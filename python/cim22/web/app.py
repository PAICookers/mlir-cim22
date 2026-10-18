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

from .manager import TaskManager
from .storage import Document


class ImportRequest(BaseModel):
    """Local paths selected by the single user."""

    weights: str
    dataset: str


class ActionRequest(BaseModel):
    """Allowed serial compute commands."""

    action: Literal["compile", "single", "full"]


def create_app(root: Path, model_dir: Path) -> FastAPI:
    """Create one local service. Its lifespan owns the exclusive worker lock."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.manager = TaskManager(root, model_dir)
        try:
            yield
        finally:
            app.state.manager.close()

    app = FastAPI(title="CIM22 · DVS 本地对比", lifespan=lifespan)
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
    def defaults() -> dict[str, str]:
        return {
            "weights": str(
                model_dir / "weights/dvsgesture_mobilenet_snn_qat_int8.pt"
            ),
            "dataset": str(model_dir / "data/test"),
        }

    @app.post("/api/import")
    def import_model(body: ImportRequest) -> Document:
        try:
            return app.state.manager.import_model(
                Path(body.weights), Path(body.dataset)
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
