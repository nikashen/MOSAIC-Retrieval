from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from mosaic import __version__

from .engine import MosaicSearchEngine, SearchFailure
from .video_evidence import VideoEvidenceCatalog


LOGGER = logging.getLogger("mosaic.serving")


def configure_logging(root: Path) -> Path:
    default = root / "logs"
    log_dir = Path(os.environ.get("MOSAIC_LOG_DIR", default)).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "mosaic_8050.jsonl"
    if not any(getattr(handler, "_mosaic_path", None) == str(path) for handler in LOGGER.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, "_mosaic_path", str(path))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    return path


def close_logging(path: Path) -> None:
    """Detach and close the handler owned by one application instance."""
    target = str(path)
    for handler in list(LOGGER.handlers):
        if getattr(handler, "_mosaic_path", None) != target:
            continue
        LOGGER.removeHandler(handler)
        handler.close()


class VectorSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector: list[float] = Field(min_length=2, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=100)


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


def create_app(root: Path | None = None, *, device: str = "cpu") -> FastAPI:
    root = Path(root or Path(__file__).resolve().parents[3]).resolve()
    log_path = configure_logging(root)
    engine = MosaicSearchEngine(root, device=device)
    video_evidence = VideoEvidenceCatalog(root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            close_logging(log_path)

    app = FastAPI(
        title="MOSAIC-Retrieval Workbench",
        version=__version__,
        description="Offline COCO retrieval and frozen MSR-VTT evidence workbench.",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.video_evidence = video_evidence
    app.state.close_logging = lambda: close_logging(log_path)

    @app.middleware("http")
    async def observability(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.2f}"
        LOGGER.info(
            json.dumps(
                {
                    "event": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed_ms, 3),
                    "request_id": request_id,
                },
                ensure_ascii=False,
            )
        )
        return response

    @app.exception_handler(SearchFailure)
    async def search_failure(_: Request, exc: SearchFailure) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_failure(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": {"code": "validation_failed", "details": exc.errors()}})

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        video_summary = video_evidence.summary()
        return {
            **engine.health(),
            "video_evidence_ready": video_evidence.ready,
            "video_samples": len(video_summary["samples"]),
            "structured_logging": True,
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return engine.models()

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        report = root / "reports" / "mosaic_external_final_v1.json"
        if not report.is_file():
            return {"status": "not_ready", "reports": []}
        return {"status": "ready", "report": json.loads(report.read_text(encoding="utf-8"))}

    @app.get("/api/video/evidence")
    def video_evidence_summary() -> dict[str, Any]:
        return video_evidence.summary()

    @app.get("/api/video/sample/{sample_id}")
    def video_sample(sample_id: str) -> FileResponse:
        try:
            path = video_evidence.sample_path(sample_id)
        except KeyError as exc:
            raise SearchFailure(
                "video_sample_not_found",
                "video sample is not available",
                404,
            ) from exc
        response = FileResponse(path, media_type="video/mp4")
        response.headers["Cache-Control"] = "private, max-age=3600"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.post("/api/search/vector")
    def vector_search(payload: VectorSearchRequest) -> dict[str, Any]:
        return {"mode": "vector", "results": engine.search_vector(np.asarray(payload.vector, dtype=np.float32), payload.top_k)}

    @app.post("/api/search/text")
    def text_search(payload: TextSearchRequest) -> dict[str, Any]:
        return {"mode": "text", "results": engine.search_text(payload.query, payload.top_k)}

    @app.get("/api/content/{content_id}/image")
    def content_image(content_id: int) -> FileResponse:
        return FileResponse(engine.image_path(content_id))

    static = Path(__file__).resolve().parent / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    return app


__all__ = ["create_app"]
