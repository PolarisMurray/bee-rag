"""
FastAPI application entrypoint for Beekeeper Research Intelligence Platform.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from beekeeper_intel.models import MultiTurnContextState
from beekeeper_intel.orchestration import PlatformOrchestrator

from .demo_retriever import DemoRetriever
from .errors import register_exception_handlers
from .metrics import ApiMetrics
from .routes import IngestionService, router


logger = logging.getLogger(__name__)


def create_app(
    *,
    orchestrator: Optional[PlatformOrchestrator] = None,
    ingestion_service: Optional[IngestionService] = None,
) -> FastAPI:
    """
    Create and configure FastAPI app.

    The app is intentionally dependency-injectable for easier testing and deployment.
    """

    app = FastAPI(
        title="Beekeeper Research Intelligence Platform API",
        version="0.1.0",
    )

    app.state.orchestrator = orchestrator or PlatformOrchestrator(retriever=DemoRetriever())
    app.state.ingestion_service = ingestion_service
    app.state.metrics = ApiMetrics()
    app.state.session_store: dict[str, MultiTurnContextState] = {}

    @app.middleware("http")
    async def request_logging_and_metrics(request: Request, call_next) -> Response:
        start = time.perf_counter()
        request_id = str(uuid4())
        request.state.request_id = request_id
        logger.info("api.request.start", extra={"request_id": request_id, "path": request.url.path, "method": request.method})
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        app.state.metrics.record(request.url.path, response.status_code, latency_ms)
        logger.info(
            "api.request.end",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        response.headers["x-request-id"] = request_id
        return response

    app.include_router(router)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    register_exception_handlers(app)
    return app


app = create_app()

