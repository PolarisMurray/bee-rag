"""
Centralized API error handling.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import ApiErrorResponse


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach exception handlers to FastAPI app."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        payload = ApiErrorResponse(
            error="http_error",
            detail=str(exc.detail),
            request_id=request_id,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API exception")
        request_id = getattr(request.state, "request_id", None)
        payload = ApiErrorResponse(
            error="internal_error",
            detail="Internal server error.",
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

