"""
Lightweight in-memory API metrics.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict

from .schemas import MetricsResponse


class ApiMetrics:
    """Thread-safe request counters and latency aggregates."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total = 0
        self._requests_ok = 0
        self._requests_error = 0
        self._latency_total_ms = 0.0
        self._per_route: Dict[str, int] = {}

    def record(self, route: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._requests_total += 1
            self._latency_total_ms += max(0.0, latency_ms)
            self._per_route[route] = self._per_route.get(route, 0) + 1
            if 200 <= status_code < 400:
                self._requests_ok += 1
            else:
                self._requests_error += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            avg = self._latency_total_ms / self._requests_total if self._requests_total else 0.0
            return MetricsResponse(
                requests_total=self._requests_total,
                requests_ok=self._requests_ok,
                requests_error=self._requests_error,
                average_latency_ms=avg,
                per_route=dict(self._per_route),
            )

