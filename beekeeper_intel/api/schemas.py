"""
Typed request/response schemas for FastAPI interface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from beekeeper_intel.memory.followup_rewriter import FollowupRewriteResult
from beekeeper_intel.orchestration.orchestrator import OrchestrationTraceEvent, PipelineMode


class ApiErrorResponse(BaseModel):
    """Normalized API error payload."""

    error: str
    detail: str
    request_id: Optional[str] = None


class QueryRequest(BaseModel):
    """Conversational query request payload."""

    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_trace: bool = Field(False, description="Include orchestration trace in response.")


class ReportRequest(BaseModel):
    """Structured research report request payload."""

    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_trace: bool = Field(False, description="Include orchestration trace in response.")


class DocumentIngestItem(BaseModel):
    """Single document ingestion request item."""

    uri: Optional[str] = None
    text: Optional[str] = None
    source_type: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentsIngestRequest(BaseModel):
    """Batch ingestion request."""

    documents: List[DocumentIngestItem] = Field(default_factory=list)
    rebuild_indexes: bool = False


class IngestedDocumentResult(BaseModel):
    """Per-document ingestion result."""

    document_id: Optional[UUID] = None
    uri: Optional[str] = None
    status: str
    message: Optional[str] = None


class DocumentsIngestResponse(BaseModel):
    """Batch ingestion response payload."""

    accepted: int
    succeeded: int
    failed: int
    results: List[IngestedDocumentResult] = Field(default_factory=list)


class CitationView(BaseModel):
    """API-friendly citation shape."""

    citation_id: UUID
    source_title: str
    source_uri: Optional[str] = None
    quote: Optional[str] = None
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    timestamp_ms: Optional[int] = None
    section_path: List[str] = Field(default_factory=list)
    rendered: Optional[str] = None


class EvidenceView(BaseModel):
    """Evidence metadata returned by query/report endpoints."""

    evidence_id: UUID
    text: str
    postprocess_steps: List[str] = Field(default_factory=list)
    citations: List[CitationView] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Conversational query response."""

    run_id: UUID
    mode: PipelineMode
    session_id: Optional[str] = None
    answer: str
    citations: List[CitationView] = Field(default_factory=list)
    evidence: List[EvidenceView] = Field(default_factory=list)
    trace: List[OrchestrationTraceEvent] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """Research report response."""

    run_id: UUID
    mode: PipelineMode
    session_id: Optional[str] = None
    executive_summary: str
    needs_count: int = 0
    citations: List[CitationView] = Field(default_factory=list)
    evidence_map: Dict[str, List[str]] = Field(default_factory=dict)
    trace: List[OrchestrationTraceEvent] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    service: str
    version: str
    now_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MetricsResponse(BaseModel):
    """Basic API metrics payload."""

    requests_total: int
    requests_ok: int
    requests_error: int
    average_latency_ms: float
    per_route: Dict[str, int] = Field(default_factory=dict)

