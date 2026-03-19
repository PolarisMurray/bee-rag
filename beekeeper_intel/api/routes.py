"""
API routes for query, research report, ingestion, health, and metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from beekeeper_intel.models import Citation, MultiTurnContextState, RetrievedEvidence
from beekeeper_intel.orchestration import PipelineMode

from .schemas import (
    CitationView,
    DocumentsIngestRequest,
    DocumentsIngestResponse,
    EvidenceView,
    HealthResponse,
    IngestedDocumentResult,
    QueryRequest,
    QueryResponse,
    ReportRequest,
    ReportResponse,
)


logger = logging.getLogger(__name__)
router = APIRouter()


class IngestionService(Protocol):
    """Optional ingestion service adapter."""

    def ingest(self, request: DocumentsIngestRequest) -> DocumentsIngestResponse:
        ...


@router.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "beekeeper-research-intelligence-platform",
        "ui": "/ui",
        "health": "/health",
    }


@router.get("/ui")
def ui() -> FileResponse:
    page = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(str(page))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="beekeeper-research-intelligence-platform",
        version="0.1.0",
    )


@router.get("/metrics")
def metrics(request: Request):
    metrics_obj = request.app.state.metrics
    return metrics_obj.snapshot()


@router.post("/query", response_model=QueryResponse)
def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    orchestrator = request.app.state.orchestrator
    session_state = _load_or_create_session_state(request, request_body.session_id)
    result = orchestrator.run(
        query_text=request_body.query,
        context_state=session_state,
        mode=PipelineMode.conversational_qa,
        user_id=request_body.user_id,
    )
    if not result.success or result.answer_bundle is None:
        raise HTTPException(status_code=500, detail=result.error or "Failed to process query.")

    _persist_session_state(request, result.context_state)
    answer_bundle = result.answer_bundle
    response = QueryResponse(
        run_id=result.run_id,
        mode=result.mode,
        session_id=(result.context_state.session_id if result.context_state else None),
        answer=answer_bundle.rendered_answer,
        citations=_citations_from_rendered(answer_bundle.rendered_citations),
        evidence=_evidence_views(answer_bundle.answer.evidence),
        trace=(result.trace if request_body.include_trace else []),
    )
    return response


@router.post("/research/report", response_model=ReportResponse)
def research_report(request_body: ReportRequest, request: Request) -> ReportResponse:
    orchestrator = request.app.state.orchestrator
    session_state = _load_or_create_session_state(request, request_body.session_id)
    result = orchestrator.run(
        query_text=request_body.query,
        context_state=session_state,
        mode=PipelineMode.research_synthesis,
        user_id=request_body.user_id,
    )
    if not result.success or result.report_bundle is None:
        raise HTTPException(status_code=500, detail=result.error or "Failed to generate report.")

    _persist_session_state(request, result.context_state)
    report_bundle = result.report_bundle
    response = ReportResponse(
        run_id=result.run_id,
        mode=result.mode,
        session_id=(result.context_state.session_id if result.context_state else None),
        executive_summary=report_bundle.report.executive_summary,
        needs_count=len(report_bundle.report.needs),
        citations=_citations_from_rendered(report_bundle.rendered_citations),
        evidence_map=report_bundle.rendered_evidence_map,
        trace=(result.trace if request_body.include_trace else []),
    )
    return response


@router.post("/documents/ingest", response_model=DocumentsIngestResponse)
def documents_ingest(request_body: DocumentsIngestRequest, request: Request) -> DocumentsIngestResponse:
    ingestion_service: Optional[IngestionService] = getattr(request.app.state, "ingestion_service", None)
    if ingestion_service is None:
        # Graceful fallback for early integration stage.
        results: List[IngestedDocumentResult] = []
        for doc in request_body.documents:
            results.append(
                IngestedDocumentResult(
                    uri=doc.uri,
                    status="accepted_not_processed",
                    message="No ingestion service configured.",
                )
            )
        return DocumentsIngestResponse(
            accepted=len(request_body.documents),
            succeeded=0,
            failed=len(request_body.documents),
            results=results,
        )
    return ingestion_service.ingest(request_body)


def _load_or_create_session_state(request: Request, session_id: Optional[str]) -> MultiTurnContextState:
    store: Dict[str, MultiTurnContextState] = request.app.state.session_store
    sid = session_id or str(uuid4())
    state = store.get(sid)
    if state is None:
        state = MultiTurnContextState(session_id=sid)
        store[sid] = state
    return state


def _persist_session_state(request: Request, state: Optional[MultiTurnContextState]) -> None:
    if state is None:
        return
    request.app.state.session_store[state.session_id] = state


def _citations_from_rendered(rendered_citations) -> List[CitationView]:
    out: List[CitationView] = []
    for rc in rendered_citations:
        c = rc.citation
        out.append(
            CitationView(
                citation_id=c.citation_id,
                source_title=c.source_title,
                source_uri=c.source_uri,
                quote=c.quote,
                page_number=c.anchor.page_number,
                slide_number=c.anchor.slide_number,
                timestamp_ms=c.anchor.timestamp_ms,
                section_path=list(c.anchor.section_path),
                rendered=rc.rendered,
            )
        )
    return out


def _citation_view(c: Citation) -> CitationView:
    return CitationView(
        citation_id=c.citation_id,
        source_title=c.source_title,
        source_uri=c.source_uri,
        quote=c.quote,
        page_number=c.anchor.page_number,
        slide_number=c.anchor.slide_number,
        timestamp_ms=c.anchor.timestamp_ms,
        section_path=list(c.anchor.section_path),
    )


def _evidence_views(evidence_items: List[RetrievedEvidence]) -> List[EvidenceView]:
    out: List[EvidenceView] = []
    for e in evidence_items:
        out.append(
            EvidenceView(
                evidence_id=e.evidence_id,
                text=e.evidence_text,
                postprocess_steps=list(e.postprocess_steps),
                citations=[_citation_view(c) for c in e.citations],
            )
        )
    return out

