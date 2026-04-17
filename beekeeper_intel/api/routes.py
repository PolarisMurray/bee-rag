"""
API routes for query, research report, ingestion, health, and metrics.
"""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from beekeeper_intel.models import Citation, MultiTurnContextState, NeedInsight, RetrievedEvidence
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
    ReportDistributionsView,
    ReportNeedResultView,
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
        llm_provider=request_body.llm_provider,
        llm_api_key=request_body.llm_api_key,
        llm_model=request_body.llm_model,
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
        llm_provider=request_body.llm_provider,
        llm_api_key=request_body.llm_api_key,
        llm_model=request_body.llm_model,
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
        key_needs=[need.statement for need in report_bundle.report.needs[:8]],
        gaps_and_unknowns=list(report_bundle.report.gaps_and_unknowns),
        citations=_citations_from_rendered(report_bundle.rendered_citations),
        evidence_map=report_bundle.rendered_evidence_map,
        results=[_report_need_result_view(need) for need in report_bundle.report.needs],
        distributions=_report_distributions(report_bundle.report.needs),
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


def _report_need_result_view(need: NeedInsight) -> ReportNeedResultView:
    supporting_quotes = []
    for citation in need.citations:
        if citation.quote and citation.quote not in supporting_quotes:
            supporting_quotes.append(citation.quote)
        if len(supporting_quotes) >= 3:
            break

    return ReportNeedResultView(
        statement=need.statement,
        persona=need.persona.value,
        topic=need.topic.value,
        workflow_stage=(need.workflow_stage.value if need.workflow_stage is not None else None),
        pain_severity_1_5=need.pain_severity_1_5,
        frequency_1_5=need.frequency_1_5,
        confidence=need.confidence,
        unmet_need=need.unmet_need,
        current_workaround=need.current_workaround,
        product_signal=need.product_signal,
        evidence_count=need.evidence_count,
        citation_count=len(need.citations),
        source_titles=list(need.source_titles),
        source_type_distribution=dict(need.source_type_distribution),
        is_multi_source_signal=need.is_multi_source_signal,
        supporting_quotes=supporting_quotes,
    )


def _report_distributions(needs: List[NeedInsight]) -> ReportDistributionsView:
    persona_counts = Counter()
    topic_counts = Counter()
    workflow_counts = Counter()
    frequency_counts = Counter()
    source_type_counts = Counter()
    density_counts = Counter()

    for need in needs:
        persona_counts[need.persona.value] += 1
        topic_counts[need.topic.value] += 1
        if need.workflow_stage is not None:
            workflow_counts[need.workflow_stage.value] += 1
        if need.frequency_1_5 is not None:
            frequency_counts[str(need.frequency_1_5)] += 1
        for source_type, count in need.source_type_distribution.items():
            source_type_counts[source_type] += count
        density_counts[_density_bucket(need.evidence_count)] += 1

    return ReportDistributionsView(
        personas=dict(persona_counts),
        topics=dict(topic_counts),
        workflow_stages=dict(workflow_counts),
        frequency_1_5=dict(frequency_counts),
        source_types=dict(source_type_counts),
        evidence_density=dict(density_counts),
    )


def _density_bucket(evidence_count: int) -> str:
    if evidence_count <= 1:
        return "single_source"
    if evidence_count == 2:
        return "paired_signal"
    if evidence_count <= 4:
        return "clustered_3_4"
    return "dense_5_plus"
