"""
Integration helpers for adding citation explainability to final answer/report outputs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from beekeeper_intel.models import Citation, FinalAnswer, FinalResearchReport, RetrievedEvidence

from .models import (
    CitationFormat,
    CitationProvenanceRecord,
    ExplainableAnswerBundle,
    ExplainableReportBundle,
)
from .renderer import render_citations, render_inline_citation_block


def build_answer_bundle(
    final_answer: FinalAnswer,
    *,
    fmt: CitationFormat = CitationFormat.compact,
) -> ExplainableAnswerBundle:
    """
    Build explainable conversational answer output.

    - Ensures citations are populated from evidence if missing.
    - Produces inline citation block for user-facing answer.
    - Produces provenance records for audit/debug.
    """

    citations = final_answer.citations or _collect_citations_from_evidence(final_answer.evidence)
    if not final_answer.citations and citations:
        final_answer.citations = citations

    rendered = render_citations(citations, fmt=fmt)
    inline = render_inline_citation_block(citations, fmt=fmt)
    rendered_answer = final_answer.answer if not inline else f"{final_answer.answer}\n\n{inline}"

    provenance = _build_provenance_records(
        claim_ref="final_answer",
        citations=citations,
        evidence=final_answer.evidence,
    )
    return ExplainableAnswerBundle(
        answer=final_answer,
        rendered_answer=rendered_answer,
        rendered_citations=rendered,
        provenance=provenance,
    )


def build_report_evidence_map(
    report: FinalResearchReport,
    *,
    section_to_evidence: Optional[Dict[str, List[RetrievedEvidence]]] = None,
) -> Dict[str, List[Citation]]:
    """
    Build/augment section-level evidence map for FinalResearchReport.

    If explicit section_to_evidence is provided, it takes precedence.
    Otherwise uses report.evidence_map as-is and ensures report-level citations are represented.
    """

    if section_to_evidence:
        out: Dict[str, List[Citation]] = {}
        for section, evidence_items in section_to_evidence.items():
            out[section] = _collect_citations_from_evidence(evidence_items)
        report.evidence_map = out
        # refresh report-level citations
        report.citations = _dedupe_citations([c for cites in out.values() for c in cites])
        return out

    # ensure citations from explicit report.citations are represented at least in a fallback section
    if report.citations and not report.evidence_map:
        report.evidence_map = {"report_summary": list(report.citations)}
    return report.evidence_map


def build_report_bundle(
    report: FinalResearchReport,
    *,
    fmt: CitationFormat = CitationFormat.compact,
    section_to_evidence: Optional[Dict[str, List[RetrievedEvidence]]] = None,
) -> ExplainableReportBundle:
    """
    Build explainable report output with rendered evidence map + provenance.
    """

    evidence_map = build_report_evidence_map(report, section_to_evidence=section_to_evidence)
    # collect report-level citations from evidence map if needed
    if not report.citations:
        report.citations = _dedupe_citations([c for cites in evidence_map.values() for c in cites])

    rendered_citations = render_citations(report.citations, fmt=fmt)
    rendered_map = {section: [rc.rendered for rc in render_citations(cites, fmt=fmt)] for section, cites in evidence_map.items()}

    provenance: List[CitationProvenanceRecord] = []
    for section, cites in evidence_map.items():
        provenance.extend(_build_provenance_records(claim_ref=section, citations=cites, evidence=[]))

    return ExplainableReportBundle(
        report=report,
        rendered_citations=rendered_citations,
        rendered_evidence_map=rendered_map,
        provenance=provenance,
    )


def _collect_citations_from_evidence(evidence_items: Iterable[RetrievedEvidence]) -> List[Citation]:
    citations: List[Citation] = []
    for e in evidence_items:
        citations.extend(e.citations)
    return _dedupe_citations(citations)


def _build_provenance_records(
    *,
    claim_ref: str,
    citations: List[Citation],
    evidence: List[RetrievedEvidence],
) -> List[CitationProvenanceRecord]:
    # map citation_id to one evidence record if possible
    ev_by_citation = {}
    for e in evidence:
        for c in e.citations:
            ev_by_citation[c.citation_id] = e

    out: List[CitationProvenanceRecord] = []
    for c in citations:
        e = ev_by_citation.get(c.citation_id)
        rank = None
        score = None
        steps: List[str] = []
        evidence_id = None
        if e is not None:
            evidence_id = e.evidence_id
            steps = list(e.postprocess_steps)
            if e.source_candidates:
                rank = e.source_candidates[0].rank
                score = e.source_candidates[0].score
        out.append(
            CitationProvenanceRecord(
                claim_ref=claim_ref,
                evidence_id=evidence_id,
                citation_id=c.citation_id,
                source_title=c.source_title,
                retrieval_rank=rank,
                retrieval_score=score,
                postprocess_steps=steps,
            )
        )
    return out


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    seen = set()
    out: List[Citation] = []
    for c in citations:
        key = (
            c.document_id,
            c.source_title,
            c.anchor.page_number,
            c.anchor.slide_number,
            c.anchor.timestamp_ms,
            tuple(c.anchor.section_path),
            c.quote,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

