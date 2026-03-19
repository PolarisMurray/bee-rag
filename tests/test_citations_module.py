from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from beekeeper_intel.citations import (
    build_answer_bundle,
    build_report_bundle,
    render_citation,
)
from beekeeper_intel.citations.models import CitationFormat
from beekeeper_intel.models import (
    Anchor,
    Citation,
    FinalAnswer,
    FinalResearchReport,
    IntentType,
    ResearchIntent,
    ResearchPlan,
    ResearchQuery,
    RetrievedEvidence,
    RetrievalMethod,
)


def _base_query() -> ResearchQuery:
    return ResearchQuery(
        text="What do hobbyist beekeepers struggle with in varroa monitoring?",
        session_id="s-cite-1",
    )


def _base_plan(query_id):
    return ResearchPlan(
        query_id=query_id,
        intent=ResearchIntent(intent_type=IntentType.need_discovery, confidence=0.9),
        retrieval_method=RetrievalMethod.hybrid_fusion,
    )


def test_render_formats_document_slide_transcript_interview():
    c_doc = Citation(
        document_id=uuid4(),
        source_title="Varroa Field Guide",
        anchor=Anchor(page_number=7),
    )
    c_slide = Citation(
        document_id=uuid4(),
        source_title="Training Deck",
        anchor=Anchor(slide_number=4),
    )
    c_ts = Citation(
        document_id=uuid4(),
        source_title="Interview transcript A",
        anchor=Anchor(timestamp_ms=12 * 60 * 1000 + 31 * 1000),
    )
    c_int = Citation(
        document_id=uuid4(),
        source_title="Interview Note - Apiary Ops",
        anchor=Anchor(section_path=["Pain Points", "Inspection Burden"]),
    )

    assert render_citation(c_doc) == "[Varroa Field Guide p.7]"
    assert render_citation(c_slide) == "[PPT slide 4]"
    assert render_citation(c_ts) == "[Transcript 00:12:31]"
    assert render_citation(c_int) == "[Interview Note section Inspection Burden]"


def test_build_answer_bundle_appends_sources_and_provenance():
    q = _base_query()
    plan = _base_plan(q.query_id)
    citation = Citation(
        document_id=uuid4(),
        source_title="Forum thread: varroa monitoring",
        anchor=Anchor(page_number=1, section_path=["Thread"]),
        quote="I struggle to do alcohol washes regularly because it takes too much time after work.",
    )
    evidence = RetrievedEvidence(
        query_id=q.query_id,
        evidence_text="I struggle to do alcohol washes regularly because it takes too much time after work.",
        citations=[citation],
        source_candidates=[],
        postprocess_steps=["expand_neighbors"],
    )
    ans = FinalAnswer(query=q, plan=plan, answer="Hobbyists report time burden in routine varroa checks.", evidence=[evidence])
    bundle = build_answer_bundle(ans, fmt=CitationFormat.compact)

    assert bundle.answer.citations
    assert "Sources:" in bundle.rendered_answer
    assert "[Forum thread: varroa monitoring p.1]" in bundle.rendered_answer
    assert bundle.provenance and bundle.provenance[0].claim_ref == "final_answer"


def test_build_report_bundle_renders_evidence_map():
    q = _base_query()
    plan = _base_plan(q.query_id)
    report = FinalResearchReport(
        query=q,
        plan=plan,
        executive_summary="Varroa monitoring burden appears across hobbyist and commercial contexts.",
    )

    c1 = Citation(document_id=uuid4(), source_title="Varroa Extension Bulletin", anchor=Anchor(page_number=12))
    c2 = Citation(document_id=uuid4(), source_title="Interview transcript B", anchor=Anchor(timestamp_ms=15_000))
    e1 = RetrievedEvidence(query_id=q.query_id, evidence_text="Guidance text...", citations=[c1], source_candidates=[], postprocess_steps=[])
    e2 = RetrievedEvidence(query_id=q.query_id, evidence_text="User quote...", citations=[c2], source_candidates=[], postprocess_steps=[])

    bundle = build_report_bundle(
        report,
        section_to_evidence={
            "workflow_frictions": [e1, e2],
        },
    )

    assert "workflow_frictions" in bundle.rendered_evidence_map
    rendered = " ".join(bundle.rendered_evidence_map["workflow_frictions"])
    assert "[Varroa Extension Bulletin p.12]" in rendered
    assert "[Transcript 00:00:15]" in rendered
    assert bundle.provenance

