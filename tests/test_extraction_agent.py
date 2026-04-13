from __future__ import annotations

from uuid import uuid4

from beekeeper_intel.agents import ExtractionAgent
from beekeeper_intel.models import (
    Anchor,
    Citation,
    RetrievedEvidence,
    SourceMetadata,
    SourceType,
    DocumentFormat,
    OCRSource,
)


def _citation(title: str, page: int) -> Citation:
    return Citation(
        document_id=uuid4(),
        source_title=title,
        source_uri=f"file:///{title.replace(' ', '_')}.pdf",
        anchor=Anchor(page_number=page, section_path=["Varroa", "Monitoring"]),
        quote=None,
        confidence=0.9,
    )


def test_extract_forum_varroa_monitoring_pain_workaround():
    agent = ExtractionAgent()
    ev = RetrievedEvidence(
        query_id=uuid4(),
        evidence_text=(
            "I struggle to do alcohol washes regularly because it takes too much time after work. "
            "I just eyeball mites on sticky boards and hope it's enough. "
            "I wish there was a simple tool to tell me when to treat."
        ),
        citations=[_citation("Forum thread: varroa monitoring", 1)],
        source_candidates=[],
        postprocess_steps=["expand_neighbors"],
    )
    src = SourceMetadata(
        source_type=SourceType.forum,
        document_format=DocumentFormat.html,
        title="Forum thread: varroa monitoring",
        authors=[],
        ocr_used=False,
        ocr_source=OCRSource.none,
    )
    insights = agent.extract_from_evidence(ev, src)
    assert insights
    ins = insights[0]
    assert ins.directness == "direct_user_voice"
    assert ins.pain_severity_1_5 >= 3
    assert ins.current_workaround is not None
    assert ins.product_signal is not None
    assert ins.unmet_need is True
    assert ins.supporting_snippets


def test_extract_extension_guidance_varroa_thresholds_barriers():
    agent = ExtractionAgent()
    ev = RetrievedEvidence(
        query_id=uuid4(),
        evidence_text=(
            "Extension guidance: Monitor varroa using alcohol wash or sugar roll. "
            "Treatment thresholds vary by season; delays increase colony loss risk. "
            "However, handling chemicals requires safety training and proper equipment."
        ),
        citations=[_citation("State Extension Varroa Guide", 12)],
        source_candidates=[],
        postprocess_steps=[],
    )
    src = SourceMetadata(
        source_type=SourceType.extension_resource,
        document_format=DocumentFormat.pdf,
        title="State Extension Varroa Guide",
        authors=["State Extension"],
        ocr_used=False,
        ocr_source=OCRSource.none,
    )
    insights = agent.extract_from_evidence(ev, src)
    assert insights
    ins = insights[0]
    assert ins.directness == "expert_guidance"
    assert any("safety" in b.lower() or "equipment" in b.lower() for b in ins.barriers)


def test_aggregation_clusters_similar_varroa_monitoring_pain_across_sources():
    agent = ExtractionAgent()

    ev1 = RetrievedEvidence(
        query_id=uuid4(),
        evidence_text="I struggle with varroa monitoring because it is time-consuming and I forget. I just do a quick sugar roll.",
        citations=[_citation("Interview note A", 2)],
        source_candidates=[],
        postprocess_steps=[],
    )
    src1 = SourceMetadata(
        source_type=SourceType.interview_transcript,
        document_format=DocumentFormat.vtt,
        title="Interview note A",
        authors=[],
        ocr_used=False,
        ocr_source=OCRSource.asr_transcript,
    )
    ins1 = agent.extract_from_evidence(ev1, src1)

    ev2 = RetrievedEvidence(
        query_id=uuid4(),
        evidence_text="Forum: Varroa testing is hard after work. I just skip alcohol wash most weeks.",
        citations=[_citation("Forum thread B", 1)],
        source_candidates=[],
        postprocess_steps=[],
    )
    src2 = SourceMetadata(
        source_type=SourceType.forum,
        document_format=DocumentFormat.html,
        title="Forum thread B",
        authors=[],
        ocr_used=False,
        ocr_source=OCRSource.none,
    )
    ins2 = agent.extract_from_evidence(ev2, src2)

    clusters = agent.aggregate([*ins1, *ins2])
    assert clusters
    # should form 1 cluster for similar monitoring pain
    assert clusters[0].evidence_count >= 2
    assert clusters[0].is_multi_source_signal is True
    assert "forum" in clusters[0].source_types


def test_extract_overwintering_labor_burden():
    agent = ExtractionAgent()
    ev = RetrievedEvidence(
        query_id=uuid4(),
        evidence_text=(
            "Overwintering is difficult because feeding and wrapping takes all day across multiple yards. "
            "We keep notes in a spreadsheet, but it's still confusing which colonies got what."
        ),
        citations=[_citation("Commercial operations notes", 5)],
        source_candidates=[],
        postprocess_steps=[],
    )
    src = SourceMetadata(
        source_type=SourceType.internal_note,
        document_format=DocumentFormat.txt,
        title="Commercial operations notes",
        authors=[],
        ocr_used=False,
        ocr_source=OCRSource.none,
    )
    insights = agent.extract_from_evidence(ev, src)
    assert insights
    assert insights[0].pain_severity_1_5 >= 3
    assert insights[0].current_workaround is not None or insights[0].barriers

