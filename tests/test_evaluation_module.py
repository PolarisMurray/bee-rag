from __future__ import annotations

from uuid import uuid4

from beekeeper_intel.evaluation.eval_grounding import evaluate_grounding
from beekeeper_intel.evaluation.eval_report import evaluate_report
from beekeeper_intel.evaluation.eval_retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from beekeeper_intel.evaluation.schemas import GroundingInput, ReportEvaluationInput
from beekeeper_intel.models import (
    Anchor,
    Citation,
    IntentType,
    NeedInsight,
    ResearchIntent,
    ResearchPlan,
    ResearchQuery,
    FinalResearchReport,
    RetrievalMethod,
    PersonaType,
    ResearchTopic,
)


def test_retrieval_metrics_basic():
    g1 = uuid4()
    g2 = uuid4()
    pred = [uuid4(), g1, uuid4(), g2]
    gold = {g1, g2}
    assert mean_reciprocal_rank(pred, gold) == 0.5
    assert precision_at_k(pred, gold, 2) == 0.5
    assert recall_at_k(pred, gold, 2) == 0.5
    assert 0.0 <= ndcg_at_k(pred, gold, 4) <= 1.0


def test_grounding_eval_with_supported_claims():
    quote = "Hobbyist beekeepers report that alcohol wash monitoring is too time-consuming after work."
    inp = GroundingInput(
        answer_text="Hobbyists face time burden in varroa monitoring.",
        citations=[
            Citation(
                document_id=uuid4(),
                source_title="Interview Note A",
                anchor=Anchor(section_path=["Varroa", "Monitoring burden"]),
                quote=quote,
            )
        ],
        evidence_texts=[quote],
    )
    out = evaluate_grounding(inp)
    assert out.citation_presence > 0
    assert out.citation_grounding > 0.5
    assert out.evidence_coverage > 0.5


def test_report_eval_outputs_score():
    query = ResearchQuery(text="Compare hobbyist vs commercial varroa monitoring pain points.")
    plan = ResearchPlan(
        query_id=query.query_id,
        intent=ResearchIntent(intent_type=IntentType.need_discovery, confidence=0.95),
        retrieval_method=RetrievalMethod.hybrid_fusion,
    )
    report = FinalResearchReport(
        query=query,
        plan=plan,
        executive_summary="Monitoring burden appears across personas.",
        needs=[
            NeedInsight(
                persona=PersonaType.hobbyist,
                topic=ResearchTopic.varroa_management,
                statement="Hobbyists struggle to perform regular alcohol washes due to time constraints.",
                pain_severity_1_5=4,
                unmet_need=True,
                confidence=0.82,
            )
        ],
        citations=[
            Citation(
                document_id=uuid4(),
                source_title="Interview transcript",
                anchor=Anchor(timestamp_ms=12_000),
                quote="I often skip the alcohol wash because it takes too long after work.",
            )
        ],
    )
    out = evaluate_report(ReportEvaluationInput(report=report))
    assert 0.0 <= out.overall_score <= 1.0
    assert out.section_completeness > 0.0

