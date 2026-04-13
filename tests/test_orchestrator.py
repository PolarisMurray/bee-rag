from __future__ import annotations

from uuid import uuid4

from beekeeper_intel.models import Anchor, Citation, RetrievedEvidence
from beekeeper_intel.orchestration import OrchestratorConfig, PipelineMode, PlatformOrchestrator


class FakeRetriever:
    def retrieve(self, plan, qp):  # pragma: no cover - signature is protocol-driven
        c1 = Citation(
            document_id=uuid4(),
            source_title="Forum: Varroa Burden Thread",
            anchor=Anchor(page_number=1),
            quote="I struggle to do alcohol washes every week because it is too time-consuming after work.",
        )
        c2 = Citation(
            document_id=uuid4(),
            source_title="Extension Varroa Bulletin",
            anchor=Anchor(page_number=12),
            quote="Monitoring compliance drops when labor constraints increase.",
        )
        return [
            RetrievedEvidence(
                query_id=plan.query_id,
                evidence_text=(
                    "I struggle to do alcohol washes every week because it is too time-consuming after work. "
                    "I wish there was a faster way to track mite load."
                ),
                citations=[c1],
                postprocess_steps=["hybrid_retrieve", "rerank"],
            ),
            RetrievedEvidence(
                query_id=plan.query_id,
                evidence_text=(
                    "Extension guidance notes labor burden and low consistency in mite monitoring. "
                    "Commercial operators use delegated inspections but still report bottlenecks."
                ),
                citations=[c2],
                postprocess_steps=["hybrid_retrieve", "rerank"],
            ),
        ]


def test_orchestrator_conversational_qa_pipeline():
    orch = PlatformOrchestrator(
        cfg=OrchestratorConfig(max_iterations=2),
        retriever=FakeRetriever(),
    )
    out = orch.run(
        query_text="How do hobbyist beekeepers currently deal with varroa monitoring burden?",
        mode=PipelineMode.conversational_qa,
    )
    assert out.success is True
    assert out.answer_bundle is not None
    assert out.answer_bundle.answer.citations
    assert "Sources:" in out.answer_bundle.rendered_answer
    assert out.trace and any(t.step == "critic" for t in out.trace)


def test_orchestrator_research_synthesis_pipeline():
    orch = PlatformOrchestrator(
        cfg=OrchestratorConfig(max_iterations=2),
        retriever=FakeRetriever(),
    )
    out = orch.run(
        query_text="Compare hobbyist vs commercial pain points in varroa monitoring and unmet needs.",
        mode=PipelineMode.research_synthesis,
    )
    assert out.success is True
    assert out.report_bundle is not None
    assert out.report_bundle.report.needs
    assert out.report_bundle.report.citations
    assert "needs" in out.report_bundle.rendered_evidence_map

