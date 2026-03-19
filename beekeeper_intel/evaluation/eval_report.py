"""
Report quality evaluation for research synthesis outputs.
"""

from __future__ import annotations

from typing import List

from beekeeper_intel.evaluation.eval_grounding import evaluate_grounding
from beekeeper_intel.evaluation.schemas import (
    GroundingInput,
    ReportEvaluationInput,
    ReportEvaluationResult,
)


def evaluate_report(inp: ReportEvaluationInput) -> ReportEvaluationResult:
    """
    Evaluate FinalResearchReport quality:
    - citation presence
    - citation grounding
    - evidence coverage
    - unsupported claim detection
    - section completeness
    """

    report = inp.report
    # Flatten report text approximation for grounding check.
    text_parts = [report.executive_summary]
    text_parts.extend([n.statement for n in report.needs])
    text_parts.extend([f.description for f in report.workflow_frictions])
    text_parts.extend([o.problem_statement for o in report.opportunities])
    report_text = "\n".join([x for x in text_parts if x])

    # Evidence texts are not embedded in report; use citation quotes as proxy where available.
    evidence_texts = [c.quote for c in report.citations if c.quote]
    g = evaluate_grounding(
        GroundingInput(
            answer_text=report_text,
            citations=report.citations,
            evidence_texts=evidence_texts,
        )
    )

    section_completeness = _section_completeness(report, inp.required_sections)
    weaknesses: List[str] = list(g.weaknesses)
    if section_completeness < 1.0:
        weaknesses.append("missing_required_sections")

    # Weighted overall score
    overall = (
        0.25 * g.citation_presence
        + 0.30 * g.citation_grounding
        + 0.30 * g.evidence_coverage
        + 0.15 * section_completeness
    )

    return ReportEvaluationResult(
        citation_presence=g.citation_presence,
        citation_grounding=g.citation_grounding,
        evidence_coverage=g.evidence_coverage,
        unsupported_claim_count=g.unsupported_claim_count,
        unsupported_claims=g.unsupported_claims,
        section_completeness=section_completeness,
        overall_score=max(0.0, min(1.0, overall)),
        weaknesses=weaknesses,
    )


def _section_completeness(report, required_sections: List[str]) -> float:
    present = 0
    total = max(1, len(required_sections))
    for s in required_sections:
        if s == "executive_summary" and bool(report.executive_summary.strip()):
            present += 1
        elif s == "needs" and len(report.needs) > 0:
            present += 1
        elif s == "workflow_frictions" and len(report.workflow_frictions) > 0:
            present += 1
        elif s == "opportunities" and len(report.opportunities) > 0:
            present += 1
        elif s == "persona_insights" and len(report.persona_insights) > 0:
            present += 1
    return present / total

