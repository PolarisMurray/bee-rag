"""
Debug-friendly retriever used when no production retriever is injected.
"""

from __future__ import annotations

from typing import List
from uuid import uuid4

from beekeeper_intel.models import Anchor, Citation, RetrievedEvidence


class DemoRetriever:
    """
    Minimal retriever adapter for local API debugging.

    This keeps `/query` and `/research/report` functional before wiring
    the real BM25/vector/hybrid retrieval stack.
    """

    def retrieve(self, plan, qp) -> List[RetrievedEvidence]:
        base_query = plan.rewritten_query or ""
        qid = plan.query_id
        requested_top_k = getattr(plan, "top_k", 7)
        target_count = max(7, min(12, requested_top_k))
        source_rows = [
            {
                "title": "Beekeeper Forum - Varroa Monitoring Thread",
                "uri": "forum://beekeeper/varroa-monitoring",
                "page": 1,
                "sections": ["Community Reports"],
                "quote": "I struggle to run alcohol washes every week because it takes too much time after work.",
                "text": (
                    f"Query focus: {base_query}. Hobbyist operators report routine varroa checks are time-intensive. "
                    "Many rely on irregular manual checks and personal notes."
                ),
            },
            {
                "title": "Extension Bulletin - Varroa Best Practices",
                "uri": "docs://extension/varroa-best-practices",
                "page": 12,
                "sections": ["Monitoring", "Operational Constraints"],
                "quote": "Monitoring compliance often drops when labor is tight, especially during peak field workload.",
                "text": (
                    "Commercial teams delegate inspections and standardize treatment windows, but still report labor "
                    "and timing bottlenecks in monitoring workflows."
                ),
            },
            {
                "title": "Reddit Beekeeping - Weekly Inspection Habits",
                "uri": "reddit://beekeeping/inspection-habits",
                "page": 3,
                "sections": ["Inspection Routine"],
                "quote": "I usually skip a formal mite count unless something already looks off because the process eats half my evening.",
                "text": (
                    "Small-scale keepers frequently defer formal mite counts until obvious colony stress appears, which leads to reactive "
                    "instead of scheduled monitoring."
                ),
            },
            {
                "title": "State Apiary Survey - Seasonal Workflow Frictions",
                "uri": "survey://state-apiary/seasonal-frictions",
                "page": 8,
                "sections": ["Seasonal Labor"],
                "quote": "Spring splits, feeding, and honey prep compress the window for careful monitoring.",
                "text": (
                    "During spring buildup and honey flow prep, monitoring competes with splits, feeding, and equipment setup, reducing "
                    "consistency even among experienced operators."
                ),
            },
            {
                "title": "Commercial Apiary Ops Interview Notes",
                "uri": "interview://commercial-apiary/ops-notes",
                "page": 4,
                "sections": ["Delegation", "Crew Coordination"],
                "quote": "Even with crews, someone still has to centralize counts and decide when the threshold means action.",
                "text": (
                    "Commercial operations can distribute inspection labor, but decision bottlenecks remain around aggregating counts and "
                    "triggering treatment windows across yards."
                ),
            },
            {
                "title": "Extension Workshop Transcript - Recordkeeping Pain Points",
                "uri": "transcript://extension-workshop/recordkeeping",
                "page": 6,
                "sections": ["Records", "Monitoring"],
                "quote": "People are writing counts on gloves, lids, or scraps of paper and re-entering them later.",
                "text": (
                    "Both hobbyist and sideliner beekeepers describe fragmented recordkeeping, with mite counts captured on paper first and "
                    "digitized later, creating data loss and follow-up errors."
                ),
            },
            {
                "title": "Beekeeper Slack Archive - Treatment Timing",
                "uri": "slack://beekeeper/treatment-timing",
                "page": 9,
                "sections": ["Treatment Decision"],
                "quote": "The hard part is not knowing the threshold, it's lining up weather, labor, and the right week to act.",
                "text": (
                    "Operators say the friction is less about understanding treatment thresholds and more about fitting monitoring and action "
                    "into weather, labor, and honey production constraints."
                ),
            },
            {
                "title": "Regional Co-op Notes - Multi-yard Monitoring",
                "uri": "coop://regional/multi-yard-monitoring",
                "page": 11,
                "sections": ["Field Ops"],
                "quote": "Once you have several yards, the driving and batching become as painful as the sampling itself.",
                "text": (
                    "As apiaries expand to multiple yards, travel time and batching of counts become core burdens, increasing the payoff of "
                    "faster sampling or centralized tracking."
                ),
            },
            {
                "title": "University Pilot Study - Sensor Interest",
                "uri": "study://university/sensor-interest",
                "page": 14,
                "sections": ["Opportunity Signals"],
                "quote": "Participants consistently asked for a lower-touch way to decide when manual sampling is worth doing.",
                "text": (
                    "Respondents show interest in triage tools that help decide when a full mite count is necessary, suggesting demand for "
                    "screening, reminders, and alerting products."
                ),
            },
        ]

        evidence_items: List[RetrievedEvidence] = []
        for row in source_rows[:target_count]:
            citation = Citation(
                document_id=uuid4(),
                source_title=row["title"],
                source_uri=row["uri"],
                anchor=Anchor(page_number=row["page"], section_path=list(row["sections"])),
                quote=row["quote"],
            )
            evidence_items.append(
                RetrievedEvidence(
                    query_id=qid,
                    evidence_text=row["text"],
                    citations=[citation],
                    postprocess_steps=["demo_retriever", "hybrid_stub", "rerank_stub"],
                )
            )

        return evidence_items
