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

        c1 = Citation(
            document_id=uuid4(),
            source_title="Beekeeper Forum - Varroa Monitoring Thread",
            source_uri="forum://beekeeper/varroa-monitoring",
            anchor=Anchor(page_number=1, section_path=["Community Reports"]),
            quote="I struggle to run alcohol washes every week because it takes too much time after work.",
        )
        c2 = Citation(
            document_id=uuid4(),
            source_title="Extension Bulletin - Varroa Best Practices",
            source_uri="docs://extension/varroa-best-practices",
            anchor=Anchor(page_number=12, section_path=["Monitoring", "Operational Constraints"]),
            quote="Monitoring compliance often drops when labor is tight, especially during peak field workload.",
        )

        return [
            RetrievedEvidence(
                query_id=qid,
                evidence_text=(
                    f"Query focus: {base_query}. Hobbyist operators report routine varroa checks are time-intensive. "
                    "Many rely on irregular manual checks and personal notes."
                ),
                citations=[c1],
                postprocess_steps=["demo_retriever", "hybrid_stub", "rerank_stub"],
            ),
            RetrievedEvidence(
                query_id=qid,
                evidence_text=(
                    "Commercial teams delegate inspections and standardize treatment windows, but still report labor "
                    "and timing bottlenecks in monitoring workflows."
                ),
                citations=[c2],
                postprocess_steps=["demo_retriever", "hybrid_stub", "rerank_stub"],
            ),
        ]

