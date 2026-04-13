"""
Citation/explainability models.

These models complement core domain models with rendering + provenance payloads used
by answer/report synthesis layers and audit tooling.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from beekeeper_intel.models import Citation, FinalAnswer, FinalResearchReport


class CitationFormat(str, Enum):
    """Supported user-facing citation output styles."""

    compact = "compact"
    verbose = "verbose"


class RenderedCitation(BaseModel):
    """Citation plus rendered display text."""

    citation: Citation = Field(..., description="Original structured citation.")
    rendered: str = Field(..., description="Rendered display string, e.g. '[Doc p.12]'.")


class CitationProvenanceRecord(BaseModel):
    """
    Provenance record for debugging/auditability.

    Captures how an answer/report claim was grounded in evidence.
    """

    record_id: UUID = Field(default_factory=uuid4)
    claim_ref: str = Field(..., description="Claim identifier or section key.")
    evidence_id: Optional[UUID] = Field(None, description="RetrievedEvidence id when available.")
    citation_id: UUID = Field(..., description="Citation id used for this grounding.")
    source_title: str = Field(..., description="Human-readable source title.")
    retrieval_rank: Optional[int] = Field(None, description="Candidate rank if known.")
    retrieval_score: Optional[float] = Field(None, description="Candidate score if known.")
    postprocess_steps: List[str] = Field(default_factory=list, description="Context postprocessing lineage.")


class ExplainableAnswerBundle(BaseModel):
    """FinalAnswer with rendered citations and provenance info."""

    answer: FinalAnswer
    rendered_answer: str = Field(..., description="Answer text with appended inline citation block.")
    rendered_citations: List[RenderedCitation] = Field(default_factory=list)
    provenance: List[CitationProvenanceRecord] = Field(default_factory=list)


class ExplainableReportBundle(BaseModel):
    """FinalResearchReport with evidence-map rendering and provenance."""

    report: FinalResearchReport
    rendered_citations: List[RenderedCitation] = Field(default_factory=list)
    rendered_evidence_map: Dict[str, List[str]] = Field(
        default_factory=dict, description="Section -> list of rendered citation strings."
    )
    provenance: List[CitationProvenanceRecord] = Field(default_factory=list)

