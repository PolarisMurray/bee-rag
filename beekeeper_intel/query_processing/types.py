"""
Types for query processing.

These are intentionally stable, strongly-typed interfaces so the planner/retriever layers can
depend on them without importing implementation details.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from beekeeper_intel.models import PersonaType, ResearchTopic, WorkflowStage


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for model defaults."""

    return datetime.now(UTC)


class SupportedIntent(str, Enum):
    """Supported query intents for Beekeeper Research Intelligence Platform."""

    problem_discovery = "problem_discovery"
    persona_comparison = "persona_comparison"
    workflow_analysis = "workflow_analysis"
    evidence_synthesis = "evidence_synthesis"
    opportunity_framing = "opportunity_framing"
    follow_up_clarification = "follow_up_clarification"
    document_lookup = "document_lookup"


class QueryProcessingConfig(BaseModel):
    """
    Configuration for query processing.

    Keep this config serializable so it can be logged and used for reproducibility in evaluations.
    """

    enable_typo_correction: bool = Field(True, description="Whether to attempt typo correction.")
    enable_synonym_expansion: bool = Field(True, description="Whether to expand domain synonyms.")
    enable_hyde: bool = Field(True, description="Whether HyDE generation is allowed by policy.")

    max_expansions: int = Field(12, ge=0, description="Maximum number of expansions to emit.")
    max_decomposition: int = Field(6, ge=0, description="Maximum number of subqueries to emit.")

    # Heuristics
    short_query_token_threshold: int = Field(
        6, ge=1, description="Queries shorter than this are treated as potentially vague/follow-up."
    )


class HyDEConfig(BaseModel):
    """Configuration for HyDE hypothetical document generation."""

    desired_length_chars: int = Field(900, ge=100, description="Target length for HyDE document.")
    style: str = Field(
        "evidence_note",
        description="HyDE style (used by LLM-based generators; ignored by deterministic generator).",
    )


class HyDEResult(BaseModel):
    """Result of HyDE generation."""

    used: bool = Field(..., description="Whether HyDE was used.")
    hypothetical_document: Optional[str] = Field(None, description="Generated hypothetical document text.")
    generator: str = Field(..., description="Generator identifier (e.g., 'deterministic_v1', 'llm_v1').")


class QueryProcessingTrace(BaseModel):
    """Debug/observability payload for query processing decisions."""

    trace_id: UUID = Field(default_factory=uuid4, description="Trace id for this processing run.")
    created_at: datetime = Field(default_factory=utc_now, description="When trace was created.")
    steps: List[str] = Field(default_factory=list, description="Step names executed (in order).")
    notes: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary debug notes (safe to log).")


class QueryProcessingResult(BaseModel):
    """
    Structured output consumed by planner/retriever.

    This is intentionally richer than a simple rewritten query: it captures constraints, expansions,
    follow-up resolution, and HyDE outputs for explainability.
    """

    request_id: UUID = Field(default_factory=uuid4, description="Unique id for this processing result.")

    # Intent + confidence
    intent: SupportedIntent = Field(..., description="Classified intent.")
    intent_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for intent classification.")
    intent_signals: List[str] = Field(default_factory=list, description="Matched signals/rules for intent.")

    # Resolved query (the main one to retrieve against)
    original_query: str = Field(..., description="Original user query.")
    resolved_query: str = Field(..., description="Resolved query after follow-up rewriting.")
    rewritten_query: str = Field(..., description="Final retrieval-optimized query string.")

    # Optional additional subqueries (for evidence synthesis / decomposition)
    subqueries: List[str] = Field(default_factory=list, description="Additional subqueries for retrieval.")

    # Expansion + corrections
    typo_corrections: List[Tuple[str, str]] = Field(
        default_factory=list, description="List of (original, corrected) terms."
    )
    expansions: List[str] = Field(default_factory=list, description="Synonym/keyword expansions.")

    # Extracted domain constraints (used for metadata-aware retrieval)
    persona: Optional[PersonaType] = Field(None, description="Detected/target persona.")
    topics: List[ResearchTopic] = Field(default_factory=list, description="Detected topics.")
    workflow_stage: Optional[WorkflowStage] = Field(None, description="Detected workflow stage.")

    # HyDE
    hyde: HyDEResult = Field(..., description="HyDE output (may be unused).")

    # Observability
    trace: QueryProcessingTrace = Field(default_factory=QueryProcessingTrace, description="Debug trace.")
