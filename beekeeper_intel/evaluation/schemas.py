"""
Evaluation schemas for Beekeeper Research Intelligence Platform.

Includes:
- retrieval benchmarking schemas
- grounding evaluation schemas
- report quality schemas
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from beekeeper_intel.models import Citation, FinalResearchReport, ResearchQuery


class EvaluationExample(BaseModel):
    """
    One labeled example for offline evaluation.

    Manual labels should include relevant chunk ids and/or document ids.
    """

    example_id: UUID = Field(default_factory=uuid4)
    query: ResearchQuery
    gold_chunk_ids: List[UUID] = Field(default_factory=list)
    gold_document_ids: List[UUID] = Field(default_factory=list)
    notes: Optional[str] = None


class EvaluationDataset(BaseModel):
    """Evaluation dataset loaded from manually labeled files."""

    dataset_id: str
    version: str = "v1"
    split: str = "test"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    examples: List[EvaluationExample] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    """
    Retrieval strategy switch set for experiments.
    """

    strategy_name: str
    retrieval_mode: str = Field(
        ..., description="Expected values: bm25_only | vector_only | hybrid | hybrid_rerank"
    )
    use_query_rewrite: bool = False
    use_hyde: bool = False
    use_rerank: bool = False
    top_k: int = 10
    extra: Dict[str, float | int | str | bool] = Field(default_factory=dict)


class RetrievalPrediction(BaseModel):
    """
    Ranked retrieval output for one query under one strategy.
    """

    example_id: UUID
    strategy_name: str
    retrieved_chunk_ids: List[UUID] = Field(default_factory=list)
    retrieved_document_ids: List[UUID] = Field(default_factory=list)


class RetrievalRunRecord(BaseModel):
    """
    Metrics for one query under one strategy.
    """

    record_id: UUID = Field(default_factory=uuid4)
    example_id: UUID
    strategy_name: str
    mrr: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0


class StrategyResult(BaseModel):
    """Aggregate retrieval metrics over a strategy."""

    strategy_name: str
    n_examples: int
    avg_mrr: float
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_ndcg_at_k: float
    per_example: List[RetrievalRunRecord] = Field(default_factory=list)


class RetrievalExperimentConfig(BaseModel):
    """Configuration for offline retrieval benchmarking."""

    dataset_path: str
    strategies: List[StrategyConfig]
    top_k: int = 10
    output_path: Optional[str] = None


class GroundingInput(BaseModel):
    """
    Input for answer grounding evaluation.
    """

    answer_text: str
    citations: List[Citation] = Field(default_factory=list)
    evidence_texts: List[str] = Field(default_factory=list)


class GroundingEvaluationResult(BaseModel):
    """
    Grounding evaluation output with interpretable diagnostics.
    """

    supported: bool
    citation_presence: float = Field(..., ge=0.0, le=1.0)
    citation_grounding: float = Field(..., ge=0.0, le=1.0)
    evidence_coverage: float = Field(..., ge=0.0, le=1.0)
    unsupported_claim_count: int = 0
    unsupported_claims: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class ReportEvaluationInput(BaseModel):
    """
    Input for report quality evaluation.
    """

    report: FinalResearchReport
    required_sections: List[str] = Field(
        default_factory=lambda: ["executive_summary", "needs", "workflow_frictions", "opportunities"]
    )


class ReportEvaluationResult(BaseModel):
    """
    Structured report quality result.
    """

    citation_presence: float = Field(..., ge=0.0, le=1.0)
    citation_grounding: float = Field(..., ge=0.0, le=1.0)
    evidence_coverage: float = Field(..., ge=0.0, le=1.0)
    unsupported_claim_count: int = 0
    unsupported_claims: List[str] = Field(default_factory=list)
    section_completeness: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    weaknesses: List[str] = Field(default_factory=list)

