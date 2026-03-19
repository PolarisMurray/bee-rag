"""
Pydantic domain models for the Beekeeper Research Intelligence Platform.

Design goals:
- Strong typing for industrial RAG pipelines (ingestion -> retrieval -> generation -> eval).
- Evidence traceability (anchors + citations are first-class).
- Beekeeper-domain research synthesis (pain points, workflows, unmet needs, opportunities).

Notes:
- These models target Pydantic v2 (`pydantic>=2`).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# -----------------------------
# Enums (domain + system)
# -----------------------------


class SourceType(str, Enum):
    """High-level source category for filtering and synthesis provenance."""

    research_paper = "research_paper"
    extension_resource = "extension_resource"
    forum = "forum"
    interview_transcript = "interview_transcript"
    internal_note = "internal_note"
    slide_deck = "slide_deck"
    report = "report"
    dataset = "dataset"
    other = "other"


class DocumentFormat(str, Enum):
    """Original file format of an ingested document."""

    pdf = "pdf"
    scanned_pdf = "scanned_pdf"
    pptx = "pptx"
    image = "image"
    txt = "txt"
    html = "html"
    srt = "srt"
    vtt = "vtt"
    csv = "csv"
    docx = "docx"
    other = "other"


class OCRSource(str, Enum):
    """How text was obtained for a chunk/document."""

    none = "none"  # native digital text
    ocr_engine = "ocr_engine"  # text produced by an OCR engine
    vision_caption = "vision_caption"  # text produced by vision captioning (figures/tables)
    asr_transcript = "asr_transcript"  # automatic speech recognition transcript


class PersonaType(str, Enum):
    """Beekeeper persona type for need discovery and comparisons."""

    hobbyist = "hobbyist"
    sideliner = "sideliner"
    commercial = "commercial"
    queen_breeder = "queen_breeder"
    pollination_operator = "pollination_operator"
    beekeeper_researcher = "beekeeper_researcher"
    extension_agent = "extension_agent"
    supplier = "supplier"
    unknown = "unknown"


class ResearchTopic(str, Enum):
    """Common beekeeper research topics used for tagging and routing."""

    varroa_management = "varroa_management"
    nutrition_feeding = "nutrition_feeding"
    overwintering = "overwintering"
    queen_health_reproduction = "queen_health_reproduction"
    disease_pest_general = "disease_pest_general"
    equipment_tools = "equipment_tools"
    labor_operations = "labor_operations"
    honey_production = "honey_production"
    pollination_services = "pollination_services"
    regulatory_compliance = "regulatory_compliance"
    economics_markets = "economics_markets"
    safety = "safety"
    climate_environment = "climate_environment"
    other = "other"


class WorkflowStage(str, Enum):
    """Workflow stage for structuring friction points and interventions."""

    planning = "planning"
    monitoring = "monitoring"
    diagnosis = "diagnosis"
    treatment = "treatment"
    follow_up = "follow_up"
    recordkeeping = "recordkeeping"
    procurement = "procurement"
    transport = "transport"
    apiary_management = "apiary_management"
    extraction_processing = "extraction_processing"
    sales_distribution = "sales_distribution"
    other = "other"


class IntentType(str, Enum):
    """Top-level intent categories for routing QA vs synthesis workflows."""

    conversational_qa = "conversational_qa"
    evidence_synthesis = "evidence_synthesis"
    need_discovery = "need_discovery"
    workflow_analysis = "workflow_analysis"
    persona_comparison = "persona_comparison"
    opportunity_generation = "opportunity_generation"
    retrieval_debug = "retrieval_debug"


class RetrievalMethod(str, Enum):
    """Retrieval method used for a candidate/evaluation record."""

    vector = "vector"
    bm25 = "bm25"
    hybrid_fusion = "hybrid_fusion"
    graph = "graph"


class RerankMethod(str, Enum):
    """Reranking strategy used after initial retrieval."""

    none = "none"
    cross_encoder = "cross_encoder"
    llm = "llm"
    keyword = "keyword"


# -----------------------------
# Shared low-level structs
# -----------------------------


class TextSpan(BaseModel):
    """A byte/character span within some text field."""

    start: int = Field(..., ge=0, description="Start offset (inclusive).")
    end: int = Field(..., ge=0, description="End offset (exclusive).")


class Anchor(BaseModel):
    """
    A precise location marker inside a source.

    Anchors are critical for explainability and citation: they tie retrieved evidence back to
    document structure (page/slide/section) or time (transcripts).
    """

    page_number: Optional[int] = Field(None, ge=1, description="1-indexed page number if applicable.")
    slide_number: Optional[int] = Field(None, ge=1, description="1-indexed slide number if applicable.")
    timestamp_ms: Optional[int] = Field(
        None, ge=0, description="Timestamp in milliseconds for audio/video transcripts or subtitles."
    )
    section_path: List[str] = Field(
        default_factory=list,
        description="Hierarchical section titles, e.g. ['Varroa', 'Treatment options', 'Oxalic acid']",
    )
    block_id: Optional[str] = Field(
        None,
        description="Parser-specific identifier for a structural block (table id, figure id, paragraph id).",
    )
    span: Optional[TextSpan] = Field(
        None, description="Optional span within the chunk text for pinpoint citations."
    )


class Citation(BaseModel):
    """
    A human- and machine-readable citation for an evidence item.

    Citations should remain stable across runs by using stable ids and anchors.
    """

    citation_id: UUID = Field(default_factory=uuid4, description="Stable id for this citation instance.")
    document_id: UUID = Field(..., description="Refers to ParsedDocument.document_id.")
    source_title: str = Field(..., description="Human-readable source title (e.g., report name).")
    source_uri: Optional[str] = Field(
        None, description="URI or path for the source (file path, S3 key, etc.)."
    )
    anchor: Anchor = Field(..., description="Where this citation points into the source.")
    quote: Optional[str] = Field(
        None,
        description="Optional verbatim quote used as evidence (preferred for research reporting).",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in citation correctness (parser/OCR confidence or heuristic).",
    )


class SourceMetadata(BaseModel):
    """
    Metadata attached to a source document (for filtering, provenance, and synthesis).
    """

    source_type: SourceType = Field(..., description="Category of the source.")
    document_format: DocumentFormat = Field(..., description="Original format of the file.")
    title: str = Field(..., description="Document title for UI and citations.")
    authors: List[str] = Field(default_factory=list, description="Authors or creators.")
    published_at: Optional[datetime] = Field(None, description="Publication or creation time.")
    org: Optional[str] = Field(None, description="Publishing organization (university extension, company, etc.).")
    region: Optional[str] = Field(None, description="Geography (state/province/country) if relevant.")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es').")
    tags: List[str] = Field(default_factory=list, description="Free-form tags for filtering.")
    beekeeper_persona_hint: Optional[PersonaType] = Field(
        None, description="If source is strongly associated with a persona."
    )
    topics: List[ResearchTopic] = Field(default_factory=list, description="Topics covered by the source.")

    # OCR + extraction provenance
    ocr_used: bool = Field(False, description="Whether OCR was applied to produce core text.")
    ocr_source: OCRSource = Field(OCRSource.none, description="How text was produced (OCR, caption, etc.).")
    ocr_engine: Optional[str] = Field(None, description="OCR engine identifier/version if used.")
    ocr_avg_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Average OCR confidence across pages/regions if available."
    )

    # Internal governance
    access_level: Optional[str] = Field(
        None, description="Access classification (public/internal/confidential) for governance."
    )


class ParsedDocument(BaseModel):
    """
    Canonical representation of an ingested document.

    This is the output of the ingestion/parsing pipeline prior to chunking/indexing.
    """

    document_id: UUID = Field(default_factory=uuid4, description="Stable id for the document.")
    source: SourceMetadata = Field(..., description="Source metadata and provenance.")
    uri: Optional[str] = Field(None, description="Storage location or file path for this document.")
    extracted_at: datetime = Field(default_factory=datetime.utcnow, description="When ingestion/extraction ran.")

    # Parsed structure
    section_index: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Normalized section tree/index produced by parser (implementation-defined).",
    )
    page_count: Optional[int] = Field(None, ge=0, description="Page count if applicable.")
    slide_count: Optional[int] = Field(None, ge=0, description="Slide count if applicable.")
    duration_ms: Optional[int] = Field(None, ge=0, description="Duration for transcripts/audio/video if applicable.")

    # Raw extracted artifacts (optional references)
    extracted_text: Optional[str] = Field(
        None, description="Full extracted text (may be omitted for large docs)."
    )
    extracted_tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables (normalized).")
    extracted_images: List[Dict[str, Any]] = Field(
        default_factory=list, description="Extracted images/figures metadata (paths, anchors)."
    )


class DocumentChunk(BaseModel):
    """
    A chunked unit of text used for indexing and retrieval.

    Chunk ids must be stable and carry the anchor/section metadata needed for citations.
    """

    chunk_id: UUID = Field(default_factory=uuid4, description="Stable id for this chunk.")
    document_id: UUID = Field(..., description="Parent ParsedDocument.document_id.")
    text: str = Field(..., description="Chunk text (may include cleaned OCR text).")
    anchor: Anchor = Field(..., description="Primary anchor for this chunk.")

    # chunk metadata
    chunk_index: Optional[int] = Field(None, ge=0, description="Ordinal chunk index within document.")
    token_count: Optional[int] = Field(None, ge=0, description="Approx token count for context budgeting.")
    char_count: int = Field(..., ge=0, description="Character count for quick sizing.")
    chunker: str = Field(..., description="Chunking strategy name/version (e.g., 'structure_aware_v1').")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in chunk correctness (OCR/layout/heuristics)."
    )

    # domain enrichment at chunk-level (optional)
    topics: List[ResearchTopic] = Field(default_factory=list, description="Topics detected for this chunk.")
    persona_hint: Optional[PersonaType] = Field(None, description="Persona implied by the chunk.")
    workflow_stage_hint: Optional[WorkflowStage] = Field(None, description="Workflow stage implied by the chunk.")


class RetrievalCandidate(BaseModel):
    """
    Candidate chunk returned from the retrieval layer (pre- or post-rerank).
    """

    candidate_id: UUID = Field(default_factory=uuid4, description="Stable id for this candidate instance.")
    query_id: UUID = Field(..., description="Refers to ResearchQuery.query_id.")
    chunk_id: UUID = Field(..., description="Refers to DocumentChunk.chunk_id.")
    document_id: UUID = Field(..., description="Refers to ParsedDocument.document_id.")

    retrieval_method: RetrievalMethod = Field(..., description="How this candidate was retrieved.")
    rerank_method: RerankMethod = Field(RerankMethod.none, description="How this candidate was reranked (if any).")

    # scoring + ranks (store all to support debugging and evaluation)
    rank: int = Field(..., ge=1, description="Final rank presented to the generator.")
    score: float = Field(..., description="Final score used for ordering (post-fusion/post-rerank).")
    vector_score: Optional[float] = Field(None, description="Raw dense retrieval score if applicable.")
    bm25_score: Optional[float] = Field(None, description="Raw BM25 score if applicable.")
    rerank_score: Optional[float] = Field(None, description="Reranker score if applicable.")

    # evidence payload
    snippet: Optional[str] = Field(None, description="Short snippet for UI/debug (not necessarily the full chunk).")
    anchor: Anchor = Field(..., description="Anchor for the candidate (copied from chunk).")

    # filters + provenance
    applied_filters: List[str] = Field(default_factory=list, description="Names of metadata filters applied.")
    dedupe_key: Optional[str] = Field(None, description="Key used for deduplication across candidates.")


class RetrievedEvidence(BaseModel):
    """
    Evidence bundle prepared for generation or synthesis.

    This is the “post-retrieval, post-processing” unit: it may represent a raw chunk, a neighbor-expanded window,
    a compressed excerpt, or a contiguous segment selection. Critically, it must preserve citations.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Stable id for this evidence instance.")
    query_id: UUID = Field(..., description="Refers to ResearchQuery.query_id.")

    # content prepared for the LLM
    evidence_text: str = Field(..., description="Text used as context (may be compressed or expanded).")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting this evidence text.")

    # lineage
    source_candidates: List[RetrievalCandidate] = Field(
        default_factory=list,
        description="Candidates from which this evidence was derived (for auditability).",
    )
    postprocess_steps: List[str] = Field(
        default_factory=list,
        description="Applied steps (e.g., 'expand_neighbors', 'compress_selective', 'rse_segment').",
    )
    compression_ratio: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="If compressed, proportion removed from original context."
    )


class ResearchQuery(BaseModel):
    """
    User/system query object for both QA and research synthesis.
    """

    query_id: UUID = Field(default_factory=uuid4, description="Stable id for this query.")
    session_id: Optional[str] = Field(None, description="Conversation/session identifier for multi-turn flows.")
    asked_at: datetime = Field(default_factory=datetime.utcnow, description="Time the query was issued.")

    text: str = Field(..., description="User-provided query text.")
    language: Optional[str] = Field(None, description="Language code of the query (e.g., 'en').")

    # domain constraints (critical for research, not generic chat)
    persona: Optional[PersonaType] = Field(None, description="Target persona for this query/synthesis.")
    topics: List[ResearchTopic] = Field(default_factory=list, description="Topics to focus retrieval on.")
    workflow_stage: Optional[WorkflowStage] = Field(None, description="Workflow stage focus if applicable.")
    region: Optional[str] = Field(None, description="Geographic constraint (e.g., 'US-CA').")
    time_horizon: Optional[str] = Field(None, description="Temporal scope (e.g., 'last 5 years', '2023 season').")

    # governance / usage context
    requester: Optional[str] = Field(None, description="User/team identifier for audit logs.")
    purpose: Optional[str] = Field(None, description="Declared purpose (research ticket id, project name, etc.).")


class ResearchIntent(BaseModel):
    """
    Classified intent for routing and planning.
    """

    intent_type: IntentType = Field(..., description="High-level intent category.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence.")
    rationale: Optional[str] = Field(
        None, description="Short rationale for debugging/observability (not shown to end users by default)."
    )

    # task-specific structured hints
    target_personas: List[PersonaType] = Field(default_factory=list, description="Personas to include/compare.")
    target_topics: List[ResearchTopic] = Field(default_factory=list, description="Topics to prioritize.")
    output_schema: Optional[str] = Field(
        None, description="Requested output schema/template (e.g., 'need_discovery_report_v1')."
    )


class ResearchPlan(BaseModel):
    """
    Execution plan for a query: transformations + retrieval policy + synthesis steps.

    This is the core orchestration artifact you log and evaluate.
    """

    plan_id: UUID = Field(default_factory=uuid4, description="Stable id for this plan.")
    query_id: UUID = Field(..., description="Refers to ResearchQuery.query_id.")
    intent: ResearchIntent = Field(..., description="Intent classification result.")

    # query optimization pipeline decisions
    rewritten_query: Optional[str] = Field(None, description="Final rewritten query string used for retrieval.")
    expansions: List[str] = Field(default_factory=list, description="Synonyms/keyword expansions applied.")
    typo_corrections: List[Tuple[str, str]] = Field(
        default_factory=list, description="List of (original, corrected) pairs."
    )
    hyde_document: Optional[str] = Field(None, description="HyDE hypothetical document text (if used).")
    decomposition: List[str] = Field(default_factory=list, description="Subqueries if decomposition was used.")

    # retrieval parameters
    retrieval_method: RetrievalMethod = Field(..., description="Primary retrieval method.")
    rerank_method: RerankMethod = Field(RerankMethod.none, description="Reranking method.")
    top_k: int = Field(10, ge=1, description="Number of candidates to retrieve before post-processing.")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters (source_type, persona, date range, region, etc.).",
    )
    postprocess: List[str] = Field(
        default_factory=list,
        description="Post-retrieval steps (neighbors/compress/segment selection).",
    )

    # observability
    trace_id: Optional[str] = Field(None, description="Trace identifier for end-to-end observability.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the plan was created.")


class NeedInsight(BaseModel):
    """
    A single extracted need/pain point insight grounded in evidence.
    """

    insight_id: UUID = Field(default_factory=uuid4, description="Stable id for this insight.")
    persona: PersonaType = Field(..., description="Which persona this insight applies to.")
    topic: ResearchTopic = Field(..., description="Primary topic for this insight.")
    workflow_stage: Optional[WorkflowStage] = Field(None, description="Relevant workflow stage, if applicable.")

    statement: str = Field(..., description="Crisp statement of the need/pain point.")
    pain_severity_1_5: int = Field(..., ge=1, le=5, description="Severity rating for prioritization.")
    frequency_1_5: Optional[int] = Field(None, ge=1, le=5, description="How often it occurs, if known.")
    unmet_need: bool = Field(..., description="Whether the need is currently unmet.")
    current_workaround: Optional[str] = Field(None, description="What beekeepers do today to cope.")
    product_signal: Optional[str] = Field(
        None, description="Signal of product opportunity (requests, willingness to pay, hacks, tool mentions)."
    )

    # evidence and explainability
    evidence: List[RetrievedEvidence] = Field(default_factory=list, description="Supporting evidence bundles.")
    citations: List[Citation] = Field(default_factory=list, description="Flattened citations for the insight.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in this insight.")


class PersonaInsight(BaseModel):
    """
    A persona-level synthesis (needs, constraints, goals) grounded in evidence.
    """

    persona: PersonaType = Field(..., description="Persona synthesized.")
    summary: str = Field(..., description="High-level persona summary relevant to beekeeping operations.")
    key_needs: List[NeedInsight] = Field(default_factory=list, description="Top needs/pain points for persona.")
    constraints: List[str] = Field(default_factory=list, description="Constraints (budget, labor, compliance, etc.).")
    success_metrics: List[str] = Field(default_factory=list, description="What success looks like for this persona.")
    evidence_citations: List[Citation] = Field(default_factory=list, description="Citations supporting the synthesis.")


class WorkflowFriction(BaseModel):
    """
    A workflow-specific friction point (where and why the workflow breaks down).
    """

    friction_id: UUID = Field(default_factory=uuid4, description="Stable id for this friction.")
    workflow_stage: WorkflowStage = Field(..., description="Stage where friction occurs.")
    persona: Optional[PersonaType] = Field(None, description="Persona most affected (if applicable).")
    topic: Optional[ResearchTopic] = Field(None, description="Topic area (e.g., varroa_management).")

    description: str = Field(..., description="What goes wrong / what is difficult.")
    root_causes: List[str] = Field(default_factory=list, description="Likely root causes.")
    consequences: List[str] = Field(default_factory=list, description="Operational consequences (losses, time, risk).")
    pain_severity_1_5: int = Field(..., ge=1, le=5, description="Severity for prioritization.")
    evidence: List[RetrievedEvidence] = Field(default_factory=list, description="Supporting evidence bundles.")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting the friction.")


class ProductOpportunity(BaseModel):
    """
    A product opportunity hypothesis grounded in needs and evidence.
    """

    opportunity_id: UUID = Field(default_factory=uuid4, description="Stable id for this opportunity.")
    title: str = Field(..., description="Short name for the opportunity.")
    target_personas: List[PersonaType] = Field(default_factory=list, description="Personas targeted.")
    topics: List[ResearchTopic] = Field(default_factory=list, description="Topics related to the opportunity.")
    workflow_stages: List[WorkflowStage] = Field(default_factory=list, description="Workflow stages addressed.")

    problem_statement: str = Field(..., description="What problem this opportunity addresses.")
    proposed_solution: str = Field(..., description="High-level proposed solution concept.")
    differentiation: Optional[str] = Field(None, description="Why this is better than current alternatives.")
    adoption_barriers: List[str] = Field(default_factory=list, description="Barriers (cost, regulation, trust).")
    evidence_strength_1_5: int = Field(..., ge=1, le=5, description="How strong the evidence is.")

    supporting_needs: List[NeedInsight] = Field(default_factory=list, description="Needs that motivate the opportunity.")
    supporting_frictions: List[WorkflowFriction] = Field(default_factory=list, description="Workflow frictions addressed.")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting the opportunity.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence this is a real opportunity.")


class CriticReview(BaseModel):
    """
    A structured critique of an answer/report for grounding and usefulness.
    """

    review_id: UUID = Field(default_factory=uuid4, description="Stable id for this review.")
    reviewer: Literal["system", "llm", "human"] = Field(..., description="Who produced this critique.")
    reviewed_at: datetime = Field(default_factory=datetime.utcnow, description="When review occurred.")

    grounding_score_0_1: float = Field(..., ge=0.0, le=1.0, description="How well content is supported by evidence.")
    citation_coverage_0_1: float = Field(..., ge=0.0, le=1.0, description="Fraction of key claims covered by citations.")
    hallucination_risk_0_1: float = Field(..., ge=0.0, le=1.0, description="Estimated hallucination risk.")
    missing_evidence: List[str] = Field(default_factory=list, description="What evidence is missing.")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions found vs evidence.")
    suggested_followups: List[str] = Field(default_factory=list, description="Follow-up questions to reduce uncertainty.")


class MultiTurnContextState(BaseModel):
    """
    Conversation state for multi-turn grounded QA and iterative research sessions.
    """

    session_id: str = Field(..., description="Conversation/session identifier.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time.")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time.")

    # conversation turns
    turns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw turn log (role, text, timestamps). Keep minimal or store references externally.",
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Persisted constraints (persona, region, timeframe, topics) inferred across turns.",
    )
    pinned_citations: List[Citation] = Field(
        default_factory=list, description="Evidence the user/system decided to keep across turns."
    )
    pinned_evidence_ids: List[UUID] = Field(
        default_factory=list, description="Evidence ids persisted across turns (references)."
    )
    entity_memory: Dict[str, Any] = Field(
        default_factory=dict,
        description="Lightweight entity memory (apiary sizes, treatment names, products, orgs).",
    )


class FinalAnswer(BaseModel):
    """
    Output of conversational grounded Q&A.
    """

    answer_id: UUID = Field(default_factory=uuid4, description="Stable id for this answer.")
    query: ResearchQuery = Field(..., description="Original query object.")
    plan: ResearchPlan = Field(..., description="Plan used to produce the answer.")

    answer: str = Field(..., description="Final grounded answer text.")
    citations: List[Citation] = Field(default_factory=list, description="Citations referenced in the answer.")
    evidence: List[RetrievedEvidence] = Field(
        default_factory=list, description="Evidence bundles used to generate the answer."
    )

    # generation metadata
    model: Optional[str] = Field(None, description="LLM model identifier used for generation.")
    prompt_id: Optional[str] = Field(None, description="Prompt template/version used.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Answer creation time.")

    # quality + safety
    critic_review: Optional[CriticReview] = Field(None, description="Grounding/usefulness critique.")


class FinalResearchReport(BaseModel):
    """
    Output of structured research synthesis for need discovery.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Stable id for the report.")
    query: ResearchQuery = Field(..., description="Research request that generated the report.")
    plan: ResearchPlan = Field(..., description="Plan used to generate the report.")

    executive_summary: str = Field(..., description="High-level summary of findings.")
    persona_insights: List[PersonaInsight] = Field(default_factory=list, description="Persona-level synthesis.")
    workflow_frictions: List[WorkflowFriction] = Field(default_factory=list, description="Workflow friction points.")
    needs: List[NeedInsight] = Field(default_factory=list, description="Extracted needs/pain points.")
    opportunities: List[ProductOpportunity] = Field(default_factory=list, description="Opportunity hypotheses.")

    evidence_map: Dict[str, List[Citation]] = Field(
        default_factory=dict,
        description="Mapping from section/claim ids to citations (for explainability).",
    )
    citations: List[Citation] = Field(default_factory=list, description="All citations referenced in the report.")
    gaps_and_unknowns: List[str] = Field(default_factory=list, description="Known gaps/uncertainties.")

    # generation + review
    model: Optional[str] = Field(None, description="LLM model identifier used for synthesis.")
    prompt_id: Optional[str] = Field(None, description="Prompt template/version used.")
    critic_review: Optional[CriticReview] = Field(None, description="Grounding/usefulness critique.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Report creation time.")


class RetrievalEvaluationRecord(BaseModel):
    """
    A single retrieval evaluation record for observability and regression testing.

    Supports offline evaluation with gold answers/passages and standard IR metrics.
    """

    record_id: UUID = Field(default_factory=uuid4, description="Stable id for this evaluation record.")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow, description="When evaluation ran.")

    query_id: UUID = Field(..., description="ResearchQuery.query_id being evaluated.")
    dataset_id: Optional[str] = Field(None, description="Evaluation dataset identifier/version.")
    split: Optional[Literal["train", "val", "test"]] = Field(None, description="Dataset split.")

    retrieval_method: RetrievalMethod = Field(..., description="Retrieval method evaluated.")
    rerank_method: RerankMethod = Field(RerankMethod.none, description="Reranking method evaluated.")
    top_k: int = Field(..., ge=1, description="K used for metrics.")

    # gold labels (optional, depending on dataset)
    gold_chunk_ids: List[UUID] = Field(default_factory=list, description="Gold relevant chunk ids, if available.")
    gold_document_ids: List[UUID] = Field(
        default_factory=list, description="Gold relevant document ids, if available."
    )

    # retrieved results
    retrieved_chunk_ids: List[UUID] = Field(default_factory=list, description="Retrieved chunk ids (ranked).")
    candidates: List[RetrievalCandidate] = Field(
        default_factory=list, description="Full candidate objects for debugging."
    )

    # IR metrics
    precision_at_k: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision@K.")
    recall_at_k: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall@K.")
    mrr: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mean Reciprocal Rank for this query.")
    ndcg_at_k: Optional[float] = Field(None, ge=0.0, le=1.0, description="NDCG@K.")

    # grounding checks for generation (optional)
    citation_coverage_0_1: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="If generation happened, citation coverage score."
    )
    notes: Optional[str] = Field(None, description="Free-form notes about anomalies or failures.")
    trace_id: Optional[str] = Field(None, description="Trace id linking to pipeline logs/spans.")

