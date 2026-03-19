"""
Production-oriented orchestration layer for Beekeeper Research Intelligence Platform.

Combines:
- ingestion pipeline (pluggable)
- multi-turn context manager
- query processing
- planner
- retriever
- extractor
- critic
- citation renderer
- final answer/report synthesizer

Pipeline goals:
1) conversational grounded Q&A
2) structured research synthesis for beekeeper need discovery
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from beekeeper_intel.agents.extraction_agent import ExtractedNeedInsight, ExtractionAgent
from beekeeper_intel.citations.integration import build_answer_bundle, build_report_bundle
from beekeeper_intel.citations.models import ExplainableAnswerBundle, ExplainableReportBundle
from beekeeper_intel.memory.context_state import (
    ConversationTurn,
    EntityTracker,
    RetrievalContextHints,
    TurnRole,
    update_context_state,
)
from beekeeper_intel.memory.followup_rewriter import FollowupRewriter
from beekeeper_intel.models import (
    Citation,
    FinalAnswer,
    FinalResearchReport,
    IntentType,
    MultiTurnContextState,
    NeedInsight,
    PersonaType,
    ResearchIntent,
    ResearchPlan,
    ResearchQuery,
    ResearchTopic,
    RetrievedEvidence,
    RetrievalMethod,
    RerankMethod,
    WorkflowStage,
)
from beekeeper_intel.query_processing.processor import QueryProcessor
from beekeeper_intel.query_processing.types import QueryProcessingResult, SupportedIntent


logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Pipeline execution mode."""

    auto = "auto"
    conversational_qa = "conversational_qa"
    research_synthesis = "research_synthesis"


class OrchestratorConfig(BaseModel):
    """Global orchestration config."""

    max_iterations: int = Field(2, ge=1, description="Maximum retrieval/critic refinement iterations.")
    min_evidence_items: int = Field(1, ge=0, description="Minimum evidence items before synthesis.")
    enable_ingestion_precheck: bool = Field(False, description="Whether to call ingestion precheck hook.")
    strict_failure_mode: bool = Field(False, description="If true, raise on failures instead of graceful fallback.")


class OrchestrationTraceEvent(BaseModel):
    """One trace event for observability."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step: str
    message: str = ""
    data: Dict[str, str] = Field(default_factory=dict)


class OrchestratorResult(BaseModel):
    """
    Unified orchestration output.

    Exactly one of `answer_bundle` or `report_bundle` is expected for successful runs.
    """

    run_id: UUID = Field(default_factory=uuid4)
    mode: PipelineMode
    success: bool
    answer_bundle: Optional[ExplainableAnswerBundle] = None
    report_bundle: Optional[ExplainableReportBundle] = None
    context_state: Optional[MultiTurnContextState] = None
    trace: List[OrchestrationTraceEvent] = Field(default_factory=list)
    error: Optional[str] = None


# -------------------------
# Pluggable interfaces
# -------------------------


class IngestionPipeline(Protocol):
    """Optional ingestion hook interface."""

    def ensure_ready(self) -> None:
        """Ensure indices/artifacts are ready before querying."""


class Planner(Protocol):
    """Planner interface."""

    def create_plan(
        self,
        query: ResearchQuery,
        qp: QueryProcessingResult,
        hints: RetrievalContextHints,
    ) -> ResearchPlan:
        ...


class Retriever(Protocol):
    """Retriever interface."""

    def retrieve(self, plan: ResearchPlan, qp: QueryProcessingResult) -> List[RetrievedEvidence]:
        ...


class CriticDecision(BaseModel):
    """Critic decision for iterative retrieval control."""

    evidence_sufficient: bool
    retrieval_gap: Optional[str] = None
    confidence_adjustment: float = 0.0
    recommended_next_action: str = "proceed"
    weaknesses: List[str] = Field(default_factory=list)


class Critic(Protocol):
    """Critic interface."""

    def review(
        self,
        *,
        query: ResearchQuery,
        plan: ResearchPlan,
        evidence: List[RetrievedEvidence],
        extracted: List[ExtractedNeedInsight],
        iteration: int,
    ) -> CriticDecision:
        ...


# -------------------------
# Default implementations
# -------------------------


class DefaultPlanner:
    """Simple planner mapping query processing results into a ResearchPlan."""

    def create_plan(self, query: ResearchQuery, qp: QueryProcessingResult, hints: RetrievalContextHints) -> ResearchPlan:
        intent = _map_intent(qp.intent)
        retrieval_method = RetrievalMethod.hybrid_fusion
        rerank_method = RerankMethod.cross_encoder
        filters = dict(hints.hard_filters)
        if query.region:
            filters["region"] = query.region
        if query.time_horizon:
            filters["time_horizon"] = query.time_horizon

        return ResearchPlan(
            query_id=query.query_id,
            intent=ResearchIntent(intent_type=intent, confidence=qp.intent_confidence),
            rewritten_query=qp.rewritten_query,
            expansions=qp.expansions,
            typo_corrections=qp.typo_corrections,
            hyde_document=qp.hyde.hypothetical_document if qp.hyde.used else None,
            decomposition=qp.subqueries,
            retrieval_method=retrieval_method,
            rerank_method=rerank_method,
            filters=filters,
            postprocess=["expand_neighbors", "dedupe", "rerank"],
        )


class DefaultCritic:
    """
    Deterministic critic for reliability gating.

    This is intentionally conservative and transparent.
    """

    def review(
        self,
        *,
        query: ResearchQuery,
        plan: ResearchPlan,
        evidence: List[RetrievedEvidence],
        extracted: List[ExtractedNeedInsight],
        iteration: int,
    ) -> CriticDecision:
        if not evidence:
            return CriticDecision(
                evidence_sufficient=False,
                retrieval_gap="no_evidence",
                confidence_adjustment=-0.25,
                recommended_next_action="retrieve_more",
                weaknesses=["no_evidence_returned"],
            )

        source_titles = set()
        for ev in evidence:
            for c in ev.citations:
                source_titles.add(c.source_title)

        multi_source = len(source_titles) >= 2
        if len(extracted) == 0:
            return CriticDecision(
                evidence_sufficient=multi_source and len(evidence) >= 2,
                retrieval_gap="no_extractable_need_signals",
                confidence_adjustment=-0.10,
                recommended_next_action="retrieve_more" if iteration == 1 else "proceed_with_caution",
                weaknesses=["no_structured_insights_extracted"],
            )

        # Penalize when only one source and single evidence item (anecdotal risk)
        anecdotal = len(evidence) == 1 and not multi_source
        if anecdotal:
            return CriticDecision(
                evidence_sufficient=False,
                retrieval_gap="anecdotal_only",
                confidence_adjustment=-0.15,
                recommended_next_action="retrieve_more",
                weaknesses=["single_source_anecdotal_evidence"],
            )

        return CriticDecision(
            evidence_sufficient=True,
            confidence_adjustment=0.0,
            recommended_next_action="proceed",
            weaknesses=[],
        )


# -------------------------
# Orchestrator
# -------------------------


class PlatformOrchestrator:
    """
    End-to-end orchestrator for QA + research synthesis.

    Interfaces are explicit and pluggable to avoid monoliths.
    """

    def __init__(
        self,
        *,
        cfg: Optional[OrchestratorConfig] = None,
        ingestion: Optional[IngestionPipeline] = None,
        query_processor: Optional[QueryProcessor] = None,
        planner: Optional[Planner] = None,
        retriever: Optional[Retriever] = None,
        extractor: Optional[ExtractionAgent] = None,
        critic: Optional[Critic] = None,
        followup_rewriter: Optional[FollowupRewriter] = None,
        entity_tracker: Optional[EntityTracker] = None,
    ) -> None:
        self.cfg = cfg or OrchestratorConfig()
        self.ingestion = ingestion
        self.query_processor = query_processor or QueryProcessor()
        self.planner = planner or DefaultPlanner()
        self.retriever = retriever
        self.extractor = extractor or ExtractionAgent()
        self.critic = critic or DefaultCritic()
        self.followup_rewriter = followup_rewriter or FollowupRewriter()
        self.entity_tracker = entity_tracker or EntityTracker()

    def run(
        self,
        *,
        query_text: str,
        context_state: Optional[MultiTurnContextState] = None,
        mode: PipelineMode = PipelineMode.auto,
        user_id: Optional[str] = None,
    ) -> OrchestratorResult:
        trace: List[OrchestrationTraceEvent] = []
        state = context_state or MultiTurnContextState(session_id=str(uuid4()))
        run_mode = mode

        try:
            if self.cfg.enable_ingestion_precheck and self.ingestion is not None:
                self._trace(trace, "ingestion_precheck", "Ensuring ingestion artifacts are ready.")
                self.ingestion.ensure_ready()

            # maintain conversation state with incoming user query
            self._trace(trace, "context_update", "Updating conversation state.")
            update_context_state(
                state,
                ConversationTurn(role=TurnRole.user, text=query_text),
                tracker=self.entity_tracker,
            )

            # explicit follow-up rewrite for retrieval quality
            self._trace(trace, "followup_rewrite", "Resolving follow-up ambiguity.")
            follow = self.followup_rewriter.rewrite(query_text, state)

            # query processing (intent, rewrite, expand, hyde)
            self._trace(trace, "query_processing", "Running query processing.")
            prior = follow.referent_text
            qp = self.query_processor.process(
                follow.rewritten_query,
                context_state=state,
                prior_user_utterance=prior,
            )

            # mode selection
            if run_mode == PipelineMode.auto:
                run_mode = _decide_mode_from_intent(qp.intent)
            self._trace(trace, "mode_selection", f"Selected mode: {run_mode.value}")

            # planning
            self._trace(trace, "planning", "Creating research plan.")
            query = ResearchQuery(
                text=query_text,
                session_id=state.session_id,
                requester=user_id,
                persona=qp.persona,
                topics=qp.topics,
                workflow_stage=qp.workflow_stage,
            )
            hints = follow.hints
            plan = self.planner.create_plan(query, qp, hints)

            if self.retriever is None:
                raise RuntimeError("No retriever configured for orchestration.")

            # iterative retrieve -> extract -> critic loop
            evidence: List[RetrievedEvidence] = []
            extracted: List[ExtractedNeedInsight] = []
            critic_decision: Optional[CriticDecision] = None
            for iteration in range(1, self.cfg.max_iterations + 1):
                self._trace(trace, "retrieve", f"Retrieving evidence (iteration {iteration}).")
                evidence = self.retriever.retrieve(plan, qp)
                self._trace(trace, "extract", f"Extracting structured insights from evidence ({len(evidence)} items).")
                extracted = []
                for ev in evidence:
                    extracted.extend(self.extractor.extract_from_evidence(ev))

                self._trace(trace, "critic", "Validating evidence sufficiency.")
                critic_decision = self.critic.review(
                    query=query,
                    plan=plan,
                    evidence=evidence,
                    extracted=extracted,
                    iteration=iteration,
                )
                if len(evidence) < self.cfg.min_evidence_items:
                    critic_decision.evidence_sufficient = False
                    critic_decision.retrieval_gap = critic_decision.retrieval_gap or "below_min_evidence_threshold"
                    if "below_min_evidence_threshold" not in critic_decision.weaknesses:
                        critic_decision.weaknesses.append("below_min_evidence_threshold")
                if critic_decision.evidence_sufficient:
                    break
                if iteration < self.cfg.max_iterations:
                    # lightweight iterative adjustment: broaden plan if critic flags gaps
                    plan.top_k = min(50, plan.top_k + 5)
                    plan.postprocess = list(dict.fromkeys(plan.postprocess + ["broaden_retrieval"]))

            # synthesis
            if run_mode == PipelineMode.conversational_qa:
                self._trace(trace, "synthesis_qa", "Synthesizing final conversational answer.")
                final_answer = self._synthesize_answer(query, plan, evidence, extracted, critic_decision)
                answer_bundle = build_answer_bundle(final_answer)
                self._trace(trace, "done", "Completed conversational QA pipeline.")
                return OrchestratorResult(
                    mode=run_mode,
                    success=True,
                    answer_bundle=answer_bundle,
                    context_state=state,
                    trace=trace,
                )

            self._trace(trace, "synthesis_report", "Synthesizing final structured report.")
            report = self._synthesize_report(query, plan, evidence, extracted, critic_decision)
            section_to_evidence = {
                "needs": evidence,
                "workflow_frictions": evidence,
                "opportunities": evidence,
            }
            report_bundle = build_report_bundle(report, section_to_evidence=section_to_evidence)
            self._trace(trace, "done", "Completed research synthesis pipeline.")
            return OrchestratorResult(
                mode=run_mode,
                success=True,
                report_bundle=report_bundle,
                context_state=state,
                trace=trace,
            )

        except Exception as exc:
            self._trace(trace, "error", f"{type(exc).__name__}: {exc}")
            logger.exception("Orchestration failure")
            if self.cfg.strict_failure_mode:
                raise
            return OrchestratorResult(
                mode=run_mode,
                success=False,
                context_state=state,
                trace=trace,
                error=str(exc),
            )

    def _synthesize_answer(
        self,
        query: ResearchQuery,
        plan: ResearchPlan,
        evidence: List[RetrievedEvidence],
        extracted: List[ExtractedNeedInsight],
        critic: Optional[CriticDecision],
    ) -> FinalAnswer:
        if extracted:
            top = extracted[:3]
            lines = [f"- {x.problem}" for x in top]
            ans = "Grounded findings from retrieved evidence:\n" + "\n".join(lines)
        elif evidence:
            ans = "Grounded evidence found, but structured need signals are limited. Review sources for details."
        else:
            ans = "Insufficient grounded evidence found for this query."
        if critic and critic.confidence_adjustment < 0:
            ans += "\n\nNote: Confidence reduced due to evidence limitations."
        citations = _collect_citations(evidence)
        return FinalAnswer(
            query=query,
            plan=plan,
            answer=ans,
            citations=citations,
            evidence=evidence,
        )

    def _synthesize_report(
        self,
        query: ResearchQuery,
        plan: ResearchPlan,
        evidence: List[RetrievedEvidence],
        extracted: List[ExtractedNeedInsight],
        critic: Optional[CriticDecision],
    ) -> FinalResearchReport:
        clusters = self.extractor.aggregate(extracted)
        needs: List[NeedInsight] = []
        for c in clusters[:10]:
            persona = c.persona or PersonaType.unknown
            topic = c.topic or ResearchTopic.other
            needs.append(
                NeedInsight(
                    persona=persona,
                    topic=topic,
                    workflow_stage=c.workflow_stage,
                    statement=c.canonical_problem,
                    pain_severity_1_5=c.merged_pain_severity_1_5,
                    unmet_need=True,
                    current_workaround=(c.merged_workarounds[0] if c.merged_workarounds else None),
                    product_signal=(c.merged_product_signals[0] if c.merged_product_signals else None),
                    confidence=max(0.1, min(0.95, 0.7 + (0.1 if c.is_multi_source_signal else -0.1))),
                )
            )

        summary = (
            f"Extracted {len(needs)} prioritized need insights from {len(evidence)} evidence bundles. "
            f"Top themes include workflow friction, labor burden, and unmet product support."
        )
        if critic and critic.retrieval_gap:
            summary += f" Retrieval gap noted: {critic.retrieval_gap}."

        return FinalResearchReport(
            query=query,
            plan=plan,
            executive_summary=summary,
            needs=needs,
            citations=_collect_citations(evidence),
            gaps_and_unknowns=([critic.retrieval_gap] if critic and critic.retrieval_gap else []),
        )

    @staticmethod
    def _trace(trace: List[OrchestrationTraceEvent], step: str, message: str, **data: str) -> None:
        trace.append(OrchestrationTraceEvent(step=step, message=message, data=data))


def _map_intent(intent: SupportedIntent) -> IntentType:
    mapping = {
        SupportedIntent.problem_discovery: IntentType.need_discovery,
        SupportedIntent.persona_comparison: IntentType.persona_comparison,
        SupportedIntent.workflow_analysis: IntentType.workflow_analysis,
        SupportedIntent.evidence_synthesis: IntentType.evidence_synthesis,
        SupportedIntent.opportunity_framing: IntentType.opportunity_generation,
        SupportedIntent.follow_up_clarification: IntentType.conversational_qa,
        SupportedIntent.document_lookup: IntentType.retrieval_debug,
    }
    return mapping.get(intent, IntentType.conversational_qa)


def _decide_mode_from_intent(intent: SupportedIntent) -> PipelineMode:
    if intent in {
        SupportedIntent.problem_discovery,
        SupportedIntent.persona_comparison,
        SupportedIntent.workflow_analysis,
        SupportedIntent.evidence_synthesis,
        SupportedIntent.opportunity_framing,
    }:
        return PipelineMode.research_synthesis
    return PipelineMode.conversational_qa


def _collect_citations(evidence: List[RetrievedEvidence]) -> List[Citation]:
    seen = set()
    out: List[Citation] = []
    for e in evidence:
        for c in e.citations:
            key = (
                c.document_id,
                c.source_title,
                c.anchor.page_number,
                c.anchor.slide_number,
                c.anchor.timestamp_ms,
                tuple(c.anchor.section_path),
                c.quote,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out

