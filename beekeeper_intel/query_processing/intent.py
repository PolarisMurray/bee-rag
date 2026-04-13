"""
Intent classification for beekeeper-domain research queries.

This implementation is production-oriented in two ways:
- deterministic & testable by default (rule-based)
- pluggable interface to swap in an ML/LLM classifier later
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from beekeeper_intel.models import PersonaType, ResearchTopic, WorkflowStage
from beekeeper_intel.query_processing.types import SupportedIntent


@dataclass(frozen=True)
class IntentPrediction:
    intent: SupportedIntent
    confidence: float
    signals: List[str]
    persona: Optional[PersonaType] = None
    topics: Optional[List[ResearchTopic]] = None
    workflow_stage: Optional[WorkflowStage] = None


class IntentClassifier:
    """Interface for intent classification."""

    def predict(
        self, query: str, *, prior_user_utterance: Optional[str] = None
    ) -> IntentPrediction:
        raise NotImplementedError


class RuleBasedIntentClassifier(IntentClassifier):
    """
    Deterministic rule-based classifier tuned for beekeeper research workflows.

    This is not meant to be perfect; it provides a robust baseline that is:
    - cheap
    - explainable (signals)
    - stable (good for tests/evals)
    """

    _re_doc_lookup = re.compile(
        r"\b(document|doc|report|paper|slide|slides|page|pages|figure|table|appendix|section)\b",
        re.IGNORECASE,
    )
    _re_followup = re.compile(r"\b(this|that|it|they|those|these|such)\b", re.IGNORECASE)
    _re_evidence = re.compile(r"\b(evidence|proof|data|stud(y|ies)|research|citation|source)\b", re.IGNORECASE)
    _re_workflow = re.compile(r"\b(process|workflow|steps|procedure|how do|how does)\b", re.IGNORECASE)
    _re_compare = re.compile(r"\b(different|compare|vs\.?|versus|contrast)\b", re.IGNORECASE)
    _re_opportunity = re.compile(
        r"\b(opportunity|product|solution|build|startup|market|feature|tool)\b", re.IGNORECASE
    )
    _re_problem = re.compile(r"\b(struggle|pain|pain point|problem|challenge|friction)\b", re.IGNORECASE)

    def predict(self, query: str, *, prior_user_utterance: Optional[str] = None) -> IntentPrediction:
        q = (query or "").strip()
        q_low = q.lower()
        signals: List[str] = []

        persona = _detect_persona(q_low)
        topics = _detect_topics(q_low)
        stage = _detect_workflow_stage(q_low)

        # Precompute problem signal so it can override workflow routing
        has_problem_signal = bool(self._re_problem.search(q)) or "unmet need" in q_low or "needs" in q_low

        # 1) document lookup
        if self._re_doc_lookup.search(q):
            signals.append("doc_lookup_keywords")
            return IntentPrediction(SupportedIntent.document_lookup, 0.85, signals, persona, topics, stage)

        # 2) explicit opportunity framing
        if self._re_opportunity.search(q):
            signals.append("opportunity_keywords")
            # if user also asks for evidence, this is often "opportunity_framing" not synthesis
            return IntentPrediction(SupportedIntent.opportunity_framing, 0.78, signals, persona, topics, stage)

        # 3) persona comparison
        if persona and self._re_compare.search(q):
            signals.append("persona_compare_keywords")
            return IntentPrediction(SupportedIntent.persona_comparison, 0.82, signals, persona, topics, stage)

        # 4) workflow analysis
        if (stage or self._re_workflow.search(q)) and not has_problem_signal:
            signals.append("workflow_keywords")
            # follow-up like "What does that process look like?" is often workflow_analysis but needs resolution
            if _looks_like_followup(q, prior_user_utterance):
                signals.append("followup_pronouns_or_short")
                return IntentPrediction(SupportedIntent.follow_up_clarification, 0.75, signals, persona, topics, stage)
            return IntentPrediction(SupportedIntent.workflow_analysis, 0.76, signals, persona, topics, stage)

        # 5) evidence synthesis
        if self._re_evidence.search(q):
            signals.append("evidence_keywords")
            # "Is there evidence that..." tends to be synthesis
            return IntentPrediction(SupportedIntent.evidence_synthesis, 0.8, signals, persona, topics, stage)

        # 6) problem discovery
        if has_problem_signal:
            signals.append("problem_keywords")
            return IntentPrediction(SupportedIntent.problem_discovery, 0.72, signals, persona, topics, stage)

        # 7) follow-up clarification if short/vague or pronoun-heavy and we have prior context
        if _looks_like_followup(q, prior_user_utterance):
            signals.append("followup_pronouns_or_short")
            return IntentPrediction(SupportedIntent.follow_up_clarification, 0.7, signals, persona, topics, stage)

        # Default: problem discovery (most common in need research)
        signals.append("default_problem_discovery")
        return IntentPrediction(SupportedIntent.problem_discovery, 0.55, signals, persona, topics, stage)


def _looks_like_followup(q: str, prior: Optional[str]) -> bool:
    if not prior:
        return False
    tokens = _simple_tokens(q)
    if len(tokens) <= 5:
        return True
    if RuleBasedIntentClassifier._re_followup.search(q):
        return True
    return False


def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text or "").strip().lower()) if t]


def _detect_persona(q_low: str) -> Optional[PersonaType]:
    if "hobbyist" in q_low:
        return PersonaType.hobbyist
    if "commercial" in q_low:
        return PersonaType.commercial
    if "queen breeder" in q_low or "queen-breeder" in q_low:
        return PersonaType.queen_breeder
    if "pollination" in q_low:
        return PersonaType.pollination_operator
    return None


def _detect_topics(q_low: str) -> List[ResearchTopic]:
    topics: List[ResearchTopic] = []
    if "varroa" in q_low:
        topics.append(ResearchTopic.varroa_management)
    if "feed" in q_low or "feeding" in q_low or "nutrition" in q_low:
        topics.append(ResearchTopic.nutrition_feeding)
    if "overwinter" in q_low or "winter" in q_low:
        topics.append(ResearchTopic.overwintering)
    if "queen" in q_low:
        topics.append(ResearchTopic.queen_health_reproduction)
    return topics


def _detect_workflow_stage(q_low: str) -> Optional[WorkflowStage]:
    # very lightweight mapping; can be replaced by learned classifier later
    if "monitor" in q_low or "monitoring" in q_low or "check" in q_low:
        return WorkflowStage.monitoring
    if "diagnos" in q_low or "test" in q_low or "measure" in q_low:
        return WorkflowStage.diagnosis
    if "treat" in q_low or "treatment" in q_low or "apply" in q_low:
        return WorkflowStage.treatment
    if "record" in q_low or "log" in q_low:
        return WorkflowStage.recordkeeping
    return None

