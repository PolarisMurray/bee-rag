"""
Production-grade query processing orchestrator.

This module composes:
1) intent classification
2) follow-up resolution
3) typo correction
4) synonym expansion
5) query rewriting
6) HyDE generation

It returns a structured `QueryProcessingResult` consumable by planner and retriever layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from beekeeper_intel.models import MultiTurnContextState

from .expand import default_synonyms, expand_query_keywords
from .followups import resolve_followup
from .hyde import DeterministicHyDEGenerator, HyDEGenerator
from .intent import IntentClassifier, RuleBasedIntentClassifier
from .normalize import TypoCorrectionRule, normalize_whitespace, apply_typo_corrections, unique_preserve_order
from .rewrite import DeterministicQueryRewriter, QueryRewriter, RewriteContext
from .types import HyDEConfig, QueryProcessingConfig, QueryProcessingResult, HyDEResult, SupportedIntent


def default_typo_rules() -> list[TypoCorrectionRule]:
    """
    Curated typo/variant rules.

    These are intentionally conservative. Add entries based on real logs.
    """

    return [
        TypoCorrectionRule("varoa", "varroa"),
        TypoCorrectionRule("varroah", "varroa"),
        TypoCorrectionRule("oxalic", "oxalic acid"),
        TypoCorrectionRule("formic pro", "formic pro"),
        TypoCorrectionRule("alchohol wash", "alcohol wash"),
        TypoCorrectionRule("alcohol washing", "alcohol wash"),
    ]


@dataclass
class QueryProcessorDeps:
    """
    Dependency bundle for `QueryProcessor`.

    In production you can inject learned/LLM components here.
    """

    intent_classifier: IntentClassifier = RuleBasedIntentClassifier()
    rewriter: QueryRewriter = DeterministicQueryRewriter()
    hyde_generator: HyDEGenerator = DeterministicHyDEGenerator()


class QueryProcessor:
    """High-level query processing entrypoint."""

    def __init__(
        self,
        *,
        cfg: Optional[QueryProcessingConfig] = None,
        hyde_cfg: Optional[HyDEConfig] = None,
        deps: Optional[QueryProcessorDeps] = None,
    ) -> None:
        self.cfg = cfg or QueryProcessingConfig()
        self.hyde_cfg = hyde_cfg or HyDEConfig()
        self.deps = deps or QueryProcessorDeps()

        self._synonyms = default_synonyms()
        self._typo_rules = default_typo_rules()

    def process(
        self,
        query: str,
        *,
        context_state: Optional[MultiTurnContextState] = None,
        prior_user_utterance: Optional[str] = None,
    ) -> QueryProcessingResult:
        """
        Process a user query into a retrieval-ready plan artifact.
        """

        trace_steps: list[str] = []

        original = normalize_whitespace(query)
        if not original:
            # produce a minimal, safe output for empty inputs
            out = QueryProcessingResult(
                intent=SupportedIntent.problem_discovery,
                intent_confidence=0.0,
                intent_signals=["empty_query"],
                original_query="",
                resolved_query="",
                rewritten_query="",
                hyde=HyDEResult(used=False, hypothetical_document=None, generator="none"),
            )
            out.trace.steps = ["normalize_empty"]
            return out

        # 1) follow-up resolution
        trace_steps.append("followup_resolution")
        follow = resolve_followup(original, context_state=context_state, prior_user_utterance=prior_user_utterance)
        resolved = follow.resolved_query

        # 2) intent classification (uses prior utterance for follow-up detection)
        trace_steps.append("intent_classification")
        pred = self.deps.intent_classifier.predict(resolved, prior_user_utterance=prior_user_utterance)

        # 3) typo correction
        corrected = resolved
        corrections = []
        if self.cfg.enable_typo_correction:
            trace_steps.append("typo_correction")
            corrected, corrections = apply_typo_corrections(corrected, self._typo_rules)

        # 4) synonym expansion
        expansions = []
        if self.cfg.enable_synonym_expansion:
            trace_steps.append("synonym_expansion")
            expansions = expand_query_keywords(corrected, self._synonyms, self.cfg.max_expansions)

        # 5) rewrite for retrieval
        trace_steps.append("rewrite")
        rw_ctx = RewriteContext(persona=pred.persona, topics=pred.topics or [], workflow_stage=pred.workflow_stage)
        rewritten = self.deps.rewriter.rewrite(corrected, rw_ctx)

        # 6) HyDE (policy-controlled)
        trace_steps.append("hyde")
        hyde: HyDEResult
        if self.cfg.enable_hyde:
            hyde = self.deps.hyde_generator.generate(rewritten, cfg=self.hyde_cfg)
        else:
            hyde = HyDEResult(used=False, hypothetical_document=None, generator="disabled")

        # Subqueries (basic, deterministic): for evidence synthesis, also query workarounds / constraints
        subqueries: list[str] = []
        synthesis_like = pred.intent in (
            SupportedIntent.evidence_synthesis,
            SupportedIntent.problem_discovery,
            SupportedIntent.opportunity_framing,
        )
        # If we classified as follow-up but user is explicitly asking for evidence/unmet need, treat as synthesis-like
        if (not synthesis_like) and ("evidence" in rewritten.lower() or "unmet need" in rewritten.lower()):
            synthesis_like = True

        if synthesis_like:
            trace_steps.append("subquery_generation")
            base = rewritten
            subqueries = unique_preserve_order(
                [
                    base,
                    f"{base} current workaround",
                    f"{base} pain points",
                    f"{base} unmet need evidence",
                ]
            )[: self.cfg.max_decomposition]

        result = QueryProcessingResult(
            intent=pred.intent,
            intent_confidence=pred.confidence,
            intent_signals=pred.signals,
            original_query=original,
            resolved_query=resolved,
            rewritten_query=rewritten,
            subqueries=subqueries,
            typo_corrections=corrections,
            expansions=expansions,
            persona=pred.persona,
            topics=pred.topics or [],
            workflow_stage=pred.workflow_stage,
            hyde=hyde,
        )

        result.trace.steps = trace_steps
        result.trace.notes.update(
            {
                "followup_used": follow.used,
                "followup_notes": follow.notes,
                "resolved_from_prior": bool(prior_user_utterance),
            }
        )
        return result

