"""
Follow-up rewriter for retrieval-quality standalone queries.

This module resolves ambiguous references ("this", "that", "it", "they") by leveraging:
- recent user turns
- tracked entities/constraints in MultiTurnContextState
- domain-aware pattern heuristics
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from beekeeper_intel.models import MultiTurnContextState
from beekeeper_intel.memory.context_state import EntityTracker, RetrievalContextHints, build_retrieval_hints


class FollowupRewriteResult(BaseModel):
    """Structured rewrite result for planner/retriever consumption."""

    original_query: str = Field(..., description="Original user follow-up query.")
    rewritten_query: str = Field(..., description="Standalone explicit query.")
    used_context: bool = Field(..., description="Whether prior context was used.")
    referent_text: Optional[str] = Field(None, description="Resolved referent phrase from prior turns.")
    ambiguity_score_0_1: float = Field(..., ge=0.0, le=1.0, description="Estimated ambiguity in original query.")
    hints: RetrievalContextHints = Field(..., description="Planner/retriever integration hints.")


class FollowupRewriter:
    """
    Rewrites short/ambiguous follow-up questions into explicit standalone queries.

    Target: retrieval accuracy improvement rather than conversational style polishing.
    """

    _pronoun_re = re.compile(r"\b(this|that|it|they|those|these|them)\b", re.IGNORECASE)
    _compare_re = re.compile(r"\b(different|difference|compare|vs\.?|versus)\b", re.IGNORECASE)
    _how_re = re.compile(r"\b(how|what about|is this|is that)\b", re.IGNORECASE)

    def __init__(self, tracker: Optional[EntityTracker] = None) -> None:
        self.tracker = tracker or EntityTracker()

    def rewrite(self, query: str, state: MultiTurnContextState) -> FollowupRewriteResult:
        q = _clean(query)
        ambiguity = _ambiguity_score(q)
        referent = self._resolve_referent(state)
        active_entities = self.tracker.extract_entities(q)
        hints = build_retrieval_hints(state, active_query_entities=active_entities)

        used_context = ambiguity > 0.25 or len(q.split()) <= 6
        rewritten = q

        if used_context:
            if referent:
                rewritten = _replace_pronouns_with_referent(q, referent)
            # apply domain shape upgrades for common follow-up forms
            rewritten = self._shape_followup(rewritten, q, referent, hints)

            # prepend retrieval prefix if not already present
            if hints.query_prefix:
                rewritten = f"{hints.query_prefix} {rewritten}".strip()

        rewritten = _clean(rewritten)
        return FollowupRewriteResult(
            original_query=q,
            rewritten_query=rewritten,
            used_context=used_context,
            referent_text=referent,
            ambiguity_score_0_1=ambiguity,
            hints=hints,
        )

    def _resolve_referent(self, state: MultiTurnContextState) -> Optional[str]:
        """
        Resolve best referent from recent user turns.
        """

        # prefer last user turn text
        for turn in reversed(state.turns):
            if turn.get("role") == "user":
                t = _clean(str(turn.get("text", "")))
                if t:
                    return t
        return None

    def _shape_followup(
        self,
        rewritten: str,
        original: str,
        referent: Optional[str],
        hints: RetrievalContextHints,
    ) -> str:
        """
        Apply domain-specific rewrite patterns for frequent follow-up forms.
        """

        low = original.lower()
        persona = hints.resolved_entities.get("persona")
        season = hints.resolved_entities.get("season")

        # "What about commercial ones?"
        if "what about" in low and persona:
            return f"How does {referent or 'the issue'} differ for {persona} beekeepers?"

        # "Is this still a problem in winter?"
        if "still a problem" in low and ("winter" in low or season == "winter"):
            return f"Is {referent or 'this issue'} still a problem in winter conditions for beekeepers?"

        # "How is that different?"
        if self._compare_re.search(low):
            return f"How is {referent or 'this'} different across beekeeper personas and operation scale?"

        # "How do they do that?"
        if "how do they do that" in low:
            return f"What is the step-by-step workflow for {referent or 'this process'}?"

        # generic pronoun follow-up
        if self._pronoun_re.search(low) or self._how_re.search(low):
            return f"{referent or ''} {rewritten}".strip()

        return rewritten


def _ambiguity_score(q: str) -> float:
    """
    Heuristic ambiguity score for follow-up detection.
    """

    if not q:
        return 1.0

    score = 0.0
    token_count = len(q.split())
    if token_count <= 6:
        score += 0.35
    if FollowupRewriter._pronoun_re.search(q):
        score += 0.45
    if FollowupRewriter._how_re.search(q.lower()):
        score += 0.2
    return min(1.0, score)


def _replace_pronouns_with_referent(query: str, referent: str) -> str:
    if not referent:
        return query
    return re.sub(r"\b(this|that|it|they|those|these|them)\b", referent, query, flags=re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

