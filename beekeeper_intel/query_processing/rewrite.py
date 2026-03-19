"""
Query rewriting for retrieval.

This module focuses on:
- making vague questions more specific using detected domain constraints
- keeping output stable and explainable (no hidden LLM magic required)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from beekeeper_intel.models import PersonaType, ResearchTopic, WorkflowStage

from .normalize import normalize_whitespace


@dataclass(frozen=True)
class RewriteContext:
    persona: Optional[PersonaType] = None
    topics: Optional[List[ResearchTopic]] = None
    workflow_stage: Optional[WorkflowStage] = None


class QueryRewriter:
    """Interface for rewriting queries for retrieval."""

    def rewrite(self, query: str, ctx: RewriteContext) -> str:
        raise NotImplementedError


class DeterministicQueryRewriter(QueryRewriter):
    """
    Deterministic baseline rewriter.

    Strategy:
    - preserve original user phrasing
    - prepend explicit constraints (persona/topic/stage) as retrieval keywords
    - add domain framing ("beekeeper needs", "pain points", "workarounds", etc.) when appropriate
    """

    def rewrite(self, query: str, ctx: RewriteContext) -> str:
        q = normalize_whitespace(query)
        if not q:
            return ""

        prefixes: List[str] = []
        if ctx.persona:
            prefixes.append(f"{ctx.persona.value} beekeepers")
        if ctx.topics:
            # keep unique topic values
            topic_bits = [t.value.replace("_", " ") for t in ctx.topics]
            prefixes.extend(topic_bits)
        if ctx.workflow_stage:
            prefixes.append(f"{ctx.workflow_stage.value.replace('_', ' ')} workflow")

        # If user query is short or generic, add a research framing keyword
        if len(q.split()) <= 6:
            prefixes.append("pain points unmet needs workarounds")

        if prefixes:
            return normalize_whitespace(" ".join(prefixes) + " " + q)
        return q

