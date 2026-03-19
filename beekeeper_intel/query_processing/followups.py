"""
Follow-up query resolution for multi-turn conversations.

Goal: resolve vague pronoun-based follow-ups ("this", "that", "it") into a concrete query
that can be retrieved effectively.

Design:
- deterministic baseline that uses prior user utterance and persisted constraints
- pluggable later to add LLM-based coreference resolution if needed
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from beekeeper_intel.models import MultiTurnContextState

from .normalize import normalize_whitespace


_PRONOUN_RE = re.compile(r"\b(this|that|it|they|those|these|such)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FollowUpResolution:
    resolved_query: str
    used: bool
    notes: str


def resolve_followup(
    query: str,
    *,
    context_state: Optional[MultiTurnContextState] = None,
    prior_user_utterance: Optional[str] = None,
) -> FollowUpResolution:
    """
    Resolve a follow-up query using conversation context.

    Heuristics:
    - If the query is short or contains pronouns, append the prior utterance and constraints.
    - Prefer last user utterance as the referent.
    """

    q = normalize_whitespace(query)
    if not q:
        return FollowUpResolution(resolved_query="", used=False, notes="empty_query")

    prior = normalize_whitespace(prior_user_utterance or "")
    has_pronoun = bool(_PRONOUN_RE.search(q))
    is_short = len(q.split()) <= 6

    if not (has_pronoun or is_short):
        return FollowUpResolution(resolved_query=q, used=False, notes="not_followup")

    # Build constraint prefix from context_state if available
    constraint_bits = []
    if context_state and context_state.constraints:
        persona = context_state.constraints.get("persona")
        topic = context_state.constraints.get("topic") or context_state.constraints.get("topics")
        stage = context_state.constraints.get("workflow_stage")
        if persona:
            constraint_bits.append(str(persona))
        if topic:
            if isinstance(topic, list):
                constraint_bits.extend([str(t) for t in topic])
            else:
                constraint_bits.append(str(topic))
        if stage:
            constraint_bits.append(str(stage))

    # If we lack prior utterance, still return the query with constraints (better than nothing)
    if not prior:
        resolved = normalize_whitespace(" ".join(constraint_bits + [q]))
        return FollowUpResolution(resolved_query=resolved, used=True, notes="followup_no_prior_used_constraints")

    # Combine: prior utterance provides referent; q provides new question framing
    resolved = normalize_whitespace(" ".join(constraint_bits + [prior, q]))
    return FollowUpResolution(resolved_query=resolved, used=True, notes="followup_combined_with_prior")

