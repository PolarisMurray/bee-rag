"""
Context state management for multi-turn retrieval accuracy.

This module is retrieval-centric:
- tracks domain entities and constraints
- keeps explicit state for follow-up rewriting
- emits retrieval hints for planner/retriever layers
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from beekeeper_intel.models import MultiTurnContextState, PersonaType, ResearchTopic, WorkflowStage


class TurnRole(str, Enum):
    """Conversation role."""

    user = "user"
    assistant = "assistant"
    system = "system"


class ConversationTurn(BaseModel):
    """Normalized conversation turn for state updates."""

    role: TurnRole = Field(..., description="Role of turn speaker.")
    text: str = Field(..., description="Raw turn text.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Turn timestamp.")


class RetrievalContextHints(BaseModel):
    """
    Planner/retriever integration payload derived from state.

    This is intentionally concise and machine-friendly.
    """

    session_id: str
    resolved_entities: Dict[str, Any] = Field(default_factory=dict, description="Resolved entities for current query.")
    hard_filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters to apply in retrieval.")
    boosted_terms: List[str] = Field(default_factory=list, description="Terms to boost in sparse/hybrid retrieval.")
    query_prefix: str = Field("", description="Prefix string that can be prepended before retrieval rewriting.")


class EntityTracker:
    """
    Lightweight entity tracker for beekeeper-domain conversations.

    It extracts and maintains:
    - persona focus (hobbyist/commercial/etc.)
    - topic focus (varroa/overwintering/queen...)
    - workflow stage (monitoring/treatment/recordkeeping...)
    - temporal qualifiers (winter/season/year)
    """

    _season_terms = {"winter", "spring", "summer", "fall", "autumn"}

    def extract_entities(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        entities: Dict[str, Any] = {}

        persona = self._detect_persona(t)
        if persona:
            entities["persona"] = persona.value

        topics = self._detect_topics(t)
        if topics:
            entities["topics"] = [x.value for x in topics]

        stage = self._detect_stage(t)
        if stage:
            entities["workflow_stage"] = stage.value

        season = self._detect_season(t)
        if season:
            entities["season"] = season

        return entities

    def merge_into_state(self, state: MultiTurnContextState, extracted: Dict[str, Any]) -> None:
        """Merge extracted entities into persistent state constraints/entity memory."""

        if not extracted:
            return

        # constraints are retrieval-facing
        if "persona" in extracted:
            state.constraints["persona"] = extracted["persona"]
        if "topics" in extracted:
            state.constraints["topics"] = extracted["topics"]
        if "workflow_stage" in extracted:
            state.constraints["workflow_stage"] = extracted["workflow_stage"]
        if "season" in extracted:
            state.constraints["season"] = extracted["season"]

        # entity_memory keeps richer evolving references
        for k, v in extracted.items():
            state.entity_memory[k] = v

    def _detect_persona(self, t: str) -> Optional[PersonaType]:
        if "hobbyist" in t or "backyard" in t:
            return PersonaType.hobbyist
        if "commercial" in t or "migratory" in t:
            return PersonaType.commercial
        if "queen breeder" in t:
            return PersonaType.queen_breeder
        if "pollination" in t:
            return PersonaType.pollination_operator
        return None

    def _detect_topics(self, t: str) -> List[ResearchTopic]:
        out: List[ResearchTopic] = []
        if "varroa" in t or "mite" in t:
            out.append(ResearchTopic.varroa_management)
        if "overwinter" in t or "winter" in t:
            out.append(ResearchTopic.overwintering)
        if "queen" in t:
            out.append(ResearchTopic.queen_health_reproduction)
        if "inspect" in t or "inspection" in t:
            out.append(ResearchTopic.labor_operations)
        return out

    def _detect_stage(self, t: str) -> Optional[WorkflowStage]:
        if any(x in t for x in ["monitor", "monitoring", "check", "testing", "inspection"]):
            return WorkflowStage.monitoring
        if any(x in t for x in ["treat", "treatment", "apply", "oxalic", "formic"]):
            return WorkflowStage.treatment
        if any(x in t for x in ["record", "log", "spreadsheet", "note"]):
            return WorkflowStage.recordkeeping
        return None

    def _detect_season(self, t: str) -> Optional[str]:
        for s in self._season_terms:
            if s in t:
                return s
        return None


def update_context_state(
    state: MultiTurnContextState,
    turn: ConversationTurn,
    *,
    tracker: Optional[EntityTracker] = None,
) -> MultiTurnContextState:
    """
    Update conversation state with a new turn.

    - appends normalized turn
    - updates entity tracking for user turns
    - refreshes updated_at
    """

    tracker = tracker or EntityTracker()
    state.turns.append({"role": turn.role.value, "text": turn.text, "timestamp": turn.timestamp.isoformat()})
    if turn.role == TurnRole.user:
        extracted = tracker.extract_entities(turn.text)
        tracker.merge_into_state(state, extracted)
    state.updated_at = datetime.now(UTC)
    return state


def build_retrieval_hints(
    state: MultiTurnContextState,
    *,
    active_query_entities: Optional[Dict[str, Any]] = None,
) -> RetrievalContextHints:
    """
    Build integration payload for planner + retriever.

    Planner can use:
    - `query_prefix` for follow-up rewriting
    Retriever can use:
    - `hard_filters` to constrain corpora
    - `boosted_terms` for BM25/hybrid boosting
    """

    active_query_entities = active_query_entities or {}
    merged = {**state.entity_memory, **active_query_entities}

    hard_filters: Dict[str, Any] = {}
    boosted_terms: List[str] = []

    persona = merged.get("persona")
    if persona:
        hard_filters["persona"] = persona
        boosted_terms.append(f"{persona} beekeepers")

    topics = merged.get("topics") or []
    if topics:
        hard_filters["topics"] = topics
        boosted_terms.extend([str(t).replace("_", " ") for t in topics])

    stage = merged.get("workflow_stage")
    if stage:
        hard_filters["workflow_stage"] = stage
        boosted_terms.append(str(stage).replace("_", " "))

    season = merged.get("season")
    if season:
        hard_filters["season"] = season
        boosted_terms.append(str(season))

    prefix_parts = []
    if persona:
        prefix_parts.append(f"{persona} beekeepers")
    if topics:
        prefix_parts.extend([str(t).replace("_", " ") for t in topics])
    if stage:
        prefix_parts.append(f"{str(stage).replace('_', ' ')} workflow")
    if season:
        prefix_parts.append(f"{season} season")

    return RetrievalContextHints(
        session_id=state.session_id,
        resolved_entities=merged,
        hard_filters=hard_filters,
        boosted_terms=_dedupe_preserve(boosted_terms),
        query_prefix=" ".join(prefix_parts).strip(),
    )


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

