from __future__ import annotations

from datetime import UTC, datetime

from beekeeper_intel.memory import (
    ConversationTurn,
    FollowupRewriter,
    TurnRole,
    update_context_state,
)
from beekeeper_intel.models import MultiTurnContextState


def _new_state() -> MultiTurnContextState:
    return MultiTurnContextState(session_id="s-mt-1")


def test_tracks_entities_across_turns():
    state = _new_state()
    update_context_state(
        state,
        ConversationTurn(
            role=TurnRole.user,
            text="What do hobbyist beekeepers struggle with in varroa monitoring?",
            timestamp=datetime.now(UTC),
        ),
    )

    assert state.constraints.get("persona") == "hobbyist"
    assert "varroa_management" in (state.constraints.get("topics") or [])
    assert state.constraints.get("workflow_stage") == "monitoring"


def test_followup_rewrite_how_do_they_do_that():
    state = _new_state()
    update_context_state(
        state,
        ConversationTurn(role=TurnRole.user, text="How do hobbyist beekeepers monitor varroa mites?"),
    )
    rewriter = FollowupRewriter()
    out = rewriter.rewrite("How do they do that?", state)

    assert out.used_context is True
    assert "workflow" in out.rewritten_query.lower() or "monitor" in out.rewritten_query.lower()
    assert "varroa" in out.rewritten_query.lower()


def test_followup_rewrite_commercial_ones():
    state = _new_state()
    update_context_state(
        state,
        ConversationTurn(role=TurnRole.user, text="What do hobbyist beekeepers struggle with in varroa monitoring?"),
    )
    update_context_state(
        state,
        ConversationTurn(role=TurnRole.user, text="What about commercial ones?"),
    )

    rewriter = FollowupRewriter()
    out = rewriter.rewrite("What about commercial ones?", state)
    # should become explicit comparison-like query
    assert "commercial" in out.rewritten_query.lower()


def test_followup_rewrite_winter_problem():
    state = _new_state()
    update_context_state(
        state,
        ConversationTurn(role=TurnRole.user, text="Overwintering labor burden is a major challenge."),
    )
    rewriter = FollowupRewriter()
    out = rewriter.rewrite("Is this still a problem in winter?", state)
    assert "winter" in out.rewritten_query.lower()
    assert "overwinter" in out.rewritten_query.lower() or "labor burden" in out.rewritten_query.lower()


def test_followup_rewrite_difference():
    state = _new_state()
    update_context_state(
        state,
        ConversationTurn(role=TurnRole.user, text="Varroa treatment choices are often delayed."),
    )
    rewriter = FollowupRewriter()
    out = rewriter.rewrite("How is that different?", state)
    assert "different" in out.rewritten_query.lower()
    assert out.hints.query_prefix != ""


def test_integration_hints_for_retriever():
    state = _new_state()
    update_context_state(
        state,
        ConversationTurn(role=TurnRole.user, text="For commercial beekeepers, varroa monitoring in winter is difficult."),
    )
    rewriter = FollowupRewriter()
    out = rewriter.rewrite("How is that different?", state)
    # retrieval-facing hints should be present
    assert out.hints.hard_filters.get("persona") == "commercial"
    assert "winter" in out.hints.boosted_terms
    assert out.hints.query_prefix

