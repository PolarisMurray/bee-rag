from __future__ import annotations

import pytest

from beekeeper_intel.models import MultiTurnContextState
from beekeeper_intel.query_processing import QueryProcessor, SupportedIntent


def test_intent_problem_discovery_varroa_hobbyist():
    qp = QueryProcessor()
    out = qp.process("What do hobbyist beekeepers struggle with most in varroa monitoring?")
    assert out.intent in {SupportedIntent.problem_discovery, SupportedIntent.evidence_synthesis}
    assert out.persona is not None and out.persona.value == "hobbyist"
    assert any(t.value == "varroa_management" for t in out.topics)
    assert out.rewritten_query
    assert out.hyde.used is True


def test_intent_persona_comparison():
    qp = QueryProcessor()
    out = qp.process("How is this different for commercial beekeepers?", prior_user_utterance="Varroa monitoring pain points")
    assert out.intent in {SupportedIntent.persona_comparison, SupportedIntent.follow_up_clarification}


def test_followup_resolution_process_like_that():
    qp = QueryProcessor()
    out = qp.process("What does that process look like?", prior_user_utterance="Varroa monitoring via alcohol wash")
    # should resolve into something containing prior context
    assert "alcohol wash" in out.resolved_query.lower()
    assert out.intent in {SupportedIntent.follow_up_clarification, SupportedIntent.workflow_analysis}


def test_followup_resolution_uses_context_constraints():
    qp = QueryProcessor()
    state = MultiTurnContextState(
        session_id="s1",
        constraints={"persona": "commercial", "topic": "varroa_management", "workflow_stage": "monitoring"},
    )
    out = qp.process("How do they currently deal with this?", context_state=state, prior_user_utterance="Varroa monitoring challenges")
    assert "commercial" in out.resolved_query.lower()
    assert "varroa" in out.resolved_query.lower()


def test_evidence_synthesis_intent():
    qp = QueryProcessor()
    out = qp.process("Is there evidence that this is a real unmet need?", prior_user_utterance="Varroa monitoring pain points for hobbyists")
    assert out.intent in {SupportedIntent.evidence_synthesis, SupportedIntent.follow_up_clarification}
    # should generate subqueries for synthesis-oriented intents
    assert len(out.subqueries) >= 1


def test_document_lookup_intent():
    qp = QueryProcessor()
    out = qp.process("In the document, what does page 12 say about varroa thresholds?")
    assert out.intent == SupportedIntent.document_lookup

