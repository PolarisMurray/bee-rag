"""
Answer grounding evaluation.

Supports:
- citation presence
- citation grounding (citation text appears in evidence context)
- evidence coverage
- unsupported claim detection (heuristic)
"""

from __future__ import annotations

import re
from typing import List

from beekeeper_intel.evaluation.schemas import GroundingEvaluationResult, GroundingInput


def evaluate_grounding(inp: GroundingInput) -> GroundingEvaluationResult:
    """
    Evaluate answer grounding quality with deterministic heuristics.
    """

    answer = (inp.answer_text or "").strip()
    evidence = [e.strip() for e in inp.evidence_texts if e and e.strip()]
    citations = inp.citations or []

    citation_presence = 1.0 if citations else 0.0
    if answer and citations:
        # soft ratio relative to sentence count
        n_claims = max(1, len(_split_claims(answer)))
        citation_presence = min(1.0, len(citations) / n_claims)

    citation_grounding = _citation_grounding_ratio(inp)
    unsupported_claims = _unsupported_claims(answer, evidence)
    unsupported_count = len(unsupported_claims)

    coverage = _evidence_coverage(answer, evidence)

    weaknesses: List[str] = []
    if citation_presence < 0.6:
        weaknesses.append("low_citation_presence")
    if citation_grounding < 0.6:
        weaknesses.append("low_citation_grounding")
    if coverage < 0.6:
        weaknesses.append("low_evidence_coverage")
    if unsupported_count > 0:
        weaknesses.append("unsupported_claims_detected")

    supported = (unsupported_count == 0) and citation_grounding >= 0.6 and coverage >= 0.6
    return GroundingEvaluationResult(
        supported=supported,
        citation_presence=citation_presence,
        citation_grounding=citation_grounding,
        evidence_coverage=coverage,
        unsupported_claim_count=unsupported_count,
        unsupported_claims=unsupported_claims,
        weaknesses=weaknesses,
    )


def _citation_grounding_ratio(inp: GroundingInput) -> float:
    citations = inp.citations or []
    evidence_text = "\n".join(inp.evidence_texts or []).lower()
    if not citations:
        return 0.0

    grounded = 0
    for c in citations:
        q = (c.quote or "").strip().lower()
        if q and q in evidence_text:
            grounded += 1
            continue
        # no quote -> if anchor exists, count partial grounding
        if c.anchor.page_number is not None or c.anchor.slide_number is not None or c.anchor.timestamp_ms is not None:
            grounded += 0.5
    return min(1.0, grounded / len(citations))


def _evidence_coverage(answer: str, evidence: List[str]) -> float:
    claims = _split_claims(answer)
    if not claims:
        return 1.0
    ev_tokens = set(_key_tokens(" ".join(evidence)))
    covered = 0
    for c in claims:
        toks = _key_tokens(c)
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in ev_tokens)
        ratio = overlap / len(toks)
        if ratio >= 0.5:
            covered += 1
    return covered / len(claims)


def _unsupported_claims(answer: str, evidence: List[str]) -> List[str]:
    claims = _split_claims(answer)
    if not claims:
        return []
    ev_tokens = set(_key_tokens(" ".join(evidence)))
    unsupported = []
    for c in claims:
        toks = _key_tokens(c)
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in ev_tokens)
        if (overlap / len(toks)) < 0.35:
            unsupported.append(c.strip())
    return unsupported


def _split_claims(text: str) -> List[str]:
    chunks = re.split(r"[.\n;]+", text)
    return [x.strip() for x in chunks if x and x.strip()]


def _key_tokens(text: str) -> List[str]:
    tokens = re.split(r"[^a-zA-Z0-9_]+", text.lower())
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "is", "are", "be", "that", "this"}
    out: List[str] = []
    norm_map = {
        "hobbyists": "hobbyist",
        "beekeepers": "beekeeper",
        "monitor": "monitoring",
        "timeconsuming": "time",
        "burden": "time",
    }
    for tok in tokens:
        if not tok or len(tok) <= 2 or tok in stop:
            continue
        # cheap singularization
        if tok.endswith("s") and len(tok) > 4:
            tok = tok[:-1]
        tok = norm_map.get(tok, tok)
        out.append(tok)
    return out

