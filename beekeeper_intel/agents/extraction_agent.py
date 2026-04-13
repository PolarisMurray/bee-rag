"""
ExtractionAgent: derive structured beekeeper user-need insights from retrieved evidence.

This is NOT a summarizer. The agent extracts actionable need signals:
- persona, topic, workflow_stage
- problem/pain severity
- current workaround + barriers
- unmet need + product signal
- whether evidence is direct user voice vs expert guidance
- supporting verbatim snippets + provenance

Design notes:
- The default implementation is deterministic + testable (no network calls).
- Prompt templates are included for an optional LLM-based extractor upgrade later.
- Aggregation clusters similar insights across sources, preserving evidence provenance and
  distinguishing anecdotal vs repeated multi-source signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from beekeeper_intel.models import (
    Citation,
    PersonaType,
    ResearchTopic,
    RetrievedEvidence,
    SourceMetadata,
    SourceType,
    WorkflowStage,
)
from beekeeper_intel.query_processing.normalize import basic_tokenize, unique_preserve_order


class Directness(str):
    """Whether the evidence is direct user voice, expert guidance, or mixed."""

    direct_user_voice = "direct_user_voice"
    expert_guidance = "expert_guidance"
    mixed = "mixed"


class ExtractedNeedInsight(BaseModel):
    """
    One extracted user-need insight derived from a single evidence bundle.

    This is intentionally “researcher ready”: concrete problem + workaround + barriers + signals.
    """

    insight_id: UUID = Field(default_factory=uuid4, description="Unique id for this extracted insight.")
    persona: Optional[PersonaType] = Field(None, description="Persona inferred from evidence.")
    topic: Optional[ResearchTopic] = Field(None, description="Primary topic inferred from evidence.")
    workflow_stage: Optional[WorkflowStage] = Field(None, description="Workflow stage inferred from evidence.")

    problem: str = Field(..., description="Concrete user problem/pain point.")
    pain_severity_1_5: int = Field(..., ge=1, le=5, description="Severity rating for prioritization.")
    current_workaround: Optional[str] = Field(None, description="Current workaround described or implied.")
    barriers: List[str] = Field(default_factory=list, description="Barriers/constraints that keep pain unresolved.")
    unmet_need: bool = Field(..., description="Whether this appears to be unmet.")
    product_signal: Optional[str] = Field(None, description="Signals pointing to product/service opportunity.")
    directness: str = Field(..., description="direct_user_voice vs expert_guidance vs mixed.")
    confidence_0_1: float = Field(..., ge=0.0, le=1.0, description="Confidence in extraction.")

    supporting_snippets: List[str] = Field(
        default_factory=list,
        description="Verbatim snippets supporting the insight (short quotes).",
    )

    # provenance
    evidence_id: UUID = Field(..., description="RetrievedEvidence.evidence_id this came from.")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting the insight.")
    source_type: Optional[SourceType] = Field(None, description="Source type for provenance and weighting.")
    source_title: Optional[str] = Field(None, description="Source title for reporting.")


class AggregatedNeedCluster(BaseModel):
    """
    Cluster of similar need insights merged across sources.

    Key features:
    - merges overlapping pain points
    - counts multi-source support
    - preserves provenance (citations, evidence ids)
    - distinguishes anecdotal vs repeated signal
    """

    cluster_id: UUID = Field(default_factory=uuid4, description="Cluster id.")
    canonical_problem: str = Field(..., description="Normalized/merged problem statement.")

    persona: Optional[PersonaType] = Field(None, description="Representative persona for this cluster.")
    topic: Optional[ResearchTopic] = Field(None, description="Representative topic.")
    workflow_stage: Optional[WorkflowStage] = Field(None, description="Representative workflow stage.")

    merged_pain_severity_1_5: int = Field(..., ge=1, le=5, description="Merged severity (e.g., median/max).")
    merged_workarounds: List[str] = Field(default_factory=list, description="Observed workarounds across sources.")
    merged_barriers: List[str] = Field(default_factory=list, description="Observed barriers across sources.")
    merged_product_signals: List[str] = Field(default_factory=list, description="Observed product signals.")

    evidence_count: int = Field(..., ge=0, description="Number of contributing extracted insights.")
    source_types: Dict[str, int] = Field(default_factory=dict, description="Counts per source type.")
    is_multi_source_signal: bool = Field(..., description="True if supported by multiple sources/types.")

    supporting_insights: List[ExtractedNeedInsight] = Field(
        default_factory=list, description="Underlying insights (for drilldown)."
    )
    citations: List[Citation] = Field(default_factory=list, description="All citations supporting the cluster.")


class ExtractionConfig(BaseModel):
    """Config for extraction + aggregation heuristics."""

    max_snippets: int = Field(4, ge=0, description="Max supporting snippets per extracted insight.")
    snippet_max_chars: int = Field(220, ge=40, description="Max characters per snippet.")

    # Clustering/merging heuristics
    cluster_similarity_threshold: float = Field(
        0.45, ge=0.0, le=1.0, description="Jaccard similarity threshold for clustering problems."
    )
    max_clusters: int = Field(200, ge=1, description="Safety cap for clustering output.")


class ExtractionAgent:
    """
    Main entrypoint for need extraction.

    Default behavior:
    - deterministic extraction from each evidence bundle
    - aggregation into clusters
    """

    def __init__(self, cfg: Optional[ExtractionConfig] = None) -> None:
        self.cfg = cfg or ExtractionConfig()

    def extract_from_evidence(self, evidence: RetrievedEvidence, source: Optional[SourceMetadata] = None) -> List[ExtractedNeedInsight]:
        """
        Extract one or more need insights from a single evidence bundle.

        In production, you can swap this method to use an LLM extractor that returns the same schema.
        """

        text = (evidence.evidence_text or "").strip()
        if not text:
            return []

        directness = _infer_directness(text, source)
        persona = _infer_persona(text, source)
        topic = _infer_topic(text, source)
        stage = _infer_stage(text)

        # Identify candidate “problem sentences”
        problem_sents = _extract_problem_sentences(text)
        if not problem_sents:
            return []

        insights: List[ExtractedNeedInsight] = []
        for ps in problem_sents:
            pain = _estimate_pain_severity(ps, text)
            workaround = _extract_workaround(text)
            barriers = _extract_barriers(text)
            unmet = _infer_unmet_need(ps, text)
            product_signal = _extract_product_signal(text)
            snippets = _supporting_snippets(text, ps, max_snippets=self.cfg.max_snippets, max_chars=self.cfg.snippet_max_chars)

            confidence = _estimate_confidence(
                directness=directness,
                has_quote=bool(snippets),
                has_workaround=bool(workaround),
                has_barriers=bool(barriers),
                source=source,
            )

            insights.append(
                ExtractedNeedInsight(
                    persona=persona,
                    topic=topic,
                    workflow_stage=stage,
                    problem=_normalize_problem(ps),
                    pain_severity_1_5=pain,
                    current_workaround=workaround,
                    barriers=barriers,
                    unmet_need=unmet,
                    product_signal=product_signal,
                    directness=directness,
                    confidence_0_1=confidence,
                    supporting_snippets=snippets,
                    evidence_id=evidence.evidence_id,
                    citations=list(evidence.citations),
                    source_type=source.source_type if source else None,
                    source_title=source.title if source else None,
                )
            )

        return insights

    def aggregate(self, insights: Sequence[ExtractedNeedInsight]) -> List[AggregatedNeedCluster]:
        """
        Cluster similar insights and merge overlapping pain points.
        """

        clusters: List[AggregatedNeedCluster] = []
        for ins in insights:
            placed = False
            for c in clusters:
                sim = jaccard_similarity(_key_tokens(ins.problem), _key_tokens(c.canonical_problem))

                # Domain-aware shortcut: if both point to same topic+stage, allow looser threshold.
                loose_ok = False
                if ins.topic is not None and c.topic is not None and ins.topic == c.topic:
                    if ins.workflow_stage is not None and c.workflow_stage is not None and ins.workflow_stage == c.workflow_stage:
                        # When topic+stage match, a lower textual threshold is acceptable because wording differs across sources.
                        loose_ok = sim >= max(0.18, self.cfg.cluster_similarity_threshold - 0.27)
                    else:
                        # Even without explicit stage alignment, merge if both clearly refer to the same core activity.
                        it = _key_tokens(ins.problem)
                        ct = _key_tokens(c.canonical_problem)
                        if {"varroa", "monitoring"}.issubset(it.union(ct)) and ("varroa" in it and "varroa" in ct):
                            if ("monitoring" in it or "monitor" in it) and ("monitoring" in ct or "monitor" in ct):
                                loose_ok = True

                if sim >= self.cfg.cluster_similarity_threshold or loose_ok:
                    _merge_into_cluster(c, ins)
                    placed = True
                    break
            if not placed:
                clusters.append(_new_cluster_from_insight(ins))
            if len(clusters) >= self.cfg.max_clusters:
                break

        # finalize flags + minor normalization
        for c in clusters:
            c.merged_workarounds = unique_preserve_order([w for w in c.merged_workarounds if w])
            c.merged_barriers = unique_preserve_order([b for b in c.merged_barriers if b])
            c.merged_product_signals = unique_preserve_order([p for p in c.merged_product_signals if p])
            c.citations = _dedupe_citations(c.citations)
            c.is_multi_source_signal = _is_multi_source(c)

        # order by evidence_count then severity
        clusters.sort(key=lambda x: (x.evidence_count, x.merged_pain_severity_1_5), reverse=True)
        return clusters


# -----------------------------
# Extraction helpers (heuristics)
# -----------------------------


_PROBLEM_PATTERNS = [
    r"\bstruggle\b",
    r"\bproblem\b",
    r"\bchallenge\b",
    r"\bpain\b",
    r"\bpain point\b",
    r"\bfriction\b",
    r"\bhard\b",
    r"\bdifficult\b",
    r"\btime[- ]consuming\b",
    r"\bexpensive\b",
    r"\bunsafe\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bnever\b",
    r"\balways\b",
    r"\bnot sure\b",
    r"\bconfusing\b",
]

_WORKAROUND_PATTERNS = [
    r"\bwe\s+just\b",
    r"\bi\s+just\b",
    r"\bwhat i do\b",
    r"\bworkaround\b",
    r"\bhack\b",
    r"\bmanual\b",
    r"\bspreadsheets?\b",
    r"\bwrite\s+it\s+down\b",
    r"\bextra\s+trip\b",
]

_BARRIER_PATTERNS = [
    r"\bcost\b",
    r"\bexpensive\b",
    r"\btime\b",
    r"\blabor\b",
    r"\bsafety\b",
    r"\bcompliance\b",
    r"\bregulat",
    r"\bavailability\b",
    r"\bcomplex\b",
    r"\btraining\b",
    r"\bweather\b",
    r"\bseason\b",
]

_PRODUCT_SIGNAL_PATTERNS = [
    r"\bi wish\b",
    r"\bneed\b.+\btool\b",
    r"\bshould be\b.+\bapp\b",
    r"\bwould pay\b",
    r"\blooking for\b.+\bsolution\b",
    r"\bno good\b.+\boption\b",
    r"\bthere isn't\b.+\bproduct\b",
]


def _infer_directness(text: str, source: Optional[SourceMetadata]) -> str:
    """
    Determine whether evidence is direct user voice vs expert guidance.
    """

    # Source-type prior
    if source:
        if source.source_type in {SourceType.forum, SourceType.interview_transcript, SourceType.internal_note}:
            return Directness.direct_user_voice
        if source.source_type in {SourceType.extension_resource, SourceType.research_paper, SourceType.report, SourceType.slide_deck}:
            return Directness.expert_guidance

    # Text-based cues (first-person)
    t = text.lower()
    first_person = any(x in t for x in [" i ", " i'm ", " i've ", " we ", " my ", " our "])
    if first_person:
        return Directness.direct_user_voice
    return Directness.expert_guidance


def _infer_persona(text: str, source: Optional[SourceMetadata]) -> Optional[PersonaType]:
    if source and source.beekeeper_persona_hint:
        return source.beekeeper_persona_hint
    t = text.lower()
    if "hobbyist" in t or "backyard" in t:
        return PersonaType.hobbyist
    if "commercial" in t or "migratory" in t or "hundreds of hives" in t:
        return PersonaType.commercial
    return None


def _infer_topic(text: str, source: Optional[SourceMetadata]) -> Optional[ResearchTopic]:
    if source and source.topics:
        return source.topics[0]
    t = text.lower()
    if "varroa" in t or "mite" in t:
        return ResearchTopic.varroa_management
    if "overwinter" in t or "winter" in t:
        return ResearchTopic.overwintering
    if "queen" in t:
        return ResearchTopic.queen_health_reproduction
    if "feed" in t or "feeding" in t or "nutrition" in t:
        return ResearchTopic.nutrition_feeding
    return None


def _infer_stage(text: str) -> Optional[WorkflowStage]:
    t = text.lower()
    if any(k in t for k in ["monitor", "monitoring", "testing", "test", "alcohol wash", "sugar roll", "mite count", "sampling"]):
        return WorkflowStage.monitoring
    if any(k in t for k in ["treat", "treatment", "apply", "oxalic", "formic", "thymol", "miticide"]):
        return WorkflowStage.treatment
    if any(k in t for k in ["inspect", "inspection", "check frames", "hive inspection"]):
        return WorkflowStage.monitoring
    if any(k in t for k in ["record", "log", "spreadsheet", "notes"]):
        return WorkflowStage.recordkeeping
    return None


def _extract_problem_sentences(text: str) -> List[str]:
    """
    Extract sentences that likely encode a pain point/problem.
    """

    sents = _split_sentences(text)
    if not sents:
        return []
    patt = re.compile("|".join(_PROBLEM_PATTERNS), re.IGNORECASE)
    out = [s for s in sents if patt.search(s)]
    # If none match, but the text is clearly “complaint-ish”, take the first sentence
    if not out and len(text) < 400:
        out = [sents[0]]
    return out[:3]


def _extract_workaround(text: str) -> Optional[str]:
    patt = re.compile("|".join(_WORKAROUND_PATTERNS), re.IGNORECASE)
    for s in _split_sentences(text):
        if patt.search(s):
            return s.strip()
    return None


def _extract_barriers(text: str) -> List[str]:
    patt = re.compile("|".join(_BARRIER_PATTERNS), re.IGNORECASE)
    barriers: List[str] = []
    for s in _split_sentences(text):
        if patt.search(s):
            # keep short
            b = s.strip()
            if len(b) > 260:
                b = b[:260].rstrip() + "…"
            barriers.append(b)
    return unique_preserve_order(barriers)[:5]


def _extract_product_signal(text: str) -> Optional[str]:
    patt = re.compile("|".join(_PRODUCT_SIGNAL_PATTERNS), re.IGNORECASE)
    for s in _split_sentences(text):
        if patt.search(s):
            return s.strip()
    return None


def _infer_unmet_need(problem_sentence: str, text: str) -> bool:
    t = (problem_sentence + " " + text).lower()
    if "unmet need" in t or "no good option" in t or "there isn't" in t:
        return True
    if "i wish" in t or "wish there was" in t:
        return True
    if "workaround" in t or "hack" in t:
        return True
    return False


def _estimate_pain_severity(problem_sentence: str, text: str) -> int:
    """
    Map cues to a 1-5 severity score (heuristic).
    """

    t = (problem_sentence + " " + text).lower()
    score = 2
    # explicit struggle/friction should not be "low"
    if any(x in t for x in ["struggle", "pain point", "friction", "challenge", "problem"]):
        score = max(score, 3)
    if any(x in t for x in ["unsafe", "risk", "die", "loss", "colony loss", "wipe out"]):
        score = max(score, 5)
    if any(x in t for x in ["time-consuming", "hours", "all day", "labor", "too much work"]):
        score = max(score, 4)
    if any(x in t for x in ["too much time", "takes too much time", "after work"]):
        score = max(score, 4)
    if any(x in t for x in ["expensive", "cost", "can't afford"]):
        score = max(score, 4)
    if any(x in t for x in ["confusing", "not sure", "hard to know", "uncertain"]):
        score = max(score, 3)
    return min(5, max(1, score))


def _estimate_confidence(
    *,
    directness: str,
    has_quote: bool,
    has_workaround: bool,
    has_barriers: bool,
    source: Optional[SourceMetadata],
) -> float:
    """
    Confidence heuristic prioritizing direct user voice and presence of concrete supporting snippets.
    """

    base = 0.55
    if directness == Directness.direct_user_voice:
        base += 0.15
    if has_quote:
        base += 0.15
    if has_workaround:
        base += 0.08
    if has_barriers:
        base += 0.05
    if source and source.ocr_used and source.ocr_avg_confidence is not None and source.ocr_avg_confidence < 0.7:
        base -= 0.12
    return float(min(0.95, max(0.1, base)))


def _supporting_snippets(text: str, problem_sentence: str, *, max_snippets: int, max_chars: int) -> List[str]:
    """
    Extract short verbatim snippets supporting the insight.
    """

    sents = _split_sentences(text)
    snippets: List[str] = []

    # Always include the problem sentence (or a truncated version)
    ps = problem_sentence.strip()
    if ps:
        snippets.append(_truncate(ps, max_chars))

    # Add up to N-1 additional sentences containing key patterns (workaround, barriers, product signal)
    patt = re.compile(
        "|".join(_WORKAROUND_PATTERNS + _BARRIER_PATTERNS + _PRODUCT_SIGNAL_PATTERNS),
        re.IGNORECASE,
    )
    for s in sents:
        if len(snippets) >= max_snippets:
            break
        if patt.search(s) and s.strip() not in snippets:
            snippets.append(_truncate(s.strip(), max_chars))

    return unique_preserve_order(snippets)[:max_snippets]


def _split_sentences(text: str) -> List[str]:
    # very simple; good enough for deterministic baseline
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def _normalize_problem(s: str) -> str:
    s2 = re.sub(r"\s+", " ", s.strip())
    # remove leading discourse markers
    s2 = re.sub(r"^(but|so|and)\s+", "", s2, flags=re.IGNORECASE)
    # normalize frequent near-synonyms to improve clustering
    s2 = re.sub(r"\btesting\b", "monitoring", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\btest\b", "monitor", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\bvarroa\s+testing\b", "varroa monitoring", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\balcohol\s+wash(es)?\b", "alcohol wash", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\bsugar\s+roll(s)?\b", "sugar roll", s2, flags=re.IGNORECASE)
    return s2


# -----------------------------
# Clustering / merging helpers
# -----------------------------


def _key_tokens(text: str) -> set[str]:
    toks = basic_tokenize(text)
    # drop very generic words
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "is", "are", "be"}
    # normalize a few domain equivalents to help clustering
    norm_map = {
        "testing": "monitoring",
        "test": "monitor",
        "mite": "varroa",
        "mites": "varroa",
    }
    toks = [norm_map.get(t, t) for t in toks]
    return {t for t in toks if t not in stop and len(t) > 2}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between token sets."""

    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _new_cluster_from_insight(ins: ExtractedNeedInsight) -> AggregatedNeedCluster:
    st = ins.source_type.value if ins.source_type else "unknown"
    return AggregatedNeedCluster(
        canonical_problem=ins.problem,
        persona=ins.persona,
        topic=ins.topic,
        workflow_stage=ins.workflow_stage,
        merged_pain_severity_1_5=ins.pain_severity_1_5,
        merged_workarounds=[ins.current_workaround] if ins.current_workaround else [],
        merged_barriers=list(ins.barriers),
        merged_product_signals=[ins.product_signal] if ins.product_signal else [],
        evidence_count=1,
        source_types={st: 1},
        is_multi_source_signal=False,
        supporting_insights=[ins],
        citations=_dedupe_citations(list(ins.citations)),
    )


def _merge_into_cluster(cluster: AggregatedNeedCluster, ins: ExtractedNeedInsight) -> None:
    cluster.supporting_insights.append(ins)
    cluster.evidence_count += 1
    st = ins.source_type.value if ins.source_type else "unknown"
    cluster.source_types[st] = cluster.source_types.get(st, 0) + 1

    # update representative fields cautiously
    if cluster.persona is None and ins.persona is not None:
        cluster.persona = ins.persona
    if cluster.topic is None and ins.topic is not None:
        cluster.topic = ins.topic
    if cluster.workflow_stage is None and ins.workflow_stage is not None:
        cluster.workflow_stage = ins.workflow_stage

    # severity: use max as a conservative “worst-case pain” signal
    cluster.merged_pain_severity_1_5 = max(cluster.merged_pain_severity_1_5, ins.pain_severity_1_5)
    if ins.current_workaround:
        cluster.merged_workarounds.append(ins.current_workaround)
    cluster.merged_barriers.extend(ins.barriers)
    if ins.product_signal:
        cluster.merged_product_signals.append(ins.product_signal)

    cluster.citations.extend(ins.citations)


def _is_multi_source(cluster: AggregatedNeedCluster) -> bool:
    """
    Determine whether a cluster is supported by multiple sources/types.

    We consider multi-source if:
    - >=2 evidence items AND
    - either >=2 source types OR >=2 different documents (approximated by citation document ids)
    """

    if cluster.evidence_count < 2:
        return False
    if len(cluster.source_types) >= 2:
        return True
    doc_ids = {c.document_id for c in cluster.citations}
    return len(doc_ids) >= 2


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    seen = set()
    out: List[Citation] = []
    for c in citations:
        key = (c.document_id, c.anchor.page_number, c.anchor.slide_number, c.anchor.timestamp_ms, tuple(c.anchor.section_path), c.quote)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

