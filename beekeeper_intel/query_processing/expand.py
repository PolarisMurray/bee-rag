"""
Domain-aware synonym expansion for beekeeper research queries.

This module is deliberately deterministic and lightweight:
- it emits expansions (keywords/phrases) for hybrid retrieval
- it does not require external NLP libraries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .normalize import basic_tokenize, unique_preserve_order


@dataclass(frozen=True)
class SynonymEntry:
    canonical: str
    synonyms: List[str]


def default_synonyms() -> List[SynonymEntry]:
    """
    Curated starting synonym set.

    Expand over time based on real beekeeper corpora (forums/interviews/extension docs).
    """

    return [
        SynonymEntry("varroa", ["varroa mite", "varroa destructor", "mite load", "mites"]),
        SynonymEntry("monitoring", ["monitor", "check", "testing", "mite count", "sampling"]),
        SynonymEntry("alcohol wash", ["alcohol wash", "alcohol roll", "wash test"]),
        SynonymEntry("sugar roll", ["sugar roll", "powdered sugar roll"]),
        SynonymEntry("oxalic acid", ["oxalic", "OAV", "oxalic acid vaporization", "oxalic dribble"]),
        SynonymEntry("formic acid", ["formic", "formic pro", "MAQS"]),
        SynonymEntry("thymol", ["apiguard", "thymol treatment"]),
        SynonymEntry("treatment", ["treat", "treatment", "intervention", "miticide"]),
        SynonymEntry("unmet need", ["unmet need", "pain point", "friction", "struggle", "challenge"]),
        SynonymEntry("commercial", ["commercial beekeeper", "large-scale", "migratory", "operation"]),
        SynonymEntry("hobbyist", ["hobbyist beekeeper", "backyard beekeeper", "small-scale"]),
    ]


def expand_query_keywords(query: str, entries: List[SynonymEntry], max_expansions: int) -> List[str]:
    """
    Produce a list of expansion phrases to add to retrieval.

    Output are phrases, not a single rewritten string, so downstream retrievers can decide:
    - expand BM25 query terms
    - create multiple vector queries
    - filter by topic/persona
    """

    q_low = (query or "").lower()
    tokens = set(basic_tokenize(q_low))

    expansions: List[str] = []
    for e in entries:
        # trigger if canonical appears or any synonym appears
        trig = e.canonical.lower() in q_low
        if not trig:
            for s in e.synonyms:
                if s.lower() in q_low:
                    trig = True
                    break
        if trig:
            expansions.extend(e.synonyms)

    # also expand token-level forms for common abbreviations if present
    if "oav" in tokens and "oxalic acid vaporization" not in expansions:
        expansions.append("oxalic acid vaporization")

    expansions = unique_preserve_order(expansions)
    return expansions[:max_expansions]

