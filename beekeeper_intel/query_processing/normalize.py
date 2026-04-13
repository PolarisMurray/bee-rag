"""
Normalization helpers for query processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and strip."""

    return _WS_RE.sub(" ", (text or "").strip())


def basic_tokenize(text: str) -> List[str]:
    """
    Simple, deterministic tokenizer (no external deps).

    Returns lowercase alnum tokens; intended for heuristics, not linguistics.
    """

    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text or "").lower()) if t]


@dataclass(frozen=True)
class TypoCorrectionRule:
    src: str
    dst: str


def apply_typo_corrections(text: str, rules: List[TypoCorrectionRule]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Apply simple string-based typo corrections.

    Returns corrected text and list of (src,dst) corrections that fired.
    """

    corrected = text
    applied: List[Tuple[str, str]] = []
    for r in rules:
        if r.src.lower() in corrected.lower():
            # replace case-insensitively by a regex with word boundaries when possible
            pattern = re.compile(re.escape(r.src), re.IGNORECASE)
            new = pattern.sub(r.dst, corrected)
            if new != corrected:
                corrected = new
                applied.append((r.src, r.dst))
    return corrected, applied


def unique_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""

    seen = set()
    out: List[str] = []
    for x in items:
        key = x.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

