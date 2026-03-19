"""
Citation rendering functions.
"""

from __future__ import annotations

from typing import Iterable, List

from beekeeper_intel.models import Citation, SourceType

from .models import CitationFormat, RenderedCitation


def render_citation(citation: Citation, fmt: CitationFormat = CitationFormat.compact) -> str:
    """
    Render a structured citation into a user-facing bracket form.

    Supported patterns include:
    - [DocumentName p.X]
    - [PPT slide N]
    - [Transcript 00:12:31]
    - [Interview Note section Y]
    """

    title = citation.source_title.strip() if citation.source_title else "Source"
    anchor = citation.anchor

    # timestamp has priority for transcript-style refs
    if anchor.timestamp_ms is not None:
        ts = _format_ms(anchor.timestamp_ms)
        base = f"[Transcript {ts}]"
    elif anchor.slide_number is not None:
        base = f"[PPT slide {anchor.slide_number}]"
    elif anchor.page_number is not None:
        base = f"[{title} p.{anchor.page_number}]"
    elif anchor.section_path:
        section_name = anchor.section_path[-1]
        # explicit interview note style if title suggests that, otherwise generic section style
        if "interview" in title.lower() or "note" in title.lower():
            base = f"[Interview Note section {section_name}]"
        else:
            base = f"[{title} section {section_name}]"
    else:
        base = f"[{title}]"

    if fmt == CitationFormat.verbose and citation.quote:
        return f"{base} \"{citation.quote}\""
    return base


def render_citations(citations: Iterable[Citation], fmt: CitationFormat = CitationFormat.compact) -> List[RenderedCitation]:
    """
    Render citations and deduplicate by rendered text order-preservingly.
    """

    out: List[RenderedCitation] = []
    seen = set()
    for c in citations:
        r = render_citation(c, fmt=fmt)
        if r in seen:
            continue
        seen.add(r)
        out.append(RenderedCitation(citation=c, rendered=r))
    return out


def render_inline_citation_block(citations: Iterable[Citation], fmt: CitationFormat = CitationFormat.compact) -> str:
    """
    Render citations as a compact inline block for conversational answers.

    Example:
    Sources: [Doc A p.12] [PPT slide 4] [Transcript 00:12:31]
    """

    rendered = render_citations(citations, fmt=fmt)
    if not rendered:
        return ""
    return "Sources: " + " ".join(x.rendered for x in rendered)


def _format_ms(ms: int) -> str:
    total_seconds = max(0, int(ms // 1000))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

