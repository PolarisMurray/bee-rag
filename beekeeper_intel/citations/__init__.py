"""
Citation and explainability utilities for grounded QA and research synthesis.
"""

from .models import (  # noqa: F401
    CitationFormat,
    CitationProvenanceRecord,
    ExplainableAnswerBundle,
    ExplainableReportBundle,
    RenderedCitation,
)
from .renderer import (  # noqa: F401
    render_citation,
    render_citations,
    render_inline_citation_block,
)
from .integration import (  # noqa: F401
    build_report_bundle,
    build_report_evidence_map,
    build_answer_bundle,
)

