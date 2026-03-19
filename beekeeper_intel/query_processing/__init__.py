"""
Query processing module for the Beekeeper Research Intelligence Platform.

Responsibilities:
- Intent classification (domain + RAG routing)
- Query rewriting + follow-up resolution (multi-turn)
- Synonym expansion + typo correction (domain aware)
- HyDE hypothetical document generation (pluggable)
"""

from .types import (  # noqa: F401
    HyDEConfig,
    HyDEResult,
    QueryProcessingConfig,
    QueryProcessingResult,
    QueryProcessingTrace,
    SupportedIntent,
)
from .processor import QueryProcessor  # noqa: F401

