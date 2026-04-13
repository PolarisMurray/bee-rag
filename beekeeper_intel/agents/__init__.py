"""
Agent layer for the Beekeeper Research Intelligence Platform.

Agents are higher-level components that transform evidence into researcher-ready outputs.
"""

from .extraction_agent import (  # noqa: F401
    AggregatedNeedCluster,
    ExtractedNeedInsight,
    ExtractionAgent,
    ExtractionConfig,
)

