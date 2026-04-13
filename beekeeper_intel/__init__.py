"""
Beekeeper Research Intelligence Platform (backend package).

This package is intended to evolve from the current set of prototype scripts into a
production-grade, modular RAG + research synthesis backend.
"""

from .models import (  # noqa: F401
    Citation,
    CriticReview,
    DocumentChunk,
    FinalAnswer,
    FinalResearchReport,
    MultiTurnContextState,
    ParsedDocument,
    PersonaInsight,
    ProductOpportunity,
    ResearchIntent,
    ResearchPlan,
    ResearchQuery,
    RetrievalCandidate,
    RetrievalEvaluationRecord,
    RetrievedEvidence,
    SourceMetadata,
    WorkflowFriction,
    NeedInsight,
)

from .evaluation import (  # noqa: F401
    EvaluationDataset,
    EvaluationExample,
    GroundingEvaluationResult,
    GroundingInput,
    ReportEvaluationInput,
    ReportEvaluationResult,
    RetrievalExperimentConfig,
    RetrievalPrediction,
    RetrievalRunRecord,
    StrategyConfig,
    StrategyResult,
)

from .orchestration import (  # noqa: F401
    OrchestratorConfig,
    OrchestratorResult,
    PipelineMode,
    PlatformOrchestrator,
)

