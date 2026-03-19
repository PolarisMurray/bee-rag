"""
HyDE (Hypothetical Document Embeddings) generator.

Production design:
- interface-based, so you can swap deterministic HyDE for LLM-based HyDE later
- safe default: deterministic generation (no network calls)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import HyDEConfig, HyDEResult
from .normalize import normalize_whitespace


class HyDEGenerator:
    """Interface for HyDE generation."""

    def generate(self, query: str, *, cfg: HyDEConfig) -> HyDEResult:
        raise NotImplementedError


class DeterministicHyDEGenerator(HyDEGenerator):
    """
    Deterministic HyDE generator.

    It produces an "evidence-like" note that expands the query into plausible terminology
    without asserting specific facts.
    """

    def generate(self, query: str, *, cfg: HyDEConfig) -> HyDEResult:
        q = normalize_whitespace(query)
        if not q:
            return HyDEResult(used=False, hypothetical_document=None, generator="deterministic_v1")

        # Keep it intentionally cautious: useful for embedding expansion, not for truth.
        doc = normalize_whitespace(
            f"""
            Research note (hypothetical, for semantic retrieval only):
            The question is: "{q}".

            This likely concerns beekeeper workflows, constraints, pain points, and evidence of unmet needs.
            Relevant concepts may include: varroa monitoring methods (alcohol wash, sugar roll), thresholds,
            treatment decision-making (oxalic acid vaporization / dribble, formic treatments), seasonality,
            labor and time costs, equipment availability, compliance/safety constraints, and differences by
            operation scale (hobbyist vs commercial).

            The objective is to retrieve grounded sources describing real beekeeper experiences, documented
            challenges, current workarounds, and signals of product opportunity.
            """
        )

        # Trim/expand to approximate desired length (character-based, deterministic).
        if len(doc) > cfg.desired_length_chars:
            doc = doc[: cfg.desired_length_chars].rstrip()
        else:
            # pad lightly with neutral keywords if very short
            while len(doc) < cfg.desired_length_chars:
                doc += " Evidence. Workflow. Pain point. Workaround. Unmet need."
                if len(doc) > cfg.desired_length_chars:
                    doc = doc[: cfg.desired_length_chars].rstrip()
                    break

        return HyDEResult(used=True, hypothetical_document=doc, generator="deterministic_v1")

