"""
OpenAI-compatible chat client.

Used to call DeepSeek and OpenAI via the OpenAI Chat Completions interface:
- POST {base_url}/v1/chat/completions

This module keeps key handling out of logs; callers must ensure they do not
persist secrets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OpenAICompatibleClientConfig:
    """Connection/configuration for an OpenAI-compatible chat endpoint."""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 800


class OpenAICompatibleChatClient:
    """Minimal chat client for openai-compatible providers."""

    def __init__(self, *, cfg: OpenAICompatibleClientConfig) -> None:
        self.cfg = cfg

    def generate_answer(
        self,
        *,
        query: str,
        extracted_signals: List[Dict[str, Any]],
        evidence_texts: List[str],
    ) -> str:
        """
        Generate a narrative grounded answer.

        Note: citations are appended by the citation renderer after this method returns.
        """

        system = (
            "You are a research assistant for beekeeper need discovery. "
            "Write a clear, evidence-grounded answer in natural English. "
            "Do not invent citations. Do not include citation brackets. "
            "Use only the provided evidence and extracted signals."
        )

        user = {
            "research_question": query,
            "extracted_signals": extracted_signals,
            "retrieved_evidence": evidence_texts,
            "instructions": [
                "Answer directly and concisely.",
                "If evidence is insufficient, say so explicitly.",
                "Focus on pain points, workflows, workaround, barriers, and unmet needs.",
            ],
        }

        return self._chat(system=system, user=json.dumps(user, ensure_ascii=False))

    def generate_report_summary(
        self,
        *,
        query: str,
        needs_signals: List[Dict[str, Any]],
        evidence_texts: List[str],
    ) -> str:
        """
        Generate a narrative executive summary for a structured research report.

        Note: the rest of report structure (needs list, sections) is produced
        by deterministic extractors; this function only writes `executive_summary`.
        """

        system = (
            "You are a research assistant producing an executive summary for a beekeeper "
            "need discovery report. Write a factual, evidence-grounded summary. "
            "Do not include citations brackets."
        )

        user = {
            "report_request": query,
            "prioritized_need_signals": needs_signals,
            "retrieved_evidence": evidence_texts,
            "instructions": [
                "Summarize key themes across personas/workflow stages.",
                "Highlight the most important unmet needs and bottlenecks.",
                "Avoid claims that are not supported by evidence.",
            ],
        }

        return self._chat(system=system, user=json.dumps(user, ensure_ascii=False))

    def _chat(self, *, system: str, user: str) -> str:
        # Lazy import so tests can run without httpx unless LLM is invoked.
        import httpx  # type: ignore

        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code < 200 or resp.status_code >= 300:
                raise RuntimeError(f"LLM request failed: {resp.status_code}: {resp.text}")
            data = resp.json()

        # OpenAI-compatible shape: choices[0].message.content
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc

