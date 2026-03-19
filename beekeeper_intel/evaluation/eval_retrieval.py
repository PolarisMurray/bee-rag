"""
Retrieval evaluation utilities.

Supports:
- MRR
- Precision@K
- Recall@K
- NDCG@K
- loading manually labeled datasets
- offline benchmarking over multiple strategy configs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set
from uuid import UUID

from beekeeper_intel.evaluation.schemas import (
    EvaluationDataset,
    EvaluationExample,
    RetrievalPrediction,
    RetrievalRunRecord,
    StrategyConfig,
    StrategyResult,
)
from beekeeper_intel.models import ResearchQuery


def precision_at_k(pred_ids: Sequence[UUID], gold_ids: Set[UUID], k: int) -> float:
    """Precision@K."""

    if k <= 0:
        return 0.0
    top = list(pred_ids[:k])
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in gold_ids)
    return hits / len(top)


def recall_at_k(pred_ids: Sequence[UUID], gold_ids: Set[UUID], k: int) -> float:
    """Recall@K."""

    if not gold_ids:
        return 0.0
    top = list(pred_ids[:k])
    hits = sum(1 for x in top if x in gold_ids)
    return hits / len(gold_ids)


def mean_reciprocal_rank(pred_ids: Sequence[UUID], gold_ids: Set[UUID]) -> float:
    """MRR for a single query."""

    if not gold_ids:
        return 0.0
    for idx, pid in enumerate(pred_ids, start=1):
        if pid in gold_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(pred_ids: Sequence[UUID], gold_ids: Set[UUID], k: int) -> float:
    """
    NDCG@K with binary relevance.
    """

    if k <= 0 or not gold_ids:
        return 0.0
    top = list(pred_ids[:k])
    dcg = 0.0
    for i, pid in enumerate(top, start=1):
        rel = 1.0 if pid in gold_ids else 0.0
        if rel > 0:
            dcg += rel / _log2(i + 1)

    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / _log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_prediction(
    pred: RetrievalPrediction,
    example: EvaluationExample,
    *,
    k: int,
) -> RetrievalRunRecord:
    """Evaluate one prediction against one labeled example."""

    gold = set(example.gold_chunk_ids)
    mrr = mean_reciprocal_rank(pred.retrieved_chunk_ids, gold)
    p = precision_at_k(pred.retrieved_chunk_ids, gold, k)
    r = recall_at_k(pred.retrieved_chunk_ids, gold, k)
    n = ndcg_at_k(pred.retrieved_chunk_ids, gold, k)
    return RetrievalRunRecord(
        example_id=example.example_id,
        strategy_name=pred.strategy_name,
        mrr=mrr,
        precision_at_k=p,
        recall_at_k=r,
        ndcg_at_k=n,
    )


def aggregate_strategy(records: List[RetrievalRunRecord], strategy_name: str) -> StrategyResult:
    """Aggregate per-example records into strategy-level averages."""

    n = len(records)
    if n == 0:
        return StrategyResult(
            strategy_name=strategy_name,
            n_examples=0,
            avg_mrr=0.0,
            avg_precision_at_k=0.0,
            avg_recall_at_k=0.0,
            avg_ndcg_at_k=0.0,
            per_example=[],
        )
    return StrategyResult(
        strategy_name=strategy_name,
        n_examples=n,
        avg_mrr=sum(r.mrr for r in records) / n,
        avg_precision_at_k=sum(r.precision_at_k for r in records) / n,
        avg_recall_at_k=sum(r.recall_at_k for r in records) / n,
        avg_ndcg_at_k=sum(r.ndcg_at_k for r in records) / n,
        per_example=records,
    )


def load_labeled_dataset(path: str) -> EvaluationDataset:
    """
    Load manual-labeled dataset from JSON or JSONL.

    JSON format:
    {
      "dataset_id": "...",
      "version": "v1",
      "split": "test",
      "examples": [ ... EvaluationExample-compatible dicts ... ]
    }

    JSONL format:
      one EvaluationExample-compatible dict per line.
      dataset_id/version/split are inferred from filename defaults.
    """

    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        examples = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(EvaluationExample.model_validate(json.loads(line)))
        return EvaluationDataset(dataset_id=p.stem, examples=examples)

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return EvaluationDataset.model_validate(data)


RetrieverFn = Callable[[ResearchQuery, StrategyConfig], RetrievalPrediction]


def run_offline_benchmark(
    dataset: EvaluationDataset,
    strategies: List[StrategyConfig],
    retriever_fn: RetrieverFn,
    *,
    top_k: int,
) -> Dict[str, StrategyResult]:
    """
    Run offline benchmark for multiple retrieval strategies.

    retriever_fn should execute the configured strategy and return ranked predictions.
    """

    results: Dict[str, List[RetrievalRunRecord]] = {s.strategy_name: [] for s in strategies}

    for ex in dataset.examples:
        for strategy in strategies:
            pred = retriever_fn(ex.query, strategy)
            rec = evaluate_prediction(pred, ex, k=top_k)
            results[strategy.strategy_name].append(rec)

    return {name: aggregate_strategy(records, name) for name, records in results.items()}


def _log2(x: float) -> float:
    import math

    return math.log(x, 2)

