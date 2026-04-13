"""
Sample offline experiment runner for retrieval strategy comparison.

Compares:
- BM25 only
- vector only
- hybrid
- hybrid + rerank
- with vs without query rewrite
- with vs without HyDE

Usage:
    python -m beekeeper_intel.evaluation.sample_experiment_runner --dataset data/eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from beekeeper_intel.evaluation.eval_retrieval import load_labeled_dataset, run_offline_benchmark
from beekeeper_intel.evaluation.schemas import RetrievalPrediction, StrategyConfig
from beekeeper_intel.models import ResearchQuery


def make_default_strategies(top_k: int) -> List[StrategyConfig]:
    return [
        StrategyConfig(strategy_name="bm25_only", retrieval_mode="bm25_only", top_k=top_k),
        StrategyConfig(strategy_name="vector_only", retrieval_mode="vector_only", top_k=top_k),
        StrategyConfig(strategy_name="hybrid", retrieval_mode="hybrid", top_k=top_k),
        StrategyConfig(
            strategy_name="hybrid_rerank",
            retrieval_mode="hybrid_rerank",
            use_rerank=True,
            top_k=top_k,
        ),
        StrategyConfig(
            strategy_name="hybrid_rewrite",
            retrieval_mode="hybrid",
            use_query_rewrite=True,
            top_k=top_k,
        ),
        StrategyConfig(
            strategy_name="hybrid_hyde",
            retrieval_mode="hybrid",
            use_hyde=True,
            top_k=top_k,
        ),
    ]


def mock_retriever(query: ResearchQuery, strategy: StrategyConfig) -> RetrievalPrediction:
    """
    Placeholder retriever for demonstration.

    Replace this with your real pipeline:
      QueryProcessor -> Retriever -> Reranker -> result ids.
    """

    # Deterministic synthetic ranking from query hash + strategy name.
    import random

    seed = hash((query.text, strategy.strategy_name)) & 0xFFFFFFFF
    rnd = random.Random(seed)
    # fake UUID list; in real usage these come from retrieval output
    retrieved_chunks = [UUID(int=rnd.getrandbits(128)) for _ in range(strategy.top_k)]
    return RetrievalPrediction(
        example_id=query.query_id,
        strategy_name=strategy.strategy_name,
        retrieved_chunk_ids=retrieved_chunks,
        retrieved_document_ids=[],
    )


def run(dataset_path: str, top_k: int = 10, output_path: str | None = None) -> Dict[str, dict]:
    dataset = load_labeled_dataset(dataset_path)
    strategies = make_default_strategies(top_k)
    results = run_offline_benchmark(dataset, strategies, mock_retriever, top_k=top_k)

    # serialize
    out = {name: r.model_dump() for name, r in results.items()}
    if output_path:
        p = Path(output_path)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to labeled dataset (.json or .jsonl)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="", help="Optional output path for results json")
    args = parser.parse_args()

    out = run(args.dataset, top_k=args.top_k, output_path=(args.output or None))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

