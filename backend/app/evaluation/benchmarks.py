"""
Benchmark runner and evaluation harness.
"""
from typing import List, Dict, Any, Optional
import time
from backend.app.models import (
    RAGStrategy,
    BenchmarkDataset,
    BenchmarkPair,
    EvaluationReport,
    RetrievalMetrics,
    AnswerMetrics,
    RetrievedChunk
)
from backend.app.chunking.factory import ChunkerFactory
from backend.app.embeddings.factory import EmbeddingFactory
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.reranking.factory import RerankerFactory
from backend.app.generation.llm_providers import GeneratorFactory
from backend.app.evaluation.retrieval_eval import RetrievalEvaluator
from backend.app.evaluation.answer_eval import AnswerEvaluator
from backend.app.models import Document

class BenchmarkRunner:
    @classmethod
    def run_strategy_benchmark(
        cls,
        strategy: RAGStrategy,
        documents: List[Document],
        benchmark: BenchmarkDataset,
        weights: Optional[Dict[str, float]] = None
    ) -> EvaluationReport:
        weights = weights or {
            "faithfulness": 0.35,
            "relevance": 0.25,
            "hit_rate": 0.20,
            "mrr": 0.15,
            "latency_penalty": 0.05
        }

        # 1. Chunk documents according to strategy
        chunker = ChunkerFactory.get_chunker(
            strategy=strategy.chunking_strategy,
            chunk_size=strategy.chunk_size,
            chunk_overlap=strategy.chunk_overlap
        )
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk_document(doc))

        # 2. Setup retriever
        emb_provider = EmbeddingFactory.get_provider(strategy.embedding_provider)
        retriever = HybridRetriever(
            embedding_provider=emb_provider,
            alpha=strategy.hybrid_alpha if strategy.retrieval_strategy == "hybrid" else (1.0 if strategy.retrieval_strategy == "dense" else 0.0)
        )
        retriever.index_chunks(all_chunks)

        # 3. Setup reranker and generator
        reranker = RerankerFactory.get_reranker(strategy.reranker_type) if strategy.reranker_enabled else None
        generator = GeneratorFactory.get_generator("heuristic")

        # 4. Run through benchmark pairs
        hit_rates, mrrs, precisions, recalls, ndcgs = [], [], [], [], []
        faithfulnesses, relevances, similarities = [], [], []
        latencies = []
        failures = {
            "retrieval_miss": 0,
            "rerank_drop": 0,
            "low_faithfulness": 0,
            "low_relevance": 0
        }

        for pair in benchmark.pairs:
            t0 = time.time()

            # Retrieve
            raw_retrieved = retriever.retrieve(pair.query, top_k=strategy.top_k)

            # Rerank
            if reranker and strategy.reranker_enabled:
                candidate_chunks = raw_retrieved
                final_retrieved = reranker.rerank(pair.query, candidate_chunks, top_n=strategy.rerank_top_n)
                # Check if reranker dropped a relevant chunk
                if any(RetrievalEvaluator.is_chunk_relevant(c.chunk.content, pair.ground_truth_contexts) for c in candidate_chunks) and not any(RetrievalEvaluator.is_chunk_relevant(c.chunk.content, pair.ground_truth_contexts) for c in final_retrieved):
                    failures["rerank_drop"] += 1
            else:
                final_retrieved = raw_retrieved[:strategy.top_k]

            # Retrieval eval
            r_metrics = RetrievalEvaluator.evaluate_query(final_retrieved, pair.ground_truth_contexts, k=len(final_retrieved))
            hit_rates.append(r_metrics.hit_rate)
            mrrs.append(r_metrics.mrr)
            precisions.append(r_metrics.context_precision)
            recalls.append(r_metrics.context_recall)
            ndcgs.append(r_metrics.ndcg)

            if r_metrics.hit_rate == 0.0:
                failures["retrieval_miss"] += 1

            # Generate
            answer_text, citations = generator.generate(
                query=pair.query,
                context_chunks=final_retrieved,
                prompt_template=strategy.prompt_template,
                temperature=strategy.temperature,
                max_tokens=strategy.max_tokens
            )

            # Answer eval
            a_metrics = AnswerEvaluator.evaluate_answer(
                query=pair.query,
                answer=answer_text,
                context_chunks=final_retrieved,
                ground_truth_answer=pair.ground_truth_answer
            )
            faithfulnesses.append(a_metrics.faithfulness)
            relevances.append(a_metrics.answer_relevance)
            similarities.append(a_metrics.semantic_similarity)

            if a_metrics.faithfulness < 0.6:
                failures["low_faithfulness"] += 1
            if a_metrics.answer_relevance < 0.5:
                failures["low_relevance"] += 1

            latency = (time.time() - t0) * 1000.0
            latencies.append(latency)

        n = len(benchmark.pairs) if benchmark.pairs else 1
        avg_hit = sum(hit_rates) / n
        avg_mrr = sum(mrrs) / n
        avg_prec = sum(precisions) / n
        avg_rec = sum(recalls) / n
        avg_ndcg = sum(ndcgs) / n

        avg_faith = sum(faithfulnesses) / n
        avg_rel = sum(relevances) / n
        avg_sim = sum(similarities) / n
        avg_lat = sum(latencies) / n

        # Normalized latency penalty (0 to 1 scale, normalized around 500ms)
        latency_score = max(0.0, 1.0 - (avg_lat / 500.0))

        # Composite Fitness calculation
        fitness = (
            weights.get("faithfulness", 0.35) * avg_faith +
            weights.get("relevance", 0.25) * avg_rel +
            weights.get("hit_rate", 0.20) * avg_hit +
            weights.get("mrr", 0.15) * avg_mrr +
            weights.get("latency_penalty", 0.05) * latency_score
        )

        strategy.fitness = round(fitness, 4)
        strategy.latency_ms = round(avg_lat, 2)

        return EvaluationReport(
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            retrieval_metrics=RetrievalMetrics(
                hit_rate=round(avg_hit, 4),
                mrr=round(avg_mrr, 4),
                context_precision=round(avg_prec, 4),
                context_recall=round(avg_rec, 4),
                ndcg=round(avg_ndcg, 4)
            ),
            answer_metrics=AnswerMetrics(
                faithfulness=round(avg_faith, 4),
                answer_relevance=round(avg_rel, 4),
                semantic_similarity=round(avg_sim, 4),
                hallucination_rate=round(max(0.0, 1.0 - avg_faith), 4)
            ),
            composite_fitness=round(fitness, 4),
            avg_latency_ms=round(avg_lat, 2),
            total_queries=len(benchmark.pairs),
            failure_analysis={
                "retrieval_misses": failures["retrieval_miss"],
                "reranker_drops": failures["rerank_drop"],
                "low_faithfulness_count": failures["low_faithfulness"],
                "low_relevance_count": failures["low_relevance"],
                "retrieval_success_rate": round((n - failures["retrieval_miss"]) / n, 4)
            }
        )
