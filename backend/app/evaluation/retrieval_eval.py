"""
Retrieval evaluation metrics:
- Hit Rate@K
- Mean Reciprocal Rank (MRR@K)
- Context Precision
- Context Recall
- NDCG@K
"""
from typing import List, Dict, Any, Set
import math
import re
from backend.app.models import RetrievedChunk, RetrievalMetrics

class RetrievalEvaluator:
    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    @classmethod
    def is_chunk_relevant(cls, chunk_text: str, ground_truth_contexts: List[str], threshold: float = 0.3) -> bool:
        if not ground_truth_contexts:
            return True
        chunk_tokens = cls._tokenize(chunk_text)
        if not chunk_tokens:
            return False

        for gt in ground_truth_contexts:
            gt_tokens = cls._tokenize(gt)
            if not gt_tokens:
                continue
            overlap = len(chunk_tokens.intersection(gt_tokens))
            recall = overlap / len(gt_tokens)
            if recall >= threshold:
                return True
        return False

    @classmethod
    def compute_hit_rate(cls, retrieved_chunks: List[RetrievedChunk], ground_truth_contexts: List[str], k: int = 5) -> float:
        top_k = retrieved_chunks[:k]
        for rc in top_k:
            if cls.is_chunk_relevant(rc.chunk.content, ground_truth_contexts):
                return 1.0
        return 0.0

    @classmethod
    def compute_mrr(cls, retrieved_chunks: List[RetrievedChunk], ground_truth_contexts: List[str], k: int = 5) -> float:
        top_k = retrieved_chunks[:k]
        for idx, rc in enumerate(top_k):
            if cls.is_chunk_relevant(rc.chunk.content, ground_truth_contexts):
                return 1.0 / (idx + 1)
        return 0.0

    @classmethod
    def compute_context_precision(cls, retrieved_chunks: List[RetrievedChunk], ground_truth_contexts: List[str], k: int = 5) -> float:
        top_k = retrieved_chunks[:k]
        if not top_k:
            return 0.0

        hits = 0
        precision_sum = 0.0
        for i, rc in enumerate(top_k):
            if cls.is_chunk_relevant(rc.chunk.content, ground_truth_contexts):
                hits += 1
                precision_sum += hits / (i + 1)

        return precision_sum / hits if hits > 0 else 0.0

    @classmethod
    def compute_context_recall(cls, retrieved_chunks: List[RetrievedChunk], ground_truth_contexts: List[str], k: int = 5) -> float:
        if not ground_truth_contexts:
            return 1.0
        top_k = retrieved_chunks[:k]
        all_retrieved_tokens: Set[str] = set()
        for rc in top_k:
            all_retrieved_tokens.update(cls._tokenize(rc.chunk.content))

        total_coverage = 0.0
        for gt in ground_truth_contexts:
            gt_tokens = cls._tokenize(gt)
            if not gt_tokens:
                total_coverage += 1.0
                continue
            overlap = len(all_retrieved_tokens.intersection(gt_tokens))
            total_coverage += overlap / len(gt_tokens)

        return total_coverage / len(ground_truth_contexts)

    @classmethod
    def compute_ndcg(cls, retrieved_chunks: List[RetrievedChunk], ground_truth_contexts: List[str], k: int = 5) -> float:
        top_k = retrieved_chunks[:k]
        if not top_k:
            return 0.0

        dcg = 0.0
        relevance_list = []
        for i, rc in enumerate(top_k):
            rel = 1.0 if cls.is_chunk_relevant(rc.chunk.content, ground_truth_contexts) else 0.0
            relevance_list.append(rel)
            dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 2)

        # Ideal DCG
        ideal_relevance = sorted(relevance_list, reverse=True)
        idcg = sum((math.pow(2, rel) - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        return dcg / idcg if idcg > 0 else 0.0

    @classmethod
    def evaluate_query(cls, retrieved_chunks: List[RetrievedChunk], ground_truth_contexts: List[str], k: int = 5) -> RetrievalMetrics:
        return RetrievalMetrics(
            hit_rate=round(cls.compute_hit_rate(retrieved_chunks, ground_truth_contexts, k=k), 4),
            mrr=round(cls.compute_mrr(retrieved_chunks, ground_truth_contexts, k=k), 4),
            context_precision=round(cls.compute_context_precision(retrieved_chunks, ground_truth_contexts, k=k), 4),
            context_recall=round(cls.compute_context_recall(retrieved_chunks, ground_truth_contexts, k=k), 4),
            ndcg=round(cls.compute_ndcg(retrieved_chunks, ground_truth_contexts, k=k), 4)
        )
