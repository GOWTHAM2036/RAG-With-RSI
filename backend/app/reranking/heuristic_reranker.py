"""
High performance heuristic reranker combining token overlap density, phrase proximity, and MMR diversity.
"""
from typing import List, Set
import re
from backend.app.models import RetrievedChunk
from backend.app.reranking.base import BaseReranker

class HeuristicReranker(BaseReranker):
    def __init__(self, diversity_penalty: float = 0.2):
        self.diversity_penalty = diversity_penalty

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _jaccard_similarity(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union if union > 0 else 0.0

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = 3) -> List[RetrievedChunk]:
        if not chunks:
            return []

        q_tokens = self._tokenize(query)
        q_token_set = set(q_tokens)
        if not q_tokens:
            return chunks[:top_n]

        # Calculate base rerank scores
        scored_candidates = []
        for r_chunk in chunks:
            content = r_chunk.chunk.content
            c_tokens = self._tokenize(content)
            c_token_set = set(c_tokens)

            # Token overlap density
            overlap = len(q_token_set.intersection(c_token_set))
            density = overlap / len(q_token_set) if q_token_set else 0.0

            # Exact phrase match boost
            exact_boost = 0.25 if query.lower() in content.lower() else 0.0

            # Early mention bonus (if query terms appear early in chunk)
            early_bonus = 0.0
            for idx, token in enumerate(c_tokens[:30]):
                if token in q_token_set:
                    early_bonus += 0.01

            # Base score blending with original retrieval score
            combined_score = 0.5 * r_chunk.score + 0.35 * density + exact_boost + early_bonus
            scored_candidates.append({
                "chunk": r_chunk,
                "score": combined_score,
                "token_set": c_token_set
            })

        # Apply Maximal Marginal Relevance (MMR) selection to avoid redundant near-duplicate chunks
        selected: List[RetrievedChunk] = []
        selected_token_sets: List[Set[str]] = []
        remaining = list(scored_candidates)

        while remaining and len(selected) < top_n:
            best_idx = -1
            best_mmr_score = -float("inf")

            for i, cand in enumerate(remaining):
                relevance = cand["score"]
                # Compute maximum redundancy against already selected
                max_redundancy = 0.0
                if selected_token_sets:
                    max_redundancy = max(
                        self._jaccard_similarity(cand["token_set"], sel_tokens)
                        for sel_tokens in selected_token_sets
                    )

                mmr_val = relevance - (self.diversity_penalty * max_redundancy)
                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = i

            if best_idx >= 0:
                chosen = remaining.pop(best_idx)
                selected_chunk = chosen["chunk"]
                selected.append(
                    RetrievedChunk(
                        chunk=selected_chunk.chunk,
                        score=float(round(chosen["score"], 4)),
                        retrieval_source="reranked",
                        rank=len(selected) + 1
                    )
                )
                selected_token_sets.append(chosen["token_set"])
            else:
                break

        return selected
