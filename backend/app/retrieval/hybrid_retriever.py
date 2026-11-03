"""
Hybrid Retriever combining Dense Vector embeddings with BM25 sparse retrieval.
Supports Reciprocal Rank Fusion (RRF) and tunable alpha score blending.
"""
from typing import List, Dict, Any, Optional
from backend.app.models import Chunk, RetrievedChunk
from backend.app.retrieval.base import BaseRetriever
from backend.app.retrieval.vector_store import VectorStore
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.embeddings.base import BaseEmbeddingProvider

class HybridRetriever(BaseRetriever):
    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None, alpha: float = 0.6, rrf_k: int = 60):
        """
        alpha: Weight for dense vector search (1.0 = pure dense, 0.0 = pure bm25).
        rrf_k: Constant for Reciprocal Rank Fusion.
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.rrf_k = rrf_k
        self.vector_store = VectorStore(embedding_provider=embedding_provider)
        self.bm25 = BM25Retriever()

    def clear(self):
        self.vector_store.clear()
        self.bm25.clear()

    def index_chunks(self, chunks: List[Chunk]):
        self.clear()
        self.vector_store.add_chunks(chunks)
        self.bm25.index_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        # Fetch candidate pools from both retrievers
        candidate_k = max(top_k * 3, 20)
        vector_results = self.vector_store.search(query, top_k=candidate_k, filter_metadata=filter_metadata)
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k, filter_metadata=filter_metadata)

        # Map chunks by id
        chunk_map: Dict[str, Chunk] = {}
        vector_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for r in vector_results:
            cid = r.chunk.id
            chunk_map[cid] = r.chunk
            vector_ranks[cid] = r.rank
            vector_scores[cid] = r.score

        for r in bm25_results:
            cid = r.chunk.id
            chunk_map[cid] = r.chunk
            bm25_ranks[cid] = r.rank
            bm25_scores[cid] = r.score

        all_ids = set(chunk_map.keys())
        if not all_ids:
            return []

        # Combined scoring with alpha-weighted RRF
        fused_scores = {}
        for cid in all_ids:
            # RRF score components
            v_rank = vector_ranks.get(cid, 9999)
            b_rank = bm25_ranks.get(cid, 9999)
            rrf_vec = 1.0 / (self.rrf_k + v_rank) if v_rank < 9999 else 0.0
            rrf_bm25 = 1.0 / (self.rrf_k + b_rank) if b_rank < 9999 else 0.0

            # Raw score components
            v_score = vector_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)

            # Combined weighted score
            score = self.alpha * (0.5 * v_score + 0.5 * (rrf_vec * self.rrf_k)) + \
                    (1.0 - self.alpha) * (0.5 * b_score + 0.5 * (rrf_bm25 * self.rrf_k))
            fused_scores[cid] = score

        # Sort descending
        sorted_ids = sorted(all_ids, key=lambda x: fused_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        return [
            RetrievedChunk(
                chunk=chunk_map[cid],
                score=float(fused_scores[cid]),
                retrieval_source="hybrid",
                rank=idx + 1
            )
            for idx, cid in enumerate(top_ids)
        ]
