"""
Cross-Encoder / API reranker with fallback.
"""
from typing import List
from backend.app.models import RetrievedChunk
from backend.app.reranking.base import BaseReranker
from backend.app.reranking.heuristic_reranker import HeuristicReranker

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = HeuristicReranker()

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = 3) -> List[RetrievedChunk]:
        if not chunks:
            return []

        if self._model:
            try:
                pairs = [[query, c.chunk.content] for c in chunks]
                scores = self._model.predict(pairs)
                # Pair with chunks
                scored = list(zip(chunks, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = scored[:top_n]
                return [
                    RetrievedChunk(
                        chunk=item[0].chunk,
                        score=float(item[1]),
                        retrieval_source="cross_encoder",
                        rank=idx + 1
                    )
                    for idx, item in enumerate(top)
                ]
            except Exception:
                pass

        return self._fallback.rerank(query, chunks, top_n=top_n)
