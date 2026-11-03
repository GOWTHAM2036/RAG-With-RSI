"""
HuggingFace / SentenceTransformers embedding provider with fallback.
"""
from typing import List
from backend.app.embeddings.base import BaseEmbeddingProvider
from backend.app.embeddings.dense_tfidf_provider import DenseTFIDFEmbeddingProvider

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = DenseTFIDFEmbeddingProvider(dim=128)
        self._dim = 384

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            self._model = None
            self._dim = 128

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> List[float]:
        if self._model:
            try:
                emb = self._model.encode(query, convert_to_numpy=True)
                return emb.tolist()
            except Exception:
                pass
        return self._fallback.embed_query(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model:
            try:
                embs = self._model.encode(texts, convert_to_numpy=True)
                return embs.tolist()
            except Exception:
                pass
        return self._fallback.embed_documents(texts)
