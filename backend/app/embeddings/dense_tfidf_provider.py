"""
Dense deterministic embedding provider using character/word n-gram feature hashing and TF-IDF.
Guarantees 100% offline, zero-network, high-speed embeddings for testing, development, and evolution.
"""
from typing import List
import math
import re
from backend.app.embeddings.base import BaseEmbeddingProvider

class DenseTFIDFEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dim: int = 128):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        # Add 2-gram word pairs for richer semantic overlap
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams

    def _hash_token(self, token: str) -> int:
        # Simple fast polynomial hash
        h = 5381
        for ch in token:
            h = ((h << 5) + h) + ord(ch)
        return abs(h) % self._dim

    def _compute_vector(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        vec = [0.0] * self._dim
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        for t, count in tf.items():
            idx = self._hash_token(t)
            # Sublinear term frequency
            weight = 1.0 + math.log(count)
            vec[idx] += weight

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    def embed_query(self, query: str) -> List[float]:
        return self._compute_vector(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._compute_vector(t) for t in texts]
