"""
Base interface for embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single query."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document strings."""
        pass
