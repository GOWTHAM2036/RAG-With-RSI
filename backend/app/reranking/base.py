"""
Base reranker interface.
"""
from abc import ABC, abstractmethod
from typing import List
from backend.app.models import RetrievedChunk

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_n: int = 3) -> List[RetrievedChunk]:
        """Rerank chunks for relevance against the query and return the top-n chunks."""
        pass
