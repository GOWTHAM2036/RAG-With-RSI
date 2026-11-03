"""
Base retriever interface.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from backend.app.models import Chunk, RetrievedChunk

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        """Retrieve the top-k chunks relevant to query."""
        pass

    @abstractmethod
    def index_chunks(self, chunks: List[Chunk]):
        """Index chunks for retrieval."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all indexed data."""
        pass
