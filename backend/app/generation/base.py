"""
Base generator interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from backend.app.models import RetrievedChunk

class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        prompt_template: str = "citation_strict",
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate answer from query and context chunks.
        Returns:
            Tuple of (answer_text, citations_list)
        """
        pass
