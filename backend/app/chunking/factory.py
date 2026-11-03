"""
Factory for instantiating chunkers.
"""
from backend.app.chunking.base import BaseChunker
from backend.app.chunking.fixed_chunker import FixedSizeChunker
from backend.app.chunking.recursive_chunker import RecursiveCharacterChunker
from backend.app.chunking.semantic_chunker import SemanticChunker

class ChunkerFactory:
    @staticmethod
    def get_chunker(strategy: str = "recursive", chunk_size: int = 400, chunk_overlap: int = 50) -> BaseChunker:
        strat = strategy.lower().strip()
        if strat in ["fixed", "fixed_size"]:
            return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strat in ["semantic", "semantic_boundary"]:
            return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
