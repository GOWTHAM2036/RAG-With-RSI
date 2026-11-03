"""
Base chunker interface.
"""
from abc import ABC, abstractmethod
from typing import List
from backend.app.models import Document, Chunk

class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into raw string chunks."""
        pass

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunk objects with proper metadata."""
        raw_chunks = self.split_text(document.content)
        chunks = []
        for idx, text in enumerate(raw_chunks):
            if not text.strip():
                continue
            chunk_meta = dict(document.metadata)
            chunk_meta.update({
                "doc_title": document.title,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "char_length": len(text),
                "token_estimate": len(text.split())
            })
            chunks.append(
                Chunk(
                    doc_id=document.id,
                    content=text.strip(),
                    index=idx,
                    metadata=chunk_meta,
                    token_count=len(text.split())
                )
            )
        return chunks
