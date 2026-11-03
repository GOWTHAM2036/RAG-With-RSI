"""
Fixed size chunker with sliding character window.
"""
from typing import List
from backend.app.chunking.base import BaseChunker

class FixedSizeChunker(BaseChunker):
    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= text_len:
                break
            start += step

        return chunks
