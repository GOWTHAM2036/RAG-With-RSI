"""
Recursive character chunker that splits hierarchically on paragraphs, sentences, and words.
"""
from typing import List
from backend.app.chunking.base import BaseChunker

class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, separators: List[str] = None):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Fallback to hard slice
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character level fallback
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]

        splits = text.split(sep)
        good_chunks = []
        current_chunk = []
        current_length = 0

        for part in splits:
            part_len = len(part) + len(sep)
            if current_length + part_len <= self.chunk_size:
                current_chunk.append(part)
                current_length += part_len
            else:
                if current_chunk:
                    joined = sep.join(current_chunk)
                    if len(joined) > self.chunk_size:
                        sub_chunks = self._recursive_split(joined, remaining_seps)
                        good_chunks.extend(sub_chunks)
                    else:
                        good_chunks.append(joined)
                    
                    # Compute overlap if possible
                    overlap_accum = []
                    overlap_len = 0
                    for prev_part in reversed(current_chunk):
                        if overlap_len + len(prev_part) <= self.chunk_overlap:
                            overlap_accum.insert(0, prev_part)
                            overlap_len += len(prev_part)
                        else:
                            break
                    current_chunk = overlap_accum
                    current_length = sum(len(p) + len(sep) for p in current_chunk)
                
                # If a single part exceeds chunk size, recursively split it
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_seps)
                    good_chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk.append(part)
                    current_length += part_len

        if current_chunk:
            joined = sep.join(current_chunk)
            if len(joined) > self.chunk_size:
                sub_chunks = self._recursive_split(joined, remaining_seps)
                good_chunks.extend(sub_chunks)
            elif joined.strip():
                good_chunks.append(joined)

        return [c.strip() for c in good_chunks if c.strip()]
