"""
Semantic boundary chunker that respects document headers, markdown sections, and logical themes.
"""
import re
from typing import List
from backend.app.chunking.base import BaseChunker

class SemanticChunker(BaseChunker):
    """
    Splits text along semantic section breaks (Markdown headers, double newlines, list items)
    while merging short cohesive paragraphs to reach the target chunk_size.
    """
    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        # Split on markdown headers or double newlines
        sections = re.split(r"(?=(?:\n#{1,6}\s+))|\n\n+", text)
        sections = [s.strip() for s in sections if s and s.strip()]

        chunks = []
        current_chunk = ""

        for sec in sections:
            if not current_chunk:
                current_chunk = sec
            elif len(current_chunk) + len(sec) + 2 <= self.chunk_size:
                current_chunk += "\n\n" + sec
            else:
                chunks.append(current_chunk)
                # If the section alone is larger than chunk_size, split into sub-sentences
                if len(sec) > self.chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", sec)
                    sub = ""
                    for s in sentences:
                        if len(sub) + len(s) + 1 <= self.chunk_size:
                            sub = f"{sub} {s}".strip()
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = s
                    if sub:
                        current_chunk = sub
                    else:
                        current_chunk = ""
                else:
                    # Apply small overlap context from previous chunk if configured
                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        overlap_prefix = current_chunk[-self.chunk_overlap:].strip()
                        current_chunk = f"... {overlap_prefix}\n{sec}"
                    else:
                        current_chunk = sec

        if current_chunk:
            chunks.append(current_chunk)

        return [c.strip() for c in chunks if c.strip()]
