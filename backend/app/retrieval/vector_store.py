"""
High performance vector store with cosine similarity, persistence, and metadata filtering.
"""
from typing import List, Dict, Any, Optional
import math
import json
from pathlib import Path
from backend.app.models import Chunk, RetrievedChunk
from backend.app.embeddings.base import BaseEmbeddingProvider
from backend.app.embeddings.factory import EmbeddingFactory

class VectorStore:
    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None, storage_path: Optional[Path] = None):
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider("tfidf_dense")
        self.storage_path = storage_path
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def clear(self):
        self.chunks.clear()
        self.vectors.clear()

    def add_chunks(self, new_chunks: List[Chunk]):
        if not new_chunks:
            return

        texts_to_embed = []
        chunk_indices_to_embed = []

        for idx, chunk in enumerate(new_chunks):
            self.chunks.append(chunk)
            if chunk.embedding is not None and len(chunk.embedding) > 0:
                self.vectors.append(chunk.embedding)
            else:
                texts_to_embed.append(chunk.content)
                chunk_indices_to_embed.append(len(self.chunks) - 1)

        if texts_to_embed:
            embs = self.embedding_provider.embed_documents(texts_to_embed)
            for chunk_idx, emb in zip(chunk_indices_to_embed, embs):
                self.chunks[chunk_idx].embedding = emb
                self.vectors.append(emb)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 * norm2 < 1e-9:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self.chunks or not self.vectors:
            return []

        query_vec = self.embedding_provider.embed_query(query)
        scored_results = []

        for i, (chunk, chunk_vec) in enumerate(zip(self.chunks, self.vectors)):
            # Check metadata filter if present
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            sim = self._cosine_similarity(query_vec, chunk_vec)
            scored_results.append((chunk, sim))

        # Sort descending by similarity
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top = scored_results[:top_k]

        return [
            RetrievedChunk(
                chunk=chunk,
                score=float(sim),
                retrieval_source="vector",
                rank=idx + 1
            )
            for idx, (chunk, sim) in enumerate(top)
        ]

    def save_to_disk(self, file_path: Path):
        data = {
            "chunks": [c.model_dump() for c in self.chunks],
            "vectors": self.vectors
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_from_disk(self, file_path: Path):
        if not file_path.exists():
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = [Chunk(**c) for c in data.get("chunks", [])]
            self.vectors = data.get("vectors", [])
