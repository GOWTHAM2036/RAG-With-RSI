"""
BM25 sparse keyword retriever implementation.
"""
from typing import List, Dict, Any, Optional
import math
import re
from backend.app.models import Chunk, RetrievedChunk
from backend.app.retrieval.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def clear(self):
        self.chunks.clear()
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens.clear()
        self.doc_freqs.clear()
        self.term_freqs.clear()

    def index_chunks(self, chunks: List[Chunk]):
        self.clear()
        self.chunks = list(chunks)
        self.corpus_size = len(chunks)
        if self.corpus_size == 0:
            return

        total_len = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 1.0

    def retrieve(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self.chunks or self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # IDF calculation with Robertson-Spärck Jones formulation
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            idf = max(0.001, idf)

            for i in range(self.corpus_size):
                if filter_metadata:
                    match = True
                    for k, v in filter_metadata.items():
                        if self.chunks[i].metadata.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                tf = self.term_freqs[i].get(token, 0)
                if tf > 0:
                    doc_len = self.doc_lens[i]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[i] += idf * (numerator / denominator)

        # Min-max normalize scores between 0 and 1
        max_score = max(scores) if scores else 0.0
        normalized_scores = [s / max_score if max_score > 0 else 0.0 for s in scores]

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0.0:
                results.append(
                    RetrievedChunk(
                        chunk=self.chunks[idx],
                        score=float(normalized_scores[idx]),
                        retrieval_source="bm25",
                        rank=rank + 1
                    )
                )

        return results
