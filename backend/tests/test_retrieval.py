"""
Unit tests for retrieval modules (VectorStore, BM25, HybridRetriever).
"""
import pytest
from backend.app.models import Chunk
from backend.app.embeddings.dense_tfidf_provider import DenseTFIDFEmbeddingProvider
from backend.app.retrieval.vector_store import VectorStore
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.reranking.heuristic_reranker import HeuristicReranker

@pytest.fixture
def sample_chunks():
    texts = [
        "Quantum computing relies on quantum bits or qubits to perform calculations.",
        "Convolutional neural networks are specialized for visual imagery analysis.",
        "Reciprocal Rank Fusion combines vector search with BM25 keyword rankings.",
        "Transformer attention mechanisms allow parallel sequence processing."
    ]
    return [
        Chunk(doc_id="doc1", content=t, index=i)
        for i, t in enumerate(texts)
    ]

def test_vector_store(sample_chunks):
    emb = DenseTFIDFEmbeddingProvider(dim=64)
    vs = VectorStore(embedding_provider=emb)
    vs.add_chunks(sample_chunks)

    results = vs.search("quantum qubits calculation", top_k=2)
    assert len(results) == 2
    assert "qubits" in results[0].chunk.content.lower()
    assert results[0].score > 0.0

def test_bm25_retriever(sample_chunks):
    bm25 = BM25Retriever()
    bm25.index_chunks(sample_chunks)

    results = bm25.retrieve("convolutional neural visual imagery", top_k=2)
    assert len(results) >= 1
    assert "neural" in results[0].chunk.content.lower()
    assert results[0].score > 0.0

def test_hybrid_retriever(sample_chunks):
    hybrid = HybridRetriever(alpha=0.5)
    hybrid.index_chunks(sample_chunks)

    results = hybrid.retrieve("Reciprocal Rank Fusion keyword", top_k=3)
    assert len(results) >= 1
    assert "reciprocal" in results[0].chunk.content.lower()

def test_heuristic_reranker(sample_chunks):
    hybrid = HybridRetriever(alpha=0.5)
    hybrid.index_chunks(sample_chunks)
    retrieved = hybrid.retrieve("quantum computing qubits", top_k=3)

    reranker = HeuristicReranker()
    reranked = reranker.rerank("quantum computing qubits", retrieved, top_n=2)
    assert len(reranked) <= 2
    assert "quantum" in reranked[0].chunk.content.lower()
