"""
Unit tests for RAGAgentGraph and state machine transitions.
"""
import pytest
from backend.app.models import Document, Chunk, RAGStrategy
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.agents.rag_graph import RAGAgentGraph

def test_agent_graph_execution():
    chunks = [
        Chunk(
            doc_id="doc1",
            content="EvoRAG uses autonomous genetic optimization to find champion RAG configurations."
        )
    ]
    retriever = HybridRetriever(alpha=0.6)
    retriever.index_chunks(chunks)

    strategy = RAGStrategy(name="Test Strategy", top_k=3, reranker_enabled=False)
    graph = RAGAgentGraph(retriever, default_strategy=strategy)

    # 1. Test greeting route
    greeting_res = graph.run("Hello there!")
    assert "EvoRAG" in greeting_res.answer
    assert any(step["node"] == "Router" for step in greeting_res.agent_steps)

    # 2. Test knowledge retrieval route
    query_res = graph.run("How does EvoRAG find champion configurations?")
    assert query_res.answer != ""
    assert len(query_res.citations) >= 1
    assert query_res.faithfulness_score > 0.0
    nodes = [s["node"] for s in query_res.agent_steps]
    assert "QueryAnalyzer" in nodes
    assert "DocumentGrader" in nodes
    assert "GroundedGenerator" in nodes
    assert "SelfCorrectionGuard" in nodes
