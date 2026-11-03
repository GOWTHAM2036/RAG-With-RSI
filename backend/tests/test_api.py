"""
Integration tests for FastAPI endpoints.
"""
from fastapi.testclient import TestClient
import pytest
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "champion_strategy" in data

def test_text_ingest_and_query():
    # Ingest text
    ingest_resp = client.post(
        "/api/ingest/text",
        json={
            "title": "Quantum Physics Basics",
            "text": "Superposition enables a quantum system to exist in multiple states simultaneously until measured."
        }
    )
    assert ingest_resp.status_code == 200

    # Query with standard RAG
    query_resp = client.post(
        "/api/query",
        json={"query": "What does superposition enable in quantum physics?", "use_agent": False}
    )
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert "superposition" in data["answer"].lower()
    assert len(data["retrieved_chunks"]) > 0

    # Query with agentic RAG
    agent_resp = client.post(
        "/api/agent/query",
        json={"query": "Explain quantum superposition", "use_agent": True}
    )
    assert agent_resp.status_code == 200
    agent_data = agent_resp.json()
    assert agent_data["agent_steps"] is not None
    assert len(agent_data["agent_steps"]) > 0
