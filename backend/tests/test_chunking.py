"""
Unit tests for chunking strategies.
"""
import pytest
from backend.app.models import Document
from backend.app.chunking.fixed_chunker import FixedSizeChunker
from backend.app.chunking.recursive_chunker import RecursiveCharacterChunker
from backend.app.chunking.semantic_chunker import SemanticChunker
from backend.app.chunking.factory import ChunkerFactory

@pytest.fixture
def sample_document():
    content = (
        "# Artificial Intelligence\n\n"
        "Artificial intelligence is transforming modern computer science.\n\n"
        "## Subfields\n\n"
        "Machine learning is a subfield of AI focused on learning patterns from data. "
        "Deep learning uses artificial neural networks with multiple layers to model complex representations. "
        "Natural language processing enables computers to understand human language."
    )
    return Document(title="AI Primer", content=content)

def test_fixed_chunker(sample_document):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 100
        assert c.doc_id == sample_document.id

def test_recursive_chunker(sample_document):
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2
    assert all(c.content for c in chunks)

def test_semantic_chunker(sample_document):
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    assert any("Artificial intelligence" in c.content for c in chunks)

def test_chunker_factory():
    c1 = ChunkerFactory.get_chunker("fixed", 300, 50)
    assert isinstance(c1, FixedSizeChunker)
    c2 = ChunkerFactory.get_chunker("semantic", 300, 50)
    assert isinstance(c2, SemanticChunker)
    c3 = ChunkerFactory.get_chunker("recursive", 300, 50)
    assert isinstance(c3, RecursiveCharacterChunker)
