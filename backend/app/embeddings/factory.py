"""
Embedding provider factory.
"""
from backend.app.embeddings.base import BaseEmbeddingProvider
from backend.app.embeddings.dense_tfidf_provider import DenseTFIDFEmbeddingProvider
from backend.app.embeddings.openai_provider import OpenAIEmbeddingProvider
from backend.app.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider

class EmbeddingFactory:
    _instances = {}

    @classmethod
    def get_provider(cls, provider_type: str = "tfidf_dense") -> BaseEmbeddingProvider:
        ptype = provider_type.lower().strip()
        if ptype not in cls._instances:
            if ptype in ["openai", "text-embedding-3-small", "text-embedding-ada-002"]:
                cls._instances[ptype] = OpenAIEmbeddingProvider()
            elif ptype in ["huggingface", "sentence_transformers", "minilm"]:
                cls._instances[ptype] = HuggingFaceEmbeddingProvider()
            else:
                cls._instances[ptype] = DenseTFIDFEmbeddingProvider(dim=128)
        return cls._instances[ptype]
