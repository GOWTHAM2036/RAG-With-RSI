"""
OpenAI embedding provider with graceful fallback.
"""
from typing import List
import os
import httpx
from backend.app.embeddings.base import BaseEmbeddingProvider
from backend.app.embeddings.dense_tfidf_provider import DenseTFIDFEmbeddingProvider

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._fallback = DenseTFIDFEmbeddingProvider(dim=128)

    @property
    def dimension(self) -> int:
        if "3-small" in self.model:
            return 1536
        return 128

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_documents([query])
        return res[0] if res else [0.0] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            return self._fallback.embed_documents(texts)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "input": texts,
                "model": self.model
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return [item["embedding"] for item in data.get("data", [])]
        except Exception:
            pass

        # If API call fails or no internet/key, use fallback
        return self._fallback.embed_documents(texts)
