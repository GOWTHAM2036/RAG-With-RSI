"""
Configuration management for EvoRAG backend with automatic Local LLM auto-detection
and token pricing / compute cost accounting.
"""
import os
import httpx
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Tuple, List

def auto_detect_local_llm() -> Tuple[str, str, List[str]]:
    """
    Probes localhost to discover running Ollama (checking OLLAMA_HOST, port 11550, port 11434)
    or LM Studio (port 1234), and returns (base_url, default_model, list_of_models).
    """
    candidate_urls = []
    
    # 1. Check OLLAMA_HOST environment variable
    ollama_host = os.getenv("OLLAMA_HOST", "")
    if ollama_host:
        clean_host = ollama_host if ollama_host.startswith("http") else f"http://{ollama_host}"
        candidate_urls.append(clean_host)

    # 2. Add common local ports
    candidate_urls.extend([
        "http://127.0.0.1:11550",
        "http://localhost:11550",
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://127.0.0.1:1234",
        "http://localhost:1234",
    ])

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = [u for u in candidate_urls if not (u in seen or seen.add(u))]

    for url in unique_candidates:
        try:
            # Check Ollama tags
            r = httpx.get(f"{url}/api/tags", timeout=1.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                default_m = models[0] if models else "gemma3:4b"
                return url, default_m, models
        except Exception:
            pass

        try:
            # Check OpenAI-compatible /v1/models (LM Studio / vLLM)
            r = httpx.get(f"{url}/v1/models", timeout=1.0)
            if r.status_code == 200:
                data = r.json().get("data", [])
                models = [m.get("id") for m in data if m.get("id")]
                default_m = models[0] if models else "local-model"
                return url, default_m, models
        except Exception:
            pass

    return "http://localhost:11434", "gemma3:4b", []

_detected_url, _detected_model, _available_models = auto_detect_local_llm()

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="allow")

    APP_NAME: str = "RAG-With-RSI: Evolutionary & Self-Improving Architecture"
    VERSION: str = "1.1.0"
    DEBUG: bool = False
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    EXPERIMENTS_DIR: Path = BASE_DIR / "experiments"
    STORAGE_DIR: Path = DATA_DIR / "storage"
    
    # Provider API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    
    # Default model settings
    DEFAULT_EMBEDDING_PROVIDER: str = "tfidf_dense"  # tfidf_dense, openai, huggingface
    DEFAULT_LLM_PROVIDER: str = "local" if _available_models else "heuristic"
    LOCAL_LLM_URL: str = os.getenv("LOCAL_LLM_URL", _detected_url)
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", _detected_model)
    AVAILABLE_LOCAL_MODELS: List[str] = _available_models

    # Cost accounting: USD per 1M tokens (Standard baseline pricing / compute proxy)
    PRICE_EMBEDDING_PER_1M_TOKENS: float = 0.020    # $0.02 per 1M tokens
    PRICE_INPUT_PER_1M_TOKENS: float = 0.150        # $0.15 per 1M prompt tokens (e.g. gpt-4o-mini / local compute unit)
    PRICE_OUTPUT_PER_1M_TOKENS: float = 0.600       # $0.60 per 1M completion tokens

    # Server host and port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
