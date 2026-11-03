"""
LLM Generation Providers for EvoRAG:
- HeuristicGenerator (Fast, zero-cost, grounded citation synthesizer for offline/local)
- LocalLLMGenerator (Ollama, LM Studio, vLLM, OpenAI-compatible local endpoints with live logs)
- OpenAIGenerator (OpenAI GPT-4o / GPT-4o-mini / GPT-3.5)
- AnthropicGenerator (Claude 3.5 Sonnet / Haiku)
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import re
import time
import httpx
from backend.app.models import RetrievedChunk
from backend.app.generation.base import BaseGenerator
from backend.app.generation.prompts import format_context_chunks, get_prompt_by_template_name

class HeuristicGenerator(BaseGenerator):
    """
    Extractive-abstractive synthesizer that extracts the most pertinent sentences
    from the retrieved chunks and composes a coherent answer with grounded citations.
    Ensures 100% offline, reproducible, zero-API-cost RAG execution.
    """
    def __init__(self):
        self.last_debug_info = {"provider": "heuristic", "status": "online"}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        prompt_template: str = "citation_strict",
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.last_debug_info = {
            "provider": "heuristic",
            "context_chunks_count": len(context_chunks),
            "status": "processed"
        }

        if not context_chunks:
            return "No relevant context found to answer the query.", []

        q_words = set(self._tokenize(query))
        stop_words = {"what", "is", "the", "how", "are", "of", "and", "in", "to", "a", "for", "with", "does", "can", "why"}
        key_q_words = q_words - stop_words
        if not key_q_words:
            key_q_words = q_words

        extracted_sentences = []
        citations = []

        for idx, r_chunk in enumerate(context_chunks):
            content = r_chunk.chunk.content
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", content)
            best_sentence = ""
            best_score = -1

            for s in sentences:
                s_words = set(self._tokenize(s))
                overlap = len(key_q_words.intersection(s_words))
                if overlap > best_score and len(s.strip()) > 15:
                    best_score = overlap
                    best_sentence = s.strip()

            if best_sentence:
                citation_num = idx + 1
                extracted_sentences.append(f"{best_sentence} [{citation_num}]")
                citations.append({
                    "citation_id": f"[{citation_num}]",
                    "doc_title": r_chunk.chunk.metadata.get("doc_title", f"Doc {citation_num}"),
                    "chunk_id": r_chunk.chunk.id,
                    "score": round(r_chunk.score, 4),
                    "snippet": best_sentence[:120] + "..." if len(best_sentence) > 120 else best_sentence
                })

        if not extracted_sentences:
            first_chunk = context_chunks[0]
            answer = f"Based on the available documentation: {first_chunk.chunk.content[:200]}... [1]"
            citations.append({
                "citation_id": "[1]",
                "doc_title": first_chunk.chunk.metadata.get("doc_title", "Document 1"),
                "chunk_id": first_chunk.chunk.id,
                "score": round(first_chunk.score, 4),
                "snippet": first_chunk.chunk.content[:120]
            })
            return answer, citations

        # Format depending on template
        if prompt_template in ["cot", "chain_of_thought"]:
            answer = (
                f"Analysis of retrieved sources indicates multiple supporting facts.\n\n"
                f"{' '.join(extracted_sentences)}"
            )
        elif prompt_template in ["concise", "brief"]:
            answer = extracted_sentences[0]
            citations = citations[:1]
        else:
            answer = " ".join(extracted_sentences)

        return answer, citations


class LocalLLMGenerator(BaseGenerator):
    """
    Local LLM generator supporting Ollama, LM Studio, vLLM, and OpenAI-compatible local endpoints.
    Auto-detects live host port and model, and provides full execution debug logs.
    """
    def __init__(self, base_url: str = None, model: str = None):
        from backend.app.config import settings
        self.base_url = base_url or settings.LOCAL_LLM_URL
        self.model = model or settings.LOCAL_LLM_MODEL
        self._fallback = HeuristicGenerator()
        self.last_debug_info: Dict[str, Any] = {}

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        prompt_template: str = "citation_strict",
        temperature: float = 0.2,
        max_tokens: int = 512,
        model_override: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        t0 = time.time()
        active_model = model_override or self.model

        # Build prompt
        if context_chunks:
            formatted_context = format_context_chunks(context_chunks)
            template_str = get_prompt_by_template_name(prompt_template)
            prompt = template_str.format(context=formatted_context, query=query)
        else:
            # Handle zero context gracefully
            prompt = (
                f"You are EvoRAG AI Assistant. No specific matching context was found in the uploaded knowledge base.\n"
                f"Answer the user's question clearly, and inform them that no local documents matched this query.\n\n"
                f"User Question: {query}\n"
                f"Answer:"
            )

        base = self.base_url.rstrip("/")
        endpoints_to_try = [
            (
                f"{base}/api/generate",
                {
                    "model": active_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                },
                "ollama_format"
            ),
            (
                f"{base}/v1/chat/completions",
                {
                    "model": active_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "openai_format"
            )
        ]

        errors = []
        with httpx.Client(timeout=60.0) as client:
            for url, payload, fmt in endpoints_to_try:
                try:
                    resp = client.post(url, json=payload)
                    latency = round((time.time() - t0) * 1000.0, 1)

                    if resp.status_code == 200:
                        data = resp.json()
                        if fmt == "openai_format":
                            text = data["choices"][0]["message"]["content"]
                        else:
                            text = data.get("response", "")

                        if text and text.strip():
                            # Extract bracketed citations [1], [2], etc.
                            citations = []
                            for idx, c in enumerate(context_chunks):
                                if f"[{idx + 1}]" in text:
                                    citations.append({
                                        "citation_id": f"[{idx + 1}]",
                                        "doc_title": c.chunk.metadata.get("doc_title", f"Doc {idx + 1}"),
                                        "chunk_id": c.chunk.id,
                                        "score": round(c.score, 4),
                                        "snippet": c.chunk.content[:100] + "..."
                                    })

                            self.last_debug_info = {
                                "status": "success",
                                "provider": "local_ollama",
                                "base_url": self.base_url,
                                "endpoint_used": url,
                                "model_used": active_model,
                                "latency_ms": latency,
                                "prompt_tokens_est": len(prompt.split()),
                                "prompt_preview": prompt[:400] + ("..." if len(prompt) > 400 else ""),
                                "raw_output": text[:400] + ("..." if len(text) > 400 else ""),
                                "citations_extracted": len(citations)
                            }
                            return text, citations
                    else:
                        errors.append(f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as err:
                    errors.append(f"{url} connection error: {str(err)}")

        # If all local calls failed, log error and fall back to heuristic
        latency = round((time.time() - t0) * 1000.0, 1)
        self.last_debug_info = {
            "status": "connection_failed",
            "provider": "local_ollama",
            "base_url": self.base_url,
            "model_attempted": active_model,
            "latency_ms": latency,
            "errors": errors,
            "fallback_used": "heuristic"
        }

        fallback_ans, fallback_cits = self._fallback.generate(query, context_chunks, prompt_template, temperature, max_tokens)
        if not context_chunks:
            fallback_ans = (
                f"⚠️ **Local LLM Notice**: No context documents found in the knowledge base, "
                f"and could not connect to Local LLM at `{self.base_url}` ({', '.join(errors[:1])}).\n\n"
                f"Please load documents or verify that Ollama is running (`ollama run {active_model}`)."
            )
        return fallback_ans, fallback_cits


class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._fallback = HeuristicGenerator()
        self.last_debug_info = {"provider": "openai", "model": self.model}

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        prompt_template: str = "citation_strict",
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.api_key:
            return self._fallback.generate(query, context_chunks, prompt_template, temperature, max_tokens)

        formatted_context = format_context_chunks(context_chunks)
        template_str = get_prompt_by_template_name(prompt_template)
        prompt = template_str.format(context=formatted_context, query=query)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            with httpx.Client(timeout=30.0) as client:
                resp = client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    citations = []
                    for idx, c in enumerate(context_chunks):
                        if f"[{idx + 1}]" in text:
                            citations.append({
                                "citation_id": f"[{idx + 1}]",
                                "doc_title": c.chunk.metadata.get("doc_title", f"Doc {idx + 1}"),
                                "chunk_id": c.chunk.id,
                                "score": round(c.score, 4),
                                "snippet": c.chunk.content[:100] + "..."
                            })
                    self.last_debug_info = {"status": "success", "provider": "openai", "model": self.model}
                    return text, citations
        except Exception as e:
            self.last_debug_info = {"status": "error", "provider": "openai", "error": str(e)}

        return self._fallback.generate(query, context_chunks, prompt_template, temperature, max_tokens)


class AnthropicGenerator(BaseGenerator):
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._fallback = HeuristicGenerator()
        self.last_debug_info = {"provider": "anthropic", "model": self.model}

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        prompt_template: str = "citation_strict",
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.api_key:
            return self._fallback.generate(query, context_chunks, prompt_template, temperature, max_tokens)

        formatted_context = format_context_chunks(context_chunks)
        template_str = get_prompt_by_template_name(prompt_template)
        prompt = template_str.format(context=formatted_context, query=query)

        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            with httpx.Client(timeout=30.0) as client:
                resp = client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
                if resp.status_code == 200:
                    text = resp.json()["content"][0]["text"]
                    citations = []
                    for idx, c in enumerate(context_chunks):
                        if f"[{idx + 1}]" in text:
                            citations.append({
                                "citation_id": f"[{idx + 1}]",
                                "doc_title": c.chunk.metadata.get("doc_title", f"Doc {idx + 1}"),
                                "chunk_id": c.chunk.id,
                                "score": round(c.score, 4),
                                "snippet": c.chunk.content[:100] + "..."
                            })
                    self.last_debug_info = {"status": "success", "provider": "anthropic", "model": self.model}
                    return text, citations
        except Exception as e:
            self.last_debug_info = {"status": "error", "provider": "anthropic", "error": str(e)}

        return self._fallback.generate(query, context_chunks, prompt_template, temperature, max_tokens)


class GeneratorFactory:
    _instances = {}

    @classmethod
    def get_generator(cls, provider: str = "local", model: str = None) -> BaseGenerator:
        p = provider.lower().strip()
        key = f"{p}_{model}" if model else p
        if key not in cls._instances:
            if p in ["openai", "gpt-4o", "gpt-4o-mini"]:
                cls._instances[key] = OpenAIGenerator(model=model or "gpt-4o-mini")
            elif p in ["anthropic", "claude"]:
                cls._instances[key] = AnthropicGenerator(model=model or "claude-3-haiku-20240307")
            elif p in ["local", "ollama", "lmstudio", "vllm", "llama", "llama3", "mistral", "deepseek", "qwen", "gemma"]:
                from backend.app.config import settings
                cls._instances[key] = LocalLLMGenerator(
                    base_url=settings.LOCAL_LLM_URL,
                    model=model or settings.LOCAL_LLM_MODEL
                )
            else:
                cls._instances[key] = HeuristicGenerator()
        return cls._instances[key]
