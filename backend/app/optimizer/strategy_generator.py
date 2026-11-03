"""
Genetic Algorithm strategy generator, chromosome mutation, and crossover operations.
"""
from typing import List, Dict, Any, Optional
import random
import copy
import uuid
from backend.app.models import RAGStrategy

class StrategyGenerator:
    # Gene space definitions
    CHUNK_SIZES = [200, 350, 500, 750, 1000]
    CHUNK_OVERLAPS = [20, 50, 80, 120, 200]
    CHUNKING_STRATEGIES = ["recursive", "fixed", "semantic"]
    RETRIEVAL_STRATEGIES = ["hybrid", "dense", "bm25"]
    EMBEDDING_PROVIDERS = ["tfidf_dense"]  # Default fast local embedding; extensible to openai/hf
    RERANKER_TYPES = ["heuristic", "cross_encoder"]
    PROMPT_TEMPLATES = ["citation_strict", "cot", "concise"]
    TEMPERATURES = [0.0, 0.1, 0.2, 0.4, 0.6]

    @classmethod
    def random_strategy(cls, generation: int = 0) -> RAGStrategy:
        chunk_size = random.choice(cls.CHUNK_SIZES)
        # Ensure overlap is strictly less than chunk_size
        valid_overlaps = [o for o in cls.CHUNK_OVERLAPS if o < chunk_size]
        chunk_overlap = random.choice(valid_overlaps) if valid_overlaps else 50
        
        return RAGStrategy(
            id=f"strat-{str(uuid.uuid4())[:8]}",
            name=f"EvoCandidate-G{generation}-{random.randint(100, 999)}",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=random.choice(cls.CHUNKING_STRATEGIES),
            embedding_provider=random.choice(cls.EMBEDDING_PROVIDERS),
            retrieval_strategy=random.choice(cls.RETRIEVAL_STRATEGIES),
            hybrid_alpha=round(random.uniform(0.2, 0.8), 2),
            top_k=random.choice([3, 5, 7, 9]),
            reranker_enabled=random.choice([True, False]),
            reranker_type=random.choice(cls.RERANKER_TYPES),
            rerank_top_n=random.choice([2, 3, 4, 5]),
            prompt_template=random.choice(cls.PROMPT_TEMPLATES),
            temperature=random.choice(cls.TEMPERATURES),
            generation=generation
        )

    @classmethod
    def heuristic_seed_strategies(cls) -> List[RAGStrategy]:
        """Provides high-quality baseline configurations across different paradigms."""
        return [
            # 1. Balanced Hybrid (Industry Standard Champion)
            RAGStrategy(
                id="strat-hybrid-balanced",
                name="Balanced-Hybrid-Baseline",
                chunk_size=400,
                chunk_overlap=60,
                chunking_strategy="recursive",
                embedding_provider="tfidf_dense",
                retrieval_strategy="hybrid",
                hybrid_alpha=0.65,
                top_k=6,
                reranker_enabled=True,
                reranker_type="heuristic",
                rerank_top_n=3,
                prompt_template="citation_strict",
                temperature=0.1,
                generation=0
            ),
            # 2. Semantic Granular Precision
            RAGStrategy(
                id="strat-semantic-precision",
                name="Semantic-Precision",
                chunk_size=300,
                chunk_overlap=40,
                chunking_strategy="semantic",
                embedding_provider="tfidf_dense",
                retrieval_strategy="hybrid",
                hybrid_alpha=0.5,
                top_k=8,
                reranker_enabled=True,
                reranker_type="heuristic",
                rerank_top_n=4,
                prompt_template="cot",
                temperature=0.2,
                generation=0
            ),
            # 3. Dense Macro Context
            RAGStrategy(
                id="strat-dense-macro",
                name="Dense-Macro-Context",
                chunk_size=800,
                chunk_overlap=120,
                chunking_strategy="recursive",
                embedding_provider="tfidf_dense",
                retrieval_strategy="dense",
                hybrid_alpha=1.0,
                top_k=4,
                reranker_enabled=False,
                rerank_top_n=3,
                prompt_template="citation_strict",
                temperature=0.2,
                generation=0
            ),
            # 4. Fast BM25 Keyword Search
            RAGStrategy(
                id="strat-fast-bm25",
                name="Fast-BM25-Keyword",
                chunk_size=350,
                chunk_overlap=30,
                chunking_strategy="fixed",
                embedding_provider="tfidf_dense",
                retrieval_strategy="bm25",
                hybrid_alpha=0.0,
                top_k=5,
                reranker_enabled=False,
                rerank_top_n=3,
                prompt_template="concise",
                temperature=0.1,
                generation=0
            )
        ]

    @classmethod
    def crossover(cls, parent1: RAGStrategy, parent2: RAGStrategy, generation: int = 0) -> RAGStrategy:
        """Uniform crossover between two parent strategies."""
        child = RAGStrategy(
            id=f"strat-{str(uuid.uuid4())[:8]}",
            name=f"EvoOffspring-G{generation}-{random.randint(100, 999)}",
            chunk_size=random.choice([parent1.chunk_size, parent2.chunk_size]),
            chunk_overlap=random.choice([parent1.chunk_overlap, parent2.chunk_overlap]),
            chunking_strategy=random.choice([parent1.chunking_strategy, parent2.chunking_strategy]),
            embedding_provider=random.choice([parent1.embedding_provider, parent2.embedding_provider]),
            retrieval_strategy=random.choice([parent1.retrieval_strategy, parent2.retrieval_strategy]),
            hybrid_alpha=round(random.choice([parent1.hybrid_alpha, parent2.hybrid_alpha]), 2),
            top_k=random.choice([parent1.top_k, parent2.top_k]),
            reranker_enabled=random.choice([parent1.reranker_enabled, parent2.reranker_enabled]),
            reranker_type=random.choice([parent1.reranker_type, parent2.reranker_type]),
            rerank_top_n=random.choice([parent1.rerank_top_n, parent2.rerank_top_n]),
            prompt_template=random.choice([parent1.prompt_template, parent2.prompt_template]),
            temperature=random.choice([parent1.temperature, parent2.temperature]),
            generation=generation
        )
        if child.chunk_overlap >= child.chunk_size:
            child.chunk_overlap = max(10, child.chunk_size // 4)
        return child

    @classmethod
    def mutate(cls, strategy: RAGStrategy, mutation_rate: float = 0.25) -> RAGStrategy:
        """Mutates genes with a probability equal to mutation_rate."""
        mutated = strategy.model_copy(deep=True)
        mutated.id = f"strat-{str(uuid.uuid4())[:8]}"

        if random.random() < mutation_rate:
            mutated.chunk_size = random.choice(cls.CHUNK_SIZES)
        if random.random() < mutation_rate:
            mutated.chunking_strategy = random.choice(cls.CHUNKING_STRATEGIES)
        if random.random() < mutation_rate:
            mutated.retrieval_strategy = random.choice(cls.RETRIEVAL_STRATEGIES)
        if random.random() < mutation_rate:
            # Gaussian perturbation on hybrid alpha
            jitter = random.gauss(0, 0.15)
            mutated.hybrid_alpha = round(max(0.05, min(0.95, mutated.hybrid_alpha + jitter)), 2)
        if random.random() < mutation_rate:
            mutated.top_k = random.choice([3, 5, 7, 9, 11])
        if random.random() < mutation_rate:
            mutated.reranker_enabled = not mutated.reranker_enabled
        if random.random() < mutation_rate:
            mutated.rerank_top_n = random.choice([2, 3, 4, 5])
        if random.random() < mutation_rate:
            mutated.prompt_template = random.choice(cls.PROMPT_TEMPLATES)
        if random.random() < mutation_rate:
            mutated.temperature = random.choice(cls.TEMPERATURES)

        # Enforce valid overlap
        if mutated.chunk_overlap >= mutated.chunk_size:
            mutated.chunk_overlap = max(10, mutated.chunk_size // 4)

        return mutated
