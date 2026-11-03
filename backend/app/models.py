"""
Data models and schemas for EvoRAG.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Untitled"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str
    index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    retrieval_source: str = "vector"  # vector, bm25, hybrid, reranked
    rank: int = 0

class RAGStrategy(BaseModel):
    id: str = Field(default_factory=lambda: f"strat-{str(uuid.uuid4())[:8]}")
    name: str = "Default Strategy"
    chunk_size: int = 400
    chunk_overlap: int = 50
    chunking_strategy: str = "recursive"  # fixed, recursive, semantic
    embedding_provider: str = "tfidf_dense"  # tfidf_dense, openai, huggingface
    retrieval_strategy: str = "hybrid"  # dense, bm25, hybrid
    hybrid_alpha: float = 0.6  # 1.0 = 100% dense, 0.0 = 100% bm25
    top_k: int = 5
    reranker_enabled: bool = True
    reranker_type: str = "heuristic"  # heuristic, cross_encoder, cohere
    rerank_top_n: int = 3
    llm_provider: str = "heuristic"  # heuristic, local, openai, anthropic
    prompt_template: str = "citation_strict"  # concise, cot, citation_strict
    temperature: float = 0.2
    max_tokens: int = 512
    generation: int = 0  # Genetic algorithm generation
    fitness: Optional[float] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str
    strategy: Optional[RAGStrategy] = None
    session_id: Optional[str] = None
    use_agent: bool = False
    filter_metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    strategy_used: RAGStrategy
    latency_ms: float = 0.0
    agent_steps: Optional[List[Dict[str, Any]]] = None
    faithfulness_score: Optional[float] = None
    evaluation: Optional[Dict[str, float]] = None
    llm_debug_info: Optional[Dict[str, Any]] = None

class BenchmarkPair(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    ground_truth_contexts: List[str] = Field(default_factory=list)
    ground_truth_answer: str
    category: str = "general"

class BenchmarkDataset(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Standard QA Benchmark"
    description: str = ""
    pairs: List[BenchmarkPair] = Field(default_factory=list)

class RetrievalMetrics(BaseModel):
    hit_rate: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    ndcg: float = 0.0

class AnswerMetrics(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    semantic_similarity: float = 0.0
    hallucination_rate: float = 0.0

class EvaluationReport(BaseModel):
    strategy_id: str
    strategy_name: str
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    composite_fitness: float
    avg_latency_ms: float
    total_queries: int
    failure_analysis: Dict[str, Any] = Field(default_factory=dict)

class ExperimentConfig(BaseModel):
    name: str = "Optimization Run"
    population_size: int = 8
    num_generations: int = 4
    mutation_rate: float = 0.25
    crossover_rate: float = 0.7
    elite_ratio: float = 0.25
    benchmark_dataset_id: Optional[str] = None
    weight_faithfulness: float = 0.35
    weight_relevance: float = 0.25
    weight_hit_rate: float = 0.20
    weight_mrr: float = 0.15
    weight_latency_penalty: float = 0.05

class GenerationStats(BaseModel):
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    champion_strategy: RAGStrategy
    population: List[RAGStrategy]

class OptimizationRunSummary(BaseModel):
    id: str
    name: str
    status: str  # running, completed, failed
    generations_completed: int
    total_generations: int
    champion_strategy: RAGStrategy
    generation_history: List[GenerationStats]
    pareto_frontier: List[RAGStrategy]
    start_time: float
    end_time: Optional[float] = None
