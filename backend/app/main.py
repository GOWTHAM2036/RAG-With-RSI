"""
EvoRAG FastAPI Main Application.
Provides RESTful APIs for Document Ingestion, Retrieval, Evaluation, Optimization, and Agentic RAG.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import time
import os
import shutil
import json
from pathlib import Path

from backend.app.config import settings
from backend.app.models import (
    Document,
    Chunk,
    RAGStrategy,
    QueryRequest,
    QueryResponse,
    BenchmarkDataset,
    BenchmarkPair,
    EvaluationReport,
    ExperimentConfig,
    OptimizationRunSummary
)
from backend.app.ingestion.loader import DocumentLoader
from backend.app.chunking.factory import ChunkerFactory
from backend.app.embeddings.factory import EmbeddingFactory
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.reranking.factory import RerankerFactory
from backend.app.generation.llm_providers import GeneratorFactory
from backend.app.evaluation.benchmarks import BenchmarkRunner
from backend.app.optimizer.strategy_generator import StrategyGenerator
from backend.app.optimizer.experiment_runner import EvolutionaryExperimentRunner
from backend.app.optimizer.selector import StrategySelector
from backend.app.agents.rag_graph import RAGAgentGraph
from backend.app.memory.session_memory import SessionMemoryManager

# Initialize FastAPI App
app = FastAPI(
    title="EvoRAG - Evolutionary & Self-Optimizing RAG API",
    description="Adaptive, Self-Optimizing Retrieval-Augmented Generation Engine with Genetic Algorithm Tuning.",
    version="1.0.0"
)

# CORS middleware for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory State
class SystemState:
    def __init__(self):
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.champion_strategy: RAGStrategy = StrategyGenerator.heuristic_seed_strategies()[0]
        self.retriever: HybridRetriever = HybridRetriever(
            embedding_provider=EmbeddingFactory.get_provider("tfidf_dense"),
            alpha=self.champion_strategy.hybrid_alpha
        )
        self.memory_manager = SessionMemoryManager()
        self.active_runs: Dict[str, OptimizationRunSummary] = {}
        self.agent_graph = RAGAgentGraph(self.retriever, self.champion_strategy)

    def reindex(self):
        strat = self.champion_strategy
        chunker = ChunkerFactory.get_chunker(
            strategy=strat.chunking_strategy,
            chunk_size=strat.chunk_size,
            chunk_overlap=strat.chunk_overlap
        )
        self.chunks.clear()
        for doc in self.documents:
            self.chunks.extend(chunker.chunk_document(doc))

        self.retriever.alpha = strat.hybrid_alpha
        self.retriever.index_chunks(self.chunks)
        self.agent_graph = RAGAgentGraph(self.retriever, self.champion_strategy)

state = SystemState()

# Preload built-in sample data and benchmark on startup
def load_default_knowledge_and_benchmarks():
    sample_dir = settings.DATA_DIR / "sample_docs"
    if sample_dir.exists():
        for f in sample_dir.glob("*.md"):
            try:
                doc = DocumentLoader.load_from_file(f)
                if not any(d.title == doc.title for d in state.documents):
                    state.documents.append(doc)
            except Exception:
                pass
        state.reindex()

@app.on_event("startup")
def startup_event():
    load_default_knowledge_and_benchmarks()

# ----------------- System & Health Endpoints -----------------

@app.get("/")
def read_root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "status": "online"
    }

@app.get("/api/status")
def get_system_status():
    return {
        "status": "healthy",
        "documents_count": len(state.documents),
        "chunks_count": len(state.chunks),
        "champion_strategy": state.champion_strategy,
        "retrieval_strategy": state.champion_strategy.retrieval_strategy,
        "active_runs_count": len(state.active_runs)
    }

@app.get("/api/llm/status")
def get_llm_status():
    from backend.app.config import auto_detect_local_llm
    url, def_m, avail_m = auto_detect_local_llm()
    return {
        "status": "online" if avail_m else "offline",
        "url": url,
        "active_model": settings.LOCAL_LLM_MODEL,
        "available_models": avail_m,
        "provider": "ollama" if ("11550" in url or "11434" in url) else "local"
    }

@app.post("/api/ingest/samples")
def ingest_samples():
    sample_dir = settings.DATA_DIR / "sample_docs"
    loaded = 0
    if sample_dir.exists():
        for f in sample_dir.glob("*.md"):
            try:
                doc = DocumentLoader.load_from_file(f)
                if not any(d.title == doc.title for d in state.documents):
                    state.documents.append(doc)
                    loaded += 1
            except Exception:
                pass
    state.reindex()
    return {
        "message": f"Successfully ingested {loaded} sample documents.",
        "documents_count": len(state.documents),
        "chunks_count": len(state.chunks)
    }

@app.post("/api/ingest/rsi_dataset")
def ingest_rsi_dataset(payload: Optional[Dict[str, Any]] = None):
    limit = (payload or {}).get("limit", 25)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "rsi_rag" / "data" / "processed" / "combined.jsonl",
        Path(__file__).resolve().parent.parent.parent / "rsi_rag" / "data" / "processed" / "hotpotqa.jsonl",
        Path("C:/Users/B Gowtham Reddy/RSI-RAG/data/processed/combined.jsonl"),
        Path("C:/Users/B Gowtham Reddy/RSI-RAG/data/processed/hotpotqa.jsonl"),
    ]
    rsi_processed = next((p for p in candidates if p.exists()), None)

    if not rsi_processed or not rsi_processed.exists():
        raise HTTPException(status_code=404, detail="RSI-RAG preprocessed dataset not found.")

    loaded = 0
    with open(rsi_processed, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            record = json.loads(line)
            title = f"{record['source'].upper()}: {record['question'][:45]}..."
            content = "\n\n".join(record.get("context", []))
            if content.strip():
                doc = DocumentLoader.load_from_text(text=content, title=title, metadata=record.get("metadata"))
                state.documents.append(doc)
                loaded += 1

    state.reindex()
    return {
        "message": f"Loaded {loaded} records from RSI-RAG dataset.",
        "documents_count": len(state.documents),
        "chunks_count": len(state.chunks)
    }

# ----------------- Document Ingestion Endpoints -----------------

@app.post("/api/ingest/text", response_model=Document)
def ingest_text(payload: Dict[str, Any]):
    text = payload.get("text", "")
    title = payload.get("title", "Direct Text Input")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    doc = DocumentLoader.load_from_text(text=text, title=title)
    state.documents.append(doc)
    state.reindex()
    return doc

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    dest_path = settings.STORAGE_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc = DocumentLoader.load_from_file(dest_path)
    state.documents.append(doc)
    state.reindex()
    return {
        "message": f"Successfully ingested '{file.filename}'",
        "document_id": doc.id,
        "chunks_generated": len(state.chunks),
        "word_count": doc.metadata.get("word_count", 0)
    }

@app.get("/api/documents")
def list_documents():
    return [
        {
            "id": d.id,
            "title": d.title,
            "char_count": d.metadata.get("char_count", len(d.content)),
            "word_count": d.metadata.get("word_count", len(d.content.split())),
            "created_at": d.created_at,
            "source": d.metadata.get("source", "user_input")
        }
        for d in state.documents
    ]

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    state.documents = [d for d in state.documents if d.id != doc_id]
    state.reindex()
    return {"message": f"Document {doc_id} deleted and index refreshed."}

@app.delete("/api/documents")
def clear_all_documents():
    state.documents.clear()
    state.chunks.clear()
    state.retriever.clear()
    return {"message": "All documents and vector indices cleared."}

# ----------------- Query & Agent Endpoints -----------------

@app.post("/api/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    t0 = time.time()
    strat = request.strategy or state.champion_strategy

    if request.use_agent:
        response = state.agent_graph.run(request.query, strategy=strat, documents=state.documents)
        # Store in session memory if requested
        if request.session_id:
            session = state.memory_manager.get_session(request.session_id)
            session.add_user_message(request.query)
            session.add_assistant_message(response.answer, response.citations)
        return response

    # Standard RAG Execution
    # 1. Retrieve
    raw_retrieved = state.retriever.retrieve(
        query=request.query,
        top_k=strat.top_k,
        filter_metadata=request.filter_metadata
    )

    # 2. Rerank
    if strat.reranker_enabled and raw_retrieved:
        reranker = RerankerFactory.get_reranker(strat.reranker_type)
        final_retrieved = reranker.rerank(request.query, raw_retrieved, top_n=strat.rerank_top_n)
    else:
        final_retrieved = raw_retrieved[:strat.top_k]

    # 3. Generate
    generator = GeneratorFactory.get_generator(getattr(strat, "llm_provider", "local"))
    answer, citations = generator.generate(
        query=request.query,
        context_chunks=final_retrieved,
        prompt_template=strat.prompt_template,
        temperature=strat.temperature,
        max_tokens=strat.max_tokens
    )

    latency_ms = (time.time() - t0) * 1000.0

    response = QueryResponse(
        query=request.query,
        answer=answer,
        retrieved_chunks=final_retrieved,
        citations=citations,
        strategy_used=strat,
        latency_ms=round(latency_ms, 2),
        llm_debug_info=getattr(generator, "last_debug_info", None)
    )

    if request.session_id:
        session = state.memory_manager.get_session(request.session_id)
        session.add_user_message(request.query)
        session.add_assistant_message(response.answer, response.citations)

    return response

@app.post("/api/agent/query", response_model=QueryResponse)
def query_agentic(request: QueryRequest):
    request.use_agent = True
    return query_rag(request)

@app.get("/api/chat/history/{session_id}")
def get_chat_history(session_id: str):
    session = state.memory_manager.get_session(session_id)
    return {"session_id": session_id, "messages": session.get_messages()}

# ----------------- Evaluation & Benchmark Endpoints -----------------

@app.post("/api/evaluate", response_model=EvaluationReport)
def evaluate_strategy(payload: Dict[str, Any]):
    strategy_data = payload.get("strategy")
    strat = RAGStrategy(**strategy_data) if strategy_data else state.champion_strategy

    if not state.documents:
        raise HTTPException(status_code=400, detail="Cannot run benchmark without ingested documents.")

    benchmark_data = payload.get("benchmark")
    if benchmark_data:
        benchmark = BenchmarkDataset(**benchmark_data)
    else:
        # Generate default benchmark from ingested documents
        benchmark = get_builtin_benchmark()

    report = BenchmarkRunner.run_strategy_benchmark(
        strategy=strat,
        documents=state.documents,
        benchmark=benchmark
    )
    return report

# ----------------- Evolutionary Optimization Endpoints -----------------

@app.post("/api/optimizer/run", response_model=OptimizationRunSummary)
def run_optimization(config: ExperimentConfig):
    if not state.documents:
        raise HTTPException(status_code=400, detail="Ingest at least one document before launching optimization.")

    benchmark = get_builtin_benchmark()
    runner = EvolutionaryExperimentRunner(config=config)
    summary = runner.run_evolution(documents=state.documents, benchmark=benchmark)

    # Automatically set newly discovered champion as active champion
    state.champion_strategy = summary.champion_strategy
    state.reindex()
    state.active_runs[summary.id] = summary

    return summary

@app.get("/api/optimizer/champion", response_model=RAGStrategy)
def get_champion():
    return state.champion_strategy

@app.post("/api/optimizer/champion", response_model=RAGStrategy)
def set_champion(strategy: RAGStrategy):
    state.champion_strategy = strategy
    state.reindex()
    return state.champion_strategy

@app.get("/api/optimizer/runs")
def list_optimization_runs():
    runs = []
    # Read from storage dir
    for f in settings.EXPERIMENTS_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as json_f:
                runs.append(json.load(json_f))
        except Exception:
            pass
    return runs

@app.get("/api/optimizer/runs/{run_id}")
def get_run_details(run_id: str):
    file_path = settings.EXPERIMENTS_DIR / f"{run_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Helper function for benchmark dataset
def get_builtin_benchmark() -> BenchmarkDataset:
    benchmark_file = settings.BASE_DIR.parent / "benchmarks" / "general_qa.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return BenchmarkDataset(**data)

    # Synthetic fallback benchmark
    return BenchmarkDataset(
        name="Adaptive RAG Core Benchmark",
        description="Comprehensive evaluation testbed across core factual recall and cross-chunk synthesis.",
        pairs=[
            BenchmarkPair(
                query="What is Retrieval-Augmented Generation?",
                ground_truth_contexts=["Retrieval-Augmented Generation blends neural generation with dense/sparse document retrieval."],
                ground_truth_answer="Retrieval-Augmented Generation (RAG) is an AI architecture that enhances language models by retrieving relevant external knowledge chunks.",
                category="definition"
            ),
            BenchmarkPair(
                query="Why is Reciprocal Rank Fusion useful in hybrid search?",
                ground_truth_contexts=["Reciprocal Rank Fusion (RRF) normalizes disparate score distributions between vector and lexical retrievers."],
                ground_truth_answer="RRF combines sparse BM25 and dense vector rankings seamlessly without requiring manual score calibration.",
                category="mechanism"
            ),
            BenchmarkPair(
                query="How does evolutionary optimization improve RAG pipelines?",
                ground_truth_contexts=["Evolutionary algorithms optimize hyperparameter combinations such as chunk size, overlap, top-k, and hybrid alpha."],
                ground_truth_answer="Evolutionary algorithms iteratively mutate and select the best RAG hyperparameters against benchmark fitness metrics.",
                category="optimization"
            )
        ]
    )
