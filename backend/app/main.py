"""
RAG-With-RSI FastAPI Main Application.
Provides RESTful APIs for Document Ingestion, Hybrid Retrieval, Claim-Level Evaluation,
Pareto Multi-Objective Optimization, and Self-Reflective Agentic RAG.
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
from backend.app.evaluation.answer_eval import AnswerEvaluator
from backend.app.optimizer.strategy_generator import StrategyGenerator
from backend.app.optimizer.experiment_runner import EvolutionaryExperimentRunner
from backend.app.optimizer.selector import StrategySelector
from backend.app.agents.rag_graph import RAGAgentGraph
from backend.app.memory.session_memory import SessionMemoryManager

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_default_knowledge_and_benchmarks()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="RAG-With-RSI - Evolutionary & Self-Improving RAG API",
    description="Adaptive, Self-Optimizing Retrieval-Augmented Generation Engine with Genetic Algorithm Tuning & Claim-Level Evaluation.",
    version="1.1.0",
    lifespan=lifespan
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

# Preload built-in sample data on startup
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

# ----------------- System & Health Endpoints -----------------

@app.get("/")
def read_root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "online",
        "local_llm": {
            "endpoint": settings.LOCAL_LLM_URL,
            "default_model": settings.LOCAL_LLM_MODEL,
            "detected_models": settings.AVAILABLE_LOCAL_MODELS
        },
        "documents_count": len(state.documents),
        "chunks_indexed": len(state.chunks),
        "active_champion": state.champion_strategy.name
    }

@app.get("/api/status")
def get_system_status():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "documents_count": len(state.documents),
        "chunks_count": len(state.chunks),
        "champion_strategy": state.champion_strategy
    }

# ----------------- Document Ingestion Endpoints -----------------

@app.post("/api/ingest/text")
def ingest_raw_text(payload: Dict[str, Any]):
    text = payload.get("text", "")
    title = payload.get("title", "Raw Text Ingestion")
    metadata = payload.get("metadata", {})

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    doc = DocumentLoader.load_from_text(text=text, title=title, metadata=metadata)
    state.documents.append(doc)
    state.reindex()

    return {
        "message": f"Successfully ingested text document '{doc.title}'.",
        "document_id": doc.id,
        "total_documents": len(state.documents),
        "chunks_indexed": len(state.chunks)
    }

@app.post("/api/ingest/file")
async def ingest_uploaded_file(file: UploadFile = File(...)):
    save_path = settings.DATA_DIR / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc = DocumentLoader.load_from_file(save_path)
        state.documents.append(doc)
        state.reindex()

        return {
            "message": f"Successfully ingested file '{file.filename}'.",
            "document_id": doc.id,
            "total_documents": len(state.documents),
            "chunks_indexed": len(state.chunks)
        }
    finally:
        if save_path.exists():
            try:
                os.remove(save_path)
            except Exception:
                pass

@app.post("/api/ingest/samples")
def ingest_sample_documents():
    load_default_knowledge_and_benchmarks()
    return {
        "message": f"Loaded {len(state.documents)} sample knowledge documents.",
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

@app.get("/api/documents", response_model=List[Document])
def list_documents():
    return state.documents

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    initial_count = len(state.documents)
    state.documents = [d for d in state.documents if d.id != doc_id]
    if len(state.documents) == initial_count:
        raise HTTPException(status_code=404, detail="Document not found.")
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
        if request.session_id:
            session = state.memory_manager.get_session(request.session_id)
            session.add_user_message(request.query)
            session.add_assistant_message(response.answer, response.citations)
        return response

    # Standard RAG Execution
    # 1. Retrieve & Rerank (Isolated timing)
    t_ret_start = time.time()
    raw_retrieved = state.retriever.retrieve(
        query=request.query,
        top_k=strat.top_k,
        filter_metadata=request.filter_metadata
    )

    if strat.reranker_enabled and raw_retrieved:
        reranker = RerankerFactory.get_reranker(strat.reranker_type)
        final_retrieved = reranker.rerank(request.query, raw_retrieved, top_n=strat.rerank_top_n)
    else:
        final_retrieved = raw_retrieved[:strat.top_k]
    retrieval_latency_ms = round((time.time() - t_ret_start) * 1000.0, 2)

    # 2. Generate (Isolated timing)
    t_gen_start = time.time()
    generator = GeneratorFactory.get_generator(getattr(strat, "llm_provider", "local"))
    answer, citations = generator.generate(
        query=request.query,
        context_chunks=final_retrieved,
        prompt_template=strat.prompt_template,
        temperature=strat.temperature,
        max_tokens=strat.max_tokens
    )
    generation_latency_ms = round((time.time() - t_gen_start) * 1000.0, 2)
    total_latency_ms = round((time.time() - t0) * 1000.0, 2)

    # 3. Claim-Level Faithfulness Verification
    claim_verification = AnswerEvaluator.compute_claim_faithfulness(answer, final_retrieved)
    faithfulness_score = claim_verification["claim_faithfulness"]

    response = QueryResponse(
        query=request.query,
        answer=answer,
        retrieved_chunks=final_retrieved,
        citations=citations,
        strategy_used=strat,
        latency_ms=total_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        faithfulness_score=faithfulness_score,
        evaluation={
            "faithfulness": faithfulness_score,
            "claim_verification": claim_verification,
            "chunks_used": len(final_retrieved)
        },
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

    split_name = payload.get("split", "validation")
    sample_limit = payload.get("sample_limit", 15)

    benchmark_data = payload.get("benchmark")
    if benchmark_data:
        benchmark = BenchmarkDataset(**benchmark_data)
        docs = state.documents
    else:
        benchmark, docs = BenchmarkRunner.load_rsi_split(split_name, sample_limit=sample_limit)
        if not docs and state.documents:
            docs = state.documents

    if not docs and not state.documents:
        raise HTTPException(status_code=400, detail="Cannot run benchmark without ingested documents or RSI splits.")

    report = BenchmarkRunner.run_strategy_benchmark(
        strategy=strat,
        documents=docs if docs else state.documents,
        benchmark=benchmark,
        generator_provider=payload.get("generator_provider")
    )
    return report

@app.get("/api/benchmarks/splits")
def get_dataset_splits():
    splits = {}
    for split_name in ["train", "validation", "test_isolated", "test_blind"]:
        bench, docs = BenchmarkRunner.load_rsi_split(split_name, sample_limit=50)
        splits[split_name] = {
            "queries_count": len(bench.pairs),
            "documents_count": len(docs),
            "fingerprint": bench.fingerprint or "n/a",
            "description": bench.description
        }
    return splits

# ----------------- Evolutionary Optimization Endpoints -----------------

@app.post("/api/optimizer/run", response_model=OptimizationRunSummary)
def run_optimization(config: ExperimentConfig):
    runner = EvolutionaryExperimentRunner(config=config)
    summary = runner.run_evolution(documents=state.documents)

    # Set validated champion as active pipeline
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

@app.get("/api/optimizer/champions")
def list_registered_champions():
    return EvolutionaryExperimentRunner.load_champions_registry()

@app.get("/api/optimizer/runs")
def list_optimization_runs():
    runs = []
    for f in settings.EXPERIMENTS_DIR.glob("exp-*.json"):
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

def get_builtin_benchmark() -> BenchmarkDataset:
    return BenchmarkRunner.get_builtin_benchmark()
