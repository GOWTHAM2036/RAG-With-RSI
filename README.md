# RAG-With-RSI: Evolutionary & Recursive Self-Improving RAG Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAG-With-RSI** is an autonomous, self-optimizing Retrieval-Augmented Generation (RAG) framework paired with a **Recursive Self-Improving (RSI) dataset engineering and evaluation pipeline**. Instead of relying on static, hand-tuned hyperparameter choices (such as arbitrary chunk sizes, fixed top-k, and static similarity metrics), the architecture combines:

1. **EvoRAG**: Treats RAG configurations as evolving chromosomes using **Genetic Algorithms (GA)** to search, mutate, and discover Pareto-optimal champion configurations across multi-objective benchmarks (Composite Accuracy vs Latency vs Cost).
2. **Self-Reflective Agentic State Machine**: Performs real-time query optimization, intent routing, document grading, and hallucination self-correction guardrails with local LLM (Ollama) and cloud model support.
3. **RSI-RAG Pipeline**: Automated dataset downloading, preprocessing, and validation pipeline streaming and standardizing **HotpotQA**, **RAGBench**, and **SciFact** into isolated training, validation, and blind test partitions.

---

## 🌟 Key Features

1. **Evolutionary Optimization Engine**:
   - Chromosome representation: `chunk_size`, `chunk_overlap`, `chunking_strategy`, `retrieval_strategy`, `hybrid_alpha`, `top_k`, `reranker_enabled`, `temperature`.
   - Genetic operators: Tournament selection, uniform crossover, Gaussian mutation, elitism.
   - Multi-objective Pareto frontier selection balancing **Composite Accuracy vs Latency vs Cost**.

2. **Adaptive Ingestion & Hybrid Retrieval**:
   - Multi-format ingestion: PDF, Markdown, JSON, Text, and Web URLs.
   - Dynamic chunkers: Recursive character, fixed-size, and semantic boundary chunking.
   - Hybrid search: Reciprocal Rank Fusion (RRF) blending dense vector semantic embeddings with BM25 Okapi lexical retrieval.
   - Heuristic and cross-encoder rerankers with MMR diversity penalties to prevent redundancy.

3. **Rigorous Evaluation Framework**:
   - **Retrieval Metrics**: Hit Rate @ K, Mean Reciprocal Rank (MRR @ K), Context Precision, Context Recall, NDCG @ K.
   - **Answer Metrics**: Faithfulness (Groundedness), Answer Relevance, Semantic F1 Similarity, Hallucination Rate.
   - Automated failure diagnostics (detects retrieval misses, reranker drops, and low groundedness).

4. **Self-Reflective Agentic Guardrails**:
   - Query rewrite and intent routing.
   - Document grading to filter irrelevant or noisy context.
   - Query expansion fallback loops.
   - Post-generation faithfulness verification with auto-correction for hallucinations.

5. **Full-Stack Architecture**:
   - High-throughput **FastAPI** backend with REST endpoints.
   - Interactive **Streamlit** dashboard for live experiments, chat, and visual Pareto curves.
   - Zero-dependency local mode out of the box (with optional OpenAI/Anthropic/HuggingFace plugins).

---

## 🏛️ Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │              User Query                │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │        Agentic RAG State Machine       │
                      │  (Query Analyzer & Intent Router)      │
                      └──────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        ┌───────────────────────┐                 ┌───────────────────────┐
        │  Dense Vector Search  │                 │    Sparse BM25 Search │
        └───────────┬───────────┘                 └───────────┬───────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Hybrid Retriever (RRF + Alpha)       │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Heuristic / Cross-Encoder Reranker   │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │       Document Relevance Grader        │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │ Grounded Generator with Citations [1]  │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Self-Correction & Faithfulness Guard │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                                Final Grounded Answer
```

---

## 📂 Project Structure

```
RAG-With-RSI/
│
├── backend/
│   ├── app/
│   │   ├── ingestion/             # Loaders (PDF, MD, JSON, TXT) and normalizers
│   │   ├── chunking/              # Fixed, Recursive, Semantic chunkers & factory
│   │   ├── embeddings/            # Dense TF-IDF/SVD, OpenAI, HuggingFace providers
│   │   ├── retrieval/             # VectorStore, BM25Okapi, Hybrid RRF Retriever
│   │   ├── reranking/             # MMR Heuristic & Cross-Encoder rerankers
│   │   ├── generation/            # Heuristic citation synthesizer & LLM adapters (Ollama/OpenAI)
│   │   │
│   │   ├── evaluation/
│   │   │   ├── retrieval_eval.py  # Hit Rate, MRR, Context Precision, Recall, NDCG
│   │   │   ├── answer_eval.py     # Faithfulness, Relevance, Hallucination Rate
│   │   │   └── benchmarks.py      # Benchmark runner & composite fitness scoring
│   │   │
│   │   ├── optimizer/
│   │   │   ├── analyzer.py        # Failure mode diagnosis & bottleneck detection
│   │   │   ├── strategy_generator.py # Genetic chromosome representation, crossover, mutation
│   │   │   ├── experiment_runner.py  # Generation-by-generation evolution runner
│   │   │   └── selector.py        # Pareto frontier & tournament selection
│   │   │
│   │   ├── agents/
│   │   │   └── rag_graph.py       # Self-reflective agentic RAG state machine
│   │   │
│   │   ├── memory/                # Session and conversational history manager
│   │   ├── config.py              # Application settings and paths
│   │   ├── models.py              # Pydantic schemas
│   │   └── main.py                # FastAPI REST server with RSI dataset loader
│   │
│   ├── data/                      # Sample knowledge documents & storage
│   ├── experiments/               # Saved optimization runs, logs, and Pareto frontiers
│   └── tests/                     # Pytest test suite (18 unit/integration tests)
│
├── frontend/
│   └── app.py                     # Streamlit Interactive Studio (Chat, Logs, GA Optimizer, Docs)
│
├── rsi_rag/                       # RSI-RAG Automated Dataset & Evaluation Pipeline
│   ├── config/
│   │   └── dataset_config.yaml    # Configurable limits, seeds, storage paths
│   ├── data/
│   │   ├── raw/                   # Raw cache directory (.gitkeep)
│   │   ├── processed/             # Unified JSONL (HotpotQA, RAGBench, SciFact, Combined)
│   │   └── evaluation/            # Train, Validation, Test Isolated, Blind Test splits
│   ├── scripts/
│   │   ├── download_datasets.py   # Streamed dataset fetcher with caching and retries
│   │   ├── preprocess_datasets.py # Standardized JSONL preprocessor & isolated splitter
│   │   ├── validate_datasets.py   # Comprehensive schema & integrity validation reporter
│   │   └── utils.py               # Shared logger and text normalizer
│   └── tests/                     # Dataset pipeline validation tests (7 tests)
│
├── benchmarks/
│   ├── general_qa.json            # Core evaluation dataset
│   └── generate_benchmarks.py     # Synthetic QA generation utility
│
├── docker/
│   ├── Dockerfile.backend         # Container for FastAPI server
│   ├── Dockerfile.frontend        # Container for Streamlit UI
│   └── docker-compose.yml         # Multi-container orchestration
│
├── requirements.txt               # Unified project dependencies
├── pytest.ini                     # Root test configuration
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Installation
Clone the repository and install dependencies in a virtual environment:

```bash
cd "C:\Users\B Gowtham Reddy\.vscode\EvoRag"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Backend API
Start the FastAPI server:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 3. Launch the Frontend Dashboard
In a separate terminal window:

```bash
streamlit run frontend/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧬 Evolutionary Hyperparameter Tuning

Run an automated genetic optimization run with custom parameters:

```python
import httpx

config = {
    "name": "Production-RAG-Tuning",
    "population_size": 8,
    "num_generations": 4,
    "mutation_rate": 0.25,
    "crossover_rate": 0.7,
    "weight_faithfulness": 0.35,
    "weight_relevance": 0.25,
    "weight_hit_rate": 0.20
}

response = httpx.post("http://localhost:8000/api/optimizer/run", json=config, timeout=60.0)
summary = response.json()
print("Champion Pipeline:", summary["champion_strategy"]["name"])
print("Fitness Score:", summary["champion_strategy"]["fitness"])
```

### Chromosome Gene Space
| Hyperparameter | Search Range / Values | Impact |
|---|---|---|
| `chunk_size` | 200, 350, 500, 750, 1000 | Balances context completeness vs precision |
| `chunk_overlap` | 20, 50, 80, 120, 200 | Prevents information boundary cutoff |
| `chunking_strategy` | recursive, fixed, semantic | Preserves grammatical & thematic structure |
| `retrieval_strategy`| hybrid, dense, bm25 | Optimizes lexical vs semantic recall |
| `hybrid_alpha` | 0.05 to 0.95 | Density weight in Reciprocal Rank Fusion |
| `top_k` | 3, 5, 7, 9, 11 | Number of candidates retrieved |
| `reranker_enabled` | True, False | Eliminates irrelevant chunks prior to generation |
| `rerank_top_n` | 2, 3, 4, 5 | Candidate reduction for context window |
| `temperature` | 0.0, 0.1, 0.2, 0.4, 0.6 | Controls randomness vs deterministic grounding |

---

## 🧪 Running Tests

The test suite validates chunkers, vector search, BM25, hybrid retrieval, reranking, evaluation metrics, genetic algorithms, agent state machines, and API endpoints:

```bash
python -m pytest backend/tests -v
```

All 18 tests pass with sub-second execution:
```
backend/tests/test_agent.py::test_agent_graph_execution PASSED
backend/tests/test_api.py::test_root_endpoint PASSED
backend/tests/test_api.py::test_status_endpoint PASSED
backend/tests/test_api.py::test_text_ingest_and_query PASSED
backend/tests/test_chunking.py::test_fixed_chunker PASSED
backend/tests/test_chunking.py::test_recursive_chunker PASSED
backend/tests/test_chunking.py::test_semantic_chunker PASSED
backend/tests/test_chunking.py::test_chunker_factory PASSED
backend/tests/test_evaluation.py::test_retrieval_evaluation PASSED
backend/tests/test_evaluation.py::test_answer_evaluation PASSED
backend/tests/test_optimizer.py::test_random_strategy_generation PASSED
backend/tests/test_optimizer.py::test_crossover_and_mutation PASSED
backend/tests/test_optimizer.py::test_pareto_frontier_selection PASSED
backend/tests/test_optimizer.py::test_evolutionary_run PASSED
backend/tests/test_retrieval.py::test_vector_store PASSED
backend/tests/test_retrieval.py::test_bm25_retriever PASSED
backend/tests/test_retrieval.py::test_hybrid_retriever PASSED
backend/tests/test_retrieval.py::test_heuristic_reranker PASSED
======================= 18 passed in 0.69s =======================
```

---

## 🐳 Docker Deployment

To launch both the backend and frontend using Docker Compose:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

- Backend API: [http://localhost:8000](http://localhost:8000)
- Frontend UI: [http://localhost:8501](http://localhost:8501)

---

## 📄 License
MIT License. Free for commercial and research use.
