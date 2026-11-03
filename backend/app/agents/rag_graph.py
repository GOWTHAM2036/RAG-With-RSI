"""
Agentic RAG State Machine & Self-Corrective Graph.
Implements Adaptive RAG, Document Grading, Query Transformation, and Self-Correction.
Includes metadata query routing, local LLM parametric fallback, and execution diagnostics.
"""
from typing import List, Dict, Any, Optional
import time
import re
from backend.app.models import RAGStrategy, RetrievedChunk, QueryResponse, Document
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.reranking.factory import RerankerFactory
from backend.app.generation.llm_providers import GeneratorFactory
from backend.app.evaluation.answer_eval import AnswerEvaluator

class RAGAgentState:
    def __init__(self, query: str, strategy: RAGStrategy):
        self.original_query: str = query
        self.transformed_query: str = query
        self.strategy: RAGStrategy = strategy
        self.route: str = "retrieve"  # direct, catalog, retrieve
        self.retrieved_chunks: List[RetrievedChunk] = []
        self.graded_chunks: List[RetrievedChunk] = []
        self.generated_answer: str = ""
        self.citations: List[Dict[str, Any]] = []
        self.faithfulness_score: float = 1.0
        self.is_faithful: bool = True
        self.retry_count: int = 0
        self.max_retries: int = 1
        self.execution_trace: List[Dict[str, Any]] = []
        self.llm_debug_info: Optional[Dict[str, Any]] = None

    def log_step(self, node_name: str, description: str, details: Dict[str, Any] = None):
        self.execution_trace.append({
            "step": len(self.execution_trace) + 1,
            "node": node_name,
            "description": description,
            "timestamp": time.time(),
            "details": details or {}
        })

class RAGAgentGraph:
    def __init__(self, retriever: HybridRetriever, default_strategy: RAGStrategy):
        self.retriever = retriever
        self.default_strategy = default_strategy

    def run(self, query: str, strategy: Optional[RAGStrategy] = None, documents: Optional[List[Document]] = None) -> QueryResponse:
        t0 = time.time()
        strat = strategy or self.default_strategy
        state = RAGAgentState(query=query, strategy=strat)
        docs_list = documents or []

        # Node 1: Analyze & Route Query
        self._analyze_and_route(state, docs_list)

        if state.route == "direct":
            pass  # Already set in _analyze_and_route
        elif state.route == "catalog":
            pass  # Already set in _analyze_and_route
        else:
            # Node 2: Retrieve & Rerank
            self._retrieve_and_rerank(state)

            # Node 3: Grade Documents
            self._grade_documents(state)

            # Node 4: Query Transformation Fallback if needed
            if len(state.graded_chunks) == 0 and state.retry_count < state.max_retries:
                state.retry_count += 1
                self._transform_query(state)
                self._retrieve_and_rerank(state)
                self._grade_documents(state)

            # Node 5: Grounded Answer Generation
            self._generate_answer(state)

            # Node 6: Self-Correction & Faithfulness Verification
            self._verify_and_self_correct(state)

        total_latency = (time.time() - t0) * 1000.0

        return QueryResponse(
            query=query,
            answer=state.generated_answer,
            retrieved_chunks=state.graded_chunks if state.graded_chunks else state.retrieved_chunks,
            citations=state.citations,
            strategy_used=state.strategy,
            latency_ms=round(total_latency, 2),
            agent_steps=state.execution_trace,
            faithfulness_score=round(state.faithfulness_score, 4),
            llm_debug_info=state.llm_debug_info,
            evaluation={
                "faithfulness": round(state.faithfulness_score, 4),
                "chunks_used": len(state.graded_chunks) if state.graded_chunks else len(state.retrieved_chunks)
            }
        )

    def _analyze_and_route(self, state: RAGAgentState, documents: List[Document]):
        q = state.original_query.strip().lower()

        # 1. Check for greetings
        greetings = ["hello", "hi", "hey", "good morning", "good evening", "who are you"]
        if q in greetings or (len(q.split()) <= 2 and any(g in q for g in greetings)):
            state.route = "direct"
            state.generated_answer = (
                "Hello! I am EvoRAG. I can answer questions grounded in your ingested documents. "
                "How can I help you today?"
            )
            state.log_step("Router", "Routed query to direct conversation handler.", {"route": "direct"})
            return

        # 2. Check for catalog / knowledge inquiry
        catalog_patterns = [
            r"what is the knowledge you have",
            r"what (do you know|knowledge do you have)",
            r"what (are the )?inserted doc(um)?ents",
            r"what documents (do you have|are uploaded|are ingested)",
            r"list (the )?documents",
            r"tell me about your (knowledge|documents)"
        ]
        if any(re.search(pat, q) for pat in catalog_patterns):
            state.route = "catalog"
            if not documents:
                state.generated_answer = (
                    "⚠️ **The knowledge base is currently empty** (0 documents ingested).\n\n"
                    "Please upload files in the **📁 Document Manager** tab, or click **📥 Load Sample Knowledge** "
                    "in the sidebar to populate documents."
                )
            else:
                doc_titles = [f"- **{d.title}** ({len(d.content.split())} words)" for d in documents]
                state.generated_answer = (
                    f"📚 **Currently Ingested Knowledge Base ({len(documents)} document{'s' if len(documents) > 1 else ''}):**\n\n"
                    + "\n".join(doc_titles)
                    + "\n\nYou can ask me specific questions based on these documents!"
                )
            state.log_step("Router", "Handled metadata knowledge catalog query.", {"documents_count": len(documents)})
            return

        # 3. Standard retrieval route
        state.route = "retrieve"
        cleaned = re.sub(r"^(can you tell me|please tell me|what is|how do i|explain)\s+", "", q, flags=re.IGNORECASE)
        state.transformed_query = cleaned.strip() if cleaned.strip() else state.original_query
        state.log_step(
            "QueryAnalyzer",
            f"Analyzed intent and optimized query: '{state.transformed_query}'",
            {"original": state.original_query, "optimized": state.transformed_query}
        )

    def _retrieve_and_rerank(self, state: RAGAgentState):
        strat = state.strategy
        # 1. Retrieve
        raw_chunks = self.retriever.retrieve(state.transformed_query, top_k=strat.top_k)
        state.log_step(
            "Retriever",
            f"Retrieved {len(raw_chunks)} candidate chunks using {strat.retrieval_strategy} strategy.",
            {"candidate_count": len(raw_chunks)}
        )

        # 2. Rerank
        if strat.reranker_enabled and raw_chunks:
            reranker = RerankerFactory.get_reranker(strat.reranker_type)
            reranked = reranker.rerank(state.transformed_query, raw_chunks, top_n=strat.rerank_top_n)
            state.retrieved_chunks = reranked
            state.log_step(
                "Reranker",
                f"Reranked candidates and retained top {len(reranked)} most relevant chunks.",
                {"reranked_count": len(reranked)}
            )
        else:
            state.retrieved_chunks = raw_chunks[:strat.top_k]

    def _grade_documents(self, state: RAGAgentState):
        """Filters retrieved chunks to keep only those with tangible lexical/semantic relevance."""
        q_tokens = set(re.findall(r"\b\w+\b", state.transformed_query.lower()))
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "in", "to", "for"}
        key_tokens = q_tokens - stop_words
        if not key_tokens:
            key_tokens = q_tokens

        graded = []
        for rc in state.retrieved_chunks:
            c_tokens = set(re.findall(r"\b\w+\b", rc.chunk.content.lower()))
            overlap = len(key_tokens.intersection(c_tokens))
            if overlap > 0 or rc.score >= 0.2:
                graded.append(rc)

        state.graded_chunks = graded
        state.log_step(
            "DocumentGrader",
            f"Graded documents: {len(graded)}/{len(state.retrieved_chunks)} chunks deemed relevant.",
            {"passed": len(graded), "dropped": len(state.retrieved_chunks) - len(graded)}
        )

    def _transform_query(self, state: RAGAgentState):
        """Expands query with broader terms if initial retrieval yielded no confident matches."""
        expanded = f"{state.transformed_query} summary overview"
        state.transformed_query = expanded
        state.log_step(
            "QueryExpander",
            f"Transformed query for fallback retrieval: '{expanded}'",
            {"new_query": expanded}
        )

    def _generate_answer(self, state: RAGAgentState):
        chunks_to_use = state.graded_chunks if state.graded_chunks else state.retrieved_chunks
        provider = getattr(state.strategy, "llm_provider", "local")
        generator = GeneratorFactory.get_generator(provider)

        answer, citations = generator.generate(
            query=state.original_query,
            context_chunks=chunks_to_use,
            prompt_template=state.strategy.prompt_template,
            temperature=state.strategy.temperature,
            max_tokens=state.strategy.max_tokens
        )
        state.generated_answer = answer
        state.citations = citations
        state.llm_debug_info = getattr(generator, "last_debug_info", None)

        state.log_step(
            "GroundedGenerator",
            f"Generated synthesis using provider '{provider}' with {len(citations)} source citations.",
            {"citations_count": len(citations), "llm_provider": provider}
        )

    def _verify_and_self_correct(self, state: RAGAgentState):
        chunks_to_use = state.graded_chunks if state.graded_chunks else state.retrieved_chunks
        if not chunks_to_use:
            # If no chunks were retrieved in the first place, skip hallucination rewrite
            state.faithfulness_score = 0.0
            return

        faithfulness = AnswerEvaluator.compute_faithfulness(state.generated_answer, chunks_to_use)
        state.faithfulness_score = faithfulness

        if faithfulness < 0.4 and chunks_to_use:
            # Self-correction: rewrite answer strictly from the first most confident chunk
            first = chunks_to_use[0]
            corrected = f"According to verified context: {first.chunk.content[:220]}... [1]"
            state.generated_answer = corrected
            state.is_faithful = True
            state.faithfulness_score = 1.0
            state.log_step(
                "SelfCorrectionGuard",
                "Detected potential hallucination; self-corrected answer to strict factual quote.",
                {"original_faithfulness": round(faithfulness, 4), "action": "strict_quote_enforced"}
            )
        else:
            state.log_step(
                "SelfCorrectionGuard",
                f"Verified factual groundedness. Score: {faithfulness:.2f}",
                {"faithfulness": round(faithfulness, 4)}
            )
