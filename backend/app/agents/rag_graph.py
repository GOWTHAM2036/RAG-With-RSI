"""
Agentic RAG State Machine & Self-Corrective Graph.
Implements:
- Strict Retrieval Relevance Gate: Filters out unrelated chunks before they reach generation.
- Grounded Abstention: Explicitly abstains when evidence is missing, never fabricating or quoting random documents.
- Verified Query Reformulation: Only adopts fallback queries if they verifiably improve retrieval score.
- Bounded retry limits to eliminate infinite or futile loops.
- Claim-Level Faithfulness Verification with entity co-occurrence and negation checks.
- Granular latency decomposition (retrieval vs generation vs total).
"""
from typing import List, Dict, Any, Optional
import time
import re
from backend.app.models import RAGStrategy, RetrievedChunk, QueryResponse, Document
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.relevance_gate import RelevanceGate
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
        self.claim_verification_details: Dict[str, Any] = {}
        self.is_faithful: bool = True
        self.retry_count: int = 0
        self.max_retries: int = 1
        self.execution_trace: List[Dict[str, Any]] = []
        self.llm_debug_info: Optional[Dict[str, Any]] = None
        self.retrieval_latency_ms: float = 0.0
        self.generation_latency_ms: float = 0.0

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
            pass
        elif state.route == "catalog":
            pass
        else:
            # Node 2: Retrieve & Rerank (Isolated timing)
            t_ret_start = time.time()
            self._retrieve_and_rerank(state)

            # Node 3: Grade Documents with Strict Relevance Gate
            self._grade_documents(state)

            # Node 4: Verified Query Transformation Fallback if initial retrieval relevance is zero
            if len(state.graded_chunks) == 0 and state.retry_count < state.max_retries:
                state.retry_count += 1
                self._execute_verified_fallback(state)

            state.retrieval_latency_ms = round((time.time() - t_ret_start) * 1000.0, 2)

            # Node 5: Context Validation & Honest Abstention Check
            if len(state.graded_chunks) == 0:
                # Abstain immediately: do NOT pass unrelated chunks to the generator
                state.generated_answer = (
                    f"I do not have sufficient information in the ingested documents to answer: '{query}'. "
                    f"The current knowledge base does not contain verified context about this topic."
                )
                state.retrieved_chunks = []
                state.citations = []
                state.faithfulness_score = 1.0  # Honest abstention makes zero ungrounded assertions
                state.claim_verification_details = {
                    "status": "abstained_insufficient_evidence",
                    "reason": "Zero retrieved chunks met relevance criteria",
                    "searched_terms": list(RelevanceGate.extract_key_query_terms(state.original_query))
                }
                state.log_step(
                    "RelevanceGate",
                    f"Abstained from generation: zero chunks relevant to '{state.original_query}'.",
                    {"action": "abstained", "chunks_passed": 0}
                )
            else:
                # Node 6: Grounded Answer Generation (Isolated timing)
                t_gen_start = time.time()
                self._generate_answer(state)
                state.generation_latency_ms = round((time.time() - t_gen_start) * 1000.0, 2)

                # Node 7: Self-Correction & Claim-Level Faithfulness Verification
                self._verify_and_self_correct(state)

        total_latency = (time.time() - t0) * 1000.0

        return QueryResponse(
            query=query,
            answer=state.generated_answer,
            retrieved_chunks=state.graded_chunks,
            citations=state.citations,
            strategy_used=state.strategy,
            latency_ms=round(total_latency, 2),
            retrieval_latency_ms=state.retrieval_latency_ms,
            generation_latency_ms=state.generation_latency_ms,
            agent_steps=state.execution_trace,
            faithfulness_score=round(state.faithfulness_score, 4),
            llm_debug_info=state.llm_debug_info,
            evaluation={
                "faithfulness": round(state.faithfulness_score, 4),
                "claim_verification": state.claim_verification_details,
                "chunks_used": len(state.graded_chunks)
            }
        )

    def _analyze_and_route(self, state: RAGAgentState, documents: List[Document]):
        q = state.original_query.strip().lower()

        # 1. Check for greetings
        greetings = {"hello", "hi", "hey", "greetings", "good morning", "good evening"}
        if q in greetings or q.startswith("hello ") or q.startswith("hi "):
            state.route = "direct"
            state.generated_answer = (
                "Hello! I am your **EvoRAG / RAG-With-RSI** assistant. "
                "I can search across your uploaded documents using hybrid vector-lexical retrieval, "
                "or you can trigger evolutionary optimization to find the champion RAG pipeline."
            )
            state.log_step("Router", "Handled conversational greeting directly.", {"intent": "greeting"})
            return

        # 2. Check for knowledge base inventory queries
        catalog_intents = [
            "what documents", "what files", "list documents", "show files",
            "what do you know", "what is uploaded", "what dataset", "available documents"
        ]
        if any(intent in q for intent in catalog_intents):
            state.route = "catalog"
            if not documents:
                state.generated_answer = (
                    "📁 No documents are currently ingested in the knowledge base. "
                    "You can upload text, markdown, or PDF files, or click **Load RSI Dataset** in the Document Manager."
                )
            else:
                doc_titles = [f"- **{d.title}** ({len(d.content.split())} words)" for d in documents[:12]]
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
        raw_chunks = self.retriever.retrieve(state.transformed_query, top_k=strat.top_k)
        state.log_step(
            "Retriever",
            f"Retrieved {len(raw_chunks)} candidate chunks using {strat.retrieval_strategy} strategy.",
            {"candidate_count": len(raw_chunks)}
        )

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
        """Strictly filters retrieved chunks through the Relevance Gate."""
        passed, rejected = RelevanceGate.filter_relevant_chunks(state.transformed_query, state.retrieved_chunks)
        state.graded_chunks = passed
        state.log_step(
            "DocumentGrader",
            f"Graded documents: {len(passed)}/{len(state.retrieved_chunks)} chunks deemed relevant to '{state.transformed_query}'.",
            {"passed": len(passed), "dropped": len(rejected)}
        )

    def _reformulate_query(self, query: str) -> str:
        cleaned = re.sub(
            r"\b(can you|could you|please|what is|how do|why does|tell me about|explain the)\b",
            "",
            query,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(r"[?!.,;:]", " ", cleaned)
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 2]
        stop_words = {
            "the", "and", "for", "are", "with", "from", "that", "this", "they",
            "have", "been", "which", "about", "into", "their"
        }
        filtered = [t for t in tokens if t.lower() not in stop_words]

        if len(filtered) >= 2:
            return " ".join(filtered)
        return query.strip()

    def _execute_verified_fallback(self, state: RAGAgentState):
        previous_chunks = list(state.retrieved_chunks)
        previous_top_score = max([c.score for c in previous_chunks], default=0.0)

        reformulated = self._reformulate_query(state.original_query)
        state.transformed_query = reformulated

        state.log_step(
            "QueryReformulator",
            f"Reformulated query for verified fallback: '{reformulated}'",
            {"previous_top_score": round(previous_top_score, 4), "reformulated_query": reformulated}
        )

        strat = state.strategy
        new_raw_chunks = self.retriever.retrieve(reformulated, top_k=strat.top_k)

        if strat.reranker_enabled and new_raw_chunks:
            reranker = RerankerFactory.get_reranker(strat.reranker_type)
            new_retrieved = reranker.rerank(reformulated, new_raw_chunks, top_n=strat.rerank_top_n)
        else:
            new_retrieved = new_raw_chunks[:strat.top_k]

        passed, rejected = RelevanceGate.filter_relevant_chunks(reformulated, new_retrieved)
        new_top_score = max([c.score for c in passed], default=0.0) if passed else 0.0

        if len(passed) > 0 and new_top_score > previous_top_score:
            state.retrieved_chunks = new_retrieved
            state.graded_chunks = passed
            state.log_step(
                "FallbackVerifier",
                f"Verification Passed: Rewrite found {len(passed)} relevant chunks (score {new_top_score:.3f}).",
                {"action": "adopted_reformulation", "new_top_score": round(new_top_score, 4)}
            )
        else:
            state.retrieved_chunks = previous_chunks
            state.graded_chunks = []
            state.log_step(
                "FallbackVerifier",
                f"Verification Halted: Fallback found 0 relevant chunks. Halting retry loop to enforce honest abstention.",
                {"action": "reverted_and_abstained"}
            )

    def _generate_answer(self, state: RAGAgentState):
        chunks_to_use = state.graded_chunks
        provider = getattr(state.strategy, "llm_provider", "local")
        model = state.strategy.metadata.get("local_model") if state.strategy.metadata else None
        generator = GeneratorFactory.get_generator(provider, model=model)

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
        chunks_to_use = state.graded_chunks
        if not chunks_to_use:
            return

        verification = AnswerEvaluator.compute_claim_faithfulness(state.generated_answer, chunks_to_use)
        faithfulness = verification["claim_faithfulness"]
        state.faithfulness_score = faithfulness
        state.claim_verification_details = verification

        if faithfulness < 0.4 and chunks_to_use:
            state.generated_answer = (
                f"I cannot provide a sufficiently grounded answer based on the retrieved context. "
                f"The available documentation does not adequately verify claims regarding '{state.original_query}'."
            )
            state.is_faithful = False
            state.faithfulness_score = round(faithfulness, 4)
            state.log_step(
                "SelfCorrectionGuard",
                "Detected ungrounded propositions; refused hallucination and flagged low factual grounding.",
                {
                    "original_faithfulness": round(faithfulness, 4),
                    "unsupported_claims": verification.get("unsupported_claims", []),
                    "action": "refused_hallucination"
                }
            )
        else:
            state.log_step(
                "SelfCorrectionGuard",
                f"Verified factual groundedness across {verification.get('total_claims', 0)} atomic claims. Score: {faithfulness:.2f}",
                {
                    "faithfulness": round(faithfulness, 4),
                    "supported_claims": verification.get("supported_claims", 0),
                    "total_claims": verification.get("total_claims", 0)
                }
            )
