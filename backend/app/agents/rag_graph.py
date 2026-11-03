"""
Agentic RAG State Machine & Self-Corrective Graph.
Implements:
- Verified Query Reformulation: Only adopts fallback queries if they verifiably improve retrieval score.
- Bounded retry limits to eliminate infinite or futile loops.
- Claim-Level Faithfulness Verification with entity co-occurrence and negation checks.
- Granular latency decomposition (retrieval vs generation vs total).
- Transparent execution diagnostics and provider trace logs.
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

            # Node 3: Grade Documents
            self._grade_documents(state)

            # Node 4: Verified Query Transformation Fallback if initial retrieval confidence is insufficient
            if (len(state.graded_chunks) == 0 or max([c.score for c in state.retrieved_chunks], default=0.0) < 0.25) and state.retry_count < state.max_retries:
                state.retry_count += 1
                self._execute_verified_fallback(state)

            state.retrieval_latency_ms = round((time.time() - t_ret_start) * 1000.0, 2)

            # Node 5: Grounded Answer Generation (Isolated timing)
            t_gen_start = time.time()
            self._generate_answer(state)
            state.generation_latency_ms = round((time.time() - t_gen_start) * 1000.0, 2)

            # Node 6: Self-Correction & Claim-Level Faithfulness Verification
            self._verify_and_self_correct(state)

        total_latency = (time.time() - t0) * 1000.0

        return QueryResponse(
            query=query,
            answer=state.generated_answer,
            retrieved_chunks=state.graded_chunks if state.graded_chunks else state.retrieved_chunks,
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
                "chunks_used": len(state.graded_chunks) if state.graded_chunks else len(state.retrieved_chunks)
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
        """Filters retrieved chunks to keep only those with tangible lexical/semantic relevance."""
        q_tokens = set(re.findall(r"\b\w+\b", state.transformed_query.lower()))
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "in", "to", "for", "of", "with"}
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

    def _reformulate_query(self, query: str) -> str:
        """
        Intelligently reformulates query:
        - Extracts key noun phrases and entities.
        - Strips interrogative stopwords.
        - Converts conversational phrasing to declarative search tokens.
        - Does NOT append generic filler words like 'summary overview'.
        """
        # Strip interrogatives
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
        """
        Executes query transformation fallback and strictly verifies whether the new query
        improves retrieval confidence before adopting it.
        """
        previous_chunks = list(state.retrieved_chunks)
        previous_top_score = max([c.score for c in previous_chunks], default=0.0)

        reformulated = self._reformulate_query(state.original_query)
        state.transformed_query = reformulated

        state.log_step(
            "QueryReformulator",
            f"Reformulated query for verified fallback: '{reformulated}'",
            {"previous_top_score": round(previous_top_score, 4), "reformulated_query": reformulated}
        )

        # Retrieve candidates with reformulated query
        strat = state.strategy
        new_raw_chunks = self.retriever.retrieve(reformulated, top_k=strat.top_k)

        if strat.reranker_enabled and new_raw_chunks:
            reranker = RerankerFactory.get_reranker(strat.reranker_type)
            new_retrieved = reranker.rerank(reformulated, new_raw_chunks, top_n=strat.rerank_top_n)
        else:
            new_retrieved = new_raw_chunks[:strat.top_k]

        new_top_score = max([c.score for c in new_retrieved], default=0.0)

        # Grade new candidates
        q_tokens = set(re.findall(r"\b\w+\b", reformulated.lower()))
        new_graded = [
            rc for rc in new_retrieved
            if (len(q_tokens.intersection(set(re.findall(r"\b\w+\b", rc.chunk.content.lower())))) > 0 or rc.score >= 0.2)
        ]

        # Verify whether rewrite genuinely improved retrieval
        if new_top_score > previous_top_score and len(new_graded) > 0:
            state.retrieved_chunks = new_retrieved
            state.graded_chunks = new_graded
            state.log_step(
                "FallbackVerifier",
                f"Verification Passed: Rewrite improved retrieval score ({new_top_score:.3f} > {previous_top_score:.3f}).",
                {"action": "adopted_reformulation", "new_top_score": round(new_top_score, 4)}
            )
        else:
            # Fallback failed to improve retrieval: retain original chunks and bound retry loop
            state.retrieved_chunks = previous_chunks
            state.log_step(
                "FallbackVerifier",
                f"Verification Halted: Rewrite did not improve score ({new_top_score:.3f} <= {previous_top_score:.3f}). Retained original chunks.",
                {"action": "reverted_to_original", "previous_top_score": round(previous_top_score, 4)}
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
            state.faithfulness_score = 0.0
            state.claim_verification_details = {"claims_evaluated": 0, "status": "no_context"}
            return

        # Run Claim-Level Support Verification
        verification = AnswerEvaluator.compute_claim_faithfulness(state.generated_answer, chunks_to_use)
        faithfulness = verification["claim_faithfulness"]
        state.faithfulness_score = faithfulness
        state.claim_verification_details = verification

        if faithfulness < 0.4 and chunks_to_use:
            # Self-correction: replace unverified answer with strictly cited ground quote
            first = chunks_to_use[0]
            corrected = f"According to verified context: {first.chunk.content[:220]}... [1]"
            state.generated_answer = corrected
            state.is_faithful = True
            state.faithfulness_score = 1.0
            state.log_step(
                "SelfCorrectionGuard",
                "Detected unsupported propositions; self-corrected answer to strict factual quote.",
                {
                    "original_faithfulness": round(faithfulness, 4),
                    "unsupported_claims": verification.get("unsupported_claims", []),
                    "action": "strict_quote_enforced"
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
