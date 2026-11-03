"""
RAG-With-RSI Interactive Dashboard.
Multi-page Streamlit application providing Knowledge Ingestion, Agent Chat,
Local LLM Status & Live Logs, Evolutionary Genetic Optimizer with 3D Pareto Frontier,
and Benchmark Hub with Claim-Level Faithfulness & Latency Decomposition.
"""
import streamlit as st
import requests
import json
import pandas as pd
import time
from typing import Dict, Any

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAG-With-RSI: Evolutionary & Self-Improving Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .citation-tag {
        display: inline-block;
        background: #312E81;
        color: #A5B4FC;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to query API
def fetch_status():
    try:
        r = requests.get(f"{API_BASE}/", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_documents():
    try:
        r = requests.get(f"{API_BASE}/api/documents", timeout=2)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def fetch_champions():
    try:
        r = requests.get(f"{API_BASE}/api/optimizer/champions", timeout=2)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def fetch_splits():
    try:
        r = requests.get(f"{API_BASE}/api/benchmarks/splits", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

status_data = fetch_status()

with st.sidebar:
    st.markdown("### 🧬 RAG-With-RSI System")
    if status_data:
        st.success("🟢 Backend Connected (Port 8000)")
        col1, col2 = st.columns(2)
        col1.metric("Docs", status_data.get("documents_count", 0))
        col2.metric("Chunks", status_data.get("chunks_indexed", 0))

        st.markdown(f"**Champion:** `{status_data.get('active_champion', 'Default')}`")
    else:
        st.error("🔴 Backend Offline - Start backend server on port 8000")
        st.caption("Run: `uvicorn backend.app.main:app --port 8000`")

    st.markdown("---")
    st.markdown("### 🖥️ Local LLM (Ollama)")
    llm_info = (status_data or {}).get("local_llm", {})
    if llm_info.get("detected_models"):
        st.success(f"🟢 **Online**: `{llm_info.get('endpoint')}`")
        st.caption(f"Default Model: `{llm_info.get('default_model')}`")
        st.caption(f"Installed: {', '.join(llm_info.get('detected_models', [])[:3])}")
    else:
        st.warning("⚠️ **Local LLM Offline or Heuristic Mode**")

    st.markdown("---")
    st.markdown("### ⚡ Quick Ingest")
    if st.button("📥 Load Built-in AI Documentation", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/api/ingest/samples", timeout=5)
            if r.status_code == 200:
                st.success(r.json().get("message", "Sample knowledge loaded."))
                st.rerun()
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

    if st.button("📥 Load RSI Processed Dataset (50 items)", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/api/ingest/rsi_dataset", json={"limit": 50}, timeout=15)
            if r.status_code == 200:
                st.success(r.json().get("message", "RSI-RAG records loaded."))
                st.rerun()
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

    if st.button("🔄 Refresh System", use_container_width=True):
        st.rerun()

    if st.button("🗑️ Clear All Knowledge", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/api/documents")
            st.warning("All knowledge cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# Header
st.markdown('<div class="main-header">🧬 RAG-With-RSI: Evolutionary & Self-Improving Architecture</div>', unsafe_allow_html=True)
st.caption("Autonomous 3D Pareto Optimization • Claim-Level Verification • Isolated Evaluation Protocol")

tab_chat, tab_optimizer, tab_eval, tab_docs, tab_config = st.tabs([
    "💬 Agent & Chat Studio",
    "🧬 Evolutionary Optimizer",
    "📊 Benchmark Hub",
    "📁 Document Manager",
    "⚙️ Strategy Config"
])

# ----------------- TAB 1: Chat & Agent Studio -----------------
with tab_chat:
    st.markdown("#### 💬 Interactive Knowledge Retrieval & Synthesis")

    doc_count = status_data.get("documents_count", 0) if status_data else 0
    if doc_count == 0:
        st.warning(
            "⚠️ **Knowledge Base is Empty (Docs: 0, Chunks: 0)**: "
            "RAG requires context documents to retrieve answers. Click one of the buttons below to load documents instantly:"
        )
        col_b1, col_b2, _ = st.columns([1, 1, 2])
        if col_b1.button("📥 Load Sample Knowledge"):
            requests.post(f"{API_BASE}/api/ingest/samples", timeout=5)
            st.rerun()
        if col_b2.button("📥 Load RSI Dataset"):
            requests.post(f"{API_BASE}/api/ingest/rsi_dataset", json={"limit": 25}, timeout=10)
            st.rerun()
        st.markdown("---")

    col_input, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        use_agent = st.toggle("🧠 Enable Agentic Guardrails", value=True, help="Executes intent routing, verified query reformulator, document grader, and claim support verification.")
        
        llm_options = ["local", "heuristic", "openai", "anthropic"]
        avail = llm_info.get("detected_models", [])
        active_local_m = llm_info.get("default_model", "gemma3:4b")

        selected_llm = st.selectbox(
            "🤖 LLM Engine",
            llm_options,
            index=0 if avail else 1,
            help="Choose generator: 'local' connects to Ollama, 'heuristic' runs offline."
        )

        if selected_llm == "local" and avail:
            local_model_choice = st.selectbox("Local Model", avail, index=0)
        else:
            local_model_choice = active_local_m

        session_id = st.text_input("Session ID", value="default_user")

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Source Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c['citation_id']} {c['doc_title']}** (Score: {c.get('score', 0):.2f})")
                        st.caption(c.get("snippet", ""))
            if msg.get("chunks"):
                with st.expander("🔍 Retrieved Context Chunks"):
                    for idx, rc in enumerate(msg["chunks"]):
                        st.markdown(f"**Chunk #{idx+1}** | Score: `{rc.get('score', 0):.3f}` | Source: `{rc.get('retrieval_source', 'hybrid')}`")
                        st.caption(rc.get("chunk", {}).get("content", "")[:300] + "...")
            if msg.get("evaluation"):
                with st.expander("🔍 Factual Groundedness & Claim Verification"):
                    ev = msg["evaluation"]
                    st.write(f"**Claim Support Faithfulness:** `{ev.get('faithfulness', 0) * 100:.1f}%`")
                    cv = ev.get("claim_verification", {})
                    if cv.get("unsupported_claims"):
                        st.warning(f"Unsupported Propositions Detected: {cv['unsupported_claims']}")
                    elif cv.get("status") == "abstained_insufficient_evidence":
                        st.info("🛡️ Honest Abstention: Groundedness verified (no unsupported claims made).")
                    else:
                        st.success("All extracted factual propositions are supported by retrieved context.")
            if msg.get("llm_debug_info"):
                with st.expander("🖥️ Execution Logs & Engine Trace"):
                    st.json(msg["llm_debug_info"])

    query_input = st.chat_input("Ask a question about your ingested documents...")

    if query_input:
        st.session_state.messages.append({"role": "user", "content": query_input})
        with st.chat_message("user"):
            st.markdown(query_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and synthesizing with LLM..."):
                try:
                    active_strategy = status_data.get("champion_strategy", {}).copy() if status_data else {}
                    active_strategy["llm_provider"] = selected_llm
                    active_strategy["metadata"] = {"local_model": local_model_choice}

                    payload = {
                        "query": query_input,
                        "session_id": session_id,
                        "use_agent": use_agent,
                        "strategy": active_strategy
                    }
                    endpoint = f"{API_BASE}/api/agent/query" if use_agent else f"{API_BASE}/api/query"
                    res = requests.post(endpoint, json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "")
                        citations = data.get("citations", [])
                        chunks = data.get("retrieved_chunks", [])
                        tot_lat = data.get("latency_ms", 0)
                        ret_lat = data.get("retrieval_latency_ms", 0)
                        gen_lat = data.get("generation_latency_ms", 0)
                        faithfulness = data.get("faithfulness_score", 1.0)
                        llm_debug = data.get("llm_debug_info")
                        eval_info = data.get("evaluation", {})

                        st.markdown(answer)
                        st.caption(
                            f"⏱️ **Latency:** {tot_lat:.1f}ms (Retrieval: {ret_lat:.1f}ms | Generation: {gen_lat:.1f}ms) | "
                            f"🎯 **Claim Faithfulness:** {faithfulness * 100:.1f}%"
                        )

                        if llm_debug and "fallback" in str(llm_debug.get("status", "")).lower():
                            st.warning(f"⚠️ **Local LLM Notice**: Ollama was offline or failed to respond. Fallback to Heuristic engine occurred. Trace: `{llm_debug.get('errors')}`")

                        if citations:
                            with st.expander(f"📚 View Citations ({len(citations)})"):
                                for c in citations:
                                    st.markdown(f"**{c['citation_id']} {c['doc_title']}** (Relevance: {c.get('score', 0):.2f})")
                                    st.caption(c.get("snippet", ""))

                        if chunks:
                            with st.expander(f"🔍 Retrieved Context Chunks ({len(chunks)})"):
                                for idx, rc in enumerate(chunks):
                                    st.markdown(f"**Chunk #{idx+1}** | Score: `{rc.get('score', 0):.3f}` | Source: `{rc.get('retrieval_source', 'hybrid')}`")
                                    st.caption(rc.get("chunk", {}).get("content", "")[:300] + "...")
                        elif "insufficient" in answer.lower():
                            st.info("🛡️ **Relevance Gate Triggered**: Zero chunks passed relevance criteria. System abstained from ungrounded generation.")

                        if llm_debug:
                            with st.expander("🖥️ Engine Execution Logs"):
                                st.json(llm_debug)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "chunks": chunks,
                            "llm_debug_info": llm_debug,
                            "evaluation": eval_info
                        })
                    else:
                        st.error(f"API Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to query backend: {e}")

# ----------------- TAB 2: Evolutionary Optimizer -----------------
with tab_optimizer:
    st.markdown("#### 🧬 Genetic Algorithm RAG Pipeline Evolution")
    st.caption("Optimizes multi-objective Pareto frontier (Accuracy, Latency, Cost) with Train/Val/Test dataset isolation.")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        pop_size = st.slider("Population Size", min_value=4, max_value=16, value=6, step=2)
        num_gens = st.slider("Number of Generations", min_value=2, max_value=8, value=3, step=1)
        rand_seed = st.number_input("Deterministic Random Seed", min_value=1, max_value=9999, value=42)
    with col_opt2:
        mutation_rate = st.slider("Mutation Rate", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        crossover_rate = st.slider("Crossover Rate", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        gen_engine = st.selectbox("Evaluation Generator Engine", ["heuristic", "local"], index=0, help="'heuristic' is ultra-fast; 'local' evaluates real Ollama model latency & tokens.")
    with col_opt3:
        w_faith = st.slider("Weight: Claim Faithfulness", 0.0, 1.0, 0.35, 0.05)
        w_rel = st.slider("Weight: Answer Relevance", 0.0, 1.0, 0.25, 0.05)
        w_hit = st.slider("Weight: Retrieval NDCG", 0.0, 1.0, 0.20, 0.05)
        sample_eval_limit = st.slider("Evaluation Queries per Candidate", min_value=5, max_value=50, value=12, step=5)

    if st.button("🚀 Launch Multi-Objective Genetic Optimization", type="primary", use_container_width=True):
        with st.spinner("Simulating generations of RAG evolution across Train/Val/Test splits..."):
            try:
                opt_payload = {
                    "name": f"Run-{time.strftime('%Y%m%d-%H%M%S')}",
                    "population_size": pop_size,
                    "num_generations": num_gens,
                    "random_seed": rand_seed,
                    "mutation_rate": mutation_rate,
                    "crossover_rate": crossover_rate,
                    "generator_provider": gen_engine,
                    "eval_sample_limit": sample_eval_limit,
                    "weight_faithfulness": w_faith,
                    "weight_relevance": w_rel,
                    "weight_hit_rate": w_hit
                }
                res = requests.post(f"{API_BASE}/api/optimizer/run", json=opt_payload, timeout=120)
                if res.status_code == 200:
                    summary = res.json()
                    st.success(f"🎉 Optimization Complete! Discovered Champion: `{summary['champion_strategy']['name']}`")
                    st.session_state.latest_opt_summary = summary
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Optimization run failed: {e}")

    opt_summary = st.session_state.get("latest_opt_summary")
    if opt_summary:
        st.markdown("---")
        st.markdown("### 🏆 Discovered Champion Architecture (Validation Selected)")
        champ = opt_summary["champion_strategy"]
        test_eval = opt_summary.get("test_evaluation", {})

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Validation Accuracy", f"{champ.get('measured_accuracy', champ.get('fitness', 0)):.4f}")
        m2.metric("Total Latency", f"{champ.get('measured_latency_ms', champ.get('latency_ms', 0)):.1f}ms")
        m3.metric("Measured Cost (USD)", f"${champ.get('measured_cost_usd', 0.0001):.6f}")
        m4.metric("Total Tokens", champ.get("measured_tokens", 0))
        m5.metric("Seed", opt_summary.get("random_seed", 42))

        if test_eval:
            st.info(
                f"🛡️ **Protected Test Set Generalization Evaluation (Evaluated strictly ONCE on Champion):** "
                f"Test Accuracy: **{test_eval.get('composite_fitness', 0):.4f}** | "
                f"Claim Faithfulness: **{test_eval.get('answer_metrics', {}).get('faithfulness', 0)*100:.1f}%** | "
                f"Test Latency: **{test_eval.get('avg_latency_ms', 0):.1f}ms** (Retrieval: {test_eval.get('retrieval_latency_ms', 0):.1f}ms, Gen: {test_eval.get('generation_latency_ms', 0):.1f}ms) | "
                f"Dataset Fingerprint: `{opt_summary.get('dataset_fingerprint', '')}`"
            )

        st.markdown("#### ⚖️ 3D Pareto Frontier (Non-Dominated Trade-Offs)")
        pareto = opt_summary.get("pareto_frontier", [])
        if pareto:
            pareto_df = pd.DataFrame([
                {
                    "Strategy Name": p["name"],
                    "Accuracy (Max)": p.get("measured_accuracy", p.get("fitness", 0)),
                    "Latency ms (Min)": p.get("measured_latency_ms", p.get("latency_ms", 0)),
                    "Cost USD (Min)": p.get("measured_cost_usd", 0.0001),
                    "Tokens": p.get("measured_tokens", 0),
                    "Chunk Size": p.get("chunk_size"),
                    "Overlap": p.get("chunk_overlap"),
                    "Retrieval": p.get("retrieval_strategy"),
                    "Alpha": p.get("hybrid_alpha"),
                    "Reranker": p.get("reranker_enabled")
                }
                for p in pareto
            ])
            st.dataframe(pareto_df, use_container_width=True)

    # Persistent Champion Registry
    st.markdown("---")
    st.markdown("### 📜 Persistent Champions Registry")
    registry = fetch_champions()
    if registry:
        reg_df = pd.DataFrame([
            {
                "Version": f"v{r.get('version', 1)}",
                "Champion Name": r.get("champion_name", "Champion"),
                "Test Accuracy": r.get("test_accuracy", 0),
                "Test Faithfulness": f"{r.get('test_faithfulness', 0)*100:.1f}%",
                "Test Latency (ms)": r.get("test_latency_ms", 0),
                "Cost (USD)": f"${r.get('test_cost_usd', 0):.6f}",
                "Seed": r.get("random_seed", 42),
                "Dataset SHA256": r.get("dataset_fingerprint", "")
            }
            for r in registry
        ])
        st.dataframe(reg_df, use_container_width=True)
    else:
        st.caption("No champions registered yet. Run an optimization experiment to register persistent champions.")

# ----------------- TAB 3: Benchmark Hub -----------------
with tab_eval:
    st.markdown("#### 📊 RAG Benchmark & Performance Evaluation")
    st.caption("Rigorous isolated evaluation decomposing retrieval, generation, and claim-level factual grounding.")

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        split_choice = st.selectbox("Benchmark Partition", ["validation", "test_isolated", "train"], index=0, help="Strict isolation: Train is for optimization; Validation is for selection; Test Isolated is for unbiased evaluation.")
    with col_e2:
        eval_gen_provider = st.selectbox("Generator Provider", ["heuristic", "local"], index=0)
    with col_e3:
        bench_sample_limit = st.slider("Query Limit", 5, 50, 15)

    if st.button("🧪 Run Comprehensive Evaluation", type="primary"):
        with st.spinner(f"Evaluating active pipeline against {split_choice} partition..."):
            try:
                eval_payload = {
                    "split": split_choice,
                    "sample_limit": bench_sample_limit,
                    "generator_provider": eval_gen_provider
                }
                res = requests.post(f"{API_BASE}/api/evaluate", json=eval_payload, timeout=60)
                if res.status_code == 200:
                    st.session_state.eval_report = res.json()
                    st.success("Evaluation complete!")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Evaluation error: {e}")

    report = st.session_state.get("eval_report")
    if report:
        st.markdown("---")
        rm = report["retrieval_metrics"]
        am = report["answer_metrics"]
        failures = report.get("failure_analysis", {})

        st.markdown("##### 🔍 Retrieval Metrics")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("Hit Rate @ K", f"{rm['hit_rate'] * 100:.1f}%")
        rc2.metric("MRR @ K", f"{rm['mrr']:.3f}")
        rc3.metric("Context Precision", f"{rm['context_precision']:.3f}")
        rc4.metric("Context Recall", f"{rm['context_recall']:.3f}")
        rc5.metric("NDCG @ K", f"{rm['ndcg']:.3f}")

        st.markdown("##### 📝 Groundedness & Claim-Level Verification Metrics")
        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        ac1.metric("Claim Faithfulness", f"{am.get('claim_faithfulness', am.get('faithfulness', 0)) * 100:.1f}%")
        ac2.metric("Answer Relevance", f"{am['answer_relevance'] * 100:.1f}%")
        ac3.metric("Semantic Similarity", f"{am['semantic_similarity']:.3f}")
        ac4.metric("Token F1", f"{am.get('token_f1', 0):.3f}")
        ac5.metric("Hallucination Rate", f"{am['hallucination_rate'] * 100:.1f}%")

        st.caption(f"ℹ️ **Evaluation Note:** {am.get('evaluation_notes', 'Claim support verified against context.')}")

        st.markdown("##### ⏱️ Latency & Cost Breakdown")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        lc1.metric("Total Latency", f"{report.get('total_latency_ms', report.get('avg_latency_ms', 0)):.1f}ms")
        lc2.metric("Retrieval Latency", f"{report.get('retrieval_latency_ms', 0):.1f}ms")
        lc3.metric("Generation Latency", f"{report.get('generation_latency_ms', 0):.1f}ms")
        lc4.metric("Total Tokens", f"{report.get('total_tokens', 0)}")
        lc5.metric("Measured Cost", f"${report.get('measured_cost_usd', 0):.6f}")

        st.markdown("##### ⚠️ Pipeline Bottlenecks & Diagnostics")
        st.info(
            f"Retrieval Success: **{failures.get('retrieval_success_rate', 1.0) * 100:.1f}%** | "
            f"Retrieval Misses: **{failures.get('retrieval_misses', 0)}** | "
            f"Reranker Drops: **{failures.get('reranker_drops', 0)}** | "
            f"Low Faithfulness Count: **{failures.get('low_faithfulness_count', 0)}** | "
            f"Dataset SHA256: `{report.get('dataset_fingerprint', 'n/a')}`"
        )

# ----------------- TAB 4: Document Manager -----------------
with tab_docs:
    st.markdown("#### 📁 Knowledge Base Ingestion")
    
    col_up, col_paste = st.columns(2)
    with col_up:
        st.markdown("##### Upload File (.txt, .md, .json, .pdf)")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "json", "pdf"])
        if uploaded_file and st.button("Ingest Uploaded File"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                r = requests.post(f"{API_BASE}/api/ingest/file", files=files, timeout=10)
                if r.status_code == 200:
                    st.success(f"Ingested '{uploaded_file.name}'!")
                    st.rerun()
                else:
                    st.error(f"Ingest failed: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_paste:
        st.markdown("##### Direct Text Ingestion")
        doc_title = st.text_input("Document Title", value="Notes")
        doc_text = st.text_area("Paste Content", height=150)
        if st.button("Ingest Text"):
            if doc_text.strip():
                try:
                    r = requests.post(f"{API_BASE}/api/ingest/text", json={"title": doc_title, "text": doc_text}, timeout=10)
                    if r.status_code == 200:
                        st.success("Text document ingested successfully!")
                        st.rerun()
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

    st.markdown("---")
    st.markdown("##### Ingested Documents List")
    docs = fetch_documents()
    if docs:
        df_docs = pd.DataFrame(docs)
        st.dataframe(df_docs, use_container_width=True)
    else:
        st.info("No documents currently ingested. Ingest sample documents to start.")

# ----------------- TAB 5: Strategy Config -----------------
with tab_config:
    st.markdown("#### ⚙️ Manual Strategy Configuration")
    st.caption("Inspect and manually modify the active RAG configuration chromosome.")

    champ = (status_data or {}).get("champion_strategy", {})
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        new_chunk_size = st.number_input("Chunk Size", value=champ.get("chunk_size", 400), step=50)
        new_overlap = st.number_input("Chunk Overlap", value=champ.get("chunk_overlap", 50), step=10)
        new_chunking = st.selectbox("Chunking Strategy", ["recursive", "fixed", "semantic"], index=0)
        new_retrieval = st.selectbox("Retrieval Strategy", ["hybrid", "dense", "bm25"], index=0)
    with col_c2:
        new_alpha = st.slider("Hybrid Alpha (Dense vs BM25)", 0.0, 1.0, float(champ.get("hybrid_alpha", 0.65)), 0.05)
        new_top_k = st.slider("Top K", 1, 15, int(champ.get("top_k", 5)))
        new_rerank_enabled = st.checkbox("Enable Reranker", value=champ.get("reranker_enabled", True))
        new_rerank_top_n = st.slider("Rerank Top N", 1, 8, int(champ.get("rerank_top_n", 3)))
        new_llm = st.selectbox("LLM Generator", ["local", "heuristic", "openai", "anthropic"], index=0)

    if st.button("💾 Apply Configuration to Active Pipeline", type="primary"):
        updated_strat = {
            "name": "User-Customized Strategy",
            "chunk_size": new_chunk_size,
            "chunk_overlap": new_overlap,
            "chunking_strategy": new_chunking,
            "retrieval_strategy": new_retrieval,
            "hybrid_alpha": new_alpha,
            "top_k": new_top_k,
            "reranker_enabled": new_rerank_enabled,
            "rerank_top_n": new_rerank_top_n,
            "llm_provider": new_llm
        }
        try:
            r = requests.post(f"{API_BASE}/api/optimizer/champion", json=updated_strat, timeout=5)
            if r.status_code == 200:
                st.success("Champion strategy updated successfully!")
                st.rerun()
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))
