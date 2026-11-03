"""
EvoRAG Interactive Dashboard.
Multi-page Streamlit application providing Knowledge Ingestion, Agent Chat,
Local LLM Status & Live Logs, Evolutionary Genetic Optimizer, and Benchmark Hub.
"""
import streamlit as st
import requests
import json
import pandas as pd
import time
from typing import Dict, Any

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="EvoRAG - Evolutionary & Self-Optimizing RAG",
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
        r = requests.get(f"{API_BASE}/api/status", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_llm_status():
    try:
        r = requests.get(f"{API_BASE}/api/llm/status", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_documents():
    try:
        r = requests.get(f"{API_BASE}/api/documents", timeout=2)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

# Sidebar & System Overview
status_data = fetch_status()
llm_status = fetch_llm_status()

with st.sidebar:
    st.markdown("### 🧬 EvoRAG System")
    if status_data:
        st.success("🟢 Backend Connected (Port 8000)")
        col1, col2 = st.columns(2)
        col1.metric("Docs", status_data.get("documents_count", 0))
        col2.metric("Chunks", status_data.get("chunks_count", 0))

        champ = status_data.get("champion_strategy", {})
        st.markdown(f"**Champion Pipeline:** `{champ.get('name', 'Default')}`")
        st.caption(f"Chunk: {champ.get('chunk_size')} | Overlap: {champ.get('chunk_overlap')} | Top-K: {champ.get('top_k')} | Alpha: {champ.get('hybrid_alpha')}")
    else:
        st.error("🔴 Backend Offline - Start backend server on port 8000")
        st.caption("Run: `uvicorn backend.app.main:app --port 8000`")

    st.markdown("---")
    st.markdown("### 🖥️ Local LLM (Ollama)")
    if llm_status and llm_status.get("status") == "online":
        st.success(f"🟢 **Online**: `{llm_status.get('url')}`")
        st.caption(f"Default Model: `{llm_status.get('active_model')}`")
        if llm_status.get("available_models"):
            st.caption(f"Installed: {', '.join(llm_status['available_models'][:3])}")
    else:
        st.warning("⚠️ **Local LLM Offline**\nOllama is not responding. Start Ollama (`ollama serve`).")

    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    if st.button("📥 Load Sample AI Knowledge", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/api/ingest/samples", timeout=5)
            if r.status_code == 200:
                st.success(r.json().get("message", "Sample knowledge loaded."))
                st.rerun()
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

    if st.button("📥 Load RSI-RAG Dataset (HotpotQA)", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/api/ingest/rsi_dataset", json={"limit": 25}, timeout=10)
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
st.markdown('<div class="main-header">🧬 EvoRAG: Evolutionary & Self-Optimizing RAG</div>', unsafe_allow_html=True)
st.caption("Autonomous Hyperparameter Search • Local LLM Live Logs • Self-Reflective Agentic Guardrails")

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

    # Empty knowledge base warning banner
    doc_count = status_data.get("documents_count", 0) if status_data else 0
    if doc_count == 0:
        st.warning(
            "⚠️ **Knowledge Base is Empty (Docs: 0, Chunks: 0)**: "
            "RAG requires context documents to retrieve answers. Click one of the buttons below to load documents instantly:"
        )
        col_b1, col_b2, _ = st.columns([1, 1, 2])
        if col_b1.button("📥 Load Built-in AI Knowledge"):
            try:
                requests.post(f"{API_BASE}/api/ingest/samples", timeout=5)
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if col_b2.button("📥 Load RSI-RAG Dataset (HotpotQA)"):
            try:
                requests.post(f"{API_BASE}/api/ingest/rsi_dataset", json={"limit": 25}, timeout=10)
                st.rerun()
            except Exception as e:
                st.error(str(e))
        st.markdown("---")

    col_input, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        use_agent = st.toggle("🧠 Enable Agentic Self-RAG", value=True, help="Executes Query Rewrite, Document Grading, and Faithfulness Verification Guardrails.")
        
        # Build dynamic LLM model options
        llm_options = ["local", "heuristic", "openai", "anthropic"]
        avail = (llm_status or {}).get("available_models", [])
        active_local_m = (llm_status or {}).get("active_model", "gemma3:4b")

        selected_llm = st.selectbox(
            "🤖 LLM Engine",
            llm_options,
            index=0 if (llm_status and llm_status.get("status") == "online") else 1,
            help="Choose generator: 'local' connects to Ollama, 'heuristic' runs offline."
        )

        if selected_llm == "local" and avail:
            local_model_choice = st.selectbox("Local Model", avail, index=0)
        else:
            local_model_choice = active_local_m

        session_id = st.text_input("Session ID", value="default_user")

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Source Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c['citation_id']} {c['doc_title']}** (Relevance: {c.get('score', 0):.2f})")
                        st.caption(c.get("snippet", ""))
            if msg.get("llm_debug_info"):
                with st.expander("🖥️ Local LLM Execution Logs & Trace"):
                    st.json(msg["llm_debug_info"])

    query_input = st.chat_input("Ask a question about your ingested documents...")

    if query_input:
        st.session_state.messages.append({"role": "user", "content": query_input})
        with st.chat_message("user"):
            st.markdown(query_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and synthesizing with Local LLM..."):
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
                    res = requests.post(endpoint, json=payload, timeout=45)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "")
                        citations = data.get("citations", [])
                        chunks = data.get("retrieved_chunks", [])
                        agent_steps = data.get("agent_steps", [])
                        latency = data.get("latency_ms", 0)
                        faithfulness = data.get("faithfulness_score", 1.0)
                        llm_debug = data.get("llm_debug_info")

                        st.markdown(answer)
                        st.caption(f"⏱️ Latency: {latency:.1f}ms | 🎯 Faithfulness: {faithfulness:.2f}")

                        if citations:
                            with st.expander("📚 Source Citations & Quotes"):
                                for c in citations:
                                    st.markdown(f"**{c['citation_id']} {c['doc_title']}** (Score: {c.get('score', 0):.2f})")
                                    st.caption(c.get("snippet", ""))

                        # LLM Execution Logs & Trace
                        if llm_debug:
                            with st.expander("🖥️ Local LLM Execution Logs & Trace", expanded=(llm_debug.get("status") != "success")):
                                st.markdown(f"**Status:** `{llm_debug.get('status')}` | **Provider:** `{llm_debug.get('provider')}` | **Model:** `{llm_debug.get('model_used')}`")
                                if llm_debug.get("endpoint_used"):
                                    st.caption(f"Endpoint: `{llm_debug.get('endpoint_used')}` (Latency: {llm_debug.get('latency_ms', 0):.1f}ms)")
                                if llm_debug.get("prompt_preview"):
                                    st.markdown("**Prompt Sent to LLM:**")
                                    st.text(llm_debug.get("prompt_preview"))
                                if llm_debug.get("raw_output"):
                                    st.markdown("**Raw LLM Output:**")
                                    st.text(llm_debug.get("raw_output"))
                                if llm_debug.get("errors"):
                                    st.error("Errors encountered:\n" + "\n".join(llm_debug["errors"]))

                        if agent_steps:
                            with st.expander("🔍 Agent Reasoning Trace (State Machine)"):
                                for s in agent_steps:
                                    st.markdown(f"**Step {s['step']}: `{s['node']}`** - {s['description']}")
                                    if s.get("details"):
                                        st.json(s["details"])

                        if chunks:
                            with st.expander("📄 Retrieved Context Chunks"):
                                for idx, rc in enumerate(chunks):
                                    chunk_obj = rc["chunk"]
                                    st.markdown(f"**Rank {rc.get('rank', idx+1)}** | Source: `{rc.get('retrieval_source')}` | Score: `{rc.get('score'):.4f}`")
                                    st.text(chunk_obj.get("content", ""))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "llm_debug_info": llm_debug
                        })
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with backend: {e}")

# ----------------- TAB 2: Evolutionary Optimizer -----------------
with tab_optimizer:
    st.markdown("#### 🧬 Evolutionary Genetic Algorithm Hyperparameter Tuning")
    st.caption("Evolves chunk sizes, overlap windows, retrieval strategy, hybrid alpha, and reranking parameters to maximize benchmark fitness.")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        pop_size = st.slider("Population Size", min_value=4, max_value=16, value=6, step=2)
        num_gens = st.slider("Number of Generations", min_value=2, max_value=8, value=3, step=1)
    with col_opt2:
        mutation_rate = st.slider("Mutation Rate", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        crossover_rate = st.slider("Crossover Rate", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    with col_opt3:
        w_faith = st.slider("Weight: Faithfulness", 0.0, 1.0, 0.35, 0.05)
        w_rel = st.slider("Weight: Answer Relevance", 0.0, 1.0, 0.25, 0.05)
        w_hit = st.slider("Weight: Retrieval Hit Rate", 0.0, 1.0, 0.20, 0.05)

    if st.button("🚀 Launch Genetic Optimization Experiment", type="primary", use_container_width=True):
        with st.spinner("Simulating generations of RAG evolution across golden benchmark..."):
            try:
                opt_payload = {
                    "name": f"Run-{time.strftime('%Y%m%d-%H%M%S')}",
                    "population_size": pop_size,
                    "num_generations": num_gens,
                    "mutation_rate": mutation_rate,
                    "crossover_rate": crossover_rate,
                    "weight_faithfulness": w_faith,
                    "weight_relevance": w_rel,
                    "weight_hit_rate": w_hit
                }
                res = requests.post(f"{API_BASE}/api/optimizer/run", json=opt_payload, timeout=60)
                if res.status_code == 200:
                    summary = res.json()
                    st.success(f"🎉 Optimization Complete! Champion Discovered: `{summary['champion_strategy']['name']}`")
                    st.session_state.latest_opt_summary = summary
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Optimization run failed: {e}")

    # Display results
    opt_summary = st.session_state.get("latest_opt_summary")
    if opt_summary:
        st.markdown("---")
        st.markdown("### 🏆 Discovered Champion Architecture")
        champ = opt_summary["champion_strategy"]
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Composite Fitness", f"{champ.get('fitness', 0):.4f}")
        m2.metric("Avg Latency", f"{champ.get('latency_ms', 0):.1f}ms")
        m3.metric("Chunk Size", champ.get("chunk_size"))
        m4.metric("Hybrid Alpha", champ.get("hybrid_alpha"))
        m5.metric("Top-K", champ.get("top_k"))

        st.markdown("#### 📈 Fitness Progression Across Generations")
        history = opt_summary.get("generation_history", [])
        if history:
            chart_data = pd.DataFrame([
                {
                    "Generation": f"Gen {h['generation']}",
                    "Best Fitness": h["best_fitness"],
                    "Avg Fitness": h["avg_fitness"],
                    "Worst Fitness": h["worst_fitness"]
                }
                for h in history
            ]).set_index("Generation")
            st.line_chart(chart_data)

        st.markdown("#### ⚖️ Pareto Frontier (Accuracy vs Latency Trade-Off)")
        pareto = opt_summary.get("pareto_frontier", [])
        if pareto:
            pareto_df = pd.DataFrame([
                {
                    "Strategy Name": p["name"],
                    "Fitness": p.get("fitness", 0),
                    "Latency (ms)": p.get("latency_ms", 0),
                    "Chunk Size": p.get("chunk_size"),
                    "Overlap": p.get("chunk_overlap"),
                    "Strategy": p.get("retrieval_strategy"),
                    "Alpha": p.get("hybrid_alpha"),
                    "Reranker": p.get("reranker_enabled")
                }
                for p in pareto
            ])
            st.dataframe(pareto_df, use_container_width=True)

# ----------------- TAB 3: Benchmark Hub -----------------
with tab_eval:
    st.markdown("#### 📊 RAG Benchmark & Performance Evaluation")
    st.caption("Calculates Hit Rate@K, MRR, Context Precision, Recall, Faithfulness, and detects pipeline failure modes.")

    if st.button("🧪 Evaluate Active Pipeline Against Golden Benchmark", type="primary"):
        with st.spinner("Evaluating retrieval & generation metrics..."):
            try:
                res = requests.post(f"{API_BASE}/api/evaluate", json={}, timeout=30)
                if res.status_code == 200:
                    report = res.json()
                    st.session_state.eval_report = report
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

        st.markdown("##### 📝 Generation & Groundedness Metrics")
        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        ac1.metric("Faithfulness", f"{am['faithfulness'] * 100:.1f}%")
        ac2.metric("Answer Relevance", f"{am['answer_relevance'] * 100:.1f}%")
        ac3.metric("Semantic Similarity", f"{am['semantic_similarity']:.3f}")
        ac4.metric("Hallucination Rate", f"{am['hallucination_rate'] * 100:.1f}%")
        ac5.metric("Avg Latency", f"{report['avg_latency_ms']:.1f}ms")

        st.markdown("##### ⚠️ Pipeline Bottleneck Diagnostics")
        st.info(f"Retrieval Success Rate: **{failures.get('retrieval_success_rate', 1.0) * 100:.1f}%** | Retrieval Misses: **{failures.get('retrieval_misses', 0)}** | Reranker Drops: **{failures.get('reranker_drops', 0)}** | Low Groundedness Count: **{failures.get('low_faithfulness_count', 0)}**")

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
    st.markdown("##### 📥 Batch Ingest Preprocessed Datasets")
    col_d1, col_d2 = st.columns(2)
    if col_d1.button("📥 Load Built-in AI Documentation", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/api/ingest/samples", timeout=5)
            if r.status_code == 200:
                st.success(r.json().get("message"))
                st.rerun()
        except Exception as e:
            st.error(str(e))

    if col_d2.button("📥 Load RSI-RAG Processed Dataset (50 records)", use_container_width=True):
        try:
            r = requests.post(f"{API_BASE}/api/ingest/rsi_dataset", json={"limit": 50}, timeout=15)
            if r.status_code == 200:
                st.success(r.json().get("message"))
                st.rerun()
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

    champ = status_data.get("champion_strategy", {}) if status_data else {}
    
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
