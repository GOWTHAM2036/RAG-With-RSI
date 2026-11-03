"""
Failure mode analyzer and bottleneck diagnostic engine.
"""
from typing import Dict, Any, List
from backend.app.models import EvaluationReport, RAGStrategy

class PipelineAnalyzer:
    @staticmethod
    def analyze_report(report: EvaluationReport, strategy: RAGStrategy) -> Dict[str, Any]:
        """
        Analyzes evaluation metrics to identify the primary bottleneck in the RAG pipeline
        and generates actionable mutation suggestions.
        """
        retrieval = report.retrieval_metrics
        answer = report.answer_metrics
        failures = report.failure_analysis
        
        diagnostics = []
        mutations_suggested = []
        primary_bottleneck = "balanced"

        # 1. Retrieval Bottleneck Analysis
        if retrieval.hit_rate < 0.65 or retrieval.context_recall < 0.5:
            primary_bottleneck = "retrieval_miss"
            diagnostics.append(
                f"Low retrieval hit rate ({retrieval.hit_rate:.2f}) and recall ({retrieval.context_recall:.2f}). "
                "The retriever failed to surface relevant chunks in top-k."
            )
            # Recommendations: increase top_k, adjust chunk size, switch to hybrid
            if strategy.retrieval_strategy != "hybrid":
                mutations_suggested.append({"param": "retrieval_strategy", "value": "hybrid"})
            if strategy.top_k < 8:
                mutations_suggested.append({"param": "top_k", "value": min(10, strategy.top_k + 3)})
            if strategy.chunk_size < 500:
                mutations_suggested.append({"param": "chunk_size", "value": 600})

        # 2. Reranker Bottleneck Analysis
        if failures.get("reranker_drops", 0) > 1:
            diagnostics.append(
                f"Reranker dropped {failures['reranker_drops']} relevant chunks. Rerank top_n may be too restrictive."
            )
            mutations_suggested.append({"param": "rerank_top_n", "value": min(6, strategy.rerank_top_n + 2)})

        # 3. Groundedness / Faithfulness Analysis
        if answer.faithfulness < 0.7:
            if primary_bottleneck == "balanced":
                primary_bottleneck = "hallucination_unfaithful"
            diagnostics.append(
                f"Low faithfulness ({answer.faithfulness:.2f}) indicates potential hallucination or context truncation."
            )
            mutations_suggested.append({"param": "prompt_template", "value": "citation_strict"})
            mutations_suggested.append({"param": "temperature", "value": 0.1})
            if strategy.chunk_overlap < 80:
                mutations_suggested.append({"param": "chunk_overlap", "value": 100})

        # 4. Latency Analysis
        if report.avg_latency_ms > 400.0:
            diagnostics.append(f"High latency ({report.avg_latency_ms:.1f}ms). Consider optimizing top_k or reranker.")
            if strategy.reranker_enabled and strategy.top_k > 8:
                mutations_suggested.append({"param": "top_k", "value": 5})

        if not diagnostics:
            diagnostics.append("Pipeline achieved high precision, recall, and groundedness across benchmark queries.")

        return {
            "primary_bottleneck": primary_bottleneck,
            "diagnostics": diagnostics,
            "suggested_mutations": mutations_suggested,
            "overall_health_score": round(report.composite_fitness * 100.0, 1)
        }
