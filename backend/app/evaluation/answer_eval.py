"""
Answer evaluation metrics:
- Faithfulness (Context Groundedness)
- Answer Relevance (Alignment with user query)
- Semantic Similarity (Against ground truth answer)
- Hallucination Rate
"""
from typing import List, Dict, Any, Set
import re
from backend.app.models import RetrievedChunk, AnswerMetrics

class AnswerEvaluator:
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @classmethod
    def compute_faithfulness(cls, answer: str, context_chunks: List[RetrievedChunk]) -> float:
        """
        Calculates the proportion of sentences in the generated answer
        that are factually grounded in the provided context chunks.
        """
        if not answer.strip():
            return 0.0
        if not context_chunks:
            return 0.0

        all_context_tokens: Set[str] = set()
        for c in context_chunks:
            all_context_tokens.update(cls._tokenize(c.chunk.content))

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
        if not sentences:
            sentences = [answer]

        supported_count = 0
        stop_words = {"this", "that", "there", "is", "are", "was", "were", "the", "a", "an", "in", "on", "to", "for", "with", "by"}

        for s in sentences:
            s_tokens = set(cls._tokenize(s)) - stop_words
            if not s_tokens:
                supported_count += 1
                continue

            overlap = len(s_tokens.intersection(all_context_tokens))
            ratio = overlap / len(s_tokens)
            if ratio >= 0.5:
                supported_count += 1

        return supported_count / len(sentences)

    @classmethod
    def compute_answer_relevance(cls, query: str, answer: str) -> float:
        """
        Calculates query-answer alignment based on key intent tokens and length adequacy.
        """
        if not answer.strip() or not query.strip():
            return 0.0

        q_tokens = set(cls._tokenize(query))
        stop_words = {"what", "is", "how", "why", "when", "where", "can", "does", "the", "a", "an", "in", "to"}
        key_q_tokens = q_tokens - stop_words
        if not key_q_tokens:
            key_q_tokens = q_tokens

        a_tokens = set(cls._tokenize(answer))
        overlap = len(key_q_tokens.intersection(a_tokens))
        token_coverage = overlap / len(key_q_tokens) if key_q_tokens else 0.0

        # Length penalty if answer is trivially short (< 5 words)
        length_factor = min(1.0, len(cls._tokenize(answer)) / 8.0)

        return min(1.0, token_coverage * 0.7 + length_factor * 0.3)

    @classmethod
    def compute_semantic_similarity(cls, answer: str, ground_truth_answer: str) -> float:
        """
        Jaccard & token F1 similarity between predicted answer and ground truth answer.
        """
        if not ground_truth_answer.strip():
            return 1.0
        if not answer.strip():
            return 0.0

        pred_tokens = cls._tokenize(answer)
        gt_tokens = cls._tokenize(ground_truth_answer)

        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)

        intersection = len(pred_set.intersection(gt_set))
        if intersection == 0:
            return 0.0

        precision = intersection / len(pred_set)
        recall = intersection / len(gt_set)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    @classmethod
    def evaluate_answer(
        cls,
        query: str,
        answer: str,
        context_chunks: List[RetrievedChunk],
        ground_truth_answer: str = ""
    ) -> AnswerMetrics:
        faithfulness = cls.compute_faithfulness(answer, context_chunks)
        relevance = cls.compute_answer_relevance(query, answer)
        similarity = cls.compute_semantic_similarity(answer, ground_truth_answer)
        hallucination = max(0.0, 1.0 - faithfulness)

        return AnswerMetrics(
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(relevance, 4),
            semantic_similarity=round(similarity, 4),
            hallucination_rate=round(hallucination, 4)
        )
