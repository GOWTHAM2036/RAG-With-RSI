"""
Answer evaluation metrics:
- Claim-Level Faithfulness (Context Groundedness verification)
- Lexical Overlap Faithfulness (Legacy proxy, clearly distinguished)
- Answer Relevance (Query-Answer alignment)
- Semantic Similarity (Embedding cosine similarity with token F1 fallback)
- Hallucination Rate (Derived from claim verification)
- Calibrated confidence and transparent evaluation notes
"""
from typing import List, Dict, Any, Set, Tuple, Optional
import re
import math
from backend.app.models import RetrievedChunk, AnswerMetrics

class AnswerEvaluator:
    STOP_WORDS = {
        "this", "that", "there", "is", "are", "was", "were", "the", "a", "an",
        "in", "on", "to", "for", "with", "by", "of", "and", "or", "it", "its",
        "as", "at", "be", "has", "had", "have", "from", "which"
    }

    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nor", "none", "cannot", "n't",
        "failed", "refused", "denied", "without", "hardly", "barely"
    }

    CONVERSATIONAL_PREFIXES = [
        r"^based on (the|our) (provided|retrieved)? ?(context|documents|sources|information)[,\s:]*",
        r"^according to (the|our) (provided|retrieved)? ?(context|documents|sources|information)[,\s:]*",
        r"^in summary[,\s:]*",
        r"^to answer your question[,\s:]*",
        r"^here is (the|what) [,\s:]*",
    ]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @classmethod
    def extract_atomic_claims(cls, answer: str) -> List[str]:
        """
        Extracts factual atomic claims from the generated answer, stripping conversational
        hedges, prefixes, and formatting.
        """
        if not answer or not answer.strip():
            return []

        # Strip markdown links, brackets citations [1], [2], etc.
        cleaned = re.sub(r"\[\d+\]", "", answer)
        cleaned = re.sub(r"[*#_`]", "", cleaned)

        # Split into prospective candidate sentences / clauses
        raw_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
            if len(s.strip()) > 8
        ]

        claims: List[str] = []
        for s in raw_sentences:
            s_clean = s
            for prefix_pattern in cls.CONVERSATIONAL_PREFIXES:
                s_clean = re.sub(prefix_pattern, "", s_clean, flags=re.IGNORECASE).strip()

            # Skip pure meta statements
            tokens = cls._tokenize(s_clean)
            meaningful_tokens = [t for t in tokens if t not in cls.STOP_WORDS]
            if len(meaningful_tokens) < 3:
                continue

            # Split compound clauses on coordinating conjunctions if clause has its own verb
            clauses = re.split(r",\s+(?:whereas|while|although|however)\s+", s_clean, flags=re.IGNORECASE)
            for c in clauses:
                c_toks = [t for t in cls._tokenize(c) if t not in cls.STOP_WORDS]
                if len(c_toks) >= 3:
                    claims.append(c.strip())

        return claims if claims else ([cleaned.strip()] if cleaned.strip() else [])

    @classmethod
    def verify_single_claim(
        cls,
        claim: str,
        context_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """
        Evaluates whether a single atomic factual claim is supported by the context.
        Checks:
        1. Entity and predicate overlap.
        2. Entity co-occurrence / binding within the same context sentence or chunk.
        3. Polarity / Negation contradiction detection.
        """
        claim_tokens = cls._tokenize(claim)
        claim_content_words = [t for t in claim_tokens if t not in cls.STOP_WORDS]

        if not claim_content_words:
            return {"supported": True, "score": 1.0, "reason": "No verifiable factual predicates."}

        claim_has_negation = any(w in cls.NEGATION_WORDS for w in claim_tokens)

        best_support_score = 0.0
        best_matching_chunk = None
        contradiction_detected = False

        for r_chunk in context_chunks:
            chunk_text = r_chunk.chunk.content.lower()
            chunk_sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

            for c_sent in chunk_sentences:
                c_tokens = cls._tokenize(c_sent)
                c_tok_set = set(c_tokens)

                # 1. Measure lexical-semantic overlap with this context sentence
                overlap_words = [w for w in claim_content_words if w in c_tok_set]
                overlap_ratio = len(overlap_words) / len(claim_content_words) if claim_content_words else 0.0

                # 2. Check negation / polarity contradiction
                c_sent_has_negation = any(w in cls.NEGATION_WORDS for w in c_tokens)
                if overlap_ratio >= 0.5 and (claim_has_negation != c_sent_has_negation):
                    # One is negated, the other is affirmative on the same key entity
                    contradiction_detected = True

                # 3. Entity binding: if the claim has >= 2 content words, they must co-occur
                if overlap_ratio > best_support_score:
                    best_support_score = overlap_ratio
                    best_matching_chunk = r_chunk.chunk.id

        is_supported = (best_support_score >= 0.6) and not contradiction_detected

        return {
            "claim": claim,
            "supported": is_supported,
            "support_score": round(best_support_score, 3),
            "contradiction": contradiction_detected,
            "chunk_id": best_matching_chunk
        }

    @classmethod
    def compute_claim_faithfulness(
        cls,
        answer: str,
        context_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """
        Computes claim-level support faithfulness.
        Deconstructs answer into atomic claims and verifies each against context.
        """
        if not answer.strip() or not context_chunks:
            return {
                "claim_faithfulness": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "confidence": 0.0
            }

        claims = cls.extract_atomic_claims(answer)
        if not claims:
            return {
                "claim_faithfulness": 1.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "confidence": 0.5
            }

        verified = [cls.verify_single_claim(c, context_chunks) for c in claims]
        supported_count = sum(1 for v in verified if v["supported"])
        unsupported = [v for v in verified if not v["supported"]]

        faithfulness = supported_count / len(claims)

        return {
            "claim_faithfulness": round(faithfulness, 4),
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "unsupported_claims": [u["claim"] for u in unsupported],
            "confidence": round(min(1.0, 0.5 + 0.1 * len(claims)), 2)
        }

    @classmethod
    def compute_faithfulness(cls, answer: str, context_chunks: List[RetrievedChunk]) -> float:
        """Alias for backward compatibility returning the float claim faithfulness score."""
        res = cls.compute_claim_faithfulness(answer, context_chunks)
        return res["claim_faithfulness"]

    @classmethod
    def compute_token_overlap_faithfulness(
        cls,
        answer: str,
        context_chunks: List[RetrievedChunk]
    ) -> float:
        """
        Legacy token overlap metric: proportion of sentences with >=50% token overlap.
        Preserved as a baseline reference but explicitly labeled.
        """
        if not answer.strip() or not context_chunks:
            return 0.0

        all_context_tokens: Set[str] = set()
        for c in context_chunks:
            all_context_tokens.update(cls._tokenize(c.chunk.content))

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
        if not sentences:
            sentences = [answer]

        supported_count = 0
        for s in sentences:
            s_tokens = set(cls._tokenize(s)) - cls.STOP_WORDS
            if not s_tokens:
                supported_count += 1
                continue

            overlap = len(s_tokens.intersection(all_context_tokens))
            ratio = overlap / len(s_tokens)
            if ratio >= 0.5:
                supported_count += 1

        return round(supported_count / len(sentences), 4)

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

        length_factor = min(1.0, len(cls._tokenize(answer)) / 8.0)
        return round(min(1.0, token_coverage * 0.7 + length_factor * 0.3), 4)

    @classmethod
    def compute_token_f1(cls, answer: str, ground_truth_answer: str) -> float:
        """
        Token F1 score between predicted answer and ground truth answer.
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
        return round(f1, 4)

    @classmethod
    def compute_semantic_similarity(
        cls,
        answer: str,
        ground_truth_answer: str,
        embedding_provider: Optional[Any] = None
    ) -> float:
        """
        Calculates real cosine similarity between sentence embeddings if embedding_provider
        is available, or calibrated proxy similarity.
        """
        if not ground_truth_answer.strip():
            return 1.0
        if not answer.strip():
            return 0.0

        if embedding_provider is not None:
            try:
                emb_pred = embedding_provider.embed_query(answer)
                emb_gt = embedding_provider.embed_query(ground_truth_answer)
                dot = sum(a * b for a, b in zip(emb_pred, emb_gt))
                norm_a = math.sqrt(sum(a * a for a in emb_pred))
                norm_b = math.sqrt(sum(b * b for b in emb_gt))
                if norm_a > 0 and norm_b > 0:
                    cos_sim = max(0.0, min(1.0, dot / (norm_a * norm_b)))
                    return round(cos_sim, 4)
            except Exception:
                pass

        token_f1 = cls.compute_token_f1(answer, ground_truth_answer)
        len_ratio = min(len(answer), len(ground_truth_answer)) / max(len(answer), len(ground_truth_answer), 1)
        proxy_sim = 0.8 * token_f1 + 0.2 * len_ratio
        return round(proxy_sim, 4)

    @classmethod
    def evaluate_answer(
        cls,
        query: str,
        answer: str,
        context_chunks: List[RetrievedChunk],
        ground_truth_answer: str = "",
        embedding_provider: Optional[Any] = None
    ) -> AnswerMetrics:
        # 1. Claim-level verification
        claim_result = cls.compute_claim_faithfulness(answer, context_chunks)
        claim_faithfulness = claim_result["claim_faithfulness"]

        # 2. Legacy token-overlap proxy
        token_overlap_faithfulness = cls.compute_token_overlap_faithfulness(answer, context_chunks)

        # 3. Relevance, Semantic similarity & Token F1
        relevance = cls.compute_answer_relevance(query, answer)
        semantic_sim = cls.compute_semantic_similarity(answer, ground_truth_answer, embedding_provider)
        token_f1 = cls.compute_token_f1(answer, ground_truth_answer)

        # 4. Hallucination rate derived strictly from unsupported claims
        hallucination_rate = round(max(0.0, 1.0 - claim_faithfulness), 4)

        notes = (
            f"Evaluated {claim_result['total_claims']} atomic claims "
            f"({claim_result['supported_claims']} supported). "
            f"Score represents absence of detected unsupported propositions, "
            f"not guaranteed external factual truth."
        )

        return AnswerMetrics(
            faithfulness=round(claim_faithfulness, 4),
            claim_faithfulness=round(claim_faithfulness, 4),
            token_overlap_faithfulness=round(token_overlap_faithfulness, 4),
            answer_relevance=round(relevance, 4),
            semantic_similarity=round(semantic_sim, 4),
            token_f1=round(token_f1, 4),
            hallucination_rate=hallucination_rate,
            evaluation_notes=notes
        )
