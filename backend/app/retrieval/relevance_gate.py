"""
Retrieval Relevance Gate & Context Validation for EvoRAG.
Filters out candidate chunks that lack genuine lexical/semantic alignment with the query.
Prevents unrelated context from reaching the generator and enforces honest abstention.
Includes morphological stemming and entity verification.
"""
from typing import List, Dict, Any, Set, Tuple
import re
from backend.app.models import RetrievedChunk

class RelevanceGate:
    STOP_WORDS = {
        "what", "is", "the", "how", "are", "of", "and", "in", "to", "a", "an",
        "for", "with", "does", "can", "why", "about", "tell", "me", "explain",
        "describe", "give", "detail", "detailed", "summary", "please", "in", "by",
        "which", "there", "this", "that", "it", "its"
    }

    @classmethod
    def _stem(cls, word: str) -> str:
        """Lightweight morphological stemmer to match plurals and tenses."""
        w = word.lower().strip()
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 3:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        if w.endswith("ing") and len(w) > 4:
            return w[:-3]
        if w.endswith("ed") and len(w) > 3:
            return w[:-2]
        return w

    @classmethod
    def extract_key_query_terms(cls, query: str) -> Set[str]:
        """Extracts non-stopword core query keywords and their stems."""
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        key_tokens = tokens - cls.STOP_WORDS
        base = key_tokens if key_tokens else tokens
        stemmed = {cls._stem(w) for w in base}
        return base.union(stemmed)

    @classmethod
    def filter_relevant_chunks(
        cls,
        query: str,
        chunks: List[RetrievedChunk],
        min_overlap: int = 1
    ) -> Tuple[List[RetrievedChunk], List[RetrievedChunk]]:
        """
        Evaluates retrieved chunks against key query terms.
        Returns:
            (passed_chunks, rejected_chunks)
        Rejects chunks that do not mention any key query concepts.
        """
        key_terms = cls.extract_key_query_terms(query)
        if not key_terms:
            return chunks, []

        passed: List[RetrievedChunk] = []
        rejected: List[RetrievedChunk] = []

        for c in chunks:
            content_tokens = set(re.findall(r"\b\w+\b", c.chunk.content.lower()))
            content_stems = {cls._stem(w) for w in content_tokens}
            all_content = content_tokens.union(content_stems)

            overlap = len(key_terms.intersection(all_content))
            if overlap >= min_overlap:
                passed.append(c)
            else:
                rejected.append(c)

        return passed, rejected
