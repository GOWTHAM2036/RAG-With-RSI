"""
Factory for instantiating rerankers.
"""
from backend.app.reranking.base import BaseReranker
from backend.app.reranking.heuristic_reranker import HeuristicReranker
from backend.app.reranking.cross_encoder_reranker import CrossEncoderReranker

class RerankerFactory:
    _instances = {}

    @classmethod
    def get_reranker(cls, reranker_type: str = "heuristic") -> BaseReranker:
        rtype = reranker_type.lower().strip()
        if rtype not in cls._instances:
            if rtype in ["cross_encoder", "cross-encoder", "ms-marco"]:
                cls._instances[rtype] = CrossEncoderReranker()
            else:
                cls._instances[rtype] = HeuristicReranker()
        return cls._instances[rtype]
