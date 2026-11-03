"""
Unit tests for evaluation metrics:
- Retrieval metrics (Hit rate, MRR, Precision, Recall, NDCG)
- Claim-Level Faithfulness & Hallucination detection
- Negation and Entity-Binding verification
- Semantic similarity and Token F1
"""
import pytest
from backend.app.models import Chunk, RetrievedChunk
from backend.app.evaluation.retrieval_eval import RetrievalEvaluator
from backend.app.evaluation.answer_eval import AnswerEvaluator

@pytest.fixture
def retrieved_data():
    c1 = Chunk(doc_id="d1", content="Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.")
    c2 = Chunk(doc_id="d2", content="Mitochondria are the powerhouses of the cell producing ATP energy.")
    rc1 = RetrievedChunk(chunk=c1, score=0.9, rank=1)
    rc2 = RetrievedChunk(chunk=c2, score=0.6, rank=2)
    return [rc1, rc2]

def test_retrieval_evaluation(retrieved_data):
    gt_contexts = ["Plants use photosynthesis to make glucose from sunlight."]
    
    hit_rate = RetrievalEvaluator.compute_hit_rate(retrieved_data, gt_contexts, k=2)
    assert hit_rate == 1.0

    mrr = RetrievalEvaluator.compute_mrr(retrieved_data, gt_contexts, k=2)
    assert mrr == 1.0

    precision = RetrievalEvaluator.compute_context_precision(retrieved_data, gt_contexts, k=2)
    assert precision > 0.0

    recall = RetrievalEvaluator.compute_context_recall(retrieved_data, gt_contexts, k=2)
    assert recall > 0.0

def test_answer_evaluation(retrieved_data):
    answer = "Photosynthesis produces glucose and oxygen from sunlight and water."
    gt_answer = "Plants generate glucose through photosynthesis using light."

    faithfulness = AnswerEvaluator.compute_faithfulness(answer, retrieved_data)
    assert faithfulness >= 0.7

    relevance = AnswerEvaluator.compute_answer_relevance("What does photosynthesis produce?", answer)
    assert relevance > 0.5

    similarity = AnswerEvaluator.compute_semantic_similarity(answer, gt_answer)
    assert similarity > 0.2

def test_claim_level_unsupported_detection(retrieved_data):
    # Completely unsupported statement
    hallucinated_answer = "Photosynthesis produces rocket fuel on Mars."
    claim_res = AnswerEvaluator.compute_claim_faithfulness(hallucinated_answer, retrieved_data)
    assert claim_res["claim_faithfulness"] == 0.0
    assert len(claim_res["unsupported_claims"]) > 0

    metrics = AnswerEvaluator.evaluate_answer(
        query="What does photosynthesis produce?",
        answer=hallucinated_answer,
        context_chunks=retrieved_data
    )
    assert metrics.claim_faithfulness == 0.0
    assert metrics.hallucination_rate == 1.0

def test_negation_contradiction_detection(retrieved_data):
    # Context says: Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.
    # Negated claim contradicts context:
    contradictory_answer = "Photosynthesis does not convert sunlight into glucose."
    claim_res = AnswerEvaluator.compute_claim_faithfulness(contradictory_answer, retrieved_data)
    assert claim_res["claim_faithfulness"] < 1.0
