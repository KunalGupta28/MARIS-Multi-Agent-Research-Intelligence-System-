"""
MARIS Tests — Evaluation metrics and grounding suite tests.
"""

import pytest

from src.eval.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    validate_grounding,
    score_faithfulness,
)


class TestEvaluationMetrics:
    def test_precision_at_k(self):
        retrieved = ["paper1", "paper2", "paper3", "paper4"]
        ground_truth = ["paper2", "paper4", "paper5"]

        # Precision@1: "paper1" is not in ground truth -> 0 / 1 = 0.0
        assert calculate_precision_at_k(retrieved, ground_truth, k=1) == 0.0

        # Precision@2: "paper2" is in ground truth -> 1 / 2 = 0.5
        assert calculate_precision_at_k(retrieved, ground_truth, k=2) == 0.5

        # Precision@4: "paper2", "paper4" are in ground truth -> 2 / 4 = 0.5
        assert calculate_precision_at_k(retrieved, ground_truth, k=4) == 0.5

        # Handle out of bounds / edge cases
        assert calculate_precision_at_k(retrieved, ground_truth, k=0) == 0.0

    def test_recall_at_k(self):
        retrieved = ["paper1", "paper2", "paper3", "paper4"]
        ground_truth = ["paper2", "paper4", "paper5"]

        # Recall@2: only "paper2" retrieved -> 1 / 3 = 0.333...
        assert calculate_recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1 / 3)

        # Recall@4: "paper2" and "paper4" retrieved -> 2 / 3 = 0.666...
        assert calculate_recall_at_k(retrieved, ground_truth, k=4) == pytest.approx(2 / 3)

        # Empty ground truth -> defaults to 1.0 (perfect recall of nothing)
        assert calculate_recall_at_k(retrieved, [], k=2) == 1.0

    def test_validate_grounding_perfect(self):
        review = """
        According to the study [^arXiv:1905.04149], brain signals can be modeled.
        Additionally, Alice et al. [^2012.06753] proposed neurohaptics.
        """
        retrieved_ids = {"arXiv:1905.04149", "arXiv:2012.06753"}
        
        result = validate_grounding(review, retrieved_ids)
        assert result["grounded_score"] == 1.0
        assert result["total_citations"] == 2
        assert len(result["ungrounded_citations"]) == 0

    def test_validate_grounding_failed(self):
        review = """
        Deep learning methods are excellent [^arXiv:1905.04149].
        However, another model exists [^arXiv:2202.99999].
        """
        retrieved_ids = {"arXiv:1905.04149"}  # missing the second paper
        
        result = validate_grounding(review, retrieved_ids)
        assert result["grounded_score"] == 0.5
        assert result["total_citations"] == 2
        assert "2202.99999" in result["ungrounded_citations"]

    def test_score_faithfulness(self):
        sources = [
            "We propose a State Space Model (SSM) for non-invasive brain signal analysis on EEG benchmarks.",
            "PyMuPDF parses section blocks by comparing font configurations."
        ]
        
        # Highly faithful review (high word overlap)
        faithful_review = "SSMs are proposed for brain signal analysis on EEG benchmarks."
        score1 = score_faithfulness(faithful_review, sources)
        assert score1 > 0.6
        
        # Hallucinated review (low word overlap)
        hallucinated_review = "The solar flare temperature exceeds millions of degrees in the corona."
        score2 = score_faithfulness(hallucinated_review, sources)
        assert score2 < 0.2
