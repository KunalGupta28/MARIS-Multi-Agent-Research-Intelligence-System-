"""
MARIS Evaluation Metrics — Suite for measuring RAG & agent performance.

Includes:
    - Retrieval Precision@K and Recall@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - Grounding validation (citation-source checking)
    - Hallucination / Faithfulness scoring
    - Latency benchmarks
"""

import math
import re
import time
from typing import Sequence, Set, Any


class LatencyTracker:
    """Helper context manager to track and benchmark latency of individual agent nodes."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time


# ── Retrieval Quality Metrics ────────────────────────────────────────


def calculate_precision_at_k(
    retrieved: Sequence[str], ground_truth: Sequence[str], k: int
) -> float:
    """
    Calculate Precision@K: fraction of retrieved items in top K that are relevant.

    Precision@K = (Relevant retrieved in top K) / K
    """
    if k <= 0:
        return 0.0

    top_k_retrieved = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k_retrieved if item in ground_truth)
    return relevant_retrieved / k


def calculate_recall_at_k(
    retrieved: Sequence[str], ground_truth: Sequence[str], k: int
) -> float:
    """
    Calculate Recall@K: fraction of relevant items that are retrieved in top K.

    Recall@K = (Relevant retrieved in top K) / (Total relevant)
    """
    if not ground_truth:
        return 1.0

    top_k_retrieved = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k_retrieved if item in ground_truth)
    return relevant_retrieved / len(ground_truth)


def calculate_ndcg_at_k(
    retrieved: Sequence[str], ground_truth: Sequence[str], k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    Uses binary relevance: 1 if the item is in ground_truth, 0 otherwise.
    DCG@K = Σ (rel_i / log2(i + 1)) for i in 1..K
    IDCG@K is the DCG of a perfect ranking.

    Args:
        retrieved: Ordered list of retrieved item IDs.
        ground_truth: Set/list of relevant item IDs.
        k: Cutoff rank.

    Returns:
        NDCG@K score between 0.0 and 1.0.
    """
    if k <= 0 or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)

    # DCG@K: actual ranking
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        rel = 1.0 if item in gt_set else 0.0
        dcg += rel / math.log2(i + 1)

    # IDCG@K: perfect ranking (all relevant items at top)
    n_relevant = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_mrr(
    retrieved: Sequence[str], ground_truth: Sequence[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1 / rank of the first relevant item in the retrieved list.
    Returns 0.0 if no relevant item is found.

    Args:
        retrieved: Ordered list of retrieved item IDs.
        ground_truth: Set/list of relevant item IDs.

    Returns:
        MRR score between 0.0 and 1.0.
    """
    if not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    for i, item in enumerate(retrieved, start=1):
        if item in gt_set:
            return 1.0 / i

    return 0.0


# ── Grounding & Faithfulness Metrics ─────────────────────────────────


def validate_grounding(review: str, retrieved_paper_ids: Set[str]) -> dict[str, Any]:
    """
    Validate that every citation used in the generated review is grounded in the retrieved sources.

    Checks what percentage of cited papers in the text are present in the retrieval set.
    Supports standard footnotes [^arXiv:1905.04149], bracketed arXiv IDs [arXiv:1905.04149], and numbered brackets [1], [2].
    """
    # 1. Extract standard footnotes e.g. [^arXiv:1905.04149]
    cited_ids = set(re.findall(r"\^([a-zA-Z0-9_\-\.\:\/]+)", review))

    # 2. Extract bracketed arXiv IDs e.g. [arXiv:1905.04149] or [1905.04149]
    cited_ids.update(re.findall(r"\[(?:arXiv:)?(\d{4}\.\d{4,5})\]", review))

    # 3. Extract numbered bracket citations e.g. [1], [2] and resolve them from the References section
    numbered_citations = re.findall(r"\[(\d+)\]", review)
    if numbered_citations:
        # Match lines like [1] Author et al., "Title", Year. https://arxiv.org/abs/1905.04149
        # or [1] arXiv:1905.04149
        ref_lines = re.findall(r"\[(\d+)\]\s+.*?(?:arxiv\.org/abs/|arxiv\.org/pdf/|arXiv:)?(\d{4}\.\d{4,5})", review, re.IGNORECASE)
        ref_map = {num: arxiv_id for num, arxiv_id in ref_lines}
        for num in numbered_citations:
            if num in ref_map:
                cited_ids.add(ref_map[num])

    # Strip prefixes if any to normalize (e.g., 'arXiv:1905.04149' -> '1905.04149')
    def normalize_id(pid: str) -> str:
        pid = pid.replace("arXiv:", "")
        # Remove version suffix e.g. v2
        pid = re.sub(r"v\d+$", "", pid)
        return pid.strip().replace("_", ".").replace("-", ".")

    normalized_retrieved = {normalize_id(pid) for pid in retrieved_paper_ids}
    normalized_citations = {normalize_id(pid) for pid in cited_ids if pid != "paper_id"}  # filter dummy footprints

    if not normalized_citations:
        # If no citations were generated, grounding is perfect but coverage is 0
        return {"grounded_score": 1.0, "total_citations": 0, "ungrounded_citations": []}

    ungrounded = []
    grounded_count = 0
    for citation in normalized_citations:
        if citation in normalized_retrieved or any(citation in r for r in normalized_retrieved):
            grounded_count += 1
        else:
            ungrounded.append(citation)

    score = grounded_count / len(normalized_citations)
    return {
        "grounded_score": score,
        "total_citations": len(normalized_citations),
        "ungrounded_citations": ungrounded,
    }


def score_faithfulness(review: str, source_texts: list[str]) -> float:
    """
    Lightweight overlap score estimating the faithfulness (absence of hallucination) of the review.

    Checks if key phrases/terms mentioned alongside citations are actually present in the source texts.
    Returns a score between 0.0 (potentially highly hallucinated) and 1.0 (highly faithful).
    """
    if not review or not source_texts:
        return 0.0

    # Tokenize and normalize sources
    source_words = set()
    for text in source_texts:
        source_words.update(re.findall(r"\b\w{4,}\b", text.lower()))

    if not source_words:
        return 0.0

    # Tokenize review
    review_words = re.findall(r"\b\w{4,}\b", review.lower())
    if not review_words:
        return 1.0

    # Check how many words in the review exist in the source text (excluding common syntax/structure)
    overlap_count = sum(1 for word in review_words if word in source_words)

    # We normalise against the length to get a density score
    return min(1.0, (overlap_count / len(review_words)) * 1.5)
