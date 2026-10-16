"""Behavior -> frozen state assignment, shared by the Tree builder and inference.

Before the refactor the builder kept the raw KMeans label (Euclidean distance to
an *unnormalised* centre) while inference used cosine similarity to the
*normalised* centre, restricted to states whose majority mechanism matched.
The same window could therefore receive different states at train and test time.
``nearest_compatible_state`` is now the only rule; the builder relabels every
memory with it (``--label-assignment cosine``, default) before building sequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bt4rec.taxonomy import clustering_bucket_key, compatible_key_for_state


@dataclass
class Assignment:
    best_index: int
    best_similarity: float
    second_index: Optional[int]
    second_similarity: Optional[float]
    margin: Optional[float]
    accepted: bool
    reasons: List[str] = field(default_factory=list)


def nearest_compatible_state(
    scores: np.ndarray,
    candidate_indices: Sequence[int],
    min_similarity: float,
    min_margin: float,
) -> Assignment:
    """Top-1 cosine among compatible centroids + similarity/margin acceptance.

    ``scores`` holds the cosine similarity to EVERY centroid; only
    ``candidate_indices`` are eligible. A bucket with one centroid has no margin
    and is never rejected for margin (same as the legacy builder).
    """
    cand = np.asarray(list(candidate_indices), dtype=np.int64)
    if cand.size == 0:
        raise ValueError("No candidate states")
    local = np.asarray(scores, dtype=np.float64)[cand]
    order = np.argsort(-local, kind="stable")
    best = int(cand[order[0]])
    best_sim = float(local[order[0]])
    second = second_sim = margin = None
    if len(order) > 1:
        second = int(cand[order[1]])
        second_sim = float(local[order[1]])
        margin = best_sim - second_sim
    reasons: List[str] = []
    if best_sim < min_similarity:
        reasons.append("low_similarity")
    if margin is not None and margin < min_margin:
        reasons.append("low_margin")
    return Assignment(best, best_sim, second, second_sim, margin, not reasons, reasons)


class CompatibilityIndex:
    """Which frozen states a behavior may be mapped to (mirrors the KMeans buckets)."""

    def __init__(self, state_rows: Sequence[Dict], cluster_mode: str, constraint_level: str) -> None:
        self.cluster_mode = str(cluster_mode or "constrained")
        self.constraint_level = str(constraint_level or "mechanism")
        self.all_indices = list(range(len(state_rows)))
        self.by_key: Dict[Tuple[str, ...], List[int]] = {}
        if self.cluster_mode != "global":
            for i, row in enumerate(state_rows):
                key = compatible_key_for_state(row, self.constraint_level)
                self.by_key.setdefault(key, []).append(i)

    def candidates(self, behavior: Dict) -> Tuple[List[int], bool]:
        """Return (candidate indices, has_compatible_bucket)."""
        if self.cluster_mode == "global":
            return self.all_indices, True
        key = clustering_bucket_key(behavior, self.constraint_level)
        compatible = self.by_key.get(key)
        if compatible:
            return compatible, True
        return self.all_indices, False


def relabel_by_cosine(
    memories: Sequence[Dict],
    embeddings: np.ndarray,
    centers: np.ndarray,
    cluster_meta: Dict[int, Dict],
    cluster_mode: str,
    constraint_level: str,
) -> np.ndarray:
    """Replace KMeans labels with the nearest compatible normalised centroid.

    ``centers`` must already be L2-normalised (as returned by run_kmeans).
    Compatibility is the KMeans bucket, so a memory never leaves its bucket.
    """
    rows = [
        {"behavior_bucket_key": cluster_meta[i].get("behavior_bucket_key")}
        for i in range(len(centers))
    ]
    index = CompatibilityIndex(rows, cluster_mode, constraint_level)
    scores = np.asarray(embeddings, dtype=np.float32) @ np.asarray(centers, dtype=np.float32).T
    labels = np.empty(len(memories), dtype=np.int64)
    for i, mem in enumerate(memories):
        cand, ok = index.candidates(mem)
        if not ok:
            raise AssertionError(f"memory {i} has no compatible bucket during relabelling")
        labels[i] = nearest_compatible_state(scores[i], cand, -np.inf, -np.inf).best_index
    return labels
