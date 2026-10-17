from __future__ import annotations

import math
import statistics
from typing import Sequence


def compute_metrics(ranks: Sequence[int], ks=(1, 3, 5, 10, 15, 20)) -> dict[str, float]:
    if not ranks:
        return {}
    out: dict[str, float] = {}
    n = len(ranks)
    for k in ks:
        hits = [1.0 if r <= k else 0.0 for r in ranks]
        ndcg = [1.0 / math.log2(r + 1) if r <= k else 0.0 for r in ranks]
        # Single positive item -> Recall@K == Hit@K.
        out[f"Hit@{k}"] = sum(hits) / n
        out[f"Recall@{k}"] = out[f"Hit@{k}"]
        out[f"NDCG@{k}"] = sum(ndcg) / n
    out["MRR"] = sum(1.0 / r for r in ranks) / n
    out["MeanRank"] = sum(ranks) / n
    out["MedianRank"] = float(statistics.median(ranks))
    return out
