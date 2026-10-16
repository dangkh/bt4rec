from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def target_rank(
    ranking: Sequence[str],
    target: str,
) -> int:
    ranking = [
        str(x)
        for x in ranking
    ]
    try:
        return (
            ranking.index(
                str(target)
            )
            + 1
        )
    except ValueError:
        return len(
            ranking
        ) + 1


def metrics_from_ranks(
    ranks: List[int],
    ks: Sequence[int] = (
        1,
        3,
        5,
        10,
    ),
) -> Optional[Dict[str, float]]:
    if not ranks:
        return None

    n = len(ranks)
    out: Dict[str, float] = {}

    for k in ks:
        out[f"Hit@{k}"] = float(
            sum(
                r <= k
                for r in ranks
            )
            / n
        )
        out[f"NDCG@{k}"] = float(
            sum(
                (
                    1.0
                    / math.log2(
                        r + 1
                    )
                )
                if r <= k
                else 0.0
                for r in ranks
            )
            / n
        )

    out["MRR"] = float(
        sum(
            1.0 / r
            for r in ranks
        )
        / n
    )
    out["MeanRank"] = float(
        np.mean(ranks)
    )
    out["MedianRank"] = float(
        np.median(ranks)
    )
    out["NumUsers"] = int(n)
    return out


def aggregate_mode(
    rows: List[Dict[str, Any]],
    mode: str,
    strict_complete: bool,
) -> Optional[Dict[str, float]]:
    ranks: List[int] = []

    for row in rows:
        rec = row.get(
            "rankings",
            {},
        ).get(
            mode
        )
        if not isinstance(
            rec,
            dict,
        ):
            continue

        if (
            strict_complete
            and not rec.get(
                "parse_complete"
            )
        ):
            continue

        rank = rec.get(
            "target_rank"
        )
        if rank is not None:
            ranks.append(
                int(rank)
            )

    return metrics_from_ranks(
        ranks
    )


def metric_gain(
    a: Optional[Dict[str, float]],
    b: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not a or not b:
        return None

    out: Dict[str, float] = {}

    for k in a:
        if (
            k not in b
            or k == "NumUsers"
        ):
            continue

        if k in {
            "MeanRank",
            "MedianRank",
        }:
            # Positive = A better.
            out[k] = float(
                b[k] - a[k]
            )
        else:
            out[k] = float(
                a[k] - b[k]
            )

    return out


def aggregate_complete_intersection(
    rows: List[Dict[str, Any]],
    modes: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Metrics per mode on users whose ranking was parse-complete in ALL modes."""
    keep = []
    for row in rows:
        recs = row.get("rankings", {})
        if all(isinstance(recs.get(m), dict) and recs[m].get("parse_complete") for m in modes):
            keep.append(row)
    out: Dict[str, Any] = {"num_users": len(keep)}
    for mode in modes:
        out[mode] = aggregate_mode(keep, mode, False)
    return out


def paired_bootstrap(
    rows: List[Dict[str, Any]],
    mode: str,
    baseline: str,
    metrics: Sequence[str] = ("NDCG@10", "Hit@10", "MRR"),
    n_boot: int = 2000,
    seed: int = 42,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Paired bootstrap over users of (mode - baseline); 95% CI and two-sided p."""
    pairs = []
    for row in rows:
        recs = row.get("rankings", {})
        a, b = recs.get(mode), recs.get(baseline)
        if isinstance(a, dict) and isinstance(b, dict) and a.get("target_rank") and b.get("target_rank"):
            pairs.append((int(a["target_rank"]), int(b["target_rank"])))
    if not pairs:
        return None

    def per_user(rank: int, name: str) -> float:
        if name == "MRR":
            return 1.0 / rank
        kind, k = name.split("@")
        k = int(k)
        if kind == "Hit":
            return float(rank <= k)
        return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    out: Dict[str, Dict[str, float]] = {}
    for name in metrics:
        diff = np.asarray([per_user(a, name) - per_user(b, name) for a, b in pairs])
        boots = diff[idx].mean(axis=1)
        p = 2.0 * min(float(np.mean(boots <= 0.0)), float(np.mean(boots >= 0.0)))
        out[name] = {
            "mean_diff": float(diff.mean()),
            "ci95_low": float(np.quantile(boots, 0.025)),
            "ci95_high": float(np.quantile(boots, 0.975)),
            "p_two_sided": min(1.0, p),
            "num_users": len(pairs),
        }
    return out
