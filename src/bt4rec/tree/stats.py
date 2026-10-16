from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np


def entropy_bits(dist: Dict[str, float], eps: float = 1e-12) -> float:
    vals = np.asarray([v for v in dist.values() if v > eps], dtype=np.float64)
    if len(vals) == 0:
        return 0.0
    vals = vals / vals.sum()
    return float(-(vals * np.log2(vals)).sum())


def js_divergence_bits(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence using log2. Range [0,1]."""
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    pv = np.asarray([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.asarray([q.get(k, 0.0) for k in keys], dtype=np.float64)
    pv = pv / max(float(pv.sum()), eps)
    qv = qv / max(float(qv.sum()), eps)
    m = 0.5 * (pv + qv)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], eps))))

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def normalize_dist(weights: Dict[str, float]) -> Dict[str, float]:
    z = float(sum(weights.values()))
    if z <= 0:
        return {}
    return {str(k): float(v / z) for k, v in weights.items() if v > 0}


def interpolate_dist(
    local: Dict[str, float],
    parent: Dict[str, float],
    lam: float,
) -> Dict[str, float]:
    keys = set(local) | set(parent)
    out = {
        k: lam * local.get(k, 0.0) + (1.0 - lam) * parent.get(k, 0.0)
        for k in keys
    }
    return normalize_dist(out)


def stable_context_id(context: Tuple[str, ...]) -> str:
    if not context:
        return "ROOT"
    h = hashlib.sha1("|".join(context).encode("utf-8")).hexdigest()[:12]
    return f"CTX_{len(context)}_{h}"
