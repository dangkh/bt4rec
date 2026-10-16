"""Tiny deterministic dataset + fake LLM behaviours for offline tests.

Nothing here needs a GPU, a network connection or a real LLM:
- ``write_dataset`` creates items.json / user_sequences.json / user_negatives.json;
- ``fake_behavior`` deterministically labels a window (stands in for Gemma/Qwen);
- ``hash_embed`` is a bag-of-words hashing encoder (stands in for Qwen3-Embedding).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

GENRES = ["jazz", "rock", "classical", "hip hop", "country", "electronic", "folk", "blues"]
# Labels WITHOUT punctuation, so legacy and refactored normalisation agree.
SAFE_MECHANISMS = ["persistence", "deepening", "shifting", "broadening", "adjacent exploration"]
SAFE_SCOPES = ["same subcategory", "same creator", "related category", "mixed"]
DIRECTION_OF = {"persistence": "stable", "deepening": "deepen", "shifting": "shift",
                "broadening": "broaden", "adjacent exploration": "shift"}


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def write_dataset(root: Path, n_users: int = 60, n_items: int = 400, train_len: int = 22,
                  seed: int = 7) -> Path:
    rng = random.Random(seed)
    ds = root / "data" / "CDs"
    ds.mkdir(parents=True, exist_ok=True)
    items = {}
    for i in range(n_items):
        g = GENRES[i % len(GENRES)]
        items[f"I{i:04d}"] = {"title": f"{g.title()} Album {i}", "main_cat": f"CDs > {g}"}
    sequences, negatives = {}, {}
    ids = list(items)
    for u in range(n_users):
        uid = f"U{u:03d}"
        home = rng.randrange(len(GENRES))
        seq = []
        for t in range(train_len + 1):
            g = home if rng.random() < 0.7 else rng.randrange(len(GENRES))
            pool = [x for k, x in enumerate(ids) if k % len(GENRES) == g and x not in seq]
            seq.append(rng.choice(pool))
        sequences[uid] = {"train": seq[:-1], "test": [seq[-1]]}
        negatives[uid] = {"test_neg": rng.sample([x for x in ids if x not in seq], 60)}
    (ds / "items.json").write_text(json.dumps(items))
    (ds / "user_sequences.json").write_text(json.dumps(sequences))
    (ds / "user_negatives.json").write_text(json.dumps(negatives))
    return ds


def fake_behavior(user_id: str, items: List[Dict[str, str]], allow_punctuation: bool = False) -> Dict:
    key = _h(user_id, *[x["item_id"] for x in items])
    mech = SAFE_MECHANISMS[key % len(SAFE_MECHANISMS)]
    scope = SAFE_SCOPES[(key // 7) % len(SAFE_SCOPES)]
    if allow_punctuation and key % 5 == 0:
        scope = "same collection/series"
    cats = sorted({x["item_category"].split(">")[-1].strip() for x in items})
    sig_words = ["keeps", "anchor", "narrows", "widens", "moves", "returns", "variation", "steady"]
    sig = " ".join(sig_words[(key >> s) % len(sig_words)] for s in range(0, 24, 3))
    return {
        "behavior_explanation": f"User bought {len(items)} items in {', '.join(cats)}.",
        "recommendation_behavior": f"Prefers {' and '.join(cats)} albums with a consistent mood and style.",
        "semantic_focus": cats + ["albums"],
        "trajectory_signature": f"The user {sig} across consecutive choices",
        "mechanism": mech, "scope": scope, "direction": DIRECTION_OF[mech],
        "confidence": round(0.5 + (key % 50) / 100, 2),
    }


def hash_embed(texts: List[str], dim: int = 256) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in str(t).lower().replace(";", " ").replace("=", " ").split():
            h = _h("tok", tok)
            out[i, h % dim] += 1.0 if (h >> 20) & 1 else -1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 0, n, 1.0)


class FakeEncoder:
    """Drop-in replacement for QwenEncoder (same constructor/encode/release)."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = str(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return hash_embed(texts)

    def release(self) -> None:
        pass


class FakeRanker:
    """Stands in for LocalQwenRanker: deterministic pseudo-random JSON rankings that
    go through the REAL parser (sanitize_ranking), including some malformed outputs."""

    def __init__(self, *args, parse_fn=None, **kwargs) -> None:
        self.parse_fn = parse_fn
        self.n = 0

    def rank_jobs(self, jobs, batch_size=8, call_type="", max_retries=0):
        out = []
        for job in jobs:
            ids = [str(x["item_id"]) for x in job["candidate_items"]]
            key = _h(call_type, job["prebuilt_prompt"])
            order = sorted(range(1, len(ids) + 1), key=lambda i: _h(key, i))
            if key % 7 == 0:          # incomplete list -> repaired
                order = order[:12]
            if key % 11 == 0:         # duplicate label -> repaired
                order = order[:5] + order[:1] + order[5:]
            raw = json.dumps({"ranking": [f"C{i:02d}" for i in order], "reasoning": "x"})
            ranking, meta = self.parse_fn(None, raw, ids)
            meta["attempts"] = 1
            meta["call_type"] = call_type
            out.append((ranking, meta))
            self.n += 1
        return out

    def stats(self):
        return {"model": "fake", "generation_records": self.n}
