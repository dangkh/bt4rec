"""Uniform cost accounting for CoMemTree and every baseline.

All methods write the same ``cost_summary.json`` layout so they can be put in one
table: LLM calls per user, input/output tokens per user, embedding calls and the
number of memory entries the method creates.

Conventions
-----------
* One *LLM call* = one prompt -> completion (one sample).  Batching several users
  into one ``model.generate`` does not change the count; ``generate_batches`` is
  reported separately.
* ``phase="online"``  : test-time work done for an evaluated user.
* ``phase="offline"`` : memory/index construction done once before evaluation
  (e.g. aMem4Rec global memory build).  Reported in total and amortised over the
  evaluated users.
* ``user_id=None``    : a shared call not attributable to one user (e.g. the single
  i2Agent instruction-knowledge call, an aMem consolidation merge).
* ``upstream``        : work reused from an earlier step that is NOT re-run here
  (e.g. behaviour extraction already precomputed for CoMemTree).  It is counted
  separately so it can be added to whichever methods share it.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

COST_SCHEMA = "bt4rec_cost_v1"


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(statistics.fmean(values)), "median": float(statistics.median(values)),
            "min": float(min(values)), "max": float(max(values))}


class CostTracker:
    def __init__(self, method: str) -> None:
        self.method = method
        self.calls: List[Dict[str, Any]] = []
        self.embedding: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "texts": 0})
        self.generate_batches: Dict[str, int] = defaultdict(int)
        self.upstream: Dict[str, Any] = {}

    # ------------------------------------------------------------------ record
    def llm(self, stage: str, user_id: Optional[str], input_tokens: int, output_tokens: int,
            phase: str = "online", **extra: Any) -> None:
        self.calls.append({"stage": stage, "user_id": None if user_id is None else str(user_id),
                           "input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
                           "phase": phase, **extra})

    def batch(self, stage: str, n: int = 1) -> None:
        self.generate_batches[stage] += int(n)

    def embed(self, phase: str, n_texts: int, n_calls: int = 1) -> None:
        self.embedding[phase]["calls"] += int(n_calls)
        self.embedding[phase]["texts"] += int(n_texts)

    def set_upstream(self, **info: Any) -> None:
        self.upstream.update(info)

    # ----------------------------------------------------------------- summary
    def _phase(self, phase: str, users: List[str]) -> Dict[str, Any]:
        rows = [c for c in self.calls if c["phase"] == phase]
        per_user = {u: {"calls": 0, "input_tokens": 0, "output_tokens": 0} for u in users}
        shared = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        by_stage: Dict[str, Dict[str, Any]] = {}
        for c in rows:
            s = by_stage.setdefault(c["stage"], {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                 "shared_calls": 0})
            s["calls"] += 1
            s["input_tokens"] += c["input_tokens"]
            s["output_tokens"] += c["output_tokens"]
            uid = c["user_id"]
            target = per_user.setdefault(uid, {"calls": 0, "input_tokens": 0, "output_tokens": 0}) \
                if uid is not None else shared
            if uid is None:
                s["shared_calls"] += 1
            target["calls"] += 1
            target["input_tokens"] += c["input_tokens"]
            target["output_tokens"] += c["output_tokens"]
        n = max(1, len(users))
        tot_in = sum(c["input_tokens"] for c in rows)
        tot_out = sum(c["output_tokens"] for c in rows)
        for name, s in by_stage.items():
            s["calls_per_user"] = s["calls"] / n
            s["input_tokens_per_user"] = s["input_tokens"] / n
            s["output_tokens_per_user"] = s["output_tokens"] / n
            s["generate_batches"] = int(self.generate_batches.get(name, 0))
        eval_vals = [per_user[u] for u in users if u in per_user]
        return {
            "llm_calls_total": len(rows),
            "input_tokens_total": tot_in,
            "output_tokens_total": tot_out,
            "total_tokens_total": tot_in + tot_out,
            # Amortised over the evaluated users (includes shared calls).
            "llm_calls_per_user": len(rows) / n,
            "input_tokens_per_user": tot_in / n,
            "output_tokens_per_user": tot_out / n,
            "total_tokens_per_user": (tot_in + tot_out) / n,
            # Distribution over users of the calls attributable to that user only.
            "attributable_calls_per_user": _stats([v["calls"] for v in eval_vals]),
            "attributable_input_tokens_per_user": _stats([v["input_tokens"] for v in eval_vals]),
            "attributable_output_tokens_per_user": _stats([v["output_tokens"] for v in eval_vals]),
            "shared_calls": shared,
            "by_stage": by_stage,
            "embedding": dict(self.embedding.get(phase, {"calls": 0, "texts": 0})),
        }

    def summary(self, eval_users: Iterable[str], memory: Optional[Dict[str, Any]] = None,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        users = [str(u) for u in eval_users]
        online = self._phase("online", users)
        offline = self._phase("offline", users)
        n = max(1, len(users))
        out = {
            "schema_version": COST_SCHEMA,
            "method": self.method,
            "num_eval_users": len(users),
            "online": online,
            "offline": offline,
            "upstream_reused": self.upstream,
            "end_to_end_per_eval_user": {
                "llm_calls": (online["llm_calls_total"] + offline["llm_calls_total"]) / n,
                "input_tokens": (online["input_tokens_total"] + offline["input_tokens_total"]) / n,
                "output_tokens": (online["output_tokens_total"] + offline["output_tokens_total"]) / n,
                "note": "online + offline (amortised). Upstream reused work is listed separately.",
            },
            "memory": memory or {},
        }
        if extra:
            out.update(extra)
        return out

    def per_user_rows(self) -> List[Dict[str, Any]]:
        return list(self.calls)

    def write(self, path: str | Path, eval_users: Iterable[str],
              memory: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None,
              calls_path: Optional[str | Path] = None) -> Dict[str, Any]:
        summary = self.summary(eval_users, memory, extra)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if calls_path is not None:
            with Path(calls_path).open("w", encoding="utf-8") as f:
                for row in self.calls:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return summary


def memory_block(total: int, per_user: Dict[str, int], kinds: Optional[Dict[str, int]] = None,
                 scope: str = "per_user", **extra: Any) -> Dict[str, Any]:
    """``scope``: 'per_user' (private memories) or 'global' (shared pool)."""
    vals = list(per_user.values())
    return {"memories_created_total": int(total), "scope": scope,
            "memories_per_user": _stats([float(v) for v in vals]) if vals else None,
            "memory_kinds": kinds or {}, **extra}


def output_token_count(token_ids: List[int], eos_id: Optional[int], pad_id: Optional[int]) -> int:
    """Generated length up to and including the first EOS (padding excluded)."""
    if eos_id is not None and eos_id in token_ids:
        return token_ids.index(eos_id) + 1
    if pad_id is not None:
        return sum(1 for t in token_ids if t != pad_id)
    return len(token_ids)
