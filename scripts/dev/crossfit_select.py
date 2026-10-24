#!/usr/bin/env python3
"""Cross-fitted selection of the CoMemTree variant (no test-set peeking).

The best Tree variant/order is chosen on one half of the users and evaluated on the
other half, and vice versa (K-fold). Every user is therefore scored by a variant
chosen WITHOUT that user's label. Also reports the optimistic "pick best on all
test users" number, so the selection bias is visible.

Usage (one --run per query-order run directory, LABEL=PATH):
  python scripts/dev/crossfit_select.py \
      --run o1=results/comemtree/CDs/<run_order1> \
      --run o2=results/comemtree/CDs/<run_order2> \
      --run full=results/comemtree/CDs/<run_full> \
      --modes tree,tree_v2 \
      --baseline latest_what_only --select-metric NDCG@5 --folds 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METRICS = ("NDCG@1", "NDCG@5", "Hit@5")


def per_user(rank: int, name: str) -> float:
    kind, k = name.split("@")
    k = int(k)
    if kind == "Hit":
        return float(rank <= k)
    return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def load_ranks(run: str, modes, baseline):
    label, path = run.split("=", 1)
    p = Path(path)
    if p.is_dir():
        p = p / "rankings.jsonl"
    out = defaultdict(dict)  # (label, mode) -> {uid: rank}
    with p.open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            uid = str(row["user_id"])
            for mode, rec in row.get("rankings", {}).items():
                if mode not in modes and mode != baseline:
                    continue
                if isinstance(rec, dict) and rec.get("target_rank"):
                    key = ("base", mode) if mode == baseline else (label, mode)
                    out[key][uid] = int(rec["target_rank"])
    return out


def fold_of(uid: str, k: int, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}:{uid}".encode()).hexdigest()[:8], 16) % k


def mean_metric(ranks: dict, users, name: str) -> float:
    return float(np.mean([per_user(ranks[u], name) for u in users]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=run_dir_or_rankings.jsonl")
    ap.add_argument("--modes", required=True, help="Comma list of Tree-family modes to select among")
    ap.add_argument("--baseline", default="latest_what_only")
    ap.add_argument("--select-metric", default="NDCG@5")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]

    table = {}
    for run in a.run:
        for key, ranks in load_ranks(run, modes, a.baseline).items():
            if key in table and key[0] == "base":
                if table[key] != ranks:
                    print(f"[WARN] baseline {a.baseline} differs across runs; using the first run's")
                continue
            table[key] = ranks
    base_key = ("base", a.baseline)
    if base_key not in table:
        raise SystemExit(f"Baseline {a.baseline} not found in any run")
    configs = [k for k in table if k != base_key]
    users = set(table[base_key])
    for k in configs:
        users &= set(table[k])
    users = sorted(users)
    if not users:
        raise SystemExit("No users shared by all configs")
    print(f"[INFO] {len(configs)} candidate configs, {len(users)} shared users, {a.folds} folds")

    folds = defaultdict(list)
    for u in users:
        folds[fold_of(u, a.folds, a.seed)].append(u)

    chosen_rank = {}
    picks = []
    for f, test_users in sorted(folds.items()):
        sel_users = [u for u in users if fold_of(u, a.folds, a.seed) != f]
        best = max(configs, key=lambda c: mean_metric(table[c], sel_users, a.select_metric))
        picks.append((f, best, len(sel_users), len(test_users)))
        for u in test_users:
            chosen_rank[u] = table[best][u]

    print("\nSelected per fold (selected on the OTHER folds):")
    for f, best, ns, nt in picks:
        print(f"  fold {f}: {best[0]}/{best[1]}  (select n={ns}, eval n={nt})")
    print("  pick counts:", dict(Counter(f"{b[0]}/{b[1]}" for _, b, _, _ in picks)))

    base = table[base_key]
    print(f"\n{'Config':42s} " + " ".join(f"{m:>8s}" for m in METRICS))
    print(f"{a.baseline:42s} " + " ".join(f"{mean_metric(base, users, m):8.4f}" for m in METRICS))
    print(f"{'CoMemTree (cross-fitted selection)':42s} "
          + " ".join(f"{mean_metric(chosen_rank, users, m):8.4f}" for m in METRICS))
    best_all = max(configs, key=lambda c: mean_metric(table[c], users, a.select_metric))
    print(f"{'[optimistic] best on all test: ' + '/'.join(best_all):42s} "
          + " ".join(f"{mean_metric(table[best_all], users, m):8.4f}" for m in METRICS))

    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(users), size=(a.n_boot, len(users)))
    print(f"\nPaired bootstrap: cross-fitted CoMemTree - {a.baseline}")
    for m in METRICS:
        d = np.asarray([per_user(chosen_rank[u], m) - per_user(base[u], m) for u in users])
        boots = d[idx].mean(axis=1)
        p = min(1.0, 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0))))
        print(f"  {m}: diff={d.mean():+.4f}  CI95=[{np.quantile(boots, .025):+.4f}, "
              f"{np.quantile(boots, .975):+.4f}]  p={p:.3f}")


if __name__ == "__main__":
    main()
