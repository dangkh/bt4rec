#!/usr/bin/env python3
"""Traditional recommendation baselines (Popularity, LightGCN, SASRec) for the
CoMemTree frozen protocol, trained the standard way.

Protocol
--------
Training data
  * ALL users in user_sequences.json, not only the evaluation users.
  * Only each user's ``train`` list. ``test`` (and any legacy ``val`` /
    ``validation`` field) is never used for training or as model input.
  * No history truncation for training. SASRec reads the last ``--max-len``
    items (default 50 for Amazon, 200 for MovieLens). The 10-20 item cut is
    only for LLM prompts.
  * Vocabulary = items that occur in at least one user's train list.

Model selection
  * Penultimate-style validation: train on train[:-1] and hold out train[-1]
    for every user with >= 2 train items.
  * Validation metric: full-catalog NDCG@10 (train[:-1] items masked) on a
    seeded sample of ``--val-users`` users. Early stopping with ``--patience``.
  * ``--refit`` (default on): retrain from scratch on the full train lists for
    the selected number of epochs, then evaluate. ``--no-refit`` evaluates the
    best validation checkpoint instead.

Evaluation
  * Exactly the frozen evaluation users (``--user-ids-file``) and their frozen
    20 candidates (``--candidate-file``). Input = the user's full train list.
  * Candidates never seen in training ("cold") get the lowest score, ranked
    among themselves by frozen candidate order. The run reports cold candidate
    and cold target rates and the metrics on the subset of users whose 20
    candidates are all warm.

Usage
-----
  python comemtree_baselines.py pop      --dataset CDs
  python comemtree_baselines.py lightgcn --dataset CDs
  python comemtree_baselines.py sasrec   --dataset CDs
  python comemtree_baselines.py sasrec   --dataset ML            # max_len 200
  python comemtree_baselines.py all      --dataset Video_Games   # pop + lightgcn + sasrec

Paths default to data/<dataset>/{items.json,user_sequences.json} and
data/<dataset>/protocol/{eval_users_1000.json,user_candidates_testpool1000_seed42.json};
override any of them with the flags below.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

KS = (1, 3, 5, 10, 15, 20)
DATASET_ALIASES = {
    "cds": "CDs", "cd": "CDs",
    "video_games": "Video_Games", "videogames": "Video_Games", "vg": "Video_Games", "video": "Video_Games",
    "ml": "ML", "ml1m": "ML", "ml-1m": "ML", "movielens": "ML",
}
DEFAULT_MAX_LEN = {"ML": 200}  # everything else (Amazon) -> 50


# =============================================================================
# I/O and data
# =============================================================================

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    out = []
    for v in x:
        if isinstance(v, dict):  # tolerate {"item_id": ...} rows
            v = v.get("item_id", v.get("item", v.get("asin", v.get("id"))))
        out.append(str(v))
    return out


def load_user_ids(path: str) -> List[str]:
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [x.strip() for x in text.splitlines() if x.strip()]
    if isinstance(obj, dict):
        obj = obj["users"] if isinstance(obj.get("users"), list) else list(obj)
    return [str(x) for x in obj]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Data:
    """All-user training data + frozen evaluation set."""

    def __init__(self, args: argparse.Namespace) -> None:
        raw = load_json(args.sequences)
        self.train: Dict[str, List[str]] = {}
        ignored_val = 0
        for uid, rec in raw.items():
            uid = str(uid)
            if not isinstance(rec, dict):
                raise ValueError(f"user {uid}: expected a dict with train/test")
            tr = as_str_list(rec.get("train"))
            ignored_val += sum(len(as_str_list(rec.get(k))) for k in ("val", "valid", "validation"))
            if tr:
                self.train[uid] = tr
        self.ignored_validation_items = ignored_val
        self.users: List[str] = sorted(self.train)
        self.user_index = {u: i for i, u in enumerate(self.users)}

        counts = Counter(i for seq in self.train.values() for i in seq)
        # Deterministic vocabulary ordering: by frequency then id.
        self.items: List[str] = [i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        self.item_index = {it: k for k, it in enumerate(self.items)}
        self.item_counts = np.asarray([counts[i] for i in self.items], dtype=np.float64)
        self.n_users, self.n_items = len(self.users), len(self.items)
        self.n_interactions = int(sum(len(s) for s in self.train.values()))

        # Frozen evaluation set.
        self.eval_users = load_user_ids(args.user_ids_file)
        if args.max_eval_users > 0:
            self.eval_users = self.eval_users[: args.max_eval_users]
        cand = load_json(args.candidate_file)
        if isinstance(cand, dict) and isinstance(cand.get("users"), dict):
            cand = cand["users"]
        self.candidates: Dict[str, List[str]] = {}
        self.targets: Dict[str, str] = {}
        self.negatives_in_history = 0
        for uid in self.eval_users:
            if uid not in self.train:
                raise ValueError(f"eval user {uid} has no train interactions in {args.sequences}")
            row = cand.get(uid)
            if row is None:
                raise ValueError(f"eval user {uid} missing from {args.candidate_file}")
            c = as_str_list(row.get("candidates", row.get("candidate_item_ids")))
            t = as_str_list(row.get("target", row.get("targets", row.get("ground_truth"))))
            test = as_str_list(raw[uid].get("test"))
            if len(t) != 1 or t != test:
                raise ValueError(f"eval user {uid}: frozen target {t} != sequences test {test}")
            if len(c) != args.expected_candidates or len(set(c)) != len(c) or t[0] not in c:
                raise ValueError(f"eval user {uid}: bad candidate row (n={len(c)})")
            if set(c) & (set(self.train[uid]) - {t[0]}):
                self.negatives_in_history += 1
            self.candidates[uid] = c
            self.targets[uid] = t[0]

    # --- splits -----------------------------------------------------------
    def sequences(self, holdout_last: bool) -> Dict[str, List[int]]:
        """Encoded train sequences; with holdout_last, train[-1] is removed."""
        out = {}
        for u, seq in self.train.items():
            s = seq[:-1] if holdout_last else seq
            if s:
                out[u] = [self.item_index[i] for i in s]
        return out

    def validation_users(self, n: int, seed: int) -> List[str]:
        pool = [u for u in self.users if len(self.train[u]) >= 2]
        rng = random.Random(seed)
        if 0 < n < len(pool):
            pool = rng.sample(pool, n)
        return pool

    def stats(self) -> Dict[str, Any]:
        lens = [len(s) for s in self.train.values()]
        cand_all = [i for u in self.eval_users for i in self.candidates[u]]
        cold_c = sum(i not in self.item_index for i in cand_all)
        cold_t = sum(self.targets[u] not in self.item_index for u in self.eval_users)
        warm_users = sum(all(i in self.item_index for i in self.candidates[u]) for u in self.eval_users)
        return {
            "n_train_users": self.n_users,
            "n_train_interactions": self.n_interactions,
            "n_vocab_items": self.n_items,
            "train_length_mean": float(np.mean(lens)),
            "train_length_median": float(np.median(lens)),
            "train_length_max": int(max(lens)),
            "ignored_validation_items_in_source": self.ignored_validation_items,
            "n_eval_users": len(self.eval_users),
            "eval_users_are_part_of_training": True,
            "test_items_used_for_training": False,
            "cold_candidate_rate": cold_c / len(cand_all),
            "cold_target_rate": cold_t / len(self.eval_users),
            "n_eval_users_all_candidates_warm": warm_users,
            "n_eval_users_with_negative_in_train_history": self.negatives_in_history,
            "target_in_own_train_history_rate": sum(
                self.targets[u] in set(self.train[u]) for u in self.eval_users) / len(self.eval_users),
        }


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {}
    n = float(len(ranks))
    out: Dict[str, float] = {}
    for k in KS:
        out[f"Hit@{k}"] = sum(r <= k for r in ranks) / n
        out[f"NDCG@{k}"] = sum((1.0 / math.log2(r + 1.0)) if r <= k else 0.0 for r in ranks) / n
    out["MRR"] = sum(1.0 / r for r in ranks) / n
    out["MeanRank"] = float(np.mean(ranks))
    out["MedianRank"] = float(np.median(ranks))
    out["NumUsers"] = int(n)
    return out


def full_rank_ndcg(scores: np.ndarray, targets: np.ndarray, masks: List[np.ndarray], k: int = 10) -> float:
    """Full-catalog NDCG@k for validation; masked items (train history) excluded."""
    total = 0.0
    for r in range(scores.shape[0]):
        s = scores[r].copy()
        s[masks[r]] = -np.inf
        rank = 1 + int(np.sum(s > s[targets[r]]))
        total += (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
    return total / max(1, scores.shape[0])


def evaluate_candidates(data: Data, score_fn, method: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """score_fn(user_ids, item_index_lists) -> list of score lists (warm items only)."""
    rows, ranks, warm_ranks = [], [], []
    users = data.eval_users
    warm_idx = [[data.item_index[i] for i in data.candidates[u] if i in data.item_index] for u in users]
    all_scores = score_fn(users, warm_idx)
    for u, idxs, sc in zip(users, warm_idx, all_scores):
        cands = data.candidates[u]
        it = iter(sc)
        scores = [float(next(it)) if i in data.item_index else -math.inf for i in cands]
        order = sorted(range(len(cands)), key=lambda j: (-scores[j], j))
        ranked = [cands[j] for j in order]
        rank = ranked.index(data.targets[u]) + 1
        cold = [i for i in cands if i not in data.item_index]
        ranks.append(rank)
        if not cold:
            warm_ranks.append(rank)
        rows.append({
            "user_id": u, "method": method, "target": data.targets[u], "candidates": cands,
            "scores_in_candidate_order": [None if math.isinf(s) else s for s in scores],
            "ranked_candidates": ranked, "rank_position": rank,
            "cold_candidates": cold, "target_is_cold": data.targets[u] not in data.item_index,
        })
    return {"metrics": compute_metrics(ranks), "metrics_all_candidates_warm": compute_metrics(warm_ranks)}, rows


# =============================================================================
# Popularity (sanity check)
# =============================================================================

def run_pop(args: argparse.Namespace, data: Data) -> Dict[str, Any]:
    def score_fn(users, idx_lists):
        return [data.item_counts[idxs].tolist() for idxs in idx_lists]
    res, rows = evaluate_candidates(data, score_fn, "pop")
    return {"method": "Popularity", "hyperparameters": {"score": "train interaction count (all users)"},
            **res, "_rows": rows}


# =============================================================================
# LightGCN
# =============================================================================

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def device_of(args):
    torch, _, _ = _torch()
    return torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def build_lightgcn(data: Data, seqs: Dict[str, List[int]], dim: int, layers: int, device):
    torch, nn, _ = _torch()
    u_idx = np.concatenate([np.full(len(s), data.user_index[u], dtype=np.int64) for u, s in seqs.items()])
    i_idx = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs.values()])
    # Deduplicate repeated (user, item) pairs for the graph.
    key = np.unique(u_idx * data.n_items + i_idx)
    u_idx, i_idx = key // data.n_items, key % data.n_items
    n_nodes = data.n_users + data.n_items
    src = np.concatenate([u_idx, data.n_users + i_idx])
    dst = np.concatenate([data.n_users + i_idx, u_idx])
    deg = np.bincount(src, minlength=n_nodes).astype(np.float32)
    dinv = np.zeros_like(deg)
    dinv[deg > 0] = deg[deg > 0] ** -0.5
    vals = dinv[src] * dinv[dst]
    adj = torch.sparse_coo_tensor(torch.from_numpy(np.stack([src, dst])), torch.from_numpy(vals),
                                  (n_nodes, n_nodes)).coalesce().to(device)

    class LightGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.user = nn.Embedding(data.n_users, dim)
            self.item = nn.Embedding(data.n_items, dim)
            nn.init.normal_(self.user.weight, std=0.1)
            nn.init.normal_(self.item.weight, std=0.1)

        def propagate(self):
            x = torch.cat([self.user.weight, self.item.weight], 0)
            outs = [x]
            for _ in range(layers):
                x = torch.sparse.mm(adj, x)
                outs.append(x)
            final = torch.stack(outs, 0).mean(0)
            return final[: data.n_users], final[data.n_users:]

    return LightGCN().to(device), u_idx, i_idx


def sample_negatives(u: np.ndarray, pos_keys: np.ndarray, n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform negatives excluding the user's training positives (vectorised rejection)."""
    neg = rng.integers(0, n_items, size=len(u))
    for _ in range(20):
        bad = np.isin(u * n_items + neg, pos_keys, assume_unique=False)
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg


def train_lightgcn(args, data: Data, seqs, epochs: int, val=None, log_prefix=""):
    """Returns (model, best_epoch, history). With ``val`` -> early stopping."""
    torch, _, F = _torch()
    set_seed(args.seed)
    dev = device_of(args)
    model, u_idx, i_idx = build_lightgcn(data, seqs, args.embedding_dim, args.n_layers, dev)
    pos_keys = np.sort(u_idx * data.n_items + i_idx)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    n = len(u_idx)
    best, best_epoch, best_state, bad_evals, history = -1.0, 0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, args.batch_size):
            b = perm[s:s + args.batch_size]
            ub, pb = u_idx[b], i_idx[b]
            nb = sample_negatives(ub, pos_keys, data.n_items, rng)
            ut, pt, nt = (torch.from_numpy(x).to(dev) for x in (ub, pb, nb))
            uf, itf = model.propagate()
            pos = (uf[ut] * itf[pt]).sum(-1)
            neg = (uf[ut] * itf[nt]).sum(-1)
            loss = -F.logsigmoid(pos - neg).mean()
            reg = (model.user(ut).pow(2).sum() + model.item(pt).pow(2).sum()
                   + model.item(nt).pow(2).sum()) / (2.0 * len(b))
            loss = loss + args.reg_weight * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        rec = {"epoch": ep, "loss": tot / n}
        if val is not None and (ep % args.eval_every == 0 or ep == epochs):
            rec["val_ndcg10"] = validate_lightgcn(model, val)
            if rec["val_ndcg10"] > best:
                best, best_epoch, bad_evals = rec["val_ndcg10"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_evals += 1
        history.append(rec)
        print(f"{log_prefix}epoch {ep:4d} loss={rec['loss']:.5f}"
              + (f" val_NDCG@10={rec['val_ndcg10']:.4f}" if "val_ndcg10" in rec else ""), flush=True)
        if val is not None and bad_evals >= args.patience:
            break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch if val is not None else epochs), history


def validate_lightgcn(model, val) -> float:
    torch, _, _ = _torch()
    users, targets, masks = val
    model.eval()
    with torch.no_grad():
        uf, itf = model.propagate()
        ut = torch.as_tensor(users, device=uf.device)
        out = 0.0
        for s in range(0, len(users), 1024):
            sc = (uf[ut[s:s + 1024]] @ itf.T).float().cpu().numpy()
            out += full_rank_ndcg(sc, targets[s:s + 1024], masks[s:s + 1024]) * len(sc)
    return out / len(users)


def run_lightgcn(args: argparse.Namespace, data: Data) -> Dict[str, Any]:
    torch, _, _ = _torch()
    seqs_val = data.sequences(holdout_last=True)
    vu = data.validation_users(args.val_users, args.seed)
    val = (np.asarray([data.user_index[u] for u in vu]),
           np.asarray([data.item_index[data.train[u][-1]] for u in vu]),
           [np.asarray(seqs_val[u]) for u in vu])
    print(f"[LightGCN] model selection on train[:-1]; validation users={len(vu)}", flush=True)
    model, best_epoch, hist = train_lightgcn(args, data, seqs_val, args.epochs, val, "[LightGCN/select] ")
    if args.refit:
        print(f"[LightGCN] refit on full train for {best_epoch} epochs", flush=True)
        model, _, hist_refit = train_lightgcn(args, data, data.sequences(holdout_last=False),
                                              best_epoch, None, "[LightGCN/refit] ")
    model.eval()
    with torch.no_grad():
        uf, itf = model.propagate()

    def score_fn(users, idx_lists):
        out = []
        for u, idxs in zip(users, idx_lists):
            if not idxs:
                out.append([])
                continue
            v = itf[torch.as_tensor(idxs, device=itf.device)] @ uf[data.user_index[u]]
            out.append(v.float().cpu().tolist())
        return out

    res, rows = evaluate_candidates(data, score_fn, "lightgcn")
    return {"method": "LightGCN",
            "hyperparameters": {k: getattr(args, k) for k in
                                ("embedding_dim", "n_layers", "lr", "reg_weight", "batch_size", "epochs",
                                 "patience", "eval_every", "refit", "seed")},
            "selected_epoch": best_epoch, "best_val_ndcg10": max((h.get("val_ndcg10", -1) for h in hist), default=None),
            "train_history": hist, **res, "_rows": rows}


# =============================================================================
# SASRec
# =============================================================================

def build_sasrec(n_items: int, max_len: int, hidden: int, layers: int, heads: int, dropout: float):
    torch, nn, _ = _torch()

    class SASRec(nn.Module):
        """Item ids are shifted by +1; 0 is padding. Sequences are LEFT-padded."""

        def __init__(self):
            super().__init__()
            self.item = nn.Embedding(n_items + 1, hidden, padding_idx=0)
            self.pos = nn.Embedding(max_len, hidden)
            self.drop = nn.Dropout(dropout)
            layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.norm = nn.LayerNorm(hidden)
            nn.init.normal_(self.item.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)
            with torch.no_grad():
                self.item.weight[0].zero_()

        def forward(self, x):  # x: [B, L] left-padded
            B, L = x.shape
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            pad = x == 0
            keep = (~pad).unsqueeze(-1).to(self.item.weight.dtype)
            h = self.drop(self.item(x) + self.pos(pos)) * keep  # "timeline mask" as in SASRec
            # Causal + key-padding mask; the diagonal is always allowed so the
            # left-padded positions never get a fully-masked (NaN) attention row.
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            m = causal.unsqueeze(0) | pad.unsqueeze(1)
            idx = torch.arange(L, device=x.device)
            m[:, idx, idx] = False
            h = self.enc(h, mask=m.repeat_interleave(heads, dim=0))
            return self.norm(h) * keep

        def item_weights(self):
            return self.item.weight[1:]

    return SASRec()


def left_pad(seqs: List[List[int]], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for r, s in enumerate(seqs):
        s = s[-max_len:]
        if s:
            out[r, -len(s):] = np.asarray(s) + 1
    return out


def train_sasrec(args, data: Data, seqs: Dict[str, List[int]], epochs: int, val=None, log_prefix=""):
    torch, _, F = _torch()
    set_seed(args.seed)
    dev = device_of(args)
    L = args.max_len
    model = build_sasrec(data.n_items, L, args.hidden_size, args.n_layers, args.n_heads, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_seqs = [s[-(L + 1):] for s in seqs.values() if len(s) >= 2]
    inp = left_pad([s[:-1] for s in train_seqs], L)
    tgt = left_pad([s[1:] for s in train_seqs], L)  # 0 = no target
    rng = np.random.default_rng(args.seed)
    full_softmax = data.n_items <= args.full_softmax_max_items
    best, best_epoch, best_state, bad_evals, history = -1.0, 0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(inp))
        tot, cnt = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            x = torch.from_numpy(inp[b]).to(dev)
            y = torch.from_numpy(tgt[b]).to(dev)
            h = model(x)
            m = y > 0
            hv, yv = h[m], y[m] - 1
            W = model.item_weights()
            if full_softmax:
                logits = hv @ W.T
                loss = F.cross_entropy(logits, yv)
            else:  # sampled softmax with shared uniform negatives
                negs = torch.from_numpy(rng.integers(0, data.n_items, size=args.num_negatives)).to(dev)
                pos_logit = (hv * W[yv]).sum(-1, keepdim=True)
                neg_logit = hv @ W[negs].T
                neg_logit = neg_logit.masked_fill(negs.unsqueeze(0) == yv.unsqueeze(1), -1e9)
                loss = F.cross_entropy(torch.cat([pos_logit, neg_logit], 1),
                                       torch.zeros(len(yv), dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot += float(loss.detach()) * len(yv)
            cnt += len(yv)
        rec = {"epoch": ep, "loss": tot / max(1, cnt)}
        if val is not None and (ep % args.eval_every == 0 or ep == epochs):
            rec["val_ndcg10"] = validate_sasrec(model, val, L)
            if rec["val_ndcg10"] > best:
                best, best_epoch, bad_evals = rec["val_ndcg10"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_evals += 1
        history.append(rec)
        print(f"{log_prefix}epoch {ep:4d} loss={rec['loss']:.5f}"
              + (f" val_NDCG@10={rec['val_ndcg10']:.4f}" if "val_ndcg10" in rec else ""), flush=True)
        if val is not None and bad_evals >= args.patience:
            break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch if val is not None else epochs), history


def sasrec_hidden(model, seqs: List[List[int]], max_len: int, batch: int = 512) -> np.ndarray:
    torch, _, _ = _torch()
    dev = next(model.parameters()).device
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch):
            x = torch.from_numpy(left_pad(seqs[s:s + batch], max_len)).to(dev)
            outs.append(model(x)[:, -1, :].float())
    return torch.cat(outs, 0)


def validate_sasrec(model, val, max_len) -> float:
    histories, targets, masks = val
    h = sasrec_hidden(model, histories, max_len)
    W = model.item_weights().detach().float()
    out = 0.0
    for s in range(0, len(histories), 1024):
        sc = (h[s:s + 1024] @ W.T).cpu().numpy()
        out += full_rank_ndcg(sc, targets[s:s + 1024], masks[s:s + 1024]) * len(sc)
    return out / len(histories)


def run_sasrec(args: argparse.Namespace, data: Data) -> Dict[str, Any]:
    seqs_val = data.sequences(holdout_last=True)
    vu = data.validation_users(args.val_users, args.seed)
    val = ([seqs_val[u] for u in vu],
           np.asarray([data.item_index[data.train[u][-1]] for u in vu]),
           [np.asarray(seqs_val[u]) for u in vu])
    print(f"[SASRec] max_len={args.max_len}; model selection on train[:-1]; validation users={len(vu)}; "
          f"{'full' if data.n_items <= args.full_softmax_max_items else 'sampled'} softmax", flush=True)
    model, best_epoch, hist = train_sasrec(args, data, seqs_val, args.epochs, val, "[SASRec/select] ")
    if args.refit:
        print(f"[SASRec] refit on full train for {best_epoch} epochs", flush=True)
        model, _, _ = train_sasrec(args, data, data.sequences(holdout_last=False), best_epoch, None,
                                   "[SASRec/refit] ")
    full = data.sequences(holdout_last=False)
    W = model.item_weights().detach().float()

    def score_fn(users, idx_lists):
        torch, _, _ = _torch()
        h = sasrec_hidden(model, [full[u] for u in users], args.max_len)
        out = []
        for r, idxs in enumerate(idx_lists):
            if not idxs:
                out.append([])
                continue
            out.append((W[torch.as_tensor(idxs, device=W.device)] @ h[r]).cpu().tolist())
        return out

    res, rows = evaluate_candidates(data, score_fn, "sasrec")
    return {"method": "SASRec",
            "hyperparameters": {k: getattr(args, k) for k in
                                ("max_len", "hidden_size", "n_layers", "n_heads", "dropout", "lr", "weight_decay",
                                 "batch_size", "epochs", "patience", "eval_every", "refit",
                                 "full_softmax_max_items", "num_negatives", "seed")},
            "selected_epoch": best_epoch, "best_val_ndcg10": max((h.get("val_ndcg10", -1) for h in hist), default=None),
            "train_history": hist, **res, "_rows": rows}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=["pop", "lightgcn", "sasrec", "all"])
    p.add_argument("--dataset", required=True, help="CDs | Video_Games | ML (sets default paths and max_len)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--items", default=None, help="unused by the models; kept for provenance")
    p.add_argument("--sequences", default=None)
    p.add_argument("--user-ids-file", default=None)
    p.add_argument("--candidate-file", default=None)
    p.add_argument("--expected-candidates", type=int, default=20)
    p.add_argument("--max-eval-users", type=int, default=0, help="0 = all users in --user-ids-file")
    p.add_argument("--output-dir", default=None, help="default results/baselines/<dataset>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    # model selection
    p.add_argument("--val-users", type=int, default=10000, help="validation users sampled (0 = all)")
    p.add_argument("--epochs", type=int, default=None, help="max epochs (default: lightgcn 300, sasrec 200)")
    p.add_argument("--patience", type=int, default=10, help="evaluations without improvement")
    p.add_argument("--eval-every", type=int, default=None, help="default: lightgcn 5, sasrec 2")
    p.add_argument("--refit", action=argparse.BooleanOptionalAction, default=True)
    # LightGCN
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=None, help="default: lightgcn 3, sasrec 2")
    p.add_argument("--reg-weight", type=float, default=1e-4)
    # SASRec
    p.add_argument("--max-len", type=int, default=None, help="default 50 (Amazon) / 200 (ML)")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=None, help="default 0.5 Amazon / 0.2 ML")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--full-softmax-max-items", type=int, default=60000)
    p.add_argument("--num-negatives", type=int, default=1024)
    # shared
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=None, help="default: lightgcn 4096 edges, sasrec 256 users")
    a = p.parse_args()
    a.dataset = DATASET_ALIASES.get(a.dataset.lower(), a.dataset)
    root = Path(a.data_root) / a.dataset
    a.items = a.items or str(root / "items.json")
    a.sequences = a.sequences or str(root / "user_sequences.json")
    a.user_ids_file = a.user_ids_file or str(root / "protocol" / "eval_users_1000.json")
    a.candidate_file = a.candidate_file or str(root / "protocol" / "user_candidates_testpool1000_seed42.json")
    a.output_dir = a.output_dir or f"results/baselines/{a.dataset}"
    a.max_len = a.max_len or DEFAULT_MAX_LEN.get(a.dataset, 50)
    a.dropout = a.dropout if a.dropout is not None else (0.2 if a.dataset == "ML" else 0.5)
    return a


def method_args(a: argparse.Namespace, method: str) -> argparse.Namespace:
    b = argparse.Namespace(**vars(a))
    b.epochs = a.epochs or (300 if method == "lightgcn" else 200)
    b.eval_every = a.eval_every or (5 if method == "lightgcn" else 2)
    b.n_layers = a.n_layers or (3 if method == "lightgcn" else 2)
    b.batch_size = a.batch_size or (4096 if method == "lightgcn" else 256)
    return b


def main() -> None:
    a = parse_args()
    t0 = time.time()
    data = Data(a)
    stats = data.stats()
    print("[DATA]", json.dumps(stats, indent=2), flush=True)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "data_stats.json", stats)
    methods = ["pop", "lightgcn", "sasrec"] if a.method == "all" else [a.method]
    table = {}
    for m in methods:
        args = method_args(a, m)
        t = time.time()
        res = {"pop": run_pop, "lightgcn": run_lightgcn, "sasrec": run_sasrec}[m](args, data)
        rows = res.pop("_rows")
        res.update({"dataset": a.dataset, "data_stats": stats, "runtime_sec": round(time.time() - t, 1),
                    "protocol": "all-user train, no history cut, penultimate-item early stopping, "
                                "frozen eval users + 20 frozen candidates"})
        save_json(out_dir / f"{m}_metrics.json", res)
        write_jsonl(out_dir / f"{m}_rankings.jsonl", rows)
        table[res["method"]] = res
        mm = res["metrics"]
        print(f"[RESULT] {res['method']}: NDCG@1={mm['NDCG@1']:.4f} NDCG@5={mm['NDCG@5']:.4f} "
              f"NDCG@10={mm['NDCG@10']:.4f} Hit@5={mm['Hit@5']:.4f} Hit@10={mm['Hit@10']:.4f} "
              f"MRR={mm['MRR']:.4f}", flush=True)
    print("\n[SUMMARY] random ranking with 20 candidates: NDCG@1=0.0500 Hit@5=0.2500 Hit@10=0.5000")
    for name, res in table.items():
        mm, ww = res["metrics"], res["metrics_all_candidates_warm"]
        print(f"  {name:10s} N@1={mm['NDCG@1']:.4f} N@10={mm['NDCG@10']:.4f} H@5={mm['Hit@5']:.4f} "
              f"| all-warm subset (n={ww.get('NumUsers', 0)}): N@10={ww.get('NDCG@10', float('nan')):.4f}")
    print(f"  cold candidate rate={stats['cold_candidate_rate']:.4%} cold target rate={stats['cold_target_rate']:.4%}")
    print(f"Saved to {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
