#!/usr/bin/env python3
"""Estimate aMem4Rec memory-build LLM calls WITHOUT a GPU / without Gemma.

Only the embedding model is needed (runs on CPU). The control flow of both builders
is replayed exactly; only the Gemma answers are replaced by assumptions.

  --builder old      baseline/amem_global_memory_old.py  (flat A-Mem: link + evolve)
      per local memory: top-k cosine search -> zone from adaptive low/high thresholds
        * no LLM call when: empty pool, max_sim < low ("store_only"), no neighbour >= low
        * otherwise 1 "memory_linking" call (independent of Gemma's answer)
        * if Gemma links >= 1 evolvable memory -> +1 "memory_evolution" call
      => linking calls are (almost) exact; evolution calls lie in [0, linking calls].
         Reported for link-rate assumptions 0 / 0.25 / 0.5 / 0.75 / 1.0.
      Approximation: evolution re-embeds edited memories, which can slightly change
      later neighbourhoods; ignored here.

  --builder epsilon  baseline/amem_build_memory.py  (epsilon-net + co-occurrence)
      geometry never depends on Gemma, so spawn/absorb/merge are exact:
        * "memory_merge" calls = consolidation merges (exact)
        * "memory_evolution" calls = absorb events with count % T == 0 while the row is
          below max evolutions: exact upper bound (Gemma never accepts -> counts never
          reach the cap) and lower bound (Gemma always accepts -> cap reached fastest).

Token estimate: --avg-input-tokens / --avg-output-tokens (defaults from typical prompt
sizes) or pass the averages measured on a small GPU run.

  python baseline/amem_estimate_calls.py --dataset CDs --builder old
  python baseline/amem_estimate_calls.py --dataset CDs --builder epsilon
Embeddings are cached to <out>/embeddings_<hash>.npy, so the second call is instant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_records(path, users):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if users is None or str(r.get("user_id")) in users:
                rows.append(r)
    rows.sort(key=lambda r: int(r["precompute_id"]))
    return rows


def text_of(r):
    kw = r.get("keywords", [])
    kw = kw if isinstance(kw, list) else [str(kw)]
    return f"{r.get('behavior_explanation', '')} {r.get('pattern_description', '')} {' '.join(map(str, kw))}".strip()


def embed(texts, model, device, batch, cache):
    if cache.exists():
        e = np.load(cache)
        if len(e) == len(texts):
            print(f"[EMB] loaded cache {cache}")
            return e
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device=device)
    e = m.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=True).astype(np.float32)
    np.save(cache, e)
    return e


def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


# --------------------------------------------------------------------------- old
def simulate_old(E, users, a, link_rate, rng):
    pool, evo = [], []          # embeddings of stored memories, evolution counts
    calls = Counter()
    zones = Counter()
    stored = discarded = 0
    per_user = Counter()
    for i, e in enumerate(E):
        n = len(pool)
        if n == 0:
            pool.append(e); evo.append(0); stored += 1; zones["store_only_no_memories"] += 1
            continue
        scale = min(n / 1000.0, 1.0)
        low, high = a.low_threshold + 0.05 * scale, a.high_threshold - 0.05 * scale
        if low >= high:
            low, high = a.low_threshold, a.high_threshold
        sims = np.asarray(pool) @ e
        k = min(a.link_size, n)
        top = np.argsort(-sims)[:k]
        s = sims[top]
        mx = float(s[0])
        hp, lp = float(np.mean(s >= high)), float(np.mean(s < low))
        if mx < low:
            up, st, zone = False, True, "store_only_completely_new"
        elif mx < high:
            up, st, zone = True, True, "update_and_store_partial_overlap"
        elif hp >= 0.6:
            up, st, zone = True, False, "update_only_likely_duplicate"
        elif lp >= 0.5:
            up, st, zone = False, True, "store_only_few_relevant"
        elif hp >= 0.4:
            up, st, zone = True, True, "update_and_store_common_pattern"
        else:
            up, st, zone = True, True, "update_and_store_ambiguous"
        zones[zone] += 1
        eligible = [int(t) for t, v in zip(top, s) if v >= low]
        linked = []
        if up and eligible and not a.wo_link and "store_only" not in zone:
            calls["memory_linking"] += 1
            per_user[users[i]] += 1
            if rng.random() < link_rate:
                linked = eligible            # assumption: Gemma links all eligible
        elif up and eligible and a.wo_link:
            linked = eligible
        if up and linked:
            evolvable = [j for j in linked if evo[j] < a.max_evolutions_per_memory]
            if evolvable:
                calls["memory_evolution"] += 1
                per_user[users[i]] += 1
                for j in evolvable:          # assumption: the evolution is accepted
                    evo[j] += 1
            elif not st:
                st = True                    # all linked rows at max -> store
        if st:
            pool.append(e); evo.append(0); stored += 1
        else:
            discarded += 1
    return {"llm_calls": dict(calls), "llm_calls_total": int(sum(calls.values())),
            "stored_memories": stored, "discarded": discarded, "zones": dict(zones),
            "max_calls_one_user": max(per_user.values()) if per_user else 0}


# ----------------------------------------------------------------------- epsilon
def simulate_epsilon(E, users, a):
    n = len(E)
    rng = np.random.default_rng(a.seed)
    m = min(a.calib_sample_size, n) if a.calib_sample_size > 0 else n
    idx = np.sort(rng.choice(n, size=m, replace=False)) if m < n else np.arange(n)
    sample = E[idx]
    mu = sample.mean(axis=0).astype(np.float32) if not a.no_center else None
    s2 = l2n(sample - mu[None, :]) if mu is not None else l2n(sample)
    g = s2 @ s2.T
    np.fill_diagonal(g, -1.0)
    nn = g.max(axis=1)
    cover = a.sigma_cover if a.sigma_cover is not None else float(np.quantile(nn, a.q_cover))
    sep = a.sigma_sep if a.sigma_sep is not None else float(np.quantile(nn, a.q_sep))
    X = l2n(E - mu[None, :]) if mu is not None else l2n(E)

    sums, counts, emb = [], [], []
    trig, evo_lo = [], []            # evolution triggers per row; accepted-always counter
    calls = Counter()
    step = 0
    events = Counter()

    def consolidate():
        nonlocal sums, counts, emb, trig, evo_lo
        while len(emb) >= 2:
            C = np.stack(emb)
            G = C @ C.T
            np.fill_diagonal(G, -1.0)
            if a.max_absorb_count > 0:
                cnt = np.asarray(counts)
                G[(cnt[:, None] + cnt[None, :]) > a.max_absorb_count] = -1.0
            f = int(np.argmax(G)); i, j = divmod(f, G.shape[1])
            if G[i, j] < sep:
                break
            if i > j:
                i, j = j, i
            calls["memory_merge"] += 1
            sums[i] = sums[i] + sums[j]; counts[i] += counts[j]; emb[i] = l2n(sums[i])
            trig[i] += trig[j]; evo_lo[i] = min(a.max_evolutions_per_memory, evo_lo[i] + evo_lo[j])
            for lst in (sums, counts, emb, trig, evo_lo):
                del lst[j]

    for x in X:
        if emb:
            sims = np.stack(emb) @ x
            j = int(np.argmax(sims)); smax = float(sims[j])
        else:
            j, smax = -1, -1.0
        capped = j >= 0 and a.max_absorb_count > 0 and counts[j] >= a.max_absorb_count
        if j < 0 or smax < cover or capped:
            sums.append(x.copy()); counts.append(1); emb.append(l2n(x)); trig.append(0); evo_lo.append(0)
            events["spawn"] += 1
        else:
            sums[j] = sums[j] + x; counts[j] += 1; emb[j] = l2n(sums[j])
            events["absorb"] += 1
            if a.evolve_every_T > 0 and counts[j] % a.evolve_every_T == 0:
                trig[j] += 1                              # upper bound: always called
                if evo_lo[j] < a.max_evolutions_per_memory:
                    evo_lo[j] += 1                        # lower bound: accepted, cap reached
                    calls["memory_evolution_lower"] += 1
        step += 1
        if a.consolidate_every_P > 0 and step % a.consolidate_every_P == 0:
            consolidate()
    consolidate()
    upper = sum(trig)   # never accepted -> evolution_count stays 0 -> every trigger calls Gemma
    merges = calls["memory_merge"]
    return {"sigma_cover": cover, "sigma_sep": sep, "global_memories": len(emb), "events": dict(events),
            "memory_merge_calls": merges,
            "memory_evolution_calls_range": [int(calls["memory_evolution_lower"]), int(upper)],
            "llm_calls_total_range": [int(calls["memory_evolution_lower"] + merges), int(upper + merges)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--builder", choices=["old", "epsilon"], default="old")
    ap.add_argument("--precomputed", default=None, help="default precomputed/<DS>/train_shared.jsonl")
    ap.add_argument("--train-users-file", default=None, help="default protocol train_users_1000; 'none' = all")
    ap.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--avg-input-tokens", type=float, default=None,
                    help="per call; default 900 (old link/evolve prompts) or 450 (epsilon prompts)")
    ap.add_argument("--avg-output-tokens", type=float, default=120)
    # old builder
    ap.add_argument("--low-threshold", dest="low_threshold", type=float, default=0.65)
    ap.add_argument("--high-threshold", dest="high_threshold", type=float, default=0.80)
    ap.add_argument("--link-size", dest="link_size", type=int, default=5)
    ap.add_argument("--max-evolutions-per-memory", dest="max_evolutions_per_memory", type=int, default=10)
    ap.add_argument("--wo-link", dest="wo_link", action="store_true")
    # epsilon builder
    ap.add_argument("--q-cover", dest="q_cover", type=float, default=0.50)
    ap.add_argument("--q-sep", dest="q_sep", type=float, default=0.90)
    ap.add_argument("--sigma-cover", dest="sigma_cover", type=float, default=None)
    ap.add_argument("--sigma-sep", dest="sigma_sep", type=float, default=None)
    ap.add_argument("--calib-sample-size", dest="calib_sample_size", type=int, default=2000)
    ap.add_argument("--no-center", dest="no_center", action="store_true")
    ap.add_argument("--evolve-every-T", dest="evolve_every_T", type=int, default=3)
    ap.add_argument("--consolidate-every-P", dest="consolidate_every_P", type=int, default=200)
    ap.add_argument("--max-absorb-count", dest="max_absorb_count", type=int, default=0)
    a = ap.parse_args()

    from bt4rec.config import load_dataset_config
    cfg = load_dataset_config(a.dataset)
    ds = cfg["dataset"]
    if a.precomputed is None:
        base = Path(cfg.get("precomputed_dir", f"precomputed/{ds}"))
        cands = [base / "train_shared.jsonl", base / "qwen" / "train_shared.jsonl"]
        a.precomputed = str(next((c for c in cands if c.exists()), cands[0]))
    tu = a.train_users_file or cfg.get("train_users")
    users = None
    if tu and tu.lower() != "none":
        obj = json.load(open(tu))
        users = set(map(str, obj.get("users", list(obj)) if isinstance(obj, dict) else obj))
    out = Path(a.output_dir or f"results/baselines/{ds}/amem_estimate")
    out.mkdir(parents=True, exist_ok=True)

    recs = load_records(a.precomputed, users)
    uids = [str(r.get("user_id")) for r in recs]
    texts = [text_of(r) for r in recs]
    key = hashlib.md5((a.precomputed + a.embedding_model + str(len(texts))).encode()).hexdigest()[:10]
    E = embed(texts, a.embedding_model, a.device, a.batch_size, out / f"embeddings_{key}.npy")
    n_users = len(set(uids))
    print(f"[{ds}] local memories (windows) = {len(recs)} from {n_users} users")

    tin = a.avg_input_tokens or (900.0 if a.builder == "old" else 450.0)
    tout = a.avg_output_tokens
    if a.builder == "old":
        res = {}
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = simulate_old(E, uids, a, p, np.random.default_rng(a.seed))
            r["est_input_tokens"] = r["llm_calls_total"] * tin
            r["est_output_tokens"] = r["llm_calls_total"] * tout
            r["llm_calls_per_user"] = r["llm_calls_total"] / max(1, n_users)
            res[f"link_rate_{p}"] = r
            print(f"  link_rate={p:<4}: linking={r['llm_calls'].get('memory_linking', 0)} "
                  f"evolution={r['llm_calls'].get('memory_evolution', 0)} total={r['llm_calls_total']} "
                  f"({r['llm_calls_per_user']:.2f}/user) stored={r['stored_memories']} discarded={r['discarded']}")
    else:
        res = simulate_epsilon(E, uids, a)
        lo, hi = res["llm_calls_total_range"]
        res["llm_calls_per_user_range"] = [lo / max(1, n_users), hi / max(1, n_users)]
        res["est_input_tokens_range"] = [lo * tin, hi * tin]
        res["est_output_tokens_range"] = [lo * tout, hi * tout]
        print(f"  global memories={res['global_memories']} merges={res['memory_merge_calls']} "
              f"evolution calls in {res['memory_evolution_calls_range']} -> total {lo}-{hi} "
              f"({lo / max(1, n_users):.2f}-{hi / max(1, n_users):.2f}/user)")
    summary = {"dataset": ds, "builder": a.builder, "precomputed": a.precomputed,
               "local_memories": len(recs), "users": n_users,
               "upstream_behavior_extraction_calls": len(recs),
               "token_assumption_per_call": {"input": tin, "output": tout},
               "config": {k: v for k, v in vars(a).items()}, "result": res}
    path = out / f"estimate_{a.builder}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
