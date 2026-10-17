#!/usr/bin/env python3
"""Dataset statistics for the frozen 1000-user protocol, per method "view".

Different methods see different amounts of each user's TRAIN history:
  full     : all TRAIN interactions          -> Pop / LightGCN / SASRec (comemtree_baselines.py)
  llm      : last H TRAIN items (H=20, ML=30) -> CoMemTree, Native, i2Agent, aMem4Rec, MemRec
  allmrec  : last 20 TRAIN items             -> A-LLMRec (SASRec + LLM, capped at 20)
For every view: #users, #interactions, #unique items, history length stats, and
how many test candidates / targets never appear in that view's training items
(cold items: a CF model cannot score them better than random).

Usage (from the repo root):
  python scripts/dataset_stats.py                       # CDs, Video_Games, ML
  python scripts/dataset_stats.py --datasets CDs ML
Writes results/dataset_stats/<DS>.json, summary.csv and summary.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt4rec.config import load_dataset_config  # noqa: E402

ALLMREC_CAP = 20


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ids(value):
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(x) for x in value]


def user_list(path):
    obj = load_json(path)
    if isinstance(obj, dict):
        obj = obj.get("users", list(obj))
    return [str(x) for x in obj]


def length_stats(lengths):
    if not lengths:
        return {}
    return {"mean": round(statistics.fmean(lengths), 2), "median": float(statistics.median(lengths)),
            "min": min(lengths), "max": max(lengths)}


def view_stats(histories, cand_items, targets, n_catalog):
    items = {i for h in histories.values() for i in h}
    n_inter = sum(len(h) for h in histories.values())
    users_short = sum(len(h) < 5 for h in histories.values())
    cold_cand = [i for c in cand_items.values() for i in c if i not in items]
    all_cand = [i for c in cand_items.values() for i in c]
    cold_targets = sum(t not in items for t in targets.values())
    return {
        "users": len(histories),
        "interactions": n_inter,
        "unique_items": len(items),
        "catalog_coverage": round(len(items) / max(1, n_catalog), 4),
        "density": round(n_inter / max(1, len(histories) * max(1, len(items))), 6),
        "history_length": length_stats([len(h) for h in histories.values()]),
        "users_with_lt5_items": users_short,
        "cold_candidate_rate": round(len(cold_cand) / max(1, len(all_cand)), 4),
        "cold_target_rate": round(cold_targets / max(1, len(targets)), 4),
        "users_all_candidates_warm": sum(all(i in items for i in c) for c in cand_items.values()),
    }


def dataset_stats(name, data_root):
    cfg = load_dataset_config(name)
    ds = cfg["dataset"]
    data = Path(data_root) / ds
    items = load_json(data / "items.json")
    n_catalog = len(items) if isinstance(items, (dict, list)) else 0
    seqs = {str(k): v for k, v in load_json(data / "user_sequences.json").items()}
    eval_users = user_list(cfg["eval_users"])
    train_users = user_list(cfg["train_users"]) if Path(cfg["train_users"]).exists() else eval_users
    cands_raw = load_json(cfg["candidate_file"])
    if isinstance(cands_raw, dict) and isinstance(cands_raw.get("users"), dict):
        cands_raw = cands_raw["users"]
    cand_items, targets = {}, {}
    for u in eval_users:
        row = cands_raw[u]
        cand_items[u] = ids(row if isinstance(row, list) else
                            row.get("candidates", row.get("candidate_item_ids", row.get("candidate_ids"))))
        targets[u] = ids(seqs[u].get("test"))[0]

    # Whole dataset (all users in user_sequences.json), for reference.
    all_train = {u: ids(v.get("train")) for u, v in seqs.items()}
    all_items = {i for h in all_train.values() for i in h} | {
        i for v in seqs.values() for i in ids(v.get("val")) + ids(v.get("test"))}
    full_dataset = {
        "users": len(seqs), "catalog_items": n_catalog,
        "items_in_interactions": len(all_items),
        "train_interactions": sum(len(h) for h in all_train.values()),
        "train_length": length_stats([len(h) for h in all_train.values()]),
    }

    h_llm = int(cfg["history_size"])
    full = {u: ids(seqs[u].get("train")) for u in train_users}
    views = {
        "full_history (Pop/LightGCN/SASRec)": full,
        f"llm_last{h_llm} (CoMemTree/Native/i2Agent/aMem/MemRec)": {u: h[-h_llm:] for u, h in full.items()},
        f"allmrec_last{ALLMREC_CAP} (A-LLMRec)": {u: h[-ALLMREC_CAP:] for u, h in full.items()},
    }
    candidate_items = {i for c in cand_items.values() for i in c}
    return {
        "dataset": ds,
        "protocol": {"eval_users": len(eval_users), "train_users": len(train_users),
                     "train_equals_eval": set(train_users) == set(eval_users),
                     "candidates_per_user": length_stats([len(c) for c in cand_items.values()]),
                     "unique_candidate_items": len(candidate_items),
                     "unique_target_items": len(set(targets.values())),
                     "llm_history_size": h_llm, "allmrec_history_cap": ALLMREC_CAP},
        "full_dataset": full_dataset,
        "views": {k: view_stats(v, cand_items, targets, n_catalog) for k, v in views.items()},
        "items_1000_users_total": len({i for h in full.values() for i in h} | candidate_items),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["CDs", "Video_Games", "ML"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-dir", default="results/dataset_stats")
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.datasets:
        st = dataset_stats(name, a.data_root)
        (out / f"{st['dataset']}.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
        for view, v in st["views"].items():
            rows.append({"dataset": st["dataset"], "view": view, "users": v["users"],
                         "interactions": v["interactions"], "unique_items": v["unique_items"],
                         "avg_len": v["history_length"].get("mean"), "max_len": v["history_length"].get("max"),
                         "density": v["density"], "cold_cand_rate": v["cold_candidate_rate"],
                         "cold_target_rate": v["cold_target_rate"],
                         "catalog_items": st["full_dataset"]["catalog_items"],
                         "candidate_items": st["protocol"]["unique_candidate_items"],
                         "items_1000u_total": st["items_1000_users_total"]})
        print(f"[{st['dataset']}] catalog={st['full_dataset']['catalog_items']} "
              f"all_users={st['full_dataset']['users']} | 1000 users: "
              + " | ".join(f"{k.split(' ')[0]}: {v['unique_items']} items, {v['interactions']} inter, "
                           f"cold target {v['cold_target_rate']:.1%}" for k, v in st["views"].items()))
    cols = list(rows[0])
    (out / "summary.csv").write_text("\n".join([",".join(cols)] + [
        ",".join(f'"{r[c]}"' if c == "view" else str(r[c]) for c in cols) for r in rows]) + "\n", encoding="utf-8")
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {out}/summary.csv, summary.md and per-dataset JSON")


if __name__ == "__main__":
    main()
