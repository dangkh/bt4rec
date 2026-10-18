#!/usr/bin/env python3
"""Where do the target ranks of each mode go?  (diagnose e.g. "R@1/3/10 up but R@5 down")

Reads a CoMemTree run (rankings.jsonl from bt4rec.cli.run) and writes a report with
  A. rank histogram (1..20) and Hit@1..10 per mode, with the gap to a reference mode;
  B. top-5 movement vs the reference: users lost from / gained into the top-5 and the
     rank they moved to;
  C. slices of "tree vs reference" by: predicted mechanism, stay-vs-change (predicted
     mechanism == latest observed mechanism), Tree matched order, and -- for Amazon
     data -- whether the target is on the user's dominant platform / is an accessory
     (category path level 2 / 3, e.g. "Video Games > PlayStation 4 > Accessories").

  python scripts/analyze_ranks.py --run-dir results/comemtree/Video_Games_1000_20_fulltree_ablation_XXXX \
      --dataset Video_Games --modes native,latest_only,tree,tree_stay --ref latest_only
Output: <run-dir>/rank_analysis.md and rank_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def hit(ranks, k):
    return sum(r <= k for r in ranks) / max(1, len(ranks))


def ndcg(ranks, k):
    return sum(1 / math.log2(r + 1) for r in ranks if r <= k) / max(1, len(ranks))


def category_path(meta):
    cat = meta.get("main_cat") or meta.get("category") or meta.get("categories") or ""
    if isinstance(cat, list):
        cat = cat[0] if cat and isinstance(cat[0], list) else cat
        cat = " > ".join(map(str, cat))
    return [x.strip() for x in str(cat).split(">") if x.strip()]


def level(meta, i):
    p = category_path(meta)
    return p[i].lower() if len(p) > i else ""


ACCESSORY_WORDS = ("accessor", "controller", "headset", "cable", "charger", "case", "memory",
                   "adapter", "stand", "skin", "battery", "keyboard", "mouse", "hardware", "console")


def is_accessory(meta):
    text = " ".join(category_path(meta)).lower()
    return any(w in text for w in ACCESSORY_WORDS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default=None, help="for items.json (platform/accessory slices)")
    ap.add_argument("--items", default=None)
    ap.add_argument("--modes", default=None, help="default: all modes in the file")
    ap.add_argument("--ref", default="latest_only", help="reference mode for gaps / movements")
    ap.add_argument("--focus", default="tree", help="mode analysed in the slices")
    a = ap.parse_args()
    run = Path(a.run_dir)
    rows = read_jsonl(run / "rankings.jsonl")
    modes = a.modes.split(",") if a.modes else list(rows[0]["rankings"])
    items = {}
    items_path = a.items or (f"data/{a.dataset}/items.json" if a.dataset else None)
    if items_path and Path(items_path).exists():
        raw = json.load(open(items_path, encoding="utf-8"))
        items = {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else \
            {str(r.get("item_id")): r for r in raw}
    ranks = {m: [r["rankings"][m]["target_rank"] for r in rows] for m in modes}
    out, md = {"n_users": len(rows)}, [f"# Rank analysis: {run.name}", f"users = {len(rows)}", ""]

    # ---- A. histogram / Hit@k ----------------------------------------------------
    md += ["## A. Hit@k per mode (gap vs `%s`)" % a.ref, "",
           "| mode | " + " | ".join(f"H@{k}" for k in (1, 2, 3, 4, 5, 6, 8, 10)) + " | N@5 | N@10 | MRR |",
           "|---|" + "---|" * 11]
    out["hit"] = {}
    for m in modes:
        hk = {k: hit(ranks[m], k) for k in range(1, 21)}
        out["hit"][m] = hk
        cells = []
        for k in (1, 2, 3, 4, 5, 6, 8, 10):
            gap = "" if m == a.ref or a.ref not in ranks else f" ({hk[k] - hit(ranks[a.ref], k):+.3f})"
            cells.append(f"{hk[k]:.3f}{gap}")
        mrr = sum(1 / r for r in ranks[m]) / len(rows)
        md.append(f"| {m} | " + " | ".join(cells) + f" | {ndcg(ranks[m], 5):.3f} | {ndcg(ranks[m], 10):.3f} | {mrr:.3f} |")
    md += ["", "Rank histogram (count of users whose target is at rank r):", "",
           "| mode | " + " | ".join(str(r) for r in range(1, 13)) + " | >12 |", "|---|" + "---|" * 13]
    out["histogram"] = {}
    for m in modes:
        c = Counter(ranks[m])
        out["histogram"][m] = {r: c.get(r, 0) for r in range(1, 21)}
        md.append(f"| {m} | " + " | ".join(str(c.get(r, 0)) for r in range(1, 13))
                  + f" | {sum(v for r, v in c.items() if r > 12)} |")

    # ---- B. top-5 movement -----------------------------------------------------------
    if a.ref in ranks:
        md += ["", f"## B. Top-5 movement vs `{a.ref}`", ""]
        out["movement"] = {}
        for m in modes:
            if m == a.ref:
                continue
            lost = [(x, y) for x, y in zip(ranks[a.ref], ranks[m]) if x <= 5 < y]
            gained = [(x, y) for x, y in zip(ranks[a.ref], ranks[m]) if y <= 5 < x]
            up_within = sum(1 for x, y in zip(ranks[a.ref], ranks[m]) if y < x <= 5)
            down_within = sum(1 for x, y in zip(ranks[a.ref], ranks[m]) if x < y <= 5)
            lost_to = Counter("6-10" if y <= 10 else ">10" for _, y in lost)
            gained_from = Counter("6-10" if x <= 10 else ">10" for x, _ in gained)
            out["movement"][m] = {"lost_from_top5": len(lost), "gained_into_top5": len(gained),
                                  "lost_to": dict(lost_to), "gained_from": dict(gained_from),
                                  "moved_up_inside_top5": up_within, "moved_down_inside_top5": down_within}
            md.append(f"- **{m}**: lost from top-5 = {len(lost)} (to {dict(lost_to)}), gained = {len(gained)} "
                      f"(from {dict(gained_from)}); inside top-5 up {up_within} / down {down_within}")

    # ---- C. slices -------------------------------------------------------------------
    f = a.focus
    if f in ranks and a.ref in ranks:
        def slice_table(title, key_fn):
            groups = defaultdict(list)
            for r, x, y in zip(rows, ranks[a.ref], ranks[f]):
                groups[key_fn(r)].append((x, y))
            res = {}
            lines = ["", f"### {title}", "",
                     f"| group | n | {a.ref} H@5 | {f} H@5 | Δ H@5 | Δ H@1 | Δ H@10 |", "|---|---|---|---|---|---|---|"]
            for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                xs, ys = [p[0] for p in v], [p[1] for p in v]
                d = {k: hit(ys, k) - hit(xs, k) for k in (1, 5, 10)}
                res[str(g)] = {"n": len(v), "ref_h5": hit(xs, 5), "focus_h5": hit(ys, 5), **{f"d_h{k}": d[k] for k in d}}
                lines.append(f"| {g} | {len(v)} | {hit(xs, 5):.3f} | {hit(ys, 5):.3f} | {d[5]:+.3f} | {d[1]:+.3f} | {d[10]:+.3f} |")
            md.extend(lines)
            return res

        def how(r):
            return (r.get("guidance") or {}).get("tree") or {}

        def latest_mech(r):
            maps = [m for m in (r.get("state_mappings") or []) if m.get("assignment_accepted")]
            return (maps[-1].get("behavior_mechanism") or maps[-1].get("mechanism") if maps else None) or "unknown"

        md += ["", f"## C. Slices: `{f}` vs `{a.ref}`"]
        out["slices"] = {
            "predicted_mechanism": slice_table("Predicted next mechanism (Tree HOW)",
                                               lambda r: how(r).get("mechanism", "none")),
            "predicted_scope": slice_table("Predicted next scope", lambda r: how(r).get("scope", "none")),
            "stay_vs_change": slice_table(
                "Predicted mechanism == latest observed mechanism?",
                lambda r: "same" if how(r).get("mechanism") and how(r).get("mechanism") == latest_mech(r) else "change"),
            "matched_order": slice_table("Tree matched context order",
                                         lambda r: (r.get("tree_result") or {}).get("matched_order")),
        }
        if items:
            def tmeta(r):
                return items.get(str(r["ground_truth_item_ids"][0]), {})

            def platform_match(r):
                plats = Counter(level({"category": h.get("category", "")}, 1) for h in r.get("history_items", []))
                plats.pop("", None)
                if not plats:
                    return "unknown"
                return "target on dominant platform" if level(tmeta(r), 1) == plats.most_common(1)[0][0] \
                    else "target on other platform"
            out["slices"]["platform"] = slice_table("Target platform vs user's dominant platform (category level 2)",
                                                    platform_match)
            out["slices"]["accessory"] = slice_table(
                "Target is accessory/hardware?", lambda r: "accessory/hardware" if is_accessory(tmeta(r)) else "game/other")
            out["slices"]["target_cat_level3"] = slice_table(
                "Target category level 3", lambda r: level(tmeta(r), 2) or "none")

    (run / "rank_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (run / "rank_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {run / 'rank_analysis.md'}")


if __name__ == "__main__":
    main()
