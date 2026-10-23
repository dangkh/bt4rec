#!/usr/bin/env python3
"""Exact token count of the behaviour-extraction precompute (no GPU, tokenizer only).

The precompute does not log tokens, but every window's prompt is deterministic and
the generated text is stored (raw_response). This script rebuilds the exact chat
prompt of every UNIQUE window used by train_shared.jsonl / test_shared.jsonl,
tokenizes it with the extraction model's tokenizer, tokenizes raw_response
(+1 EOS), and reports totals, per window and per user.

  python scripts/count_behavior_tokens.py --dataset CDs
  python scripts/count_behavior_tokens.py --dataset Books --precomputed-dir precomputed/Books_w5
Output: <precomputed-dir>/behavior_token_count.json
1 LLM call = 1 window (retries, if any, are not in the files and are not counted).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt4rec.behavior.windows import Window, cache_key, make_prompt_summary  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def from_compact(x):
    return {"item_id": x["item_id"], "item_name": x.get("item", x.get("item_name", "")),
            "item_category": x.get("category", x.get("item_category", "Unknown")),
            "action_type": x.get("action", x.get("action_type", "purchase"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--precomputed-dir", default=None, help="default precomputed/<DS>")
    ap.add_argument("--backend", choices=["gemma", "qwen"], default=None,
                    help="default: inferred from behavior_prompt_version in the files")
    ap.add_argument("--tokenizer", default=None,
                    help="default unsloth/gemma-3-4b-it-unsloth-bnb-4bit (gemma) / Qwen/Qwen2.5-7B-Instruct (qwen)")
    a = ap.parse_args()
    from bt4rec.config import load_dataset_config
    ds = load_dataset_config(a.dataset)["dataset"]
    d = Path(a.precomputed_dir or f"precomputed/{ds}")
    train = read_jsonl(d / "train_shared.jsonl")
    test = read_jsonl(d / "test_shared.jsonl") if (d / "test_shared.jsonl").exists() else []
    cache = {}
    if (d / "shared_windows.cache.jsonl").exists():
        for r in read_jsonl(d / "shared_windows.cache.jsonl"):
            if r.get("interaction_sequence") is not None:
                cache[cache_key(r["user_id"], r["interaction_sequence"])] = r

    windows = {}                       # key -> (uid, items, raw_response, role)
    for r in train:
        k = cache_key(r["user_id"], r["interaction_sequence"])
        windows[k] = (str(r["user_id"]), r["interaction_sequence"], r.get("raw_response"), "train")
    test_only = 0
    for row in test:
        for w in row.get("behavior_windows", []):
            items = [from_compact(x) for x in w]
            k = cache_key(row["user_id"], items)
            if k not in windows:
                test_only += 1
                windows[k] = (str(row["user_id"]), items, (cache.get(k) or {}).get("raw_response"), "test_only")

    version = next((r.get("behavior_prompt_version") for r in train if r.get("behavior_prompt_version")), "")
    backend = a.backend or ("qwen" if "qwen" in version.lower() else "gemma")
    if backend == "gemma":
        from bt4rec.behavior import gemma as be
        for dom, suffix in getattr(be, "DOMAIN_VERSIONS", {}).items():
            if version.endswith(suffix):
                be.set_domain(dom)
        system = ("You are a behavioral memory modeling system for recommender systems. "
                  "Create both a recommendation-oriented semantic view and a "
                  "domain-agnostic trajectory view. Return only valid JSON.")
        tok_name = a.tokenizer or "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
    else:
        from bt4rec.behavior import qwen as be
        system = None
        tok_name = a.tokenizer or "Qwen/Qwen2.5-7B-Instruct"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)

    per_win_in, per_win_out, per_user = [], [], Counter()
    missing_raw = 0
    for k, (uid, items, raw, role) in windows.items():
        prompt = be.build_behavior_extraction_prompt(make_prompt_summary(Window(uid, items, [])))
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:            # templates without a system role: merge into the user turn
            text = tok.apply_chat_template([{"role": "user", "content": (system + "\n\n" if system else "") + prompt}],
                                           tokenize=False, add_generation_prompt=True)
        n_in = len(tok(text, add_special_tokens=False)["input_ids"])
        if raw is None:
            missing_raw += 1
            n_out = 0
        else:
            n_out = len(tok(str(raw), add_special_tokens=False)["input_ids"]) + 1
        per_win_in.append(n_in)
        per_win_out.append(n_out)
        per_user[uid] += n_in + n_out

    n = len(windows)
    users = len(per_user)
    res = {
        "dataset": ds, "precomputed_dir": str(d), "backend": backend, "prompt_version": version,
        "tokenizer": tok_name,
        "llm_calls_total (= unique windows)": n, "train_windows": len(train), "test_only_windows": test_only,
        "users": users, "llm_calls_per_user": n / max(1, users),
        "input_tokens_total": sum(per_win_in), "output_tokens_total": sum(per_win_out),
        "tokens_total": sum(per_win_in) + sum(per_win_out),
        "input_tokens_per_call_mean": statistics.fmean(per_win_in) if per_win_in else 0,
        "output_tokens_per_call_mean": statistics.fmean(per_win_out) if per_win_out else 0,
        "tokens_per_user_mean": (sum(per_win_in) + sum(per_win_out)) / max(1, users),
        "windows_without_raw_response": missing_raw,
        "note": "behaviour memories created = unique windows; shared by CoMemTree (all modes) and aMem4Rec.",
    }
    out = d / "behavior_token_count.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
