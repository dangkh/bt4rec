#!/usr/bin/env python3
"""Native LLM baseline = EXACTLY the ``native`` mode of CoMemTree (bt4rec.cli.run).

Same prompt template (``build_rank_prompt`` with no behaviour/HOW block), same system
prompt, same C-label parser/repair policy, same Qwen2.5-7B ranker, same frozen
history (last 20 TRAIN items; 30 for ML), same 1000 eval users and frozen 20
candidates. It needs no Tree or behaviour cache, so it can run before anything else
and its ``native_rankings.jsonl`` can be fed to i2Agent as the ranking prior
(``--initial-ranking-file``; rows store ``rankings.native``).

The old standalone script (native_llm_list_score_batch-4.py) used a different,
movie-specific prompt, JSON history, a list/score output format, history=10,
300 users and a full-fallback parser, so its numbers are NOT comparable with the
CoMemTree modes.  Keep it only to reproduce earlier tables.

Outputs (<output-dir>/):
  native_rankings.jsonl   one row per user, rankings.native = {ranked_item_ids, target_rank, ...}
  summary.json            metrics (all / parse-complete only), parse quality, config
  cost_summary.json       bt4rec_cost_v1: calls/user, tokens/user, memories (=0)

Example:
  python baseline/native_llm.py --dataset CDs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt4rec.common.io import append_jsonl, dump_json, load_json, load_user_ids_file, read_jsonl  # noqa: E402
from bt4rec.common.seeding import set_seed  # noqa: E402
from bt4rec.config import load_dataset_config  # noqa: E402
from bt4rec.data.candidates import get_candidates_for_user, load_candidate_file  # noqa: E402
from bt4rec.data.items import as_str_list, get_item_info, normalize_items  # noqa: E402
from bt4rec.eval.cost import CostTracker, memory_block  # noqa: E402
from bt4rec.eval.metrics import aggregate_mode, target_rank  # noqa: E402
from bt4rec.ranking.prompts import build_rank_prompt  # noqa: E402

SCHEMA = "native_comemtree_template_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="CDs | Video_Games | ML")
    p.add_argument("--data-root", default="data")
    p.add_argument("--items", default=None)
    p.add_argument("--sequences", default=None)
    p.add_argument("--user-ids-file", default=None, help="default: <protocol>/eval_users_1000.json")
    p.add_argument("--candidate-file", default=None)
    p.add_argument("--history-size", type=int, default=None, help="default from configs (20; ML 30)")
    p.add_argument("--max-users", type=int, default=0, help="smoke test cap; 0 = all eval users")
    p.add_argument("--candidate-size", type=int, default=20)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="local Qwen (default) or Gemma-3 id; with --ranker openai e.g. gpt-4o-mini")
    p.add_argument("--ranker", choices=["local", "openai"], default="local")
    p.add_argument("--api-workers", type=int, default=8)
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-cache", default=None, help="default <output-dir>/openai_cache.jsonl")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rank-max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--llm-batch-size", type=int, default=8, help="same default as bt4rec.cli.run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, help="default results/baselines/<DS>/native")
    p.add_argument("--prompts-only", action="store_true", help="write prompts and stop (no GPU)")
    a = p.parse_args()
    cfg = load_dataset_config(a.dataset)
    ds = cfg["dataset"]
    a.dataset = ds
    data = Path(a.data_root) / ds
    a.items = a.items or str(data / "items.json")
    a.sequences = a.sequences or str(data / "user_sequences.json")
    a.user_ids_file = a.user_ids_file or cfg["eval_users"]
    a.candidate_file = a.candidate_file or cfg["candidate_file"]
    a.history_size = a.history_size or int(cfg["history_size"])
    suffix = ""
    if not (a.ranker == "local" and a.model == "Qwen/Qwen2.5-7B-Instruct"):
        import re
        suffix = "_" + re.sub(r"[^A-Za-z0-9.]+", "-", a.model.split("/")[-1]).strip("-")
    a.output_dir = a.output_dir or str(Path("results/baselines") / ds / f"native{suffix}")
    return a


def main() -> None:
    a = parse_args()
    set_seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = normalize_items(load_json(a.items))
    seqs = {str(k): v for k, v in load_json(a.sequences).items()}
    frozen = load_candidate_file(a.candidate_file)
    users = load_user_ids_file(a.user_ids_file)
    if a.max_users > 0:
        users = users[: a.max_users]

    jobs = []
    for uid in users:
        cand_ids, targets = get_candidates_for_user(uid, seqs[uid], {}, frozen, a.seed)
        if len(cand_ids) != a.candidate_size or len(targets) != 1:
            raise ValueError(f"{uid}: expected {a.candidate_size} candidates and one target")
        hist = [get_item_info(x, items) for x in as_str_list(seqs[uid].get("train"))[-a.history_size:]]
        cmeta = [get_item_info(x, items) for x in cand_ids]
        prompt = build_rank_prompt(history_items=hist, candidate_items=cmeta,
                                   latest_behavior=None, predicted_transition=None)
        jobs.append({"user_id": uid, "prebuilt_prompt": prompt, "candidate_items": cmeta,
                     "candidate_ids": cand_ids, "targets": targets, "history_items": hist})
    print(f"[NATIVE] {a.dataset}: users={len(jobs)} history={a.history_size} "
          f"candidates={a.candidate_size}", flush=True)
    if a.prompts_only:
        (out / "prompt_example.txt").write_text(jobs[0]["prebuilt_prompt"] + "\n", encoding="utf8")
        print(f"[PROMPTS-ONLY] {out / 'prompt_example.txt'}")
        return

    from bt4rec.llm.ranker import build_ranker

    ranker = build_ranker(a, cache_dir=out)
    bs = a.llm_batch_size if a.ranker == "local" else max(a.llm_batch_size, 4 * a.api_workers)
    results = ranker.rank_jobs(jobs, batch_size=bs, call_type="native", max_retries=0)

    path = out / "native_rankings.jsonl"
    if path.exists():
        path.unlink()
    cost = CostTracker("native")
    cost.batch("ranking", (len(jobs) + a.llm_batch_size - 1) // a.llm_batch_size)
    for job, (ordered, meta) in zip(jobs, results):
        rec = {"ranked_item_ids": ordered, "target_rank": target_rank(ordered, job["targets"][0]), **meta}
        append_jsonl(path, {"schema_version": SCHEMA, "user_id": job["user_id"],
                            "ground_truth_item_ids": job["targets"],
                            "candidate_item_ids": job["candidate_ids"],
                            "history_items": job["history_items"], "rankings": {"native": rec}})
        cost.llm("ranking", job["user_id"], meta.get("input_tokens", 0), meta.get("output_tokens", 0))
    rows = read_jsonl(path)
    complete = sum(bool(r["rankings"]["native"].get("parse_complete")) for r in rows)
    summary = {"schema_version": SCHEMA, "dataset": a.dataset, "num_users": len(rows),
               "metrics": aggregate_mode(rows, "native", False),
               "metrics_complete_only": aggregate_mode(rows, "native", True),
               "parse_quality": {"parse_complete_count": complete,
                                 "parse_complete_rate": complete / max(1, len(rows))},
               "ranking_llm": ranker.stats(), "config": vars(a),
               "prompt": "identical to bt4rec.cli.run --modes native"}
    dump_json(out / "summary.json", summary)
    cost.write(out / "cost_summary.json", users,
               memory=memory_block(0, {u: 0 for u in users}, scope="none",
                                   note="Native keeps no memory; only the recent history is shown."))
    print("[RESULTS]", summary["metrics"])
    print(f"Saved: {path}\n       {out / 'summary.json'}\n       {out / 'cost_summary.json'}")


if __name__ == "__main__":
    main()
