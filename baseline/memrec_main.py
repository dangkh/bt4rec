#!/usr/bin/env python3
"""MemRec (collaborative memory agent) under the CoMemTree frozen protocol.

Pipeline (memrec/train/trainer_memrec.py::_test_frozen):
  1. Warm-up on TRAIN only (offline): for every train user, hide the last TRAIN item,
     rank it among 19 negatives with Stage-R + Stage-ReRank, and if it lands in the
     top-10 run Stage-W to write user / neighbour / item memories.
  2. Test (online): memory is frozen; each eval user is ranked over the 20 frozen
     candidates with Stage-R + Stage-ReRank (C-label list output).

Protocol: users = data/<DS>/protocol/{train,eval}_users_1000.json (identical, so
overlap is allowed), history and graph use the last 20 TRAIN items (ML 30),
item text = title + category only. Warm-up negatives are sampled from the catalog
excluding the user's TRAIN items and frozen TEST candidates (--warmup-negatives).

Outputs (<output-dir>): rankings.jsonl, metrics.json, cost_summary.json,
llm_calls.jsonl, memory_frozen.json, warmup.json, id_maps.json, config.json.

  python baseline/memrec_main.py --dataset CDs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from bt4rec.config import load_dataset_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="CDs | Video_Games | ML")
    p.add_argument("--data-root", default="data")
    p.add_argument("--history-size", type=int, default=None, help="default from configs (20; ML 30)")
    p.add_argument("--num-users", type=int, default=1000)
    p.add_argument("--train-users-file", default=None)
    p.add_argument("--eval-users-file", default=None)
    p.add_argument("--candidate-file", default=None)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-tokens", type=int, default=1000, help="max new tokens per LLM call")
    p.add_argument("--k", type=int, default=16, help="neighbours kept by the pruner")
    p.add_argument("--tau-tokens", type=int, default=1800, help="memory snippet token budget")
    p.add_argument("--n-facets", type=int, default=7)
    p.add_argument("--pruner", default="hybrid_rule", choices=["hybrid_rule", "llm_rules"])
    p.add_argument("--stage-r-candidates", default="all", choices=["all", "first10", "none"])
    p.add_argument("--warmup-rounds", type=int, default=1, help="0 disables memory construction")
    p.add_argument("--warmup-negatives", default="catalog", choices=["catalog", "declared"])
    p.add_argument("--fanout-cap", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, help="default results/baselines/<DS>/memrec")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    cfg = load_dataset_config(a.dataset)
    a.dataset = cfg["dataset"]
    a.history_size = a.history_size or int(cfg["history_size"])
    a.train_users_file = a.train_users_file or cfg["train_users"]
    a.eval_users_file = a.eval_users_file or cfg["eval_users"]
    a.candidate_file = a.candidate_file or cfg["candidate_file"]
    a.output_dir = a.output_dir or str(Path("results/baselines") / a.dataset / "memrec")
    return a


def main():
    a = parse_args()
    data_dir = Path(a.data_root) / a.dataset
    config = {
        "dataset": a.dataset, "seed": a.seed, "history_size": a.history_size,
        "sequences_file": str(data_dir / "user_sequences.json"),
        "negatives_file": str(data_dir / "user_negatives.json"),
        "items_file": str(data_dir / "items.json"),
        "train_users_file": a.train_users_file, "eval_users_file": a.eval_users_file,
        "num_train_users": a.num_users, "num_test_users": a.num_users,
        "allow_overlap": True,  # protocol: train_users_1000 == eval_users_1000 (TRAIN items only)
        "candidate_file": a.candidate_file, "n_eval_candidates": 20,
        "warmup_negative_source": a.warmup_negatives,
        "instruction_mode": "history", "item_text_mode": "category",
        "eval_feedback": "none", "topk": [1, 3, 5, 10],
        "provider": {"name": "local_hf", "model": a.model, "load_in_4bit": a.load_in_4bit},
        "memrec": {"k": a.k, "tau_tokens": a.tau_tokens, "n_facets": a.n_facets,
                   "temperature": 0.0, "max_tokens": a.max_tokens,
                   "reranker_mode": "llm", "reranker_output": "list",
                   "stage_r_candidates": a.stage_r_candidates,
                   "pruner": {"mode": a.pruner}, "write": {"fanout_cap": a.fanout_cap}},
        "warmup": {"enabled": a.warmup_rounds > 0, "rounds": a.warmup_rounds, "n_candidates": 20},
        "output_dir": a.output_dir, "debug": a.debug,
    }
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    from memrec.data.dataset_frozen import FrozenRecDataset
    from memrec.utils import set_seed
    set_seed(a.seed)
    ds = FrozenRecDataset(data_dir, config)
    print(ds, json.dumps(ds.get_stats()), flush=True)

    import torch
    from memrec.train.trainer_memrec import MemRecTrainer
    trainer = MemRecTrainer(None, ds, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train(save_dir=str(out))
    trainer.test()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
