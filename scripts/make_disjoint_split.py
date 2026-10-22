#!/usr/bin/env python3
"""User-disjoint split of the frozen 1000-user cohort (no new LLM / memory needed).

The behaviour memories of all 1000 users are already precomputed, so a disjoint
experiment only needs two NON-overlapping user lists taken from eval_users_1000.json:

  eval  = the first --n users          -> rank these (CoMemTree / LLMRank / i2Agent)
  tree  = the last  --n users          -> build the Tree only from these

  python scripts/make_disjoint_split.py --dataset CDs --n 300
writes data/<DS>/protocol/disjoint{n}_eval_users.json and disjoint{n}_tree_users.json.
The first 300 eval users are the same users as the --max-users 300 runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt4rec.common.io import load_user_ids_file  # noqa: E402
from bt4rec.config import load_dataset_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    cfg = load_dataset_config(a.dataset)
    users = load_user_ids_file(cfg["eval_users"])
    if 2 * a.n > len(users):
        raise SystemExit(f"need 2*{a.n} users, cohort has {len(users)}")
    ev, tr = users[: a.n], users[-a.n:]
    assert not set(ev) & set(tr)
    out = Path(cfg["protocol_dir"])
    pe, pt = out / f"disjoint{a.n}_eval_users.json", out / f"disjoint{a.n}_tree_users.json"
    pe.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    pt.write_text(json.dumps(tr, indent=2) + "\n", encoding="utf-8")
    print(f"{cfg['dataset']}: eval={len(ev)} -> {pe}\n{' ' * len(cfg['dataset'])}  tree={len(tr)} -> {pt}\noverlap=0")


if __name__ == "__main__":
    main()
