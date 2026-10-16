"""Print dataset/Tree presets as shell variables for scripts/*.sh.

    eval "$(python -m bt4rec.cli.config CDs)"   # -> DS, HIST, TREE_DIR, EVAL_FILE, ...
"""
from __future__ import annotations

import argparse
import shlex

from bt4rec.config import load_dataset_config, load_tree_config

DATASET_VARS = {
    "DS": "dataset", "HIST": "history_size", "WINDOW": "window_size", "TREE_DEFAULT": "tree_dir",
    "PROTOCOL_DIR": "protocol_dir", "EVAL_FILE": "eval_users", "TRAIN_IDS_DEFAULT": "train_users",
    "CAND_FILE": "candidate_file", "PRECOMP_DIR": "precomputed_dir",
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    a = p.parse_args(argv)
    cfg = load_dataset_config(a.dataset)
    lines = [f"{var}={shlex.quote(str(cfg[key]))}" for var, key in DATASET_VARS.items() if key in cfg]
    for key, value in load_tree_config().items():
        lines.append(f"TREECFG_{key.upper()}={shlex.quote(str(value).lower() if isinstance(value, bool) else str(value))}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
