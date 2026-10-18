#!/usr/bin/env python3
"""Compare Trees built with different settings (e.g. num-clusters 30/50/80/120).

  python scripts/compare_trees.py trees/VGs_k30 trees/VGs_1000_20 trees/VGs_k80 trees/VGs_k120

Reads <tree>/tree_stats.json (written by bt4rec.cli.build_tree) and prints one row per Tree:
states, accepted-assignment ratio, silhouette, state sizes, self-transition ratio
(= how often the next window keeps the same state; drives P(stay)), distinct states
per user, and active predictive context nodes per order. More clusters = finer HOW
states but sparser contexts: watch accepted_ratio, min state size and active nodes.
"""
import json
import sys
from pathlib import Path

cols = ["tree", "states", "accepted", "silhouette", "min_size", "median_size", "self_trans",
        "distinct/user", "active_nodes", "contexts_by_order"]
print(" | ".join(cols))
for d in sys.argv[1:]:
    t = json.load(open(Path(d) / "tree_stats.json"))
    q = t.get("cluster_quality", {})
    sz = t.get("cluster_size_distribution", {})
    print(" | ".join(str(x) for x in [
        Path(d).name, t.get("num_behavior_states"), f"{t.get('accepted_assignment_ratio', 0):.3f}",
        f"{q.get('silhouette_cosine', float('nan')):.3f}", sz.get("min"), sz.get("p50"),
        f"{t.get('self_transition_ratio', 0):.3f}", t.get("mean_distinct_states_per_user"),
        t.get("num_active_predictive_nodes_including_root"), t.get("contexts_by_order")]))
