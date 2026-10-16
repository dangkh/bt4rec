"""
Build collaborative reverse behavior tree DIRECTLY from dual-view precomputed
behavior memories.

Expected input schema
---------------------
Each memory must already contain BOTH:

A) Ranking-oriented semantic view
   - recommendation_behavior
   - semantic_focus

B) Tree-oriented trajectory view
   - trajectory_signature
   - behavior_signature        (alias is accepted)
   - mechanism
   - scope
   - direction
   - confidence

IMPORTANT DESIGN RULE
---------------------
Only the TRAJECTORY view is used to create cluster embeddings / state identity.
recommendation_behavior and semantic_focus are preserved only for auditing and
later ranking evidence. They never enter KMeans.

Pipeline
--------
dual-view local memories
    -> abstract trajectory text
    -> hard bucket by mechanism (default)
    -> KMeans within compatible buckets
    -> confidence/margin filtering
    -> ordered accepted state segments
    -> collaborative variable-order reverse suffix tree

Ambiguous/rejected assignments BREAK a temporal segment rather than being
silently removed and bridging the states on either side. This avoids creating
false transitions across an uncertain memory.


Refactored entry point (formerly build_reverse_behavior_tree.py). Run: python -m bt4rec.cli.build_tree --help
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from bt4rec.common.io import json_dump, jsonl_dump
from bt4rec.common.seeding import set_seed
from bt4rec.common.vectors import l2_normalize
from bt4rec.taxonomy import TAXONOMY_VERSION
from bt4rec.tree.assign import relabel_by_cosine
from bt4rec.tree.builder import (
    accumulate_contexts,
    build_context_records,
    build_reverse_tree,
    build_user_sequences,
    validate_tree_structure,
)
from bt4rec.tree.clustering import assignment_diagnostics, build_cluster_states, cluster_quality, run_kmeans
from bt4rec.tree.embeddings import create_embeddings
from bt4rec.tree.memories import DUAL_SCHEMA_NAME, dual_schema_stats, load_memories, read_user_id_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dual-view behavior cluster tree: trajectory view -> Qwen -> "
            "constrained KMeans -> reverse behavior tree"
        )
    )

    p.add_argument(
        "--input",
        default="precomputed/CDs/local_memories_gemma_dual_1000.jsonl",
        help=(
            "Dual-view precomputed behavior-memory JSONL. "
            "Recommended: use the already-precomputed 1000-user file."
        ),
    )
    p.add_argument(
        "--num-users",
        type=int,
        default=0,
        help=(
            "Use only the first N unique users in input-file order. "
            "Examples: 100, 200, 300, ..., 1000. "
            "0 means use all users in the input file."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="behavior_tree_out_cluster_k50",
    )

    p.add_argument("--include-users", default=None)
    p.add_argument("--exclude-users", default=None)
    p.add_argument("--strict-input", action="store_true")

    p.add_argument(
        "--cluster-text-field",
        choices=[
            "structured_combined",
            "trajectory_signature",
            "behavior_signature",
        ],
        default="structured_combined",
        help=(
            "Text embedded for KMeans. structured_combined is recommended and "
            "uses ONLY mechanism/scope/direction + trajectory_signature. "
            "Semantic recommendation fields are never embedded."
        ),
    )

    p.add_argument(
        "--encoder",
        choices=["qwen", "tfidf"],
        default="qwen",
    )
    p.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    p.add_argument("--embeddings-npy", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--tfidf-max-features",
        type=int,
        default=8000,
    )

    p.add_argument(
        "--num-clusters",
        type=int,
        default=50,
        help=(
            "Requested TOTAL number of clusters. With the default mechanism-level "
            "constraint, effective K remains equal to requested K whenever feasible."
        ),
    )
    p.add_argument(
        "--cluster-mode",
        choices=["constrained", "global"],
        default="constrained",
        help=(
            "constrained = hard bucket by --constraint-level before KMeans; "
            "global = old unconstrained ablation."
        ),
    )
    p.add_argument(
        "--constraint-level",
        choices=["mechanism", "mechanism_scope", "full"],
        default="mechanism",
        help=(
            "Hard constraint for constrained KMeans. Default mechanism is recommended; "
            "scope/direction remain in the embedding and can split clusters naturally."
        ),
    )
    p.add_argument(
        "--min-cluster-similarity",
        type=float,
        default=0.55,
        help="Reject a training assignment below this cosine similarity.",
    )
    p.add_argument(
        "--min-cluster-margin",
        type=float,
        default=0.02,
        help=(
            "Reject a training assignment when top1-top2 cosine margin among "
            "compatible clusters is below this value."
        ),
    )
    p.add_argument(
        "--keep-ambiguous-assignments",
        action="store_true",
        help=(
            "Diagnostic override: keep low-confidence assignments in tree "
            "sequences. Default is to skip them."
        ),
    )
    p.add_argument(
        "--kmeans-n-init",
        type=int,
        default=20,
    )
    p.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=300,
    )

    # SAME tree hyperparameters as the semantic-state builder.
    p.add_argument(
        "--collapse-consecutive-states",
        action="store_true",
    )
    p.add_argument(
        "--label-assignment",
        choices=["cosine", "kmeans"],
        default="cosine",
        help=(
            "cosine (default): relabel every memory with the SAME nearest-compatible-"
            "centroid rule used at inference (bt4rec.tree.assign). kmeans: keep raw "
            "KMeans labels (legacy behaviour; train/test assignment can disagree)."
        ),
    )
    p.add_argument(
        "--state-signature",
        choices=["medoid", "mode"],
        default="medoid",
        help=(
            "trajectory_signature shown for a state in the ranking prompt. medoid "
            "(default) = member closest to the centroid; mode = legacy most-frequent "
            "exact string."
        ),
    )
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument(
        "--count-mode",
        choices=[
            "user_normalized",
            "distinct_user",
            "occurrence",
        ],
        default="user_normalized",
    )
    p.add_argument(
        "--smoothing-kappa",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--min-support-users",
        type=int,
        default=3,
    )
    p.add_argument(
        "--min-support-occurrences",
        type=int,
        default=3,
    )
    p.add_argument(
        "--min-jsd",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--no-promote-order1",
        action="store_true",
    )
    p.add_argument(
        "--max-evidence",
        type=int,
        default=8,
    )
    p.add_argument(
        "--top-next",
        type=int,
        default=20,
    )
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    include_users = read_user_id_set(
        args.include_users
    )
    exclude_users = read_user_id_set(
        args.exclude_users
    )

    print("[1/8] Load dual-view precomputed behavior memories")
    memories = load_memories(
        path=args.input,
        cluster_text_field=args.cluster_text_field,
        include_users=include_users,
        exclude_users=exclude_users,
        skip_invalid=not args.strict_input,
    )

    # Nested training-size protocol:
    # use the first N UNIQUE users in exact input-file order.
    all_user_order = list(dict.fromkeys(
        str(m["user_id"]) for m in memories
    ))

    if args.num_users < 0:
        raise ValueError("--num-users must be >= 0")

    if args.num_users > 0:
        if args.num_users > len(all_user_order):
            raise ValueError(
                f"--num-users={args.num_users} requested, but input contains "
                f"only {len(all_user_order)} unique users after filtering"
            )

        selected_user_order = all_user_order[:args.num_users]
        selected_user_set = set(selected_user_order)
        memories = [
            m for m in memories
            if str(m["user_id"]) in selected_user_set
        ]
    else:
        selected_user_order = all_user_order

    num_users = len(selected_user_order)

    print(
        f"      memories={len(memories)} "
        f"users={num_users}"
    )
    print(
        f"      requested_num_users="
        f"{args.num_users if args.num_users > 0 else 'ALL'}"
    )
    print(
        f"      cluster_text_field="
        f"{args.cluster_text_field}"
    )

    print("[2/8] Validate dual-view schema")
    schema_stats = dual_schema_stats(
        memories
    )
    print(
        f"      unknown_profile_ratio="
        f"{schema_stats['unknown_profile_ratio']:.3f}"
    )
    print(
        f"      mechanisms="
        f"{len(schema_stats['mechanism_counts'])}"
    )

    print("[3/8] Save cluster inputs")
    jsonl_dump(
        [
            {
                "memory_id": m["memory_id"],
                "user_id": m["user_id"],
                "window_index": m["window_index"],
                "cluster_text": m["cluster_text"],

                # Semantic/ranking view (NOT used for clustering).
                "recommendation_behavior": m.get(
                    "recommendation_behavior"
                ),
                "semantic_focus": m.get(
                    "semantic_focus"
                ),

                # Trajectory/tree view.
                "trajectory_signature": m.get(
                    "trajectory_signature"
                ),
                "behavior_signature": m.get(
                    "behavior_signature"
                ),
                "behavior_explanation": m.get(
                    "behavior_explanation"
                ),
                "mechanism": m.get("mechanism"),
                "scope": m.get("scope"),
                "direction": m.get("direction"),
                "behavior_profile": m.get("behavior_profile"),
                "behavior_bucket_key": m.get("behavior_bucket_key"),
                "keywords": m.get("keywords"),
            }
            for m in memories
        ],
        out_dir / "cluster_behavior_inputs.jsonl",
    )

    print("[4/8] Embed trajectory behavior text")
    embeddings = create_embeddings(
        args,
        memories,
    )
    embeddings = l2_normalize(embeddings)
    print(
        f"      embedding_shape={embeddings.shape}"
    )

    print(
        f"[5/8] KMeans state induction "
        f"(mode={args.cluster_mode}, requested_K={args.num_clusters})"
    )
    labels, centers, inertia, cluster_meta, cluster_info = run_kmeans(
        memories=memories,
        embeddings=embeddings,
        num_clusters=args.num_clusters,
        seed=args.seed,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        cluster_mode=args.cluster_mode,
        constraint_level=args.constraint_level,
    )
    if args.label_assignment == "cosine":
        kmeans_labels = labels
        labels = relabel_by_cosine(
            memories, embeddings, centers, cluster_meta,
            args.cluster_mode, args.constraint_level,
        )
        changed = int(np.sum(kmeans_labels != labels))
        cluster_info["label_assignment"] = "cosine_nearest_compatible"
        cluster_info["labels_changed_from_kmeans"] = changed
        print(f"      relabel_by_cosine: {changed}/{len(labels)} memories changed state")
    else:
        cluster_info["label_assignment"] = "kmeans_raw"

    assignment_info = assignment_diagnostics(
        memories=memories,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
        cluster_meta=cluster_meta,
        cluster_mode=args.cluster_mode,
        constraint_level=args.constraint_level,
        min_similarity=args.min_cluster_similarity,
        min_margin=args.min_cluster_margin,
    )
    if args.keep_ambiguous_assignments:
        for row in assignment_info:
            row["assignment_accepted"] = True
            row["rejection_reason"] = None

    state_rows, idx_to_state, centroid_rows = build_cluster_states(
        memories=memories,
        embeddings=embeddings,
        labels=labels,
        centers=centers,
        cluster_meta=cluster_meta,
        assignment_info=assignment_info,
        signature_mode=args.state_signature,
    )

    state_centroids = np.stack(
        centroid_rows,
        axis=0,
    ).astype(np.float32)

    quality = cluster_quality(
        embeddings,
        labels,
    )
    quality["kmeans_inertia"] = inertia

    accepted_count = sum(1 for x in assignment_info if x["assignment_accepted"])
    rejected_count = len(assignment_info) - accepted_count
    margins = [
        x["compatible_similarity_margin"]
        for x in assignment_info
        if x["compatible_similarity_margin"] is not None
    ]
    quality["assignment_quality"] = {
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_ratio": float(accepted_count / len(assignment_info)),
        "min_cluster_similarity": args.min_cluster_similarity,
        "min_cluster_margin": args.min_cluster_margin,
        "keep_ambiguous_assignments": args.keep_ambiguous_assignments,
        "mean_compatible_margin": float(np.mean(margins)) if margins else None,
        "p10_compatible_margin": float(np.quantile(margins, 0.10)) if margins else None,
        "p50_compatible_margin": float(np.quantile(margins, 0.50)) if margins else None,
    }

    print(
        f"      states={len(state_rows)} "
        f"effective_K={cluster_info['effective_num_clusters']} "
        f"inertia={inertia:.4f}"
    )
    print(
        f"      accepted_assignments={accepted_count}/{len(assignment_info)} "
        f"({accepted_count / len(assignment_info):.3f})"
    )
    if "silhouette_cosine" in quality:
        print(
            f"      silhouette_cosine="
            f"{quality['silhouette_cosine']:.4f}"
        )

    state_payload = {
        "metadata": {
            "input": os.path.abspath(
                args.input
            ),
            "num_memories": len(memories),
            "num_users": num_users,
            "requested_num_users": (
                args.num_users if args.num_users > 0 else None
            ),
            "selected_user_order": selected_user_order,
            "num_states": len(state_rows),
            "state_induction": "constrained_kmeans" if args.cluster_mode == "constrained" else "global_kmeans",
            "requested_num_clusters": args.num_clusters,
            "effective_num_clusters": cluster_info["effective_num_clusters"],
            "cluster_mode": args.cluster_mode,
            "constraint_level": args.constraint_level,
            "constraint_fields": (
                ["mechanism"] if args.constraint_level == "mechanism" else
                ["mechanism", "scope"] if args.constraint_level == "mechanism_scope" else
                ["mechanism", "scope", "direction"]
            ),
            "input_behavior_schema": schema_stats,
            "semantic_fields_used_for_clustering": False,
            "cluster_info": cluster_info,
            "cluster_text_field": (
                args.cluster_text_field
            ),
            "encoder": args.encoder,
            "embedding_model": (
                args.embedding_model
                if args.encoder == "qwen"
                else None
            ),
            "state_representation": (
                "Behavior-constrained KMeans (configurable hard constraint) over structured behavior embeddings" if args.cluster_mode == "constrained" else "Global KMeans over Gemma behavior-text embeddings"
            ),
            "cluster_quality": quality,
            "taxonomy_version": TAXONOMY_VERSION,
            "label_assignment": args.label_assignment,
            "state_signature": args.state_signature,
            "collapse_consecutive_states": bool(args.collapse_consecutive_states),
        },
        "states": state_rows,
    }

    json_dump(
        state_payload,
        out_dir / "behavior_states.json",
    )
    np.save(
        out_dir / "behavior_state_embeddings.npy",
        state_centroids,
    )

    assignment_rows: List[Dict[str, Any]] = []
    for idx, m in enumerate(memories):
        cluster_idx = int(labels[idx])
        sid = f"C{cluster_idx:03d}"
        diag = assignment_info[idx]

        assignment_rows.append({
            "memory_id": m["memory_id"],
            "precompute_id": m.get("precompute_id"),
            "user_id": m["user_id"],
            "window_index": m["window_index"],
            "state_id": sid,
            "cluster_id": cluster_idx,
            "cluster_similarity": diag["assigned_similarity"],
            "cluster_text": m["cluster_text"],

            # Ranking-oriented semantic view; stored only for downstream use.
            "recommendation_behavior": m.get(
                "recommendation_behavior"
            ),
            "semantic_focus": m.get(
                "semantic_focus"
            ),

            # Tree-oriented trajectory view.
            "trajectory_signature": m.get(
                "trajectory_signature"
            ),
            "behavior_signature": m.get(
                "behavior_signature"
            ),
            "behavior_explanation": m.get(
                "behavior_explanation"
            ),
            "mechanism": m.get("mechanism"),
            "scope": m.get("scope"),
            "direction": m.get("direction"),
            "behavior_profile": m.get("behavior_profile"),
            "behavior_bucket_key": m.get("behavior_bucket_key"),
            "compatible_top1_cluster_id": diag["compatible_top1_cluster_id"],
            "compatible_top1_similarity": diag["compatible_top1_similarity"],
            "compatible_top2_cluster_id": diag["compatible_top2_cluster_id"],
            "compatible_top2_similarity": diag["compatible_top2_similarity"],
            "similarity_margin": diag["compatible_similarity_margin"],
            "best_alternative_similarity": diag["best_alternative_similarity"],
            "num_compatible_clusters": diag["num_compatible_clusters"],
            "nearest_compatible_agrees_with_assignment": diag["nearest_compatible_agrees_with_assignment"],
            "global_top1_cluster_id": diag["global_top1_cluster_id"],
            "global_top1_similarity": diag["global_top1_similarity"],
            "global_top2_cluster_id": diag["global_top2_cluster_id"],
            "global_top2_similarity": diag["global_top2_similarity"],
            "assignment_accepted": diag["assignment_accepted"],
            "rejection_reason": diag["rejection_reason"],
            "keywords": m.get("keywords"),
            "state_induction": (
                "constrained_kmeans" if args.cluster_mode == "constrained"
                else "global_kmeans"
            ),
        })

    jsonl_dump(
        assignment_rows,
        out_dir / "memory_state_assignments.jsonl",
    )

    print("[6/8] Reconstruct accepted state segments")
    user_sequences = build_user_sequences(
        memories,
        idx_to_state,
        collapse_consecutive=(
            args.collapse_consecutive_states
        ),
    )

    json_dump(
        {
            "metadata": {
                "num_users": len(user_sequences),
                "collapse_consecutive_states": (
                    args.collapse_consecutive_states
                ),
                "state_induction": (
                    "constrained_kmeans" if args.cluster_mode == "constrained"
                    else "global_kmeans"
                ),
                "requested_num_clusters": args.num_clusters,
                "effective_num_clusters": cluster_info["effective_num_clusters"],
                "cluster_mode": args.cluster_mode,
                "min_cluster_similarity": args.min_cluster_similarity,
                "min_cluster_margin": args.min_cluster_margin,
                "keep_ambiguous_assignments": args.keep_ambiguous_assignments,
            },
            "users": user_sequences,
        },
        out_dir / "user_behavior_sequences.json",
    )

    observed_transitions = sum(
        max(0, seg["length"] - 1)
        for u in user_sequences.values()
        for seg in u.get("segments", [])
    )

    print(
        f"      observed_adjacent_transitions="
        f"{observed_transitions}"
    )

    if observed_transitions <= 0:
        raise ValueError(
            "No transitions after sequence construction"
        )

    print(
        "[7/8] Generate overlapping "
        "context -> next observations"
    )
    ctx_stats = accumulate_contexts(
        user_sequences=user_sequences,
        max_order=args.max_order,
        max_evidence=args.max_evidence,
        seed=args.seed,
    )

    print(
        f"      unique_contexts_including_root="
        f"{len(ctx_stats)}"
    )

    print(
        "[8/8] Estimate collaborative next distributions "
        "+ suffix smoothing"
    )
    records = build_context_records(
        stats=ctx_stats,
        state_rows=state_rows,
        count_mode=args.count_mode,
        smoothing_kappa=args.smoothing_kappa,
        min_support_users=args.min_support_users,
        min_support_occurrences=(
            args.min_support_occurrences
        ),
        min_jsd=args.min_jsd,
        promote_order1=(
            not args.no_promote_order1
        ),
        top_next=args.top_next,
    )

    jsonl_dump(
        [
            records[c]
            for c in sorted(
                records,
                key=lambda x: (len(x), x),
            )
        ],
        out_dir / "context_observations.jsonl",
    )

    print("      Build ONE physical reverse suffix tree")
    tree = build_reverse_tree(records)
    validate_tree_structure(
        tree,
        records,
    )

    tree_payload = {
        "metadata": {
            "tree_type": (
                "collaborative_variable_order_"
                "reverse_suffix_behavior_tree"
            ),
            "single_tree": True,
            "time_direction_for_learning": (
                "chronological"
            ),
            "index_direction": "reverse_suffix",
            "count_mode": args.count_mode,
            "collapse_consecutive_states": bool(args.collapse_consecutive_states),
            "taxonomy_version": TAXONOMY_VERSION,
            "smoothing": {
                "method": (
                    "suffix_parent_interpolation"
                ),
                "lambda": (
                    "support_users/"
                    "(support_users+kappa)"
                ),
                "kappa": args.smoothing_kappa,
            },
            "activation": {
                "min_support_users": (
                    args.min_support_users
                ),
                "min_support_occurrences": (
                    args.min_support_occurrences
                ),
                "min_jsd_raw_to_suffix_parent_bits": (
                    args.min_jsd
                ),
                "promote_order1": (
                    not args.no_promote_order1
                ),
                "structural_nodes_are_retained_when_inactive": True,
            },
            "input": os.path.abspath(args.input),
            "num_users": len(user_sequences),
            "requested_num_users": (
                args.num_users if args.num_users > 0 else None
            ),
            "num_memories": len(memories),
            "num_states": len(state_rows),
            "state_identity": "trajectory_behavior_constrained_kmeans_cluster_id",
            "input_behavior_schema": DUAL_SCHEMA_NAME,
            "semantic_fields_used_for_clustering": False,
            "state_representation": (
                f"{args.cluster_mode} KMeans(requested_K={args.num_clusters}, "
                f"effective_K={cluster_info['effective_num_clusters']}) over "
                f"{args.cluster_text_field} embeddings"
            ),
            "constraint_level": args.constraint_level,
            "constraint_fields": (
                ["mechanism"] if args.constraint_level == "mechanism" else
                ["mechanism", "scope"] if args.constraint_level == "mechanism_scope" else
                ["mechanism", "scope", "direction"]
            ),
            "assignment_filter": {
                "min_similarity": args.min_cluster_similarity,
                "min_margin": args.min_cluster_margin,
                "keep_ambiguous": args.keep_ambiguous_assignments,
            },
            "observed_adjacent_transitions": (
                observed_transitions
            ),
            "max_order": args.max_order,
        },
        **tree,
    }

    json_dump(
        tree_payload,
        out_dir / "behavior_tree.json",
    )

    by_order = defaultdict(
        lambda: {
            "structural": 0,
            "active": 0,
        }
    )
    for ctx, rec in records.items():
        order = len(ctx)
        by_order[order]["structural"] += 1
        if rec["active_predictive"]:
            by_order[order]["active"] += 1

    state_sizes = [
        int(x["member_count"])
        for x in state_rows
    ]
    largest_state_size = (
        max(state_sizes)
        if state_sizes
        else 0
    )

    raw_self_transitions = 0
    raw_total_transitions = 0
    distinct_states_per_user: List[int] = []

    for u in user_sequences.values():
        seq = u["states"]
        distinct_states_per_user.append(
            len(set(seq))
        )
        for seg in u.get(
            "segments",
            [],
        ):
            seg_seq = seg["states"]
            for a, b in zip(
                seg_seq[:-1],
                seg_seq[1:],
            ):
                raw_total_transitions += 1
                raw_self_transitions += int(
                    a == b
                )

    cluster_size_distribution = {
        "min": int(np.min(state_sizes)),
        "p25": float(np.quantile(
            state_sizes, 0.25
        )),
        "p50": float(np.quantile(
            state_sizes, 0.50
        )),
        "p75": float(np.quantile(
            state_sizes, 0.75
        )),
        "max": int(np.max(state_sizes)),
    }

    stats_payload = {
        "input": os.path.abspath(args.input),
        "num_memories": len(memories),
        "num_users": len(user_sequences),
        "requested_num_users": (
            args.num_users if args.num_users > 0 else None
        ),
        "num_behavior_states": len(state_rows),
        "state_induction": ("constrained_kmeans" if args.cluster_mode == "constrained" else "global_kmeans"),
        "requested_num_clusters": args.num_clusters,
        "effective_num_clusters": cluster_info["effective_num_clusters"],
        "cluster_mode": args.cluster_mode,
        "constraint_level": args.constraint_level,
            "constraint_fields": (
                ["mechanism"] if args.constraint_level == "mechanism" else
                ["mechanism", "scope"] if args.constraint_level == "mechanism_scope" else
                ["mechanism", "scope", "direction"]
            ),
        "cluster_info": cluster_info,
        "cluster_text_field": (
            args.cluster_text_field
        ),
        "input_behavior_schema": (
            DUAL_SCHEMA_NAME
        ),
        "semantic_fields_used_for_clustering": False,
        "dual_schema_stats": schema_stats,
        "cluster_quality": quality,
        "accepted_assignment_count": accepted_count,
        "rejected_assignment_count": rejected_count,
        "accepted_assignment_ratio": float(accepted_count / len(assignment_info)),
        "cluster_size_distribution": (
            cluster_size_distribution
        ),
        "mean_memories_per_state": float(
            np.mean(state_sizes)
        ),
        "median_memories_per_state": float(
            np.median(state_sizes)
        ),
        "num_singleton_states": int(
            sum(1 for n in state_sizes if n == 1)
        ),
        "largest_state_size": int(
            largest_state_size
        ),
        "largest_state_ratio": float(
            largest_state_size / len(memories)
        ),
        "observed_adjacent_transitions": (
            observed_transitions
        ),
        "self_transition_count": (
            raw_self_transitions
        ),
        "self_transition_ratio": float(
            raw_self_transitions
            / raw_total_transitions
        ) if raw_total_transitions else 0.0,
        "mean_distinct_states_per_user": float(
            np.mean(distinct_states_per_user)
        ) if distinct_states_per_user else 0.0,
        "num_users_with_one_state": int(
            sum(
                1
                for x in distinct_states_per_user
                if x == 1
            )
        ),
        "num_segments": int(
            sum(
                len(u.get("segments", []))
                for u in user_sequences.values()
            )
        ),
        "num_skipped_ambiguous_memories": int(
            sum(
                u.get(
                    "num_skipped_ambiguous_memories",
                    0,
                )
                for u in user_sequences.values()
            )
        ),
        "num_structural_context_nodes_including_root": (
            len(records)
        ),
        "num_active_predictive_nodes_including_root": (
            tree["num_active_predictive_nodes"]
        ),
        "contexts_by_order": {
            str(k): v
            for k, v
            in sorted(by_order.items())
        },
        "config": vars(args),
    }

    json_dump(
        stats_payload,
        out_dir / "tree_stats.json",
    )

    print("\n[DONE] Files")
    for name in [
        "cluster_behavior_inputs.jsonl",
        "behavior_states.json",
        "behavior_state_embeddings.npy",
        "memory_state_assignments.jsonl",
        "user_behavior_sequences.json",
        "context_observations.jsonl",
        "behavior_tree.json",
        "tree_stats.json",
    ]:
        print(f"  {out_dir / name}")

    print("\nKey stats")
    print(json.dumps({
        "num_memories": len(memories),
        "num_users": len(user_sequences),
        "num_states": len(state_rows),
        "effective_num_clusters": cluster_info["effective_num_clusters"],
        "accepted_assignment_ratio": float(accepted_count / len(assignment_info)),
        "largest_state_size": int(
            largest_state_size
        ),
        "largest_state_ratio": float(
            largest_state_size / len(memories)
        ),
        "observed_adjacent_transitions": (
            observed_transitions
        ),
        "structural_nodes": len(records),
        "active_predictive_nodes": (
            tree["num_active_predictive_nodes"]
        ),
        "self_transition_ratio": float(
            raw_self_transitions
            / raw_total_transitions
        ) if raw_total_transitions else 0.0,
        "mean_distinct_states_per_user": float(
            np.mean(distinct_states_per_user)
        ) if distinct_states_per_user else 0.0,
        "silhouette_cosine": (
            quality.get("silhouette_cosine")
        ),
        "kmeans_inertia": inertia,
    }, indent=2))

if __name__ == "__main__":
    main()
