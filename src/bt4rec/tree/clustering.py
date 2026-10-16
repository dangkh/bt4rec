from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bt4rec.common.text import clean_text
from bt4rec.common.vectors import l2_normalize
from bt4rec.taxonomy import (
    abstract_behavior_text,
    behavior_bucket_key,
    clustering_bucket_key,
    get_trajectory_signature,
)


def _fit_kmeans_block(
    x: np.ndarray,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError("Need scikit-learn for KMeans clustering") from e

    if k <= 0 or k > len(x):
        raise ValueError(f"invalid k={k} for block size={len(x)}")

    if k == 1:
        raw_center = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        labels = np.zeros(len(x), dtype=np.int64)
        diff = x - raw_center[0]
        inertia = float(np.sum(diff * diff))
        return labels, l2_normalize(raw_center), inertia

    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0,
    )
    labels = km.fit_predict(x)
    centers = l2_normalize(np.asarray(km.cluster_centers_, dtype=np.float32))
    return np.asarray(labels, dtype=np.int64), centers, float(km.inertia_)


def _allocate_clusters_to_buckets(
    bucket_sizes: Dict[Tuple[str, ...], int],
    requested_k: int,
) -> Dict[Tuple[str, ...], int]:
    """
    Allocate at least one cluster per behavior bucket, then split the largest
    expected cluster until the requested/effective total K is reached.

    If the number of behavior buckets is larger than requested_k, the effective
    K is increased to the number of buckets instead of merging incompatible
    behavior profiles.
    """
    if not bucket_sizes:
        raise ValueError("No behavior buckets available")
    if requested_k <= 1:
        raise ValueError("--num-clusters must be > 1")

    if len(bucket_sizes) > requested_k:
        raise ValueError(
            f"Hard constraint creates {len(bucket_sizes)} buckets > requested K={requested_k}. "
            "Use --constraint-level mechanism (recommended), increase K, or use global mode."
        )
    effective_k = min(int(requested_k), int(sum(bucket_sizes.values())))

    alloc = {key: 1 for key in bucket_sizes}
    remaining = effective_k - len(alloc)

    while remaining > 0:
        candidates = [
            key for key, size in bucket_sizes.items()
            if alloc[key] < size
        ]
        if not candidates:
            break
        # Split the bucket with the largest expected members/cluster.
        key = max(
            candidates,
            key=lambda z: (bucket_sizes[z] / alloc[z], bucket_sizes[z], z),
        )
        alloc[key] += 1
        remaining -= 1

    return alloc


def run_global_kmeans(
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    labels, centers, inertia = _fit_kmeans_block(
        embeddings, num_clusters, seed, n_init, max_iter
    )
    meta = {
        int(i): {
            "cluster_id": int(i),
            "cluster_mode": "global",
            "behavior_profile": None,
            "behavior_bucket_key": None,
        }
        for i in range(len(centers))
    }
    return labels, centers, inertia, meta, {
        "requested_num_clusters": int(num_clusters),
        "effective_num_clusters": int(len(centers)),
        "num_behavior_buckets": None,
        "bucket_cluster_allocation": None,
    }


def run_constrained_kmeans(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
    constraint_level: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    HARD constraint according to constraint_level.
    Recommended default is mechanism-only.
    """
    bucket_to_indices: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, m in enumerate(memories):
        bucket_to_indices[clustering_bucket_key(m, constraint_level)].append(idx)

    bucket_sizes = {k: len(v) for k, v in bucket_to_indices.items()}
    allocation = _allocate_clusters_to_buckets(bucket_sizes, num_clusters)

    effective_k = sum(allocation.values())

    labels = np.full(len(memories), -1, dtype=np.int64)
    centers: List[np.ndarray] = []
    cluster_meta: Dict[int, Dict[str, Any]] = {}
    total_inertia = 0.0
    global_cluster_id = 0

    for bucket_idx, key in enumerate(sorted(bucket_to_indices.keys())):
        indices = bucket_to_indices[key]
        local_x = embeddings[indices]
        local_k = allocation[key]
        local_labels, local_centers, inertia = _fit_kmeans_block(
            local_x,
            local_k,
            seed + bucket_idx,
            n_init,
            max_iter,
        )
        total_inertia += inertia

        for local_cluster_id in range(local_k):
            gid = global_cluster_id
            global_cluster_id += 1
            centers.append(local_centers[local_cluster_id].astype(np.float32))
            cluster_meta[gid] = {
                "cluster_id": gid,
                "cluster_mode": "constrained",
                "behavior_profile": None,
                "behavior_bucket_key": list(key),
                "bucket_size": len(indices),
                "clusters_in_bucket": local_k,
                "local_cluster_id": local_cluster_id,
            }
            for pos, mem_idx in enumerate(indices):
                if int(local_labels[pos]) == local_cluster_id:
                    labels[mem_idx] = gid

    if np.any(labels < 0):
        bad = np.where(labels < 0)[0][:10].tolist()
        raise AssertionError(f"Unassigned memories after constrained KMeans: {bad}")

    centers_arr = np.stack(centers, axis=0).astype(np.float32)
    allocation_json = {
        " | ".join(key): int(v)
        for key, v in sorted(allocation.items())
    }
    info = {
        "requested_num_clusters": int(num_clusters),
        "effective_num_clusters": int(len(centers_arr)),
        "num_behavior_buckets": int(len(bucket_to_indices)),
        "constraint_level": constraint_level,
        "bucket_cluster_allocation": allocation_json,
        "bucket_sizes": {
            " | ".join(key): int(size)
            for key, size in sorted(bucket_sizes.items())
        },
    }
    return labels, centers_arr, float(total_inertia), cluster_meta, info


def run_kmeans(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    num_clusters: int,
    seed: int,
    n_init: int,
    max_iter: int,
    cluster_mode: str,
    constraint_level: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if num_clusters > len(embeddings):
        raise ValueError(
            f"--num-clusters={num_clusters} > num_memories={len(embeddings)}"
        )
    if cluster_mode == "global":
        return run_global_kmeans(
            embeddings, num_clusters, seed, n_init, max_iter
        )
    if cluster_mode == "constrained":
        return run_constrained_kmeans(
            memories, embeddings, num_clusters, seed, n_init, max_iter, constraint_level
        )
    raise ValueError(cluster_mode)


def assignment_diagnostics(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_meta: Dict[int, Dict[str, Any]],
    cluster_mode: str,
    constraint_level: str,
    min_similarity: float,
    min_margin: float,
) -> List[Dict[str, Any]]:
    """
    Compute top-1/top-2 cosine similarity and margin among COMPATIBLE clusters.
    In constrained mode, compatible means same behavior bucket. In global mode,
    all clusters are compatible.
    """
    bucket_to_clusters: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    if cluster_mode == "constrained":
        for cid, meta in cluster_meta.items():
            key = tuple(meta["behavior_bucket_key"])
            bucket_to_clusters[key].append(int(cid))

    rows: List[Dict[str, Any]] = []
    all_ids = list(range(len(centers)))

    for idx, m in enumerate(memories):
        assigned = int(labels[idx])
        key = clustering_bucket_key(m, constraint_level)
        candidate_ids = (
            sorted(bucket_to_clusters[key])
            if cluster_mode == "constrained"
            else all_ids
        )
        if assigned not in candidate_ids:
            raise AssertionError(
                f"assigned cluster {assigned} not compatible with behavior bucket {key}"
            )

        sims = np.asarray(
            [float(embeddings[idx] @ centers[cid]) for cid in candidate_ids],
            dtype=np.float64,
        )
        order = np.argsort(-sims)
        top1_pos = int(order[0])
        top1_cid = int(candidate_ids[top1_pos])
        top1_sim = float(sims[top1_pos])

        if len(order) >= 2:
            top2_pos = int(order[1])
            top2_cid: Optional[int] = int(candidate_ids[top2_pos])
            top2_sim: Optional[float] = float(sims[top2_pos])
        else:
            top2_cid = None
            top2_sim = None

        # Confidence is defined relative to the ACTUAL assigned cluster, not
        # merely top1-vs-top2. This remains valid even if standard KMeans'
        # Euclidean assignment differs slightly from cosine-to-normalized-center.
        assigned_sim = float(embeddings[idx] @ centers[assigned])
        alternative_sims = [
            float(embeddings[idx] @ centers[cid])
            for cid in candidate_ids
            if cid != assigned
        ]
        if alternative_sims:
            best_alternative_sim: Optional[float] = max(alternative_sims)
            margin: Optional[float] = float(assigned_sim - best_alternative_sim)
            margin_ok = margin >= min_margin
        else:
            best_alternative_sim = None
            margin = None
            margin_ok = True

        nearest_agrees = bool(top1_cid == assigned)
        accepted = bool(
            assigned_sim >= min_similarity
            and margin_ok
        )

        global_sims = embeddings[idx] @ centers.T
        global_order = np.argsort(-global_sims)
        global_top1 = int(global_order[0])
        global_top2 = int(global_order[1]) if len(global_order) > 1 else None

        rows.append({
            "memory_index": idx,
            "assigned_cluster_id": assigned,
            "assigned_similarity": assigned_sim,
            "compatible_top1_cluster_id": top1_cid,
            "compatible_top1_similarity": top1_sim,
            "compatible_top2_cluster_id": top2_cid,
            "compatible_top2_similarity": top2_sim,
            "compatible_similarity_margin": margin,
            "best_alternative_similarity": best_alternative_sim,
            "num_compatible_clusters": len(candidate_ids),
            "nearest_compatible_agrees_with_assignment": nearest_agrees,
            "global_top1_cluster_id": global_top1,
            "global_top1_similarity": float(global_sims[global_top1]),
            "global_top2_cluster_id": global_top2,
            "global_top2_similarity": (
                float(global_sims[global_top2]) if global_top2 is not None else None
            ),
            "assignment_accepted": accepted,
            "rejection_reason": (
                None if accepted else (
                    "low_similarity" if assigned_sim < min_similarity else
                    "low_margin"
                )
            ),
        })

    return rows


def build_cluster_states(
    memories: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster_meta: Dict[int, Dict[str, Any]],
    assignment_info: List[Dict[str, Any]],
    signature_mode: str = "medoid",
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, str],
    List[np.ndarray],
]:
    state_rows: List[Dict[str, Any]] = []
    idx_to_state: Dict[int, str] = {}
    centroid_rows: List[np.ndarray] = []

    accepted_mask = np.asarray(
        [bool(x["assignment_accepted"]) for x in assignment_info], dtype=bool
    )

    for cluster_idx in range(len(centers)):
        member_indices = np.where(labels == cluster_idx)[0].tolist()
        if not member_indices:
            continue

        sid = f"C{cluster_idx:03d}"
        centroid = centers[cluster_idx]
        mat = embeddings[member_indices]
        sims = mat @ centroid
        medoid_local = int(np.argmax(sims))
        medoid_idx = member_indices[medoid_local]
        medoid = memories[medoid_idx]

        accepted_member_indices = [i for i in member_indices if accepted_mask[i]]
        support_users = sorted({str(memories[i]["user_id"]) for i in member_indices})
        accepted_support_users = sorted({
            str(memories[i]["user_id"]) for i in accepted_member_indices
        })

        examples = list(dict.fromkeys(
            memories[i]["cluster_text"] for i in member_indices
        ))[:12]

        keyword_counts: Counter = Counter()
        signature_counts: Counter = Counter()
        semantic_focus_counts: Counter = Counter()
        recommendation_examples: List[str] = []

        for i in member_indices:
            sig = get_trajectory_signature(
                memories[i]
            )
            if sig:
                signature_counts[sig] += 1

            for sf in memories[i].get(
                "semantic_focus",
                [],
            ):
                s = clean_text(sf).lower()
                if s:
                    semantic_focus_counts[s] += 1

            rb = clean_text(
                memories[i].get(
                    "recommendation_behavior"
                )
            )
            if rb and rb not in recommendation_examples:
                recommendation_examples.append(rb)

            kws = memories[i].get("keywords") or []
            if not isinstance(kws, list):
                kws = [str(kws)]
            for kw in kws:
                s = clean_text(kw).lower()
                if s:
                    keyword_counts[s] += 1

        meta = dict(cluster_meta[int(cluster_idx)])
        # Always summarize each learned cluster using its dominant FULL structured
        # profile. The hard constraint may only be mechanism-level.
        profiles = [behavior_bucket_key(memories[i]) for i in member_indices]
        majority_key, majority_count = Counter(profiles).most_common(1)[0]
        profile = {
            "mechanism": majority_key[0],
            "scope": majority_key[1],
            "direction": majority_key[2],
        }
        meta["majority_profile_fraction"] = float(majority_count / len(member_indices))

        canonical = abstract_behavior_text(profile)
        dominant_signature = (
            signature_counts.most_common(1)[0][0]
            if signature_counts else ""
        )
        # Free-text signatures almost never repeat verbatim, so the legacy
        # "most common string" is effectively the first member in file order.
        # Default: the signature of the medoid (member closest to the centroid).
        medoid_signature = get_trajectory_signature(medoid)
        if signature_mode == "medoid" and medoid_signature:
            state_signature = medoid_signature
        elif signature_mode in {"medoid", "mode"}:
            state_signature = dominant_signature
        else:
            raise ValueError(f"Unknown signature_mode={signature_mode!r}")

        state_rows.append({
            "state_id": sid,
            "cluster_id": int(cluster_idx),
            "cluster_mode": meta.get("cluster_mode"),
            "behavior_profile": profile,
            "behavior_bucket_key": meta.get("behavior_bucket_key"),
            "mechanism": profile["mechanism"],
            "scope": profile["scope"],
            "direction": profile["direction"],
            # IMPORTANT: abstract canonical text prevents content leakage from
            # the medoid example into downstream next-behavior prompts.
            "canonical_text": canonical,
            "canonical_behavior_signature": state_signature,
            "representative_signature": state_signature,
            "signature_mode": signature_mode,
            "mode_signature": dominant_signature,
            "mode_signature_count": int(signature_counts.most_common(1)[0][1]) if signature_counts else 0,
            "medoid_signature": medoid_signature,
            # Diagnostics only: never required in the ranking prompt.
            "representative_pattern_description": clean_text(
                medoid.get("pattern_description")
            ),
            "representative_behavior_explanation": clean_text(
                medoid.get("behavior_explanation")
            ),
            "medoid_memory_id": medoid["memory_id"],
            "cluster_text_examples": examples,
            "member_count": len(member_indices),
            "accepted_member_count": len(accepted_member_indices),
            "rejected_member_count": len(member_indices) - len(accepted_member_indices),
            "support_user_count": len(support_users),
            "accepted_support_user_count": len(accepted_support_users),
            "support_users": support_users,
            "accepted_support_users": accepted_support_users,
            "memory_ids": [memories[i]["memory_id"] for i in member_indices],
            "accepted_memory_ids": [
                memories[i]["memory_id"] for i in accepted_member_indices
            ],
            # Semantic fields below are DIAGNOSTICS ONLY.
            # They were NOT used to construct the cluster embedding/state.
            "semantic_evidence_is_state_identity": False,
            "top_semantic_focus": [
                k
                for k, _
                in semantic_focus_counts.most_common(20)
            ],
            "recommendation_behavior_examples": (
                recommendation_examples[:8]
            ),
            "top_keywords": [
                k
                for k, _
                in keyword_counts.most_common(12)
            ],
            "mean_similarity_to_centroid": float(np.mean(sims)),
            "min_similarity_to_centroid": float(np.min(sims)),
            "max_similarity_to_centroid": float(np.max(sims)),
            "cluster_metadata": meta,
        })

        centroid_rows.append(centroid.astype(np.float32))
        for i in accepted_member_indices:
            idx_to_state[i] = sid

    state_rows.sort(key=lambda x: int(x["cluster_id"]))
    centroid_rows = [
        centers[int(row["cluster_id"])].astype(np.float32)
        for row in state_rows
    ]

    return state_rows, idx_to_state, centroid_rows


def cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        if len(set(labels.tolist())) > 1:
            out["silhouette_cosine"] = float(
                silhouette_score(
                    embeddings,
                    labels,
                    metric="cosine",
                )
            )
            out["calinski_harabasz"] = float(
                calinski_harabasz_score(
                    embeddings,
                    labels,
                )
            )
            out["davies_bouldin"] = float(
                davies_bouldin_score(
                    embeddings,
                    labels,
                )
            )
    except Exception as e:
        out["metric_error"] = (
            f"{type(e).__name__}: {e}"
        )

    return out
