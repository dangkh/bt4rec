from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from bt4rec.tree.stats import (
    entropy_bits,
    interpolate_dist,
    js_divergence_bits,
    normalize_dist,
    stable_context_id,
)


def build_user_sequences(
    memories: List[Dict[str, Any]],
    idx_to_state: Dict[int, str],
    collapse_consecutive: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Build accepted state sequences.

    CRITICAL:
    An ambiguous/rejected memory BREAKS the temporal segment.
    We do NOT delete it and connect the states before/after it, because that
    would fabricate a transition that was never observed.
    """
    by_user: Dict[
        str,
        List[Tuple[int, Dict[str, Any]]],
    ] = defaultdict(list)

    for idx, m in enumerate(memories):
        by_user[str(m["user_id"])].append(
            (idx, m)
        )

    out: Dict[str, Dict[str, Any]] = {}

    for uid, items in by_user.items():
        items.sort(
            key=lambda x: (
                int(x[1]["window_index"]),
                int(x[1]["_row_idx"]),
            )
        )

        segments: List[Dict[str, Any]] = []
        current_entries: List[Dict[str, Any]] = []
        skipped_memory_ids: List[str] = []
        last_sid: Optional[str] = None

        def flush_segment() -> None:
            nonlocal current_entries, last_sid
            if current_entries:
                segments.append({
                    "states": [
                        e["state_id"]
                        for e in current_entries
                    ],
                    "entries": current_entries,
                    "length": len(current_entries),
                })
            current_entries = []
            last_sid = None

        for idx, m in items:
            sid = idx_to_state.get(idx)

            if sid is None:
                skipped_memory_ids.append(
                    m["memory_id"]
                )
                flush_segment()
                continue

            if (
                collapse_consecutive
                and sid == last_sid
                and current_entries
            ):
                current_entries[-1][
                    "memory_ids"
                ].append(m["memory_id"])
                current_entries[-1][
                    "window_indices"
                ].append(
                    int(m["window_index"])
                )
                continue

            current_entries.append({
                "state_id": sid,
                "memory_ids": [
                    m["memory_id"]
                ],
                "window_indices": [
                    int(m["window_index"])
                ],
            })
            last_sid = sid

        flush_segment()

        # Compatibility view only. NEVER use this flattened list for transition
        # counting because it can cross segment boundaries.
        flat_entries = [
            e
            for seg in segments
            for e in seg["entries"]
        ]

        out[uid] = {
            "user_id": uid,
            "length": len(flat_entries),
            "states": [
                e["state_id"]
                for e in flat_entries
            ],
            "entries": flat_entries,
            "segments": segments,
            "num_segments": len(segments),
            "num_skipped_ambiguous_memories": len(
                skipped_memory_ids
            ),
            "skipped_ambiguous_memory_ids": (
                skipped_memory_ids
            ),
        }

    return out


@dataclass
class ContextStats:
    occurrence_counts: Counter = field(default_factory=Counter)
    # user_target_counts[user][target] = number occurrences for this user/context/target
    user_target_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    support_users: Set[str] = field(default_factory=set)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def add(
        self,
        uid: str,
        target: str,
        evidence: Dict[str, Any],
        max_evidence: int,
        rng: random.Random,
    ) -> None:
        self.occurrence_counts[target] += 1
        self.user_target_counts[uid][target] += 1
        self.support_users.add(uid)
        self.seen += 1

        # bounded unbiased evidence sample
        if max_evidence <= 0:
            return
        if len(self.evidence) < max_evidence:
            self.evidence.append(evidence)
        else:
            j = rng.randint(0, self.seen - 1)
            if j < max_evidence:
                self.evidence[j] = evidence


def suffix_parent(context: Tuple[str, ...]) -> Tuple[str, ...]:
    """Chronological [A,B,C] -> [B,C] -> [C] -> ROOT."""
    if len(context) <= 1:
        return ()
    return context[1:]


def accumulate_contexts(
    user_sequences: Dict[str, Dict[str, Any]],
    max_order: int,
    max_evidence: int,
    seed: int,
) -> Dict[Tuple[str, ...], ContextStats]:
    rng = random.Random(seed)
    stats: Dict[
        Tuple[str, ...],
        ContextStats,
    ] = defaultdict(ContextStats)

    for uid, rec in user_sequences.items():
        segments = rec.get("segments")

        if segments is None:
            # Backward compatibility for older sequence payloads.
            segments = [{
                "states": rec.get("states", []),
                "entries": rec.get("entries", []),
            }]

        for segment_index, seg in enumerate(
            segments
        ):
            seq = seg["states"]
            entries = seg["entries"]

            if len(seq) < 2:
                continue

            for t in range(1, len(seq)):
                target = seq[t]

                root_ev = {
                    "user_id": uid,
                    "segment_index": segment_index,
                    "context_states": [],
                    "reverse_tree_path": [],
                    "target_state": target,
                    "target_memory_ids": (
                        entries[t]["memory_ids"]
                    ),
                    "target_window_indices": (
                        entries[t]["window_indices"]
                    ),
                }
                stats[()].add(
                    uid,
                    target,
                    root_ev,
                    max_evidence,
                    rng,
                )

                for L in range(
                    1,
                    min(max_order, t) + 1,
                ):
                    context = tuple(
                        seq[t-L:t]
                    )
                    ev = {
                        "user_id": uid,
                        "segment_index": segment_index,
                        "context_states": list(
                            context
                        ),
                        "reverse_tree_path": list(
                            reversed(context)
                        ),
                        "context_memory_ids": [
                            entries[j][
                                "memory_ids"
                            ]
                            for j in range(
                                t-L,
                                t,
                            )
                        ],
                        "context_window_indices": [
                            entries[j][
                                "window_indices"
                            ]
                            for j in range(
                                t-L,
                                t,
                            )
                        ],
                        "target_state": target,
                        "target_memory_ids": (
                            entries[t][
                                "memory_ids"
                            ]
                        ),
                        "target_window_indices": (
                            entries[t][
                                "window_indices"
                            ]
                        ),
                    }
                    stats[context].add(
                        uid,
                        target,
                        ev,
                        max_evidence,
                        rng,
                    )

    return stats


def collaborative_weights(acc: ContextStats, count_mode: str) -> Dict[str, float]:
    if count_mode == "occurrence":
        return {str(k): float(v) for k, v in acc.occurrence_counts.items()}

    if count_mode == "distinct_user":
        # Each user can vote at most once per target. A user may vote for multiple targets.
        target_users: Dict[str, Set[str]] = defaultdict(set)
        for uid, counter in acc.user_target_counts.items():
            for target in counter:
                target_users[target].add(uid)
        return {target: float(len(users)) for target, users in target_users.items()}

    if count_mode == "user_normalized":
        # Each user contributes total mass exactly 1 for a given context.
        weights: Dict[str, float] = defaultdict(float)
        for uid, counter in acc.user_target_counts.items():
            total = float(sum(counter.values()))
            if total <= 0:
                continue
            for target, cnt in counter.items():
                weights[target] += float(cnt) / total
        return dict(weights)

    raise ValueError(f"Unknown count_mode={count_mode}")


def build_context_records(
    stats: Dict[Tuple[str, ...], ContextStats],
    state_rows: List[Dict[str, Any]],
    count_mode: str,
    smoothing_kappa: float,
    min_support_users: int,
    min_support_occurrences: int,
    min_jsd: float,
    promote_order1: bool,
    top_next: int,
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    if () not in stats:
        raise ValueError("No transitions found; ROOT distribution unavailable")

    state_text = {x["state_id"]: x["canonical_text"] for x in state_rows}
    state_meta = {
        x["state_id"]: {
            "behavior_profile": x.get("behavior_profile"),
            "mechanism": x.get("mechanism"),
            "scope": x.get("scope"),
            "direction": x.get("direction"),
        }
        for x in state_rows
    }
    records: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for ctx in sorted(stats.keys(), key=lambda c: (len(c), c)):
        acc = stats[ctx]
        weights = collaborative_weights(acc, count_mode)
        raw_dist = normalize_dist(weights)
        support_users = len(acc.support_users)
        occurrences = int(sum(acc.occurrence_counts.values()))

        if ctx == ():
            parent_ctx = None
            lam = 1.0
            smooth_dist = raw_dist
            parent_raw = {}
            jsd_raw = 0.0
            eligible = True
            active = True
            reason = "root"
        else:
            parent_ctx = suffix_parent(ctx)
            parent = records[parent_ctx]
            parent_raw = parent["raw_distribution_dict"]
            parent_smooth = parent["smoothed_distribution_dict"]
            lam = float(support_users / (support_users + smoothing_kappa)) if smoothing_kappa > 0 else 1.0
            smooth_dist = interpolate_dist(raw_dist, parent_smooth, lam)
            jsd_raw = js_divergence_bits(raw_dist, parent_raw)
            eligible = (
                support_users >= min_support_users
                and occurrences >= min_support_occurrences
            )
            if not eligible:
                active = False
                reason = "insufficient_support"
            elif len(ctx) == 1 and promote_order1:
                active = True
                reason = "order1_base_context"
            else:
                active = bool(jsd_raw >= min_jsd)
                reason = "predictive_gain" if active else "low_predictive_gain"

        def dist_rows(dist: Dict[str, float]) -> List[Dict[str, Any]]:
            items = sorted(dist.items(), key=lambda x: (-x[1], x[0]))
            if top_next > 0:
                items = items[:top_next]
            return [
                {
                    "state_id": sid,
                    "probability": float(p),
                    "state_text": state_text.get(sid, ""),
                    "behavior_profile": state_meta.get(sid, {}).get("behavior_profile"),
                    "mechanism": state_meta.get(sid, {}).get("mechanism"),
                    "scope": state_meta.get(sid, {}).get("scope"),
                    "direction": state_meta.get(sid, {}).get("direction"),
                    "occurrence_count": int(acc.occurrence_counts.get(sid, 0)),
                    "support_user_count": int(sum(1 for u in acc.user_target_counts if sid in acc.user_target_counts[u])),
                }
                for sid, p in items
            ]

        records[ctx] = {
            "node_id": stable_context_id(ctx),
            "context_states": list(ctx),                    # chronological
            "reverse_tree_path": list(reversed(ctx)),       # actual traversal path
            "context_order": len(ctx),
            "suffix_parent_context": list(parent_ctx) if parent_ctx is not None else None,
            "suffix_parent_id": stable_context_id(parent_ctx) if parent_ctx is not None else None,
            "support_user_count": support_users,
            "support_occurrence_count": occurrences,
            "occurrence_next_counts": {str(k): int(v) for k, v in acc.occurrence_counts.items()},
            "collaborative_next_weights": {str(k): float(v) for k, v in weights.items()},
            "raw_distribution_dict": raw_dist,
            "smoothed_distribution_dict": smooth_dist,
            "raw_next_distribution": dist_rows(raw_dist),
            "smoothed_next_distribution": dist_rows(smooth_dist),
            "raw_entropy_bits": entropy_bits(raw_dist),
            "smoothed_entropy_bits": entropy_bits(smooth_dist),
            "smoothing_lambda": lam,
            "jsd_raw_to_suffix_parent_bits": jsd_raw,
            "eligible_support": bool(eligible),
            "active_predictive": bool(active),
            "activation_reason": reason,
            "evidence": acc.evidence,
        }

    # nearest active suffix for each node
    for ctx, rec in records.items():
        if ctx == ():
            rec["active_backoff_context"] = None
            rec["active_backoff_id"] = None
            continue
        p = suffix_parent(ctx)
        while p != () and not records[p]["active_predictive"]:
            p = suffix_parent(p)
        # ROOT is always active
        rec["active_backoff_context"] = list(p)
        rec["active_backoff_id"] = stable_context_id(p)

    return records


def build_reverse_tree(records: Dict[Tuple[str, ...], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build ONE physical reverse tree.

    For chronological context [A,B,C], reverse_tree_path = [C,B,A].
    Therefore the node is reached as ROOT -> C -> B -> A.

    IMPORTANT: structural nodes are retained even if active_predictive=False.
    This ensures a deeper path is traversable and inference can track the deepest
    active node encountered along that path.
    """
    nodes: Dict[str, Dict[str, Any]] = {}

    for ctx in sorted(records.keys(), key=lambda c: (len(c), c)):
        rec = dict(records[ctx])
        rec["children"] = {}  # token -> child node id
        # Internal dict copies not needed in public JSON twice.
        rec.pop("raw_distribution_dict", None)
        rec.pop("smoothed_distribution_dict", None)
        nodes[rec["node_id"]] = rec

    # Every context [A,B,C] has a structural reverse parent [B,C].
    # In reverse traversal, [B,C] path is C->B, and child token A reaches ABC.
    for ctx in sorted(records.keys(), key=lambda c: (len(c), c)):
        if ctx == ():
            continue
        node_id = stable_context_id(ctx)
        if len(ctx) == 1:
            parent_ctx = ()
            edge_token = ctx[-1]       # ROOT -> C
        else:
            parent_ctx = suffix_parent(ctx)
            edge_token = ctx[0]        # BC -> ABC adds A at the reverse-tree end
        parent_id = stable_context_id(parent_ctx)
        nodes[parent_id]["children"][edge_token] = node_id
        nodes[node_id]["structural_parent_id"] = parent_id
        nodes[node_id]["incoming_reverse_token"] = edge_token

    nodes["ROOT"]["structural_parent_id"] = None
    nodes["ROOT"]["incoming_reverse_token"] = None

    return {
        "root_id": "ROOT",
        "nodes": nodes,
        "num_structural_nodes": len(nodes),
        "num_active_predictive_nodes": sum(1 for n in nodes.values() if n["active_predictive"]),
    }


def validate_tree_structure(tree: Dict[str, Any], records: Dict[Tuple[str, ...], Dict[str, Any]]) -> None:
    nodes = tree["nodes"]
    for ctx in records:
        if ctx == ():
            continue
        cur = "ROOT"
        for token in reversed(ctx):
            children = nodes[cur]["children"]
            if token not in children:
                raise AssertionError(
                    f"Broken reverse path for chronological context {ctx}; "
                    f"at node {cur}, missing token {token}"
                )
            cur = children[token]
        expected = stable_context_id(ctx)
        if cur != expected:
            raise AssertionError(f"Reverse traversal reached {cur}, expected {expected} for {ctx}")
