from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from bt4rec.common.io import load_json
from bt4rec.tree.vocab import ClusterStateVocabulary


def latest_accepted_state_suffix(
    mappings: List[Dict[str, Any]],
    collapse_consecutive: bool = False,
) -> List[str]:
    """
    Reproduce the Tree training semantics:
    - a rejected/ambiguous mapping breaks the sequence;
    - if the Tree was built with --collapse-consecutive-states, repeated
      consecutive states are merged exactly as in build_user_sequences.
    """
    suffix: List[str] = []
    for m in mappings:
        if m.get("assignment_accepted"):
            sid = str(m["state_id"])
            if collapse_consecutive and suffix and suffix[-1] == sid:
                continue
            suffix.append(sid)
        else:
            suffix = []
    return suffix


def _detect_collapse(payload: Dict[str, Any], tree_json: str) -> bool:
    """Read --collapse-consecutive-states from the Tree metadata.

    Trees built by the legacy builder do not store it in behavior_tree.json, only
    in user_behavior_sequences.json next to it, so fall back to that file.
    """
    meta = payload.get("metadata", {})
    if "collapse_consecutive_states" in meta:
        return bool(meta["collapse_consecutive_states"])
    seq_path = Path(tree_json).with_name("user_behavior_sequences.json")
    if seq_path.is_file():
        seq_meta = load_json(seq_path).get("metadata", {})
        if "collapse_consecutive_states" in seq_meta:
            return bool(seq_meta["collapse_consecutive_states"])
    print(
        "[WARN] Tree metadata does not record collapse_consecutive_states; assuming False.",
        flush=True,
    )
    return False


class ReverseBehaviorTree:
    def __init__(
        self,
        tree_json: str,
        vocab: ClusterStateVocabulary,
    ) -> None:
        payload = load_json(tree_json)
        self.payload = payload
        self.nodes: Dict[
            str,
            Dict[str, Any],
        ] = payload["nodes"]
        self.root_id = str(
            payload.get(
                "root_id",
                "ROOT",
            )
        )
        self.vocab = vocab
        self.collapse_consecutive_states = _detect_collapse(payload, tree_json)
        self.max_order = int(
            payload.get(
                "metadata",
                {},
            ).get(
                "max_order",
                5,
            )
        )

        if self.root_id not in self.nodes:
            raise ValueError(
                f"Tree root {self.root_id!r} missing"
            )

    def query(
        self,
        state_sequence: List[str],
        top_next: int,
        max_query_order: int = 0,
    ) -> Dict[str, Any]:
        """
        Query the deepest ACTIVE reverse-suffix context up to a requested order.

        max_query_order:
          0 -> use the Tree's configured max_order (original/full behavior)
          K > 0 -> use at most order K, while preserving active-node backoff
                   K -> ... -> order 1 -> ROOT.
        """
        requested_max_query_order = int(max_query_order)
        if requested_max_query_order < 0:
            raise ValueError("max_query_order must be >= 0")

        effective_max_query_order = (
            self.max_order
            if requested_max_query_order == 0
            else min(
                self.max_order,
                requested_max_query_order,
            )
        )

        recent = list(
            state_sequence[
                -effective_max_query_order:
            ]
        )

        cur = self.root_id
        deepest_structural = (
            self.root_id
        )
        deepest_active = (
            self.root_id
            if self.nodes[
                self.root_id
            ].get(
                "active_predictive",
                True,
            )
            else None
        )

        traversed_reverse: List[str] = []
        stop_reason = "history_exhausted"

        for token in reversed(recent):
            children = (
                self.nodes[cur]
                .get(
                    "children",
                    {},
                )
            )
            if token not in children:
                stop_reason = (
                    "missing_reverse_edge"
                )
                break

            cur = str(
                children[token]
            )
            deepest_structural = cur
            traversed_reverse.append(
                token
            )

            if self.nodes[
                cur
            ].get(
                "active_predictive",
                False,
            ):
                deepest_active = cur

        if deepest_active is None:
            deepest_active = (
                self.root_id
            )

        structural_node = self.nodes[
            deepest_structural
        ]
        active_node = self.nodes[
            deepest_active
        ]

        dist = (
            active_node.get(
                "smoothed_next_distribution"
            )
            or active_node.get(
                "raw_next_distribution"
            )
            or []
        )

        preds = []
        for x in dist[
            :max(
                0,
                int(top_next),
            )
        ]:
            sid = str(
                x.get("state_id")
            )
            ev = (
                self.vocab
                .state_evidence(sid)
            )
            preds.append({
                **ev,
                "probability": float(
                    x.get(
                        "probability",
                        0.0,
                    )
                ),
                "support_user_count": int(
                    x.get(
                        "support_user_count",
                        0,
                    )
                ),
                "occurrence_count": int(
                    x.get(
                        "occurrence_count",
                        0,
                    )
                ),
            })

        return {
            "query_state_sequence": (
                state_sequence
            ),
            "tree_configured_max_order": int(
                self.max_order
            ),
            "requested_max_query_order": int(
                requested_max_query_order
            ),
            "effective_max_query_order": int(
                effective_max_query_order
            ),
            "query_recent_suffix": (
                recent
            ),
            "reverse_tokens_traversed": (
                traversed_reverse
            ),
            "stop_reason": (
                stop_reason
            ),
            "deepest_structural_node_id": (
                deepest_structural
            ),
            "deepest_structural_order": int(
                structural_node.get(
                    "context_order",
                    0,
                )
            ),
            "matched_node_id": (
                deepest_active
            ),
            "matched_context_state_ids": (
                active_node.get(
                    "context_states",
                    [],
                )
            ),
            "matched_order": int(
                active_node.get(
                    "context_order",
                    0,
                )
            ),
            "matched_support_users": int(
                active_node.get(
                    "support_user_count",
                    0,
                )
            ),
            "matched_support_occurrences": int(
                active_node.get(
                    "support_occurrence_count",
                    0,
                )
            ),
            "activation_reason": (
                active_node.get(
                    "activation_reason",
                    "",
                )
            ),
            "smoothing_lambda": float(
                active_node.get(
                    "smoothing_lambda",
                    1.0,
                )
            ),
            "predicted_next_states": (
                preds
            ),
        }


def build_tree_rank_evidence(
    latest_behavior: Dict[str, Any],
    tree_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Only target-user semantics + abstract collaborative transition.
    """
    predicted = []

    for rank, p in enumerate(
        tree_result.get(
            "predicted_next_states",
            [],
        ),
        start=1,
    ):
        predicted.append({
            "rank": rank,
            "state_id": p.get(
                "state_id"
            ),
            "probability": round(
                float(
                    p.get(
                        "probability",
                        0.0,
                    )
                ),
                6,
            ),
            "mechanism": p.get(
                "mechanism"
            ),
            "scope": p.get(
                "scope"
            ),
            "trajectory_signature": (
                p.get(
                    "trajectory_signature"
                )
            ),
            "support_user_count": int(
                p.get(
                    "support_user_count",
                    0,
                )
            ),
        })

    return {
        "latest_observed_behavior": {
            "recommendation_behavior": (
                latest_behavior.get(
                    "recommendation_behavior",
                    "",
                )
            ),
            "semantic_focus": (
                latest_behavior.get(
                    "semantic_focus",
                    [],
                )
            ),
        },
        "predicted_next_transition": (
            predicted
        ),
        "tree_context": {
            "matched_order": int(
                tree_result.get(
                    "matched_order",
                    0,
                )
            ),
            "support_users": int(
                tree_result.get(
                    "matched_support_users",
                    0,
                )
            ),
        },
    }
