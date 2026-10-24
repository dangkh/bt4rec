from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bt4rec.common.io import load_json
from bt4rec.common.text import clean_text
from bt4rec.common.vectors import l2_normalize
from bt4rec.llm.encoder import QwenEncoder
from bt4rec.taxonomy import (
    TAXONOMY_VERSION,
    choose_cluster_text,
    normalize_direction,
    normalize_mechanism,
    normalize_scope,
)
from bt4rec.tree.assign import CompatibilityIndex, nearest_compatible_state


class ClusterStateVocabulary:
    """
    Frozen behavior-state vocabulary produced by bt4rec.cli.build_tree.
    """

    def __init__(
        self,
        states_json: str,
        state_embeddings_npy: str,
        embedding_model: Optional[str],
        embedding_device: str,
        embedding_batch_size: int,
        cluster_text_field: Optional[str],
        min_similarity: Optional[float],
        min_margin: Optional[float],
        respect_mapping_filter: bool,
    ) -> None:
        payload = load_json(states_json)

        if not (
            isinstance(payload, dict)
            and isinstance(
                payload.get("states"),
                list,
            )
        ):
            raise ValueError(
                "behavior_states.json must contain {metadata, states}"
            )

        self.metadata = payload.get(
            "metadata",
            {},
        )
        self.rows = [
            dict(x)
            for x in payload["states"]
        ]

        # Builder writes states sorted by cluster_id and centroid rows in that
        # same cluster order. Sort explicitly here for safety.
        self.rows.sort(
            key=lambda x: int(
                x.get(
                    "cluster_id",
                    str(
                        x.get(
                            "state_id",
                            "C0",
                        )
                    ).lstrip("C") or 0,
                )
            )
        )

        self.state_by_id = {
            str(r["state_id"]): r
            for r in self.rows
        }

        centroids = np.asarray(
            np.load(state_embeddings_npy),
            dtype=np.float32,
        )
        if len(centroids) != len(self.rows):
            raise ValueError(
                f"centroid/state mismatch: "
                f"{len(centroids)} != {len(self.rows)}"
            )
        self.centroids = l2_normalize(
            centroids
        )

        self.embedding_model_name = str(
            embedding_model
            or self.metadata.get(
                "embedding_model"
            )
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.cluster_text_field = str(
            cluster_text_field
            or self.metadata.get(
                "cluster_text_field"
            )
            or "structured_combined"
        )

        quality = (
            self.metadata
            .get("cluster_quality", {})
            .get("assignment_quality", {})
        )

        self.min_similarity = float(
            min_similarity
            if min_similarity is not None
            else quality.get(
                "min_cluster_similarity",
                0.55,
            )
        )
        self.min_margin = float(
            min_margin
            if min_margin is not None
            else quality.get(
                "min_cluster_margin",
                0.02,
            )
        )
        self.respect_mapping_filter = bool(
            respect_mapping_filter
        )

        self.constraint_level = str(
            self.metadata.get(
                "constraint_level",
                "mechanism",
            )
        )

        # Same compatibility rule as the KMeans buckets (bt4rec.tree.assign).
        self.cluster_mode = str(self.metadata.get("cluster_mode", "constrained"))
        self.compat = CompatibilityIndex(self.rows, self.cluster_mode, self.constraint_level)
        built_taxonomy = self.metadata.get("taxonomy_version")
        if built_taxonomy != TAXONOMY_VERSION:
            print(
                f"[WARN] Tree states were built with taxonomy_version={built_taxonomy!r}; "
                f"inference uses {TAXONOMY_VERSION!r}. Labels containing punctuation "
                "(e.g. 'same collection/series') may embed differently. Rebuild the Tree "
                "with bt4rec.cli.build_tree for exact train/test agreement.",
                flush=True,
            )

        self.encoder = QwenEncoder(
            self.embedding_model_name,
            embedding_device,
            embedding_batch_size,
        )

    def _trajectory_text(
        self,
        behavior: Dict[str, Any],
    ) -> str:
        """Identical to the builder: bt4rec.taxonomy.choose_cluster_text."""
        text = choose_cluster_text(behavior, self.cluster_text_field)
        if not text:
            raise ValueError("Empty trajectory text for state mapping")
        return text

    def state_evidence(
        self,
        sid: str,
    ) -> Dict[str, Any]:
        """
        HOW block.

        Never expose representative_pattern_description, top_semantic_focus, or
        recommendation_behavior_examples from other cluster members.
        """
        r = self.state_by_id.get(
            str(sid),
            {},
        )

        trajectory = clean_text(
            r.get(
                "canonical_behavior_signature",
                r.get(
                    "representative_signature",
                    "",
                ),
            )
        )
        if not trajectory:
            trajectory = clean_text(
                r.get("canonical_text")
            )

        return {
            "state_id": str(sid),
            "mechanism": normalize_mechanism(
                r.get("mechanism")
            ),
            "scope": normalize_scope(
                r.get("scope")
            ),
            "trajectory_signature": trajectory,
            # Audit only. Not required by the final ranking prompt.
            "direction": normalize_direction(
                r.get("direction")
            ),
        }

    def map_users(
        self,
        behaviors_by_user: Dict[
            str,
            List[Dict[str, Any]],
        ],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Encode all test behaviors in one pass, then map each behavior to the
        nearest compatible frozen centroid.
        """
        flat: List[Tuple[str, int, Dict[str, Any]]] = []
        texts: List[str] = []

        for uid, behaviors in behaviors_by_user.items():
            for i, b in enumerate(behaviors):
                flat.append(
                    (uid, i, b)
                )
                texts.append(
                    self._trajectory_text(b)
                )

        embeddings = self.encoder.encode(
            texts
        )
        all_scores = (
            embeddings
            @ self.centroids.T
        )

        out: Dict[
            str,
            List[Dict[str, Any]],
        ] = defaultdict(list)


        for row_idx, (
            uid,
            behavior_index,
            behavior,
        ) in enumerate(flat):
            mechanism = normalize_mechanism(
                behavior.get("mechanism")
            )
            candidate_indices, compatible = self.compat.candidates(behavior)
            retrieval_scope = (
                "all_centroids_global_mode"
                if self.cluster_mode == "global"
                else "same_mechanism_centroids"
                if compatible
                else "global_centroid_fallback"
            )
            res = nearest_compatible_state(
                all_scores[row_idx],
                candidate_indices,
                self.min_similarity,
                self.min_margin,
            )
            best_idx, best_sim = res.best_index, res.best_similarity
            second_idx, second_sim, margin = res.second_index, res.second_similarity, res.margin
            reasons: List[str] = []
            if not compatible:
                reasons.append("missing_compatible_mechanism_bucket")
            reasons.extend(res.reasons)
            accepted = not reasons

            if not self.respect_mapping_filter:
                accepted = True
                reasons = []

            sid = str(
                self.rows[
                    best_idx
                ]["state_id"]
            )

            out[uid].append({
                "behavior_index": int(
                    behavior_index
                ),
                "window_index": int(
                    behavior.get(
                        "window_index",
                        behavior_index,
                    )
                ),
                "state_id": sid,
                "state_evidence": (
                    self.state_evidence(sid)
                ),
                "behavior_mechanism": (
                    mechanism
                ),
                "behavior_scope": (
                    normalize_scope(
                        behavior.get(
                            "scope"
                        )
                    )
                ),
                "behavior_direction": (
                    normalize_direction(
                        behavior.get(
                            "direction"
                        )
                    )
                ),
                "behavior_trajectory_signature": (
                    clean_text(
                        behavior.get(
                            "trajectory_signature",
                            behavior.get(
                                "behavior_signature",
                                "",
                            ),
                        )
                    )
                ),
                "cluster_text": texts[
                    row_idx
                ],
                "retrieval_scope": (
                    retrieval_scope
                ),
                "best_similarity": (
                    best_sim
                ),
                "second_similarity": (
                    second_sim
                ),
                "similarity_margin": (
                    margin
                ),
                "num_compatible_states": len(
                    candidate_indices
                ),
                "assignment_accepted": bool(
                    accepted
                ),
                "rejection_reason": (
                    "+".join(reasons)
                    if reasons
                    else None
                ),
                "second_state_id": (
                    str(
                        self.rows[
                            second_idx
                        ]["state_id"]
                    )
                    if second_idx
                    is not None
                    else None
                ),
            })

        # Preserve chronological order.
        for uid in out:
            out[uid].sort(
                key=lambda x: (
                    int(
                        x.get(
                            "window_index",
                            0,
                        )
                    ),
                    int(
                        x.get(
                            "behavior_index",
                            0,
                        )
                    ),
                )
            )

        return dict(out)

    def release_encoder(self) -> None:
        self.encoder.release()
