#!/usr/bin/env python3
"""
Inference for the dual-view collaborative behavior Tree.

Core idea
---------
The dual-view precompute intentionally separates:

    WHAT the target user currently prefers
        = latest recommendation_behavior + semantic_focus

from:

    HOW that preference is expected to evolve next
        = collaborative Tree predicted abstract state
          (mechanism + scope + trajectory_signature)

The final ranking LLM receives BOTH directly. There is NO extra LLM behavior
composer, so the method uses only one ranking LLM call per condition.

Pipeline
--------
precomputed test behaviors
    -> trajectory-only mapping to frozen KMeans states
    -> query reverse suffix Tree
    -> latest semantic behavior (WHAT)
       + predicted abstract transition/state (HOW)
    -> candidate ranking

Controlled ranking conditions
-----------------------------
1) Native:
      history + candidates

2) Latest-only:
      history + latest observed semantic behavior + candidates

3) Recent-3 (additional baseline):
      history + last 3 observed semantic behaviors + candidates
      (no predicted Tree transition)

4) Tree (main):
      history + latest observed semantic behavior
      + Tree predicted next abstract state + candidates

All conditions use the SAME frozen candidates and list ranking/output protocol.
The original Native, Latest-only, and Tree prompts are unchanged.

Important implementation details
--------------------------------
- No behavior generation at inference time.
- No GT/test item is used to construct behavior evidence.
- Candidate order is loaded EXACTLY from --candidate-file and is never reshuffled
  at inference time. This keeps Tree / Latest-only / Native aligned with the same
  frozen candidate protocol used by the compared rankers.
- Test behavior -> state mapping uses the SAME structured trajectory text used
  by build_reverse_behavior_tree_dual.py.
- When the Tree was trained with mechanism-constrained KMeans, test mapping is
  restricted to states with the same mechanism whenever compatible states exist.
- Ambiguous/rejected test mappings BREAK the recent state suffix by default,
  matching the segmented transition construction used by the current Tree builder.
- Semantic fields from other users / cluster medoids are NEVER injected into the
  ranking prompt. The predicted state contributes abstract transition information only.
- Incomplete rankings are NOT retried. They are sanitized like the native baseline.
  Summary still reports parse-complete quality for audit.

Expected inputs
---------------
Dual test behavior cache:
    precompute_test_behaviors_gemma_dual_v2.py output

Tree directory:
    build_reverse_behavior_tree_dual.py output containing:
      behavior_states.json
      behavior_state_embeddings.npy
      behavior_tree.json

Example
-------
python inference_tree_latest_behavior_gemma.py \
  --items data/CDs/items.json \
  --sequences data/CDs/user_sequences_10_5000.json \
  --negatives data/CDs/user_negatives_10_5000.json \
  --candidate-file data/CDs/user_sequences_candidates_seed42.json \
  --precomputed-behaviors precomputed/CDs/test_user_behaviors_gemma_dual_v2_300.jsonl \
  --tree-dir behavior_tree_out_dual_v2_k50 \
  --output results/tree_latest_behavior_cd_300.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-users 300 \
  --top-next 1 \
  --tree-max-query-order 0 \
  --history-size 10 \
  --llm-batch-size 8 \
  --run-native \
  --run-latest-only \
  --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Initialize Unsloth before any library that may import transformers/peft
# (e.g. sentence_transformers used by the Qwen encoder).
# This applies patches only; it does NOT load the Qwen model yet.
# Unsloth is loaded lazily before Qwen/SentenceTransformer.
# This lets --validate-only run without an accelerator or Unsloth.
import torch
from tqdm.auto import tqdm


SCHEMA_VERSION = "bt4rec_controlled_behavior_source_ablation_v1"
RECENT3_PROMPT_VERSION = "recent3_semantic_qwen_clabel_json_list_v1"
RANK_PROMPT_VERSION = "bt4rec_controlled_additional_how_list_v1"


# =============================================================================
# Generic helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def normalize_label(x: Any) -> str:
    # Mirror the dual Tree builder: punctuation is not semantic.
    s = clean_text(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s).strip() or "unknown"


def normalize_mechanism(x: Any) -> str:
    s = normalize_label(x)
    aliases = {
        "switching": "shifting",
        "shift": "shifting",
        "cross category exploration": "cross category exploration",
        "collection completion": "collection expansion",
        "preference deepening": "deepening",
        "preference refinement": "refinement",
        "repeat": "repetition",
    }
    return aliases.get(s, s)


def normalize_scope(x: Any) -> str:
    s = normalize_label(x)
    aliases = {
        "same artist": "same creator",
        "same artists": "same creator",
        "same collection series": "same series",
        "same collection": "same series",
        "same series collection": "same series",
        "same broad category": "same category",
        "same broad categories": "same category",
        "same categories": "same category",
        "related category": "adjacent category",
    }
    return aliases.get(s, s)


def normalize_direction(x: Any) -> str:
    s = normalize_label(x)
    aliases = {
        "switch": "shift",
        "shifting": "shift",
        "deepening": "deepen",
        "broadening": "broaden",
        "narrowing": "narrow",
        "repetition": "repeat",
        "returning": "return",
    }
    return aliases.get(s, s)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


def extract_json_value(text: str) -> Any:
    """Recover the first complete JSON object/array from model output."""
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        pos = s.find(opener)
        if pos >= 0:
            starts.append((pos, opener, closer))

    if not starts:
        raise ValueError("No JSON value found")

    start, opener, closer = min(starts, key=lambda z: z[0])
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1], strict=False)

    raise ValueError("Unbalanced JSON")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue
            if isinstance(x, dict):
                rows.append(x)
    return rows


def processed_users_from_jsonl(path: str | Path) -> set[str]:
    out: set[str] = set()
    for row in read_jsonl(path):
        if (
            row.get("schema_version") == SCHEMA_VERSION
            and row.get("rank_prompt_version") == RANK_PROMPT_VERSION
            and row.get("user_id") is not None
        ):
            out.add(str(row["user_id"]))
    return out


def load_user_ids_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            if isinstance(obj.get("users"), list):
                return [str(x) for x in obj["users"]]
            return [str(x) for x in obj.keys()]
    except Exception:
        pass
    return [x.strip() for x in text.splitlines() if x.strip()]


# =============================================================================
# Data / items / candidates
# =============================================================================

def normalize_items(items: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in items.items()}


def get_item_info(
    item_id: Any,
    items_meta: Dict[str, Any],
) -> Dict[str, str]:
    iid = str(item_id)
    info = items_meta.get(iid, {})

    category = (
        info.get("main_cat")
        or info.get("category")
        or info.get("categories")
        or "Unknown"
    )
    if isinstance(category, list):
        if category and isinstance(category[0], list):
            category = " > ".join(
                str(v)
                for v in category[0][:5]
            )
        else:
            category = " > ".join(
                str(v)
                for v in category[:5]
            )

    title = (
        info.get("title")
        or info.get("name")
        or f"Item {iid}"
    )

    return {
        "item_id": iid,
        "title": str(title),
        "category": str(category),
    }


def load_candidate_file(
    path: Optional[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".jsonl":
        rows: Dict[str, Dict[str, Any]] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = str(
                    row.get("user_id")
                    or row.get("uid")
                )
                rows[uid] = row
        return rows

    obj = load_json(p)
    if isinstance(obj, list):
        return {
            str(
                row.get("user_id")
                or row.get("uid")
            ): row
            for row in obj
            if isinstance(row, dict)
        }
    if isinstance(obj, dict):
        # Could already be user_id -> row.
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out

    raise ValueError(
        f"Unsupported candidate file format: {path}"
    )


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    return [str(v) for v in x]


def get_candidates_for_user(
    uid: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    candidate_rows: Optional[Dict[str, Dict[str, Any]]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Load the frozen candidate order exactly as stored in --candidate-file.

    ``negative_data`` and ``seed`` are kept in the signature for compatibility
    with the previous inference pipeline, but they are deliberately NOT used to
    rebuild or reshuffle candidates here.
    """
    if candidate_rows is None:
        raise ValueError(
            "--candidate-file is required for the frozen ranking protocol"
        )
    if uid not in candidate_rows:
        raise ValueError(
            f"user={uid}: missing row in --candidate-file"
        )

    row = candidate_rows[uid]
    candidates = as_str_list(
        row.get(
            "candidates",
            row.get(
                "candidate_item_ids",
                [],
            ),
        )
    )
    targets = as_str_list(
        row.get(
            "target",
            row.get(
                "ground_truth",
                row.get(
                    "ground_truth_item_ids",
                    user_data.get("test", []),
                ),
            ),
        )
    )

    if not candidates:
        raise ValueError(
            f"user={uid}: empty candidate file row"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError(
            f"user={uid}: candidate file contains duplicate item IDs"
        )
    if not targets:
        raise ValueError(
            f"user={uid}: no ground-truth item available"
        )

    for target in targets:
        if candidates.count(str(target)) != 1:
            raise ValueError(
                f"user={uid}: target={target} must occur exactly once "
                "in the frozen candidate list"
            )

    return candidates, targets


# =============================================================================
# Dual behavior cache
# =============================================================================

def load_precomputed_behaviors(
    path: str,
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("precompute_ok") is not True:
                continue

            uid = str(row["user_id"])
            behaviors = row.get(
                "generated_behaviors",
                [],
            )
            if not isinstance(behaviors, list):
                continue

            # Ensure chronological order.
            behaviors = sorted(
                [
                    dict(x)
                    for x in behaviors
                    if isinstance(x, dict)
                ],
                key=lambda x: int(
                    x.get("window_index", 0)
                ),
            )

            if not behaviors:
                continue

            row = dict(row)
            row["generated_behaviors"] = behaviors
            rows[uid] = row

    if not rows:
        raise ValueError(
            f"No successful users in behavior cache: {path}"
        )

    return rows


def validate_behavior_cache_history(
    uid: str,
    cache_row: Dict[str, Any],
    user_data: Dict[str, Any],
    max_train_interactions: int,
    strict: bool,
) -> None:
    """
    Validate that the precomputed behavior cache belongs to the same user history.

    IMPORTANT:
    The cache is authoritative about how many train interactions were used during
    precompute. Inference may use a different --history-size or even a different
    --max-train-interactions for diagnostics; that alone must NOT invalidate an
    otherwise correct cache.

    Validation order:
      1) Compare with the cache's own max_train_interactions, when recorded.
      2) Otherwise compare against a suffix of the current train history having
         exactly len(cached) items.
      3) If neither matches, raise/warn with useful diagnostics.
    """
    cached = [
        str(x)
        for x in cache_row.get(
            "train_history_item_ids",
            [],
        )
    ]

    current_all = [
        str(x)
        for x in list(
            user_data.get("train", [])
        )
    ]

    if not cached:
        msg = (
            f"user={uid}: precomputed behavior row has no "
            "train_history_item_ids; cannot verify cache/history identity"
        )
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}", flush=True)
        return

    cache_limit_raw = cache_row.get(
        "max_train_interactions",
        None,
    )

    cache_limit: Optional[int] = None
    try:
        if cache_limit_raw is not None:
            cache_limit = int(
                cache_limit_raw
            )
            if cache_limit <= 0:
                cache_limit = None
    except Exception:
        cache_limit = None

    # Primary comparison: reproduce exactly the precompute slicing rule.
    if cache_limit is not None:
        expected_from_cache_config = (
            current_all[
                -cache_limit:
            ]
        )
        if cached == expected_from_cache_config:
            return

    # Robust fallback: the cached IDs themselves tell us exactly how many
    # interactions were used. This also handles older caches lacking metadata.
    expected_same_length = (
        current_all[
            -len(cached):
        ]
        if cached
        else []
    )

    if cached == expected_same_length:
        return

    # Diagnostic information for a genuine mismatch.
    first_diff = None
    for i, (a, b) in enumerate(
        zip(
            cached,
            expected_same_length,
        )
    ):
        if a != b:
            first_diff = {
                "index": int(i),
                "cached": a,
                "current": b,
            }
            break

    if (
        first_diff is None
        and len(cached)
        != len(expected_same_length)
    ):
        first_diff = {
            "index": min(
                len(cached),
                len(expected_same_length),
            ),
            "cached": (
                cached[
                    len(expected_same_length)
                ]
                if len(cached)
                > len(expected_same_length)
                else None
            ),
            "current": (
                expected_same_length[
                    len(cached)
                ]
                if len(expected_same_length)
                > len(cached)
                else None
            ),
        }

    msg = (
        f"user={uid}: behavior cache history truly mismatches current sequence; "
        f"cached_n={len(cached)}, current_train_n={len(current_all)}, "
        f"cache_max_train_interactions={cache_limit_raw!r}, "
        f"inference_max_train_interactions={max_train_interactions}, "
        f"first_diff={first_diff}, "
        f"cached_tail={cached[-3:]}, "
        f"current_expected_tail={expected_same_length[-3:]}. "
        "This usually means the cache and --sequences file were produced from "
        "different dataset/split versions."
    )

    if strict:
        raise ValueError(msg)

    print(
        f"[WARN] {msg}",
        flush=True,
    )


def latest_behavior_view(
    behaviors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    WHAT block for ranking.

    Deliberately use the TARGET USER'S latest semantic evidence only.
    Do not use semantic examples from a cluster or another user.
    """
    b = behaviors[-1]

    semantic_focus = b.get(
        "semantic_focus",
        b.get("keywords", []),
    )
    if not isinstance(semantic_focus, list):
        semantic_focus = (
            [str(semantic_focus)]
            if semantic_focus
            else []
        )

    return {
        "recommendation_behavior": clean_text(
            b.get(
                "recommendation_behavior",
                b.get(
                    "pattern_description",
                    "",
                ),
            )
        ),
        "semantic_focus": [
            clean_text(x)
            for x in semantic_focus
            if clean_text(x)
        ][:8],
        # Audit only; ranking prompt mainly uses the semantic fields above.
        "observed_trajectory_signature": clean_text(
            b.get(
                "trajectory_signature",
                b.get(
                    "behavior_signature",
                    "",
                ),
            )
        ),
        "window_index": int(
            b.get("window_index", 0)
        ),
    }


def recent3_behavior_views(behaviors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Last three actually observed local semantic behaviors, oldest -> newest.

    The behavior cache is chronological. Never use validation/test interactions,
    predicted states, cluster exemplars or Tree continuation for this baseline.
    """
    return [latest_behavior_view([b]) for b in behaviors[-3:]]


# =============================================================================
# Qwen encoder + frozen KMeans state mapping
# =============================================================================

class QwenEncoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        import unsloth  # noqa: F401
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.model_name = str(model_name)
        self.device = str(device)
        self.batch_size = int(batch_size)

        print(
            f"[INFO] Loading state encoder: "
            f"{self.model_name} on {self.device}"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, 1),
                dtype=np.float32,
            )
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(
            emb,
            dtype=np.float32,
        )

    def release(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ClusterStateVocabulary:
    """
    Frozen cluster vocabulary produced by build_reverse_behavior_tree_dual.py.
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

        self.state_mechanisms = [
            normalize_mechanism(
                r.get("mechanism")
            )
            for r in self.rows
        ]

        self.encoder = QwenEncoder(
            self.embedding_model_name,
            embedding_device,
            embedding_batch_size,
        )

    def _trajectory_text(
        self,
        behavior: Dict[str, Any],
    ) -> str:
        """
        MUST mirror choose_cluster_text() in the dual Tree builder.
        Semantic recommendation fields NEVER enter this text.
        """
        mechanism = normalize_mechanism(
            behavior.get("mechanism")
        )
        scope = normalize_scope(
            behavior.get("scope")
        )
        direction = normalize_direction(
            behavior.get("direction")
        )
        trajectory = clean_text(
            behavior.get(
                "trajectory_signature",
                behavior.get(
                    "behavior_signature",
                    "",
                ),
            )
        )

        if self.cluster_text_field in {
            "trajectory_signature",
            "behavior_signature",
        }:
            text = trajectory
        elif (
            self.cluster_text_field
            == "structured_combined"
        ):
            text = (
                f"mechanism={mechanism}; "
                f"scope={scope}; "
                f"direction={direction}; "
                f"trajectory={trajectory}"
            )
        else:
            raise ValueError(
                f"Unsupported cluster_text_field="
                f"{self.cluster_text_field!r}; "
                "this inference expects the dual Tree builder."
            )

        if not clean_text(text):
            raise ValueError(
                "Empty trajectory text for state mapping"
            )
        return clean_text(text)

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

        all_indices = list(
            range(len(self.rows))
        )

        for row_idx, (
            uid,
            behavior_index,
            behavior,
        ) in enumerate(flat):
            mechanism = normalize_mechanism(
                behavior.get("mechanism")
            )

            compatible = [
                i
                for i, m in enumerate(
                    self.state_mechanisms
                )
                if m == mechanism
            ]

            # Current Tree builder defaults to mechanism-constrained KMeans.
            # If an unseen/unknown mechanism has no bucket, use global fallback.
            candidate_indices = (
                compatible
                if compatible
                else all_indices
            )
            retrieval_scope = (
                "same_mechanism_centroids"
                if compatible
                else "global_centroid_fallback"
            )

            scores = all_scores[
                row_idx,
                candidate_indices,
            ]
            order_local = np.argsort(
                -scores
            )

            best_local = int(
                order_local[0]
            )
            best_idx = int(
                candidate_indices[
                    best_local
                ]
            )
            best_sim = float(
                scores[best_local]
            )

            second_sim: Optional[float] = None
            second_idx: Optional[int] = None

            if len(order_local) > 1:
                second_local = int(
                    order_local[1]
                )
                second_idx = int(
                    candidate_indices[
                        second_local
                    ]
                )
                second_sim = float(
                    scores[second_local]
                )

            margin = (
                best_sim - second_sim
                if second_sim is not None
                else None
            )

            accepted = (
                best_sim
                >= self.min_similarity
            )
            reasons: List[str] = []

            if not compatible:
                accepted = False
                reasons.append(
                    "missing_compatible_mechanism_bucket"
                )

            if (
                best_sim
                < self.min_similarity
            ):
                accepted = False
                reasons.append(
                    "low_similarity"
                )

            if (
                len(candidate_indices) > 1
                and margin is not None
                and margin < self.min_margin
            ):
                accepted = False
                reasons.append(
                    "low_margin"
                )

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


def latest_accepted_state_suffix(
    mappings: List[Dict[str, Any]],
) -> List[str]:
    """
    Reproduce the current Tree training semantics:
    a rejected/ambiguous mapping breaks the sequence.
    """
    suffix: List[str] = []
    for m in mappings:
        if m.get(
            "assignment_accepted"
        ):
            suffix.append(
                str(m["state_id"])
            )
        else:
            suffix = []
    return suffix


# =============================================================================
# Reverse suffix Tree
# =============================================================================

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


# =============================================================================
# Ranking prompt
# =============================================================================

def format_history(
    items: List[Dict[str, str]],
) -> str:
    return "\n".join(
        f"{i}. {x['title']} | {x['category']}"
        for i, x in enumerate(
            items,
            start=1,
        )
    )


def format_candidates(
    items: List[Dict[str, str]],
) -> str:
    """
    Candidate labels are an output interface only.  Raw item IDs are intentionally
    hidden from the ranking prompt so Qwen never has to copy Amazon/MovieLens IDs.
    """
    return "\n".join(
        f"[C{i:02d}] {x['title']} | {x['category']}"
        for i, x in enumerate(
            items,
            start=1,
        )
    )


def build_rank_prompt(
    *,
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    latest_behavior: Optional[Dict[str, Any]],
    predicted_transition: Optional[
        List[Dict[str, Any]]
    ],
) -> str:
    """
    ONE controlled template for Native / Latest-only / Tree.

    The only difference across conditions is which evidence blocks are populated.
    Candidate labels C01..Cxx are used only as stable ranking handles.
    """
    n = len(candidate_items)

    latest_block = (
        "Not provided."
    )
    if latest_behavior is not None:
        latest_block = json.dumps(
            {
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
            ensure_ascii=False,
            indent=2,
        )

    transition_block = (
        "Not provided."
    )
    if predicted_transition:
        transition_block = json.dumps(
            predicted_transition,
            ensure_ascii=False,
            indent=2,
        )

    if n == 20:
        example = (
            "[C07, C03, C12, C01, C18, C05, C09, C14, C02, C20, "
            "C11, C06, C16, C04, C13, C08, C19, C10, C15, C17]"
        )
    else:
        example_order = (
            list(range(2, n + 1, 2))
            + list(range(1, n + 1, 2))
        )
        example = (
            "["
            + ", ".join(
                f"C{x:02d}"
                for x in example_order
            )
            + "]"
        )

    return f"""You are ranking candidate items for the user's NEXT choice.

AVAILABLE SIGNALS

1) RECENT HISTORY
This is the user's observed item history.

{format_history(history_items)}

2) LATEST OBSERVED BEHAVIOR — WHAT the user currently prefers
{latest_block}

3) PREDICTED NEXT TRANSITION — HOW the current preference is expected to evolve
{transition_block}

INTERPRETATION RULES
- Use only the signals that are provided.
- The LATEST OBSERVED BEHAVIOR is the semantic anchor: it describes WHAT content
  the target user currently prefers.
- If a PREDICTED NEXT TRANSITION is provided, interpret it RELATIVE TO the latest
  observed behavior. It describes HOW the current preference is expected to evolve,
  not a specific content identity.
- Do NOT replace the target user's semantic preference with content from another user.
- If multiple predicted transitions are provided, earlier entries have higher
  collaborative probability and should receive more weight.
- Use recent history as supporting context and tie-breaking evidence.
- Candidate input order is RANDOM and has no ranking meaning.
- Base matching only on information actually shown in history, behavior, transition,
  title, and category.
- CANDIDATE LABELS (C01, C02, ...) are arbitrary handles and carry no preference signal.

CANDIDATE ITEMS

{format_candidates(candidate_items)}

DECISION PROCEDURE
1. Infer the target user's current semantic preference from the available personal evidence.
2. If a predicted transition is provided, apply its mechanism/scope/trajectory to that
   current semantic preference.
3. Rank candidates that best instantiate the resulting expected NEXT preference.
4. Rank all remaining candidates by decreasing compatibility.

OUTPUT REQUIREMENTS
- Rank ALL {n} candidate labels exactly once.
- Use ONLY C01, C02, ..., C{n:02d}.
- Do NOT output bare integers such as 1, 2, 3.
- Do NOT repeat any candidate label.
- Do NOT omit any candidate label.
- Return one JSON object with "ranking" and "reasoning" fields.
- "ranking" must be a list of exactly {n} quoted candidate labels, best first.
- "reasoning" must be one short sentence (at most 25 words).
- Do not output item IDs, item titles in the ranking list, code fences or a second answer.
- Apostrophes do not need escaping in JSON strings.
Return ALL labels; do not stop after the top few candidates.
"""


def build_recent3_rank_prompt(
    *,
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    recent3_behaviors: List[Dict[str, Any]],
) -> str:
    """Recent-3 baseline: three observed semantic behaviors, NO Tree transition.

    Build from the *unchanged* List ranking prompt, then replace only the
    observed-behavior section and its interpretation. The existing Native,
    Latest-only, and Tree prompts are never modified.
    """
    if not recent3_behaviors or len(recent3_behaviors) > 3:
        raise ValueError("Recent-3 requires between 1 and 3 observed behaviors")
    prompt = build_rank_prompt(
        history_items=history_items,
        candidate_items=candidate_items,
        latest_behavior=None,
        predicted_transition=None,
    )
    semantic_evidence = [
        {
            "recommendation_behavior": b.get("recommendation_behavior", ""),
            "semantic_focus": b.get("semantic_focus", []),
        }
        for b in recent3_behaviors
    ]
    recent_block = json.dumps(semantic_evidence, ensure_ascii=False, indent=2)
    old_section = "2) LATEST OBSERVED BEHAVIOR — WHAT the user currently prefers\nNot provided."
    new_section = (
        "2) RECENT OBSERVED BEHAVIORS — WHAT the user has preferred "
        "across recent local windows (oldest to newest)\n"
        + recent_block
    )
    if prompt.count(old_section) != 1:
        raise AssertionError("Recent-3 prompt failed to locate the observed-behavior section")
    prompt = prompt.replace(old_section, new_section, 1)
    old_rule = (
        "- The LATEST OBSERVED BEHAVIOR is the semantic anchor: it describes WHAT content\n"
        "  the target user currently prefers."
    )
    new_rule = (
        "- The RECENT OBSERVED BEHAVIORS are chronological personal semantic evidence.\n"
        "  Consider all available windows; emphasize newer evidence without ignoring\n"
        "  recurring patterns. These are observations, NOT predicted transitions."
    )
    if prompt.count(old_rule) != 1:
        raise AssertionError("Recent-3 prompt failed to locate the interpretation rule")
    prompt = prompt.replace(old_rule, new_rule, 1)
    # The transition section remains 'Not provided.' and no Tree-derived state
    # evidence is included. All ranking and output-format instructions stay intact.
    return prompt


def sanitize_ranking(
    raw_output: Any,
    valid_ids: Sequence[str],
) -> Tuple[
    List[str],
    bool,
    int,
    bool,
    str,
    List[int],
]:
    """
    Parse C01..Cxx from the FIRST generated ranking line and map them back to
    actual item IDs.

    Policy shared with the A-LLMRec ranker:
    - keep the FIRST occurrence of each valid candidate label;
    - ignore duplicate labels and out-of-range labels;
    - append missing labels in the ORIGINAL frozen candidate order;
    - only fall back to the full frozen candidate order when no valid Cxx label
      can be extracted at all.

    ``parse_complete`` is stricter than ``parse_ok``: it is True only when the
    model itself returned every candidate exactly once with no invalid labels.
    """
    valid = [
        str(x)
        for x in valid_ids
    ]
    n = len(valid)
    text = str(
        raw_output
        if raw_output is not None
        else ""
    ).strip()

    if not text:
        return (
            list(valid),
            False,
            0,
            False,
            "empty_output",
            [],
        )

    # Parse the JSON list only. Labels mentioned in reasoning are never ranking items.
    try:
        obj = extract_json_value(text)
    except (ValueError, TypeError):
        # Recover the common invalid apostrophe escape without inventing structure.
        try:
            obj = extract_json_value(text.replace("\\'", "'"))
        except (ValueError, TypeError):
            return list(valid), False, 0, False, "invalid_json", []
    values = obj.get("ranking", []) if isinstance(obj, dict) else obj
    if not isinstance(values, list):
        return list(valid), False, 0, False, "ranking_not_list", []
    nums = []
    for value in values:
        match = re.fullmatch(r"C(\d{2,3})", value) if isinstance(value, str) else None
        nums.append(int(match.group(1)) if match else 0)

    seen = set()
    parsed_order: List[int] = []
    duplicate_found = False
    invalid_found = False

    for x in nums:
        if not (
            1 <= x <= n
        ):
            invalid_found = True
            continue
        if x in seen:
            duplicate_found = True
            continue
        parsed_order.append(x)
        seen.add(x)

    returned_n = len(
        parsed_order
    )

    # True parse failure: no usable candidate label was returned.
    if not parsed_order:
        return (
            list(valid),
            False,
            0,
            False,
            "no_valid_candidate_labels",
            [],
        )

    missing = [
        x
        for x in range(
            1,
            n + 1,
        )
        if x not in seen
    ]

    complete = bool(
        returned_n == n
        and not duplicate_found
        and not invalid_found
        and not missing
    )

    repaired_order = (
        parsed_order
        + missing
    )
    ranking = [
        valid[x - 1]
        for x in repaired_order
    ]

    reasons: List[str] = []
    if duplicate_found:
        reasons.append(
            "duplicate"
        )
    if invalid_found:
        reasons.append(
            "invalid"
        )
    if missing:
        reasons.append(
            "missing"
        )

    parse_reason = (
        "ok_complete_ranking"
        if complete
        else (
            "repaired_"
            + "_and_".join(
                reasons
            )
        )
    )

    assert len(ranking) == n
    assert len(set(ranking)) == n
    assert set(ranking) == set(valid)

    return (
        ranking,
        complete,
        returned_n,
        True,
        parse_reason,
        repaired_order,
    )


# =============================================================================
# Batched local Qwen ranker
# =============================================================================

@dataclass
class CallStat:
    call_type: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    parse_ok: bool
    complete_ranking: bool


class LocalQwenRanker:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        rank_max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import (
            get_chat_template,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local Unsloth Qwen ranking expects CUDA."
            )

        print(
            f"[INFO] Loading ranking model: "
            f"{model_name}",
            flush=True,
        )
        print(
            "[INFO] Loading Qwen model weights/tokenizer; "
            "ranking starts immediately after this step.",
            flush=True,
        )

        self.model, self.tokenizer = (
            FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
            )
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-2.5",
        )
        self.tokenizer.padding_side = (
            "left"
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        self.device = next(
            self.model.parameters()
        ).device

        print(
            f"[INFO] Qwen loaded on {self.device}. "
            "Starting ranking batches.",
            flush=True,
        )
        self.max_seq_length = int(
            max_seq_length
        )
        self.rank_max_new_tokens = int(
            rank_max_new_tokens
        )
        self.temperature = float(
            temperature
        )
        if not 0 < self.rank_max_new_tokens < self.max_seq_length:
            raise ValueError("rank-max-new-tokens must be positive and smaller than max-seq-length")
        self.calls: List[CallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

    def _render(
        self,
        prompt: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a recommendation ranking system. "
                    "Return only the requested candidate-label ranking."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        return text

    def _generate_batch_raw(
        self,
        prompts: List[str],
        call_type: str,
    ) -> List[str]:
        if not prompts:
            return []

        rendered = [
            self._render(p)
            for p in prompts
        ]

        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length - self.rank_max_new_tokens,
            add_special_tokens=False,
        ).to(self.device)

        input_width = int(
            batch["input_ids"].shape[1]
        )

        kwargs: Dict[str, Any] = {
            "max_new_tokens": (
                self.rank_max_new_tokens
            ),
            "use_cache": True,
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
            "eos_token_id": (
                self.tokenizer.eos_token_id
            ),
        }

        if self.temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": (
                    self.temperature
                ),
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                **kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (
            time.perf_counter()
            - t0
        )

        generated = outputs[
            :,
            input_width:,
        ]

        raws = (
            self.tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        # Detailed parse/complete stats are added later.
        per_sample_time = (
            elapsed
            / max(
                1,
                len(prompts),
            )
        )

        for gen_row in generated:
            self.calls.append(
                CallStat(
                    call_type=call_type,
                    input_tokens=input_width,
                    output_tokens=int(
                        gen_row.shape[0]
                    ),
                    elapsed_sec=float(
                        per_sample_time
                    ),
                    parse_ok=False,
                    complete_ranking=False,
                )
            )

        return [
            str(x).strip()
            for x in raws
        ]

    def _parse_rank(
        self,
        raw: str,
        valid_ids: List[str],
    ) -> Tuple[
        List[str],
        Dict[str, Any],
    ]:
        (
            ranking,
            complete,
            returned_n,
            parse_ok,
            parse_reason,
            parsed_order,
        ) = sanitize_ranking(
            raw,
            valid_ids,
        )

        return ranking, {
            "parse_ok": bool(
                parse_ok
            ),
            "parse_complete": bool(
                complete
            ),
            "parse_reason": (
                parse_reason
            ),
            "model_returned_valid_ids": int(
                returned_n
            ),
            "num_candidates": len(
                valid_ids
            ),
            "parsed_candidate_order_1based": (
                parsed_order
            ),
            "parsed_candidate_labels": [
                f"C{x:02d}"
                for x in parsed_order
            ],
            "repair_used": bool(
                parse_ok
                and not complete
            ),
            "fallback_used": bool(
                not parse_ok
            ),
            "reasoning": "",
            "raw_output": raw,
        }

    def rank_jobs(
        self,
        jobs: List[Dict[str, Any]],
        batch_size: int,
        call_type: str,
        max_retries: int,
    ) -> List[
        Tuple[
            List[str],
            Dict[str, Any],
        ]
    ]:
        if not jobs:
            return []

        final: List[
            Optional[
                Tuple[
                    List[str],
                    Dict[str, Any],
                ]
            ]
        ] = [
            None
            for _ in jobs
        ]

        batch_size = max(
            1,
            int(batch_size),
        )

        batch_starts = list(
            range(
                0,
                len(jobs),
                batch_size,
            )
        )

        pbar = tqdm(
            batch_starts,
            desc=call_type,
            unit="batch",
            dynamic_ncols=True,
        )

        for start in pbar:
            end = min(
                start + batch_size,
                len(jobs),
            )
            pbar.set_postfix(
                users=f"{start + 1}-{end}/{len(jobs)}",
                refresh=False,
            )
            chunk = jobs[
                start:end
            ]

            prompts = [
                j["prebuilt_prompt"] if "prebuilt_prompt" in j else build_rank_prompt(
                    history_items=j[
                        "history_items"
                    ],
                    candidate_items=j[
                        "candidate_items"
                    ],
                    latest_behavior=j.get(
                        "latest_behavior"
                    ),
                    predicted_transition=j.get(
                        "predicted_transition"
                    ),
                )
                for j in chunk
            ]

            try:
                raws = (
                    self._generate_batch_raw(
                        prompts,
                        call_type,
                    )
                )
            except Exception as e:
                raise RuntimeError(
                    f"{call_type} generation batch {start}:{end} failed; "
                    "not substituting frozen candidate order for a runtime failure"
                ) from e

            for local_idx, (
                job,
                prompt,
                raw,
            ) in enumerate(
                zip(
                    chunk,
                    prompts,
                    raws,
                )
            ):
                global_idx = (
                    start
                    + local_idx
                )
                valid_ids = [
                    str(
                        x["item_id"]
                    )
                    for x in job[
                        "candidate_items"
                    ]
                ]

                ranking, meta = (
                    self._parse_rank(
                        raw,
                        valid_ids,
                    )
                )

                # No individual retry: keep the first batched generation result.
                # Duplicate Cxx labels keep their first occurrence; missing labels
                # are appended in frozen candidate order. Only a true Cxx parse
                # failure falls back to the full frozen candidate order.
                attempts = 1
                meta["attempts"] = attempts
                meta["call_type"] = (
                    call_type
                )
                final[
                    global_idx
                ] = (
                    ranking,
                    meta,
                )

        pbar.close()

        if any(
            x is None
            for x in final
        ):
            raise RuntimeError(
                "Internal ranking result missing"
            )

        return [
            x
            for x in final
            if x is not None
        ]

    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {
                "model": None,
                "generation_records": 0,
            }

        by_type = Counter(
            x.call_type
            for x in self.calls
        )

        return {
            "model": str(
                getattr(
                    self.model,
                    "name_or_path",
                    "local_qwen",
                )
            ),
            "generation_records": len(
                self.calls
            ),
            "input_tokens_approx": int(
                sum(
                    x.input_tokens
                    for x in self.calls
                )
            ),
            "output_tokens_approx": int(
                sum(
                    x.output_tokens
                    for x in self.calls
                )
            ),
            "elapsed_sec": float(
                sum(
                    x.elapsed_sec
                    for x in self.calls
                )
            ),
            "call_type_counts": dict(
                by_type
            ),
        }


# =============================================================================
# Metrics
# =============================================================================

def target_rank(
    ranking: Sequence[str],
    target: str,
) -> int:
    ranking = [
        str(x)
        for x in ranking
    ]
    try:
        return (
            ranking.index(
                str(target)
            )
            + 1
        )
    except ValueError:
        return len(
            ranking
        ) + 1


def metrics_from_ranks(
    ranks: List[int],
    ks: Sequence[int] = (
        1,
        3,
        5,
        10,
    ),
) -> Optional[Dict[str, float]]:
    if not ranks:
        return None

    n = len(ranks)
    out: Dict[str, float] = {}

    for k in ks:
        out[f"Hit@{k}"] = float(
            sum(
                r <= k
                for r in ranks
            )
            / n
        )
        out[f"NDCG@{k}"] = float(
            sum(
                (
                    1.0
                    / math.log2(
                        r + 1
                    )
                )
                if r <= k
                else 0.0
                for r in ranks
            )
            / n
        )

    out["MRR"] = float(
        sum(
            1.0 / r
            for r in ranks
        )
        / n
    )
    out["MeanRank"] = float(
        np.mean(ranks)
    )
    out["MedianRank"] = float(
        np.median(ranks)
    )
    out["NumUsers"] = int(n)
    return out


def aggregate_mode(
    rows: List[Dict[str, Any]],
    mode: str,
    strict_complete: bool,
) -> Optional[Dict[str, float]]:
    ranks: List[int] = []

    for row in rows:
        rec = row.get(
            "rankings",
            {},
        ).get(
            mode
        )
        if not isinstance(
            rec,
            dict,
        ):
            continue

        if (
            strict_complete
            and not rec.get(
                "parse_complete"
            )
        ):
            continue

        rank = rec.get(
            "target_rank"
        )
        if rank is not None:
            ranks.append(
                int(rank)
            )

    return metrics_from_ranks(
        ranks
    )


def metric_gain(
    a: Optional[Dict[str, float]],
    b: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not a or not b:
        return None

    out: Dict[str, float] = {}

    for k in a:
        if (
            k not in b
            or k == "NumUsers"
        ):
            continue

        if k in {
            "MeanRank",
            "MedianRank",
        }:
            # Positive = A better.
            out[k] = float(
                b[k] - a[k]
            )
        else:
            out[k] = float(
                a[k] - b[k]
            )

    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoMemTree/Qwen LIST ranking, standardized frozen train-only protocol. "
                    "Minimal: --dataset ML (other presets: CDs, Video_Games)."
    )
    p.add_argument("--dataset", required=True,
                   help="CDs | Video_Games | ML | Books | Software")
    p.add_argument("--tree", "--tree-dir", dest="tree_dir", default=None,
                   help="Optional Tree directory override. Default auto-resolved from --dataset, "
                        "e.g. trees/ML_1000_20, trees/CDs_1000_20, trees/VGs_1000_20.")
    p.add_argument("--tree-root", default="trees",
                   help="Folder holding the standardized dataset Trees (default: trees)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--precompute-root", default="precomputed")
    p.add_argument("--output-root", default="results/comemtree")
    p.add_argument("--run-dir", default=None,
                   help="Explicit run folder, chiefly for --resume; otherwise named automatically")
    p.add_argument("--behavior-model", choices=["qwen", "gemma"], default="qwen",
                   help="Only sets default cache path and expected Tree provenance; never converts caches")
    p.add_argument("--items", default=None)
    p.add_argument("--sequences", default=None)
    p.add_argument("--negatives", default=None)
    p.add_argument("--candidate-file", default=None)
    p.add_argument("--user-ids-file", default=None,
                   help="Optional frozen cohort override; by default data/<dataset>/protocol/eval_users_1000.json")
    p.add_argument("--precomputed-behaviors", default=None,
                   help="Default: precomputed/<dataset>/test_shared.jsonl; optional model subfolder fallback")
    p.add_argument("--states-json", default=None)
    p.add_argument("--state-embeddings", default=None)
    p.add_argument("--tree-json", default=None)
    p.add_argument("--allow-tree-source-mismatch", action="store_true",
                   help="Override strict source-path provenance check (not recommended)")
    p.add_argument("--require-eval-users-in-tree", action="store_true",
                   help="Optional restrictive audit: reject eval users absent from Tree training-user metadata. "
                        "Default OFF: an unseen user can query a collaborative Tree by mapping behavior states.")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--cluster-text-field", choices=["structured_combined", "trajectory_signature", "behavior_signature"], default=None)
    p.add_argument("--mapping-min-similarity", type=float, default=None)
    p.add_argument("--mapping-min-margin", type=float, default=None)
    p.add_argument("--respect-mapping-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rank-max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--llm-batch-size", type=int, default=8)
    p.add_argument("--history-size", type=int, default=None,
                   help="Default: 30 for ML, 20 for other datasets")
    p.add_argument("--max-train-interactions", type=int, default=None,
                   help="Diagnostic only: cache's own precompute history takes precedence")
    p.add_argument("--max-behaviors", type=int, default=0,
                   help="Recent HOW windows to map into Tree; 0=all. Latest WHAT always latest overall.")
    p.add_argument("--top-next", type=int, default=1)
    p.add_argument("--tree-max-query-order", type=int, default=0,
                   help="0=full Tree configured max_order, 1/2/...=order cap")
    p.add_argument("--run-native", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-latest-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-recent3", action=argparse.BooleanOptionalAction, default=True,
                   help="Additional baseline: last 3 OBSERVED semantic behaviors, without Tree prediction")
    p.add_argument("--strict-cache-history", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--start-user", type=int, default=0)
    p.add_argument("--max-users", type=int, default=1000,
                   help="Select up to N users in frozen eval_users_1000.json order; 0=all. "
                        "Use --user-ids-file to choose a different frozen cohort.")
    p.add_argument("--candidate-size", type=int, default=20)
    p.add_argument("--prompt-example-user", default=None,
                   help="User ID for the saved filled-in example; defaults to first selected user")
    p.add_argument("--save-all-prompts", action="store_true",
                   help="Optionally save filled-in prompts for every evaluated user and mode")
    p.add_argument("--prompts-only", action="store_true",
                   help="Map behaviors and save filled-in prompts, but skip Qwen ranking")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate all frozen inputs/cache without loading encoder or ranking LLM")
    p.add_argument("--resume", action="store_true",
                   help="Continue results in explicit --run-dir (requires same frozen run configuration)")
    p.add_argument("--seed", type=int, default=42)
    # Ablation-specific settings. Base prompt/Tree mapping/ranker are unchanged.
    p.add_argument("--modes", default="latest_only,latest_how,random_how,global_continuation,pmn,tree,oracle",
                   help="Comma-separated modes. Extra: recent3,recent3_tree,tree_original,native."
                        " All modes share identical frozen history/candidates/parser.")
    p.add_argument("--train-behaviors", default=None,
                   help="TRAIN-only source behavior memory for random HOW; default precomputed/<dataset>/train_shared.jsonl")
    p.add_argument("--source-user-ids-file", default=None,
                   help="Optional source cohort whitelist; auto-uses protocol/train_users_1000.json if present")
    p.add_argument("--exclude-all-eval-sources", action="store_true",
                   help="Exclude ALL 1000 frozen eval users from random/PMN source pool (strict disjoint)")
    p.add_argument("--require-user-disjoint-tree", action="store_true",
                   help="Fail if Tree metadata indicates any training/eval user overlap or lacks coverage metadata")
    p.add_argument("--pmn-summaries", default=None,
                   help="Gemma-generated target/source preference profiles from prepare_pmn_summaries.py")
    p.add_argument("--pmn-embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
                   help="Fixed PMN profile encoder; independent of Tree states and centroids")
    p.add_argument("--oracle-behaviors", default=None,
                   help="Future-aware target behavior cache; used ONLY for oracle mode")
    p.add_argument("--allow-debug-pmn", action="store_true",
                   help="Smoke-test only: accept debug/extractive preference profiles")
    return p.parse_args()


def _canonical_dataset(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"cd":"CDs","cds":"CDs","video":"Video_Games","vg":"Video_Games",
               "videogames":"Video_Games","video_games":"Video_Games",
               "ml":"ML","ml1m":"ML","movielens":"ML","movielens_1m":"ML",
               "books":"Books","book":"Books","software":"Software","soft":"Software"}
    out = aliases.get(key, value.strip())
    if not out or out in (".", "..") or "/" in out or "\\" in out or ".." in out:
        raise ValueError(f"Unsafe dataset name {value!r}")
    return out


def _resolve_tree_directory(value: str) -> Path:
    """Explicit override, including a bare folder name under ./trees/."""
    supplied = Path(value).expanduser()
    candidates = [supplied]
    if len(supplied.parts) == 1:
        candidates += [Path("trees") / supplied, Path("behavior_trees") / supplied]
    existing = [x for x in candidates if x.is_dir()]
    if not existing:
        raise FileNotFoundError("Tree directory not found. Tried: " + ", ".join(map(str, candidates)))
    if len(existing) > 1 and len({x.resolve() for x in existing}) > 1:
        raise ValueError("Ambiguous Tree name, use its explicit path: " + ", ".join(map(str, existing)))
    return existing[0]


def _auto_tree_dir(dataset: str, tree_root: str) -> Path:
    """Known dataset Tree folders; require an override rather than guessing ambiguously."""
    tree_names = {
        "ML": "ML_1000_20",
        "CDs": "CDs_1000_20",
        "Video_Games": "VGs_1000_20",
        "Books": "Books_1000_20",
        "Software": "Software_1000_20",
    }
    root = Path(tree_root).expanduser()
    preferred = root / tree_names.get(dataset, f"{dataset}_1000_20")
    if preferred.is_dir():
        return preferred

    # If preferred folder is absent, search only unambiguous trees for the same
    # dataset. Never silently select a tree trained for another dataset.
    stems = {
        "ML": ["ML_", "ML1M_", "tree_ML_"],
        "CDs": ["CDs_", "tree_CDs_"],
        "Video_Games": ["VGs_", "Video_Games_", "tree_VGs_", "tree_Video_Games_"],
        "Books": ["Books_", "tree_Books_"],
        "Software": ["Software_", "tree_Software_"],
    }.get(dataset, [dataset + "_", "tree_" + dataset + "_"])
    matches = []
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and any(folder.name.startswith(prefix) for prefix in stems):
                if all((folder / name).is_file() for name in (
                    "behavior_states.json", "behavior_state_embeddings.npy", "behavior_tree.json"
                )):
                    matches.append(folder)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple Trees for {dataset}: {', '.join(map(str, matches))}. "
            "Choose explicitly using --tree."
        )
    raise FileNotFoundError(
        f"No Tree found for dataset {dataset} under {root}. "
        f"Expected {preferred}. Pass --tree /path/to/tree to override."
    )


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve standardized dataset input paths without changing the ranking method."""
    args.dataset = _canonical_dataset(args.dataset)
    args.tree_dir = str(
        _resolve_tree_directory(args.tree_dir)
        if args.tree_dir
        else _auto_tree_dir(args.dataset, args.tree_root)
    )
    if args.history_size is None:
        args.history_size = 30 if args.dataset == "ML" else 20
    if args.max_train_interactions is None:
        args.max_train_interactions = args.history_size
    for name in ("history_size", "llm_batch_size", "candidate_size", "top_next"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    for name in ("max_users", "max_behaviors", "start_user", "tree_max_query_order"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be >=0")

    dataset_dir = Path(args.data_root) / args.dataset
    protocol_dir = dataset_dir / "protocol"
    behavior_root = Path(args.precompute_root) / args.dataset
    default_cache = behavior_root / "test_shared.jsonl"
    legacy_cache = behavior_root / args.behavior_model / "test_shared.jsonl"
    # Canonical data has no model-specific subfolder. Legacy fallback is used
    # only if the canonical file does not exist; explicit overrides take priority.
    if not default_cache.is_file() and legacy_cache.is_file():
        default_cache = legacy_cache

    default_inputs = {
        "items": dataset_dir / "items.json",
        "sequences": dataset_dir / "user_sequences.json",
        "negatives": dataset_dir / "user_negatives.json",
        "candidate_file": protocol_dir / "user_candidates_testpool1000_seed42.json",
        "user_ids_file": protocol_dir / "eval_users_1000.json",
        "precomputed_behaviors": default_cache,
    }
    for name, path in default_inputs.items():
        if getattr(args, name) is None:
            setattr(args, name, str(path))

    # Use the train cache in the *same actual folder* as the test cache to
    # prevent false provenance mismatches from the obsolete /qwen/ assumption.
    args.expected_tree_input = str(Path(args.precomputed_behaviors).parent / "train_shared.jsonl")

    for name in default_inputs:
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}. Override with --{name.replace('_','-')} "
                "or check the standardized dataset directory."
            )
    states_json = args.states_json or str(Path(args.tree_dir) / "behavior_states.json")
    states_embeddings = args.state_embeddings or str(Path(args.tree_dir) / "behavior_state_embeddings.npy")
    tree_json = args.tree_json or str(Path(args.tree_dir) / "behavior_tree.json")
    for name, path in (("states", states_json), ("embeddings", states_embeddings), ("Tree", tree_json)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} artifact: {path}")
    args.states_json = states_json
    args.state_embeddings = states_embeddings
    args.tree_json = tree_json
    return args


def validate_tree_provenance(args: argparse.Namespace, cohort: Sequence[str]) -> Dict[str, Any]:
    """Audit Tree origin and user coverage without treating the Tree as a user lookup.

    The collaborative Tree is a learned behavior-state transition model, not an
    index keyed by evaluation user IDs.  A target user with a valid behavior
    cache may map to the existing states even if absent from the training-user
    selection recorded in ``selected_user_order``.  This is different from
    missing frozen candidates/targets/behavior caches, which remain fatal.

    Always report train/eval cohort overlap: high overlap is *not* compatible
    with a user-disjoint evaluation claim, even though train-only interactions
    need not imply held-out test-label leakage.
    """
    metadata = load_json(args.states_json).get("metadata", {})
    input_path = metadata.get("input")
    source_matches = bool(input_path and Path(input_path).resolve() == Path(args.expected_tree_input).resolve())
    warnings: List[str] = []
    if input_path and not source_matches:
        msg = (f"Tree was built from {input_path!r}, but expected {args.expected_tree_input!r}. "
               "Verify that Tree and test cache use compatible behavior schema, model, and split.")
        if not args.allow_tree_source_mismatch:
            raise ValueError(msg + " Use --allow-tree-source-mismatch only after verifying provenance.")
        warnings.append(msg)
    if not input_path:
        warnings.append("Tree state metadata lacks original input path; provenance cannot be verified.")

    tree_users = metadata.get("selected_user_order")
    eval_ids = set(map(str, cohort))
    num_tree_users: Optional[int] = None
    overlap_count: Optional[int] = None
    missing_from_tree: List[str] = []
    if isinstance(tree_users, list):
        training_ids = set(map(str, tree_users))
        num_tree_users = len(training_ids)
        overlap_count = len(eval_ids & training_ids)
        missing_from_tree = sorted(eval_ids - training_ids)
        print(
            f"[TREE USER AUDIT] eval={len(eval_ids)} "
            f"present_in_tree_training={overlap_count} "
            f"absent_from_tree_training={len(missing_from_tree)}; "
            f"examples={missing_from_tree[:5]}", flush=True,
        )
        if missing_from_tree:
            msg = (
                f"Tree training metadata omits {len(missing_from_tree)} of "
                f"{len(eval_ids)} eval users; examples={missing_from_tree[:5]}. "
                "This is allowed: Tree predictions use mapped behavior states "
                "rather than a lookup by training user ID."
            )
            if getattr(args, "require_eval_users_in_tree", False):
                raise ValueError(msg + " Disable --require-eval-users-in-tree to evaluate unseen users.")
            warnings.append(msg)
        if overlap_count:
            warnings.append(
                f"Train-user/eval-user overlap: {overlap_count}/{len(eval_ids)} users. "
                "Do not describe this run as user-disjoint unless you use a disjoint Tree."
            )
    else:
        warnings.append("Tree lacks selected_user_order metadata; train/eval user overlap cannot be audited.")

    for warning in warnings:
        print(f"[TREE PROVENANCE WARN] {warning}", flush=True)
    return {
        "tree_training_source": input_path,
        "tree_matches_expected_qwen_cache": source_matches,
        "tree_users": num_tree_users,
        "eval_users": len(eval_ids),
        "eval_users_present_in_tree_training": overlap_count,
        "eval_users_absent_from_tree_training": len(missing_from_tree) if isinstance(tree_users, list) else None,
        "eval_users_absent_examples": missing_from_tree[:10],
        "user_disjoint_tree_cohort": (overlap_count == 0) if overlap_count is not None else None,
        "warnings": warnings,
    }


def validate_frozen_users(
    users: Sequence[str], sequences: Dict[str, Any],
    candidate_rows: Dict[str, Dict[str, Any]], behavior_cache: Dict[str, Dict[str, Any]],
    items_meta: Dict[str, Any], candidate_size: int, strict_cache_history: bool,
) -> Dict[str, Any]:
    target_missing = []
    for uid in tqdm(users, desc="Validate frozen protocol", unit="user", dynamic_ncols=True):
        row = sequences[uid]
        targets = as_str_list(row.get("test", []))
        if len(targets) != 1:
            raise ValueError(f"{uid}: exactly one held-out test item required; got {len(targets)}")
        candidates, candidate_targets = get_candidates_for_user(uid, row, {}, candidate_rows, 42)
        if len(candidates) != candidate_size:
            raise ValueError(f"{uid}: expected exactly {candidate_size} frozen candidates, got {len(candidates)}")
        if candidate_targets != targets:
            raise ValueError(f"{uid}: candidate target {candidate_targets} != held-out test {targets}")
        for iid in candidates:
            info = items_meta.get(iid, {})
            if not isinstance(info, dict) or not clean_text(info.get("title") or info.get("name")):
                raise ValueError(f"{uid}: frozen candidate {iid} has no real title")
        if not as_str_list(row.get("train", [])):
            raise ValueError(f"{uid}: no observed train history")
        cache_row = behavior_cache[uid]
        if strict_cache_history:
            validate_behavior_cache_history(uid, cache_row, row, 0, True)
        elif cache_row.get("train_history_item_ids"):
            validate_behavior_cache_history(uid, cache_row, row, 0, False)
        else:
            raise ValueError(f"{uid}: missing train_history_item_ids in behavior cache")
        if not cache_row.get("generated_behaviors"):
            raise ValueError(f"{uid}: empty precomputed behaviors")
    return {"users": len(users), "frozen_candidates_per_user": candidate_size,
            "train_only": True, "held_out_test_count": 1, "validation_used": False}


def _filled_prompt_sections(record: Dict[str, Any], run_native: bool, run_latest: bool,
                            run_recent3: bool = False) -> Dict[str, str]:
    h, c, latest, trans = (record["history_items"], record["candidate_items"],
                            record["latest_behavior"], record["predicted_transition"])
    sections = {"tree": build_rank_prompt(history_items=h, candidate_items=c,
                                           latest_behavior=latest, predicted_transition=trans)}
    if run_latest:
        sections["latest_only"] = build_rank_prompt(history_items=h, candidate_items=c,
                                                     latest_behavior=latest, predicted_transition=None)
    if run_native:
        sections["native"] = build_rank_prompt(history_items=h, candidate_items=c,
                                               latest_behavior=None, predicted_transition=None)
    if run_recent3:
        sections["recent3"] = build_recent3_rank_prompt(
            history_items=h, candidate_items=c,
            recent3_behaviors=record["recent3_behaviors"],
        )
    return sections


def save_prompt_examples(args: argparse.Namespace, prepared: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    if not prepared:
        raise ValueError("Nothing prepared to save prompts")
    chosen = prepared[0]
    if args.prompt_example_user:
        hits = [r for r in prepared if r["user_id"] == args.prompt_example_user]
        if not hits:
            raise ValueError(f"--prompt-example-user={args.prompt_example_user!r} not selected for inference")
        chosen = hits[0]
    sections = _filled_prompt_sections(chosen, args.run_native, args.run_latest_only,
                                        args.run_recent3)
    system_msg = "You are a recommendation ranking system. Return only the requested candidate-label ranking."
    text_parts = [f"Dataset: {args.dataset}", f"Example user: {chosen['user_id']}",
                  "System message (common to every condition):", system_msg]
    for mode, prompt in sections.items():
        text_parts += ["", "="*32 + f" {mode.upper()} " + "="*32, prompt]
    (run_dir / "prompt_example.txt").write_text("\n".join(text_parts) + "\n", encoding="utf-8")
    details = {
        "dataset": args.dataset, "user_id": chosen["user_id"],
        "system_message": system_msg, "rank_prompt_version": RANK_PROMPT_VERSION,
        "history_items": chosen["history_items"],
        "latest_observed_behavior": chosen["latest_behavior"],
        "recent3_observed_behaviors": chosen["recent3_behaviors"] if args.run_recent3 else None,
        "recent3_prompt_version": RECENT3_PROMPT_VERSION if args.run_recent3 else None,
        "tree_query": chosen["tree_result"],
        "predicted_next_states": chosen["predicted_transition"],
        "candidate_label_map": {f"C{i:02d}": x for i,x in enumerate(chosen["candidate_items"],1)},
        "prompts": sections,
        "notice": "Held-out test item is NOT identified in the model prompt. Targets are used only for metrics.",
    }
    dump_json(run_dir / "prompt_example.json", details)
    if args.save_all_prompts:
        with (run_dir / "prompts_all.jsonl").open("w", encoding="utf-8") as out:
            for r in prepared:
                out.write(json.dumps({"user_id": r["user_id"],
                                      "prompts": _filled_prompt_sections(r, args.run_native, args.run_latest_only,
                                                                         args.run_recent3)},
                                     ensure_ascii=False) + "\n")
    return {"sample_user_id": chosen["user_id"], "prompt_example_txt": str(run_dir / "prompt_example.txt"),
            "prompt_example_json": str(run_dir / "prompt_example.json"),
            "all_prompts_jsonl": str(run_dir / "prompts_all.jsonl") if args.save_all_prompts else None}



# =============================================================================
# Controlled ablation extension to the unchanged inference_bt4rec primitives.
# =============================================================================

import hashlib

AB_MODES = (
    'latest_only', 'latest_how', 'random_how', 'global_continuation',
    'pmn', 'tree', 'oracle', 'recent3', 'recent3_tree',
    'tree_original', 'native',
    'stay_only', 'tree_stay', 'tree_original_stay', 'recent3_tree_stay',
)
PMN_VERSION = 'pmn_preference_summary_train_prefix_gemma_v4_train_cache_metadata'


def _fingerprint(parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def requested_modes(raw: str) -> List[str]:
    modes = [s.strip().lower() for s in raw.split(',') if s.strip()]
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('--modes cannot be empty or contain duplicates')
    unknown = sorted(set(modes) - set(AB_MODES))
    if unknown:
        raise ValueError(f'Unknown ablation modes: {unknown}. Allowed: {AB_MODES}')
    return modes


def how_only(raw: Dict[str, Any]) -> Dict[str, str]:
    """Uniform mechanism/scope/trajectory; no user IDs, IDs, content or probabilities."""
    raw = raw or {}
    s = clean_text(raw.get('trajectory_signature') or raw.get('behavior_signature')
                   or raw.get('canonical_behavior_signature') or raw.get('representative_signature'))
    m = clean_text(raw.get('mechanism'))
    scope = clean_text(raw.get('scope'))
    if not all([s, m, scope]):
        raise ValueError(f'HOW missing mechanism/scope/signature: {raw}')
    return {'mechanism': normalize_mechanism(m), 'scope': normalize_scope(scope),
            'trajectory_signature': s}


def controlled_prompt(history_items, candidate_items, latest, how, recent3=None):
    """ALL controlled conditions use identical guidance framing and output rules.

    The original bt4rec prompt is reused verbatim, apart from the minimal,
    mode-independent changes that remove the claim that even random/observed HOW
    is a genuinely PREDICTED transition. Only the HOW block varies by condition.
    """
    if recent3 is None:
        prompt = build_rank_prompt(history_items=history_items, candidate_items=candidate_items,
                                   latest_behavior=latest, predicted_transition=None)
    else:
        prompt = build_recent3_rank_prompt(history_items=history_items,
                                           candidate_items=candidate_items,
                                           recent3_behaviors=recent3)
    old = '3) PREDICTED NEXT TRANSITION — HOW the current preference is expected to evolve\nNot provided.'
    val = (json.dumps(how, ensure_ascii=False, indent=2) if how is not None else 'Not provided.')
    new = '3) ADDITIONAL BEHAVIOR GUIDANCE — possible HOW evolution of the current preference\n' + val
    if prompt.count(old) != 1:
        raise RuntimeError('bt4rec base ranking template changed; cannot build controlled prompt')
    prompt = prompt.replace(old, new, 1)
    prompt = prompt.replace('If a PREDICTED NEXT TRANSITION is provided,',
                            'If ADDITIONAL BEHAVIOR GUIDANCE is provided,')
    prompt = prompt.replace('If multiple predicted transitions are provided, earlier entries have higher\n'
                            '  collaborative probability and should receive more weight.\n', '')
    prompt = prompt.replace('If a predicted transition is provided, apply its mechanism/scope/trajectory to that',
                            'If additional behavior guidance is provided, apply its mechanism/scope/trajectory to that')
    # Prompt has the same rule for source and Tree, and does not identify condition.
    assert 'PREDICTED NEXT TRANSITION' not in prompt
    assert 'higher collaborative probability' not in prompt
    return prompt


def read_train_how(path: str, allowed_users: Optional[set],
                   forbidden_users: set) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(dict)
    for row in read_jsonl(path):
        uid_raw = row.get('user_id', row.get('uid'))
        if uid_raw is None:
            continue
        uid = str(uid_raw)
        if (allowed_users is not None and uid not in allowed_users) or uid in forbidden_users:
            continue
        if row.get('precompute_ok') is False:
            continue
        if isinstance(row.get('generated_behaviors'), list):
            windows = row['generated_behaviors']
        else:
            b = next((row[k] for k in ('behavior','generated_behavior','local_memory',
                          'parsed_behavior','extracted_behavior','memory','output')
                      if isinstance(row.get(k), dict)), row)
            windows = [b]
        for idx, b in enumerate(windows):
            if not isinstance(b, dict):
                continue
            try:
                how = how_only(b)
            except ValueError:
                continue
            wi = b.get('window_index', row.get('window_index', idx))
            try:
                wi = int(wi)
            except (ValueError, TypeError):
                continue
            grouped[uid].setdefault(wi, how)
    result = {uid: [{'window_index':wi,'how':b} for wi,b in sorted(w.items())] for uid,w in grouped.items() if w}
    if not result:
        raise ValueError(f'No eligible train-only source HOW in {path}. Check cohort/exclusions.')
    return result


def _choose_random_how(training: Dict[str, List[Dict[str, Any]]], uid: str, seed: int):
    # Uniform sampling of SOURCE USER then one observed behavior window;
    # deterministic per target user and seed, independent of target/test labels.
    available = sorted(x for x in training if x != uid)
    if not available:
        raise ValueError(f'{uid}: random source has no other user')
    src = available[int(_fingerprint([seed, uid, 'random_user'])[:16], 16) % len(available)]
    seq = training[src]
    ix = int(_fingerprint([seed, uid, 'random_window'])[:16], 16) % len(seq)
    return seq[ix]['how'], {'source_user_id': src, 'source_window_index': seq[ix]['window_index'],
                            'selection': 'uniform_train_user_window'}


def load_oracle(path: str, users: List[str], sequences: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    expected = set(users)
    out = {}
    for row in read_jsonl(path):
        uid = str(row.get('user_id'))
        if uid not in expected:
            continue
        if uid in out:
            raise ValueError(f'Duplicate future oracle for {uid}')
        if row.get('future_aware') is not True:
            raise ValueError(f'{uid}: oracle record must declare future_aware=true')
        target = as_str_list(sequences[uid].get('test', []))
        if len(target) != 1 or str(row.get('target_item_id')) != target[0]:
            raise ValueError(f'{uid}: oracle target ID does not match frozen test item')
        out[uid] = how_only(row.get('oracle_behavior', {}))
    missing = sorted(expected - set(out))
    if missing:
        raise ValueError(f'Oracle cache missing {len(missing)} target users, first={missing[:10]}')
    return out


def load_pmn_profiles(path: str, dataset: str, users: List[str], sequences: Dict[str, Any],
                      history_size: int, allowed_users: Optional[set], forbidden_users: set,
                      allow_debug: bool, train_how: Optional[Dict[str, List[Dict[str, Any]]]]=None):
    """Validate summary provenance without using any CoMemTree/Tree state in PMN matching."""
    expected = set(users)
    targets: Dict[str, str] = {}
    sources: List[Dict[str, Any]] = []
    seen = set()
    bad = Counter()
    for r in read_jsonl(path):
        if r.get('dataset') != dataset or r.get('summary_version') != PMN_VERSION:
            continue
        role, uid = r.get('role'), str(r.get('user_id'))
        if role not in ('target', 'source') or (role == 'target' and uid not in expected):
            continue
        if (role, uid) in seen:
            raise ValueError(f'Duplicate PMN profile {role}:{uid}')
        seen.add((role, uid))
        if not allow_debug and r.get('summary_backend') != 'gemma':
            raise ValueError(f'PMN {role}:{uid} is not a Gemma profile: {r.get("summary_backend")}')
        summary = clean_text(r.get('preference_summary'))
        if not summary:
            raise ValueError(f'PMN {role}:{uid}: blank summary')
        if uid not in sequences:
            raise ValueError(f'PMN {role}:{uid}: missing frozen sequence')
        tr = as_str_list(sequences[uid].get('train', []))
        hist = as_str_list(r.get('history_item_ids'))
        if not hist or int(r.get('history_size', -1)) != history_size:
            raise ValueError(f'PMN {role}:{uid}: wrong summary history length/config')
        if role == 'target':
            used = tr[-history_size:]
            key = _fingerprint([PMN_VERSION, dataset, 'target', uid, history_size, used])
            if hist != used or r.get('summary_key') != key:
                raise ValueError(f'PMN target profile {uid} does not match current frozen observed train history')
            targets[uid] = summary
        else:
            # Source must be a training user, not a test-specific record.
            if allowed_users is not None and uid not in allowed_users:
                bad['outside_train_user_cohort'] += 1
                continue
            if uid in forbidden_users:
                bad['excluded_eval_source'] += 1
                continue
            next_item = str(r.get('terminal_train_item_id') or '')
            terminal_idx = r.get('terminal_window_index')
            if not next_item or terminal_idx is None:
                raise ValueError(f'PMN source {uid} missing continuation item/index metadata')
            raw_how = r.get('source_how')
            how = how_only(raw_how)
            matching_ends = [end for end in range(2,len(tr)+1)
                             if tr[end-1] == next_item and
                             tr[:end-1][-history_size:] == hist]
            if not matching_ends:
                raise ValueError(f'PMN source {uid}: summary prefix does not precede continuation in frozen train')
            key = _fingerprint([PMN_VERSION, dataset, 'source', uid, history_size,
                                hist, int(terminal_idx), next_item, raw_how])
            if r.get('summary_key') != key:
                raise ValueError(f'PMN source {uid}: summary fingerprint mismatch')
            if train_how is not None:
                # Validate source continuation HOW against the actual train cache.
                # The terminal window index is an identifier, not a 0-based offset.
                available = train_how.get(uid, [])
                if not any(how == x['how'] and int(terminal_idx)==x['window_index'] for x in available):
                    raise ValueError(f'PMN source {uid}: HOW/index mismatch with training memory')
            sources.append({'uid':uid,'summary':summary,'how':how,
                            'terminal_train_item_id':next_item,
                            'terminal_window_index':int(terminal_idx),
                            'matched_prefix_end_positions':matching_ends})
    missing = sorted(expected - set(targets))
    if missing:
        raise ValueError(f'PMN summaries missing {len(missing)} targets; first={missing[:8]}')
    if len(sources) < 2:
        raise ValueError(f'PMN needs >=2 source users, found {len(sources)}, excluded={dict(bad)}')
    sources.sort(key=lambda x:x['uid'])
    return targets, sources, {'num_target_profiles':len(targets),'num_eligible_source_profiles':len(sources),
                               'excluded_source_profiles':dict(bad)}


def select_pmn_neighbours(encoder, targets: Dict[str, str], sources: List[Dict[str, Any]],
                          users: List[str]):
    # Embedding of textual preference summaries only. NO Tree, states or suffixes.
    texts = [targets[uid] for uid in users] + [s['summary'] for s in sources]
    mat = l2_normalize(encoder.encode(texts))
    q = mat[:len(users)]
    s = mat[len(users):]
    sims = q @ s.T
    neighbours = {}
    for i, uid in enumerate(users):
        row = sims[i].copy()
        for j, src in enumerate(sources):
            if src['uid'] == uid:
                row[j] = -np.inf
        winner = int(np.argmax(row))
        if not np.isfinite(row[winner]):
            raise ValueError(f'PMN {uid}: no eligible other training user')
        chosen = sources[winner]
        neighbours[uid] = {'neighbor_id':chosen['uid'], 'similarity':float(row[winner]),
                           'how':chosen['how'],
                           'continuation_train_item_id':chosen['terminal_train_item_id']}
    return neighbours


def _tree_global_how(tree):
    node = tree.nodes[tree.root_id]
    dist = node.get('smoothed_next_distribution') or node.get('raw_next_distribution') or []
    if not dist:
        raise ValueError('Tree ROOT next-state distribution empty')
    return how_only(tree.vocab.state_evidence(str(dist[0]['state_id']))), {
        'global_state_id':str(dist[0]['state_id']),
        'global_probability':float(dist[0].get('probability',0.0)),
        'root_support_user_count':int(node.get('support_user_count',0)),
    }



STAY_PROMPT_VERSION = 'last_state_order1_persistence_v1'
STAY_MODES = frozenset(('stay_only', 'tree_stay', 'tree_original_stay', 'recent3_tree_stay'))


def _full_weight_probability(node: Dict[str, Any], state_id: str) -> Optional[float]:
    """Return P(state_id) from the *full* stored collaborative weight vector.

    The serialized smoothed_next_distribution may be top-K truncated. In
    particular, it is incorrect to treat a missing state as probability 0.
    """
    weights = node.get('collaborative_next_weights')
    if not isinstance(weights, dict) or not weights:
        return None
    total = sum(float(v) for v in weights.values())
    if total <= 0:
        return None
    return float(weights.get(state_id, 0.0)) / total


def compute_order1_persistence(tree: ReverseBehaviorTree, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """P(next == B | last B), independent of any longer A,B suffix.

    IMPORTANT: never infer the last state from an older accepted behavior when
    the actual latest window is rejected. Never silently use ROOT as P(.|B).
    Reconstruct the FULL smoothed distribution (B node + ROOT) from weights so
    top-K truncation of serialized rows cannot corrupt the self-probability.
    """
    row = {
        'prompt_version': STAY_PROMPT_VERSION,
        'conditioning': 'last_observed_behavior_state_only',
        'order': 1,
        'available': False,
        'last_state_id': None,
        'probability_next_same_behavior_state': None,
        'raw_probability': None,
        'root_probability': None,
        'smoothing_lambda': None,
        'support_user_count': None,
        'reason': None,
    }
    if not mappings:
        row['reason'] = 'no_last_behavior_mapping'
        return row
    last = mappings[-1]
    if not bool(last.get('assignment_accepted')):
        row['reason'] = 'latest_behavior_state_mapping_rejected'
        return row
    sid = str(last['state_id'])
    row['last_state_id'] = sid
    root = tree.nodes[tree.root_id]
    next_node_id = root.get('children', {}).get(sid)
    if next_node_id is None or str(next_node_id) not in tree.nodes:
        row['reason'] = 'order1_context_not_observed_in_training'
        return row
    node = tree.nodes[str(next_node_id)]
    if int(node.get('context_order', -1)) != 1 or [str(x) for x in node.get('context_states', [])] != [sid]:
        raise ValueError('Corrupt Tree order-1 context: last-state node mismatch')
    p_raw = _full_weight_probability(node, sid)
    p_root = _full_weight_probability(root, sid)
    if p_raw is None or p_root is None:
        row['reason'] = 'full_collaborative_weights_unavailable'
        return row
    lam = float(node.get('smoothing_lambda', 1.0))
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f'Invalid Tree smoothing lambda={lam} for {sid}')
    p_stay = lam * p_raw + (1.0 - lam) * p_root
    if not -1e-9 <= p_stay <= 1+1e-9:
        raise ValueError(f'Invalid computed P(stay|{sid})={p_stay}')
    row.update({
        'available': True,
        'probability_next_same_behavior_state': float(min(1, max(0, p_stay))),
        'raw_probability': float(p_raw),
        'root_probability': float(p_root),
        'smoothing_lambda': float(lam),
        'support_user_count': int(node.get('support_user_count', 0)),
        'reason': 'order1_smoothed_full_collaborative_weights',
    })
    return row


def append_order1_persistence_prompt(prompt: str, persistence: Dict[str, Any]) -> str:
    """Add one independent scalar without changing the original HOW prediction."""
    header = '4) LAST-BEHAVIOR PERSISTENCE — order-1, last observed HOW state only'
    if persistence.get('available'):
        value = f"{float(persistence['probability_next_same_behavior_state']):.4f}"
        block = (
            header + '\n'
            + 'P(next behavior state = latest behavior state | latest behavior state) = ' + value + '\n'
            + 'This estimates persistence of the same abstract HOW pattern, not repetition of an item.\n'
            + 'The estimate uses only the latest behavior state (order-1), independent of the\n'
            + 'longer-context next-behavior prediction above. Treat it as a soft calibration\n'
            + 'signal for continuity versus change, not as a guaranteed outcome.\n'
        )
    else:
        block = (
            header + '\n'
            + 'Not estimable from the frozen Tree for this user; do not assume zero\n'
            + 'persistence or substitute a ROOT-only probability.\n'
        )
    anchor = 'CANDIDATE ITEMS\n'
    if prompt.count(anchor) != 1:
        raise AssertionError('Expected exactly one candidate block in bt4rec prompt')
    return prompt.replace(anchor, block + '\n' + anchor, 1)

def mode_prompt(mode: str, item: Dict[str, Any]):
    h,c,latest = item['history_items'],item['candidate_items'],item['latest_behavior']
    if mode == 'tree_original':
        # Byte-for-byte same user-side Tree prompt as original inference_bt4rec.
        return build_rank_prompt(history_items=h, candidate_items=c, latest_behavior=latest,
                                 predicted_transition=item['predicted_transition'])
    if mode in STAY_MODES:
        if mode == 'tree_original_stay':
            base = build_rank_prompt(history_items=h, candidate_items=c,
                                     latest_behavior=latest,
                                     predicted_transition=item['predicted_transition'])
        elif mode == 'recent3_tree_stay':
            base = controlled_prompt(h,c,latest,item['hows']['tree'],recent3=item['recent3_behaviors'])
        elif mode == 'tree_stay':
            base = controlled_prompt(h,c,latest,item['hows']['tree'])
        else:  # stay_only: latest WHAT + persistence, without any next-HOW guidance.
            base = controlled_prompt(h,c,latest,None)
        return append_order1_persistence_prompt(base, item['last_state_persistence'])
    if mode == 'native':
        return build_rank_prompt(history_items=h, candidate_items=c,
                                 latest_behavior=None, predicted_transition=None)
    if mode == 'recent3':
        return controlled_prompt(h,c,latest,None,recent3=item['recent3_behaviors'])
    if mode == 'recent3_tree':
        return controlled_prompt(h,c,latest,item['hows']['tree'],recent3=item['recent3_behaviors'])
    return controlled_prompt(h,c,latest,item['hows'].get(mode))


def _resolve_source_files(args, modes):
    if args.train_behaviors is None and 'random_how' in modes:
        args.train_behaviors = str(Path(args.precomputed_behaviors).parent/'train_shared.jsonl')
    if args.pmn_summaries is None and 'pmn' in modes:
        args.pmn_summaries = str(Path(args.precomputed_behaviors).parent/'pmn_preference_summaries_gemma_v4.jsonl')
    if args.oracle_behaviors is None and 'oracle' in modes:
        args.oracle_behaviors = str(Path(args.precomputed_behaviors).parent/'target_user_oracle_protocol_seed42.jsonl')
    if args.source_user_ids_file is None:
        for guess in (Path(args.data_root)/args.dataset/'protocol'/'train_users_1000.json',
                      Path(args.data_root)/args.dataset/'train_users_1000.json'):
            if guess.is_file():
                args.source_user_ids_file = str(guess)
                break
    used = [('train-behaviors',args.train_behaviors,'random_how'),
            ('pmn-summaries',args.pmn_summaries,'pmn'),
            ('oracle-behaviors',args.oracle_behaviors,'oracle')]
    for name,path,mode in used:
        if mode in modes and (not path or not Path(path).is_file()):
            raise FileNotFoundError(f'--{name} required for {mode}: {path}')
    if 'random_how' in modes and Path(args.train_behaviors).resolve()==Path(args.precomputed_behaviors).resolve():
        raise ValueError('Random HOW training stream MUST differ from target test behavior cache')


def _audit_profile_source_restrictions(args, users, sequences, modes):
    allowed = set(load_user_ids_file(args.source_user_ids_file)) if args.source_user_ids_file else None
    if allowed is not None and not allowed:
        raise ValueError('Source cohort file empty')
    full_eval = set(load_user_ids_file(args.user_ids_file))
    forbidden = full_eval if args.exclude_all_eval_sources else set()
    if 'random_how' in modes or 'pmn' in modes:
        print('[SOURCE] train_cohort=', len(allowed) if allowed is not None else 'unrestricted',
              ' eval_excluded=',len(forbidden), ' source/eval overlap prohibited self-match always',flush=True)
    return allowed, forbidden


def _manifest_checkpoint(args, modes, users):
    files = {
        'candidate_file':args.candidate_file, 'behavior_cache':args.precomputed_behaviors,
        'tree_json':args.tree_json, 'states_json':args.states_json,
        'sequences':args.sequences, 'user_ids_file':args.user_ids_file,
        'train_behaviors':args.train_behaviors if 'random_how' in modes else None,
        'pmn_summaries':args.pmn_summaries if 'pmn' in modes else None,
        'oracle_behaviors':args.oracle_behaviors if 'oracle' in modes else None,
        'source_user_ids_file':args.source_user_ids_file,
    }
    return {'schema':SCHEMA_VERSION,'prompt_version':RANK_PROMPT_VERSION,'dataset':args.dataset,
            'users':users,'modes':modes, 'history_size':args.history_size,
            'tree_max_query_order':args.tree_max_query_order,'top_next':args.top_next,
            'model':args.model,'pmn_embedding_model':args.pmn_embedding_model,
            'llm_batch_size':args.llm_batch_size,
            'rank_max_new_tokens':args.rank_max_new_tokens,'temperature':args.temperature,
            'candidate_size':args.candidate_size,'seed':args.seed,
            'max_behaviors':args.max_behaviors,'respect_mapping_filter':args.respect_mapping_filter,
            'mapping_min_similarity':args.mapping_min_similarity,'mapping_min_margin':args.mapping_min_margin,
            'cluster_text_field':args.cluster_text_field,'embedding_model':args.embedding_model,
            'embedding_batch_size':args.embedding_batch_size,'max_seq_length':args.max_seq_length,
            'load_in_4bit':args.load_in_4bit,'strict_cache_history':args.strict_cache_history,
            'exclude_all_eval_sources':args.exclude_all_eval_sources,
            'require_user_disjoint_tree':args.require_user_disjoint_tree,
            'stay_prompt_version':STAY_PROMPT_VERSION if set(modes) & STAY_MODES else None,
            'files':{k:str(Path(v).resolve()) if v else None for k,v in files.items()},
            'external_file_hashes':{
                k:_file_sha256(files[k]) for k in ('train_behaviors','pmn_summaries',
                    'oracle_behaviors','source_user_ids_file') if files[k] is not None},
            }


def main():
    a = resolve_inputs(parse_args())
    modes = requested_modes(a.modes)
    set_seed(a.seed)
    _resolve_source_files(a,modes)
    items = normalize_items(load_json(a.items))
    sequences = {str(k):v for k,v in load_json(a.sequences).items()}
    test_cache = load_precomputed_behaviors(a.precomputed_behaviors)
    frozen = load_candidate_file(a.candidate_file)
    requested = load_user_ids_file(a.user_ids_file)
    if not requested or len(requested)!=len(set(requested)):
        raise ValueError('Frozen eval users list missing or duplicated')
    users = requested[a.start_user:]
    if a.max_users>0:
        users=users[:a.max_users]
    if not users:
        raise ValueError('No selected users')
    for uid in users:
        if uid not in sequences or uid not in frozen or uid not in test_cache:
            raise ValueError(f'{uid} missing sequences/candidates/test behaviors')
    prov = validate_tree_provenance(a,users)
    if a.require_user_disjoint_tree and prov.get('user_disjoint_tree_cohort') is not True:
        raise ValueError(f'--require-user-disjoint-tree failed: {prov}')
    audit = validate_frozen_users(users,sequences,frozen,test_cache,items,
                                  a.candidate_size,a.strict_cache_history)
    train_allowed, forbidden_sources = _audit_profile_source_restrictions(a,users,sequences,modes)
    train_how = (read_train_how(a.train_behaviors,train_allowed,forbidden_sources)
                 if 'random_how' in modes else None)
    if 'random_how' in modes and not any(u not in users for u in train_how):
        raise ValueError('Random HOW lacks distinct training source')
    oracles = load_oracle(a.oracle_behaviors,users,sequences) if 'oracle' in modes else {}
    pmn_targets,pmn_sources,pmn_audit = ({},[],{})
    if 'pmn' in modes:
        pmn_targets,pmn_sources,pmn_audit = load_pmn_profiles(
            a.pmn_summaries,a.dataset,users,sequences,a.history_size,
            train_allowed,forbidden_sources,a.allow_debug_pmn,train_how)

    info = {'dataset':a.dataset,'num_users':len(users),'candidate_size':a.candidate_size,
            'history_size':a.history_size,'modes':modes,'tree_provenance':prov,
            'frozen':audit,'pmn':pmn_audit,
            'num_random_training_users':len(train_how) if train_how else None,
            'oracle_cache_loaded':len(oracles),'all_eval_sources_excluded':a.exclude_all_eval_sources}
    print('[ABLATION VALIDATION]',json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    if a.validate_only:
        return

    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    order_tag = 'fulltree' if a.tree_max_query_order==0 else f'order{a.tree_max_query_order}'
    run_dir=Path(a.run_dir) if a.run_dir else Path(a.output_root)/(f'{a.dataset}_{len(users)}_{a.history_size}_{order_tag}_ablation_{stamp}')
    if a.resume and not a.run_dir:
        raise ValueError('--resume requires --run-dir')
    if run_dir.exists() and not a.resume:
        raise FileExistsError(f'Output run dir exists: {run_dir}')
    run_dir.mkdir(parents=True,exist_ok=True)
    manifest=run_dir/'run_manifest.json'
    checkpoints=_manifest_checkpoint(a,modes,users)
    if a.resume:
        if not manifest.is_file() or load_json(manifest).get('checkpoints')!=checkpoints:
            raise ValueError('Cannot resume: configuration/file hashes changed')
    else:
        dump_json(manifest,{'checkpoints':checkpoints,'config':vars(a),'validation':info,
                            'created_at_sydney':stamp,'description':'controlled bt4rec behavior guidance ablation'})
    path=run_dir/'rankings.jsonl'
    done = processed_users_from_jsonl(path) if a.resume else set()
    pending=[uid for uid in users if uid not in done]
    if not pending:
        # The previous summary is left in place instead of overwritten.
        print('[RESUME] No new users; existing outputs preserved',flush=True)
        return
    print(f'[RUN] {run_dir} | users={len(users)} pending={len(pending)} modes={modes}',flush=True)

    behaviors_by_user = {}
    for uid in pending:
        all_b=test_cache[uid]['generated_behaviors']
        behaviors_by_user[uid]=(all_b[-a.max_behaviors:] if a.max_behaviors>0 else all_b)
    vocab=ClusterStateVocabulary(states_json=a.states_json,state_embeddings_npy=a.state_embeddings,
            embedding_model=a.embedding_model,embedding_device=a.embedding_device,
            embedding_batch_size=a.embedding_batch_size,cluster_text_field=a.cluster_text_field,
            min_similarity=a.mapping_min_similarity,min_margin=a.mapping_min_margin,
            respect_mapping_filter=a.respect_mapping_filter)
    try:
        mapped=vocab.map_users(behaviors_by_user)
        tree=ReverseBehaviorTree(a.tree_json,vocab)
        global_how,global_audit=_tree_global_how(tree)
        neighbours={}
        if 'pmn' in modes:
            # PMN embeddings use a fixed pretrained model, never Tree centroids.
            # Reuse the already-loaded instance ONLY when the model name matches.
            if vocab.encoder.model_name == a.pmn_embedding_model:
                neighbours=select_pmn_neighbours(vocab.encoder,pmn_targets,pmn_sources,pending)
            else:
                pmn_encoder=QwenEncoder(a.pmn_embedding_model,a.embedding_device,a.embedding_batch_size)
                try:
                    neighbours=select_pmn_neighbours(pmn_encoder,pmn_targets,pmn_sources,pending)
                finally:
                    pmn_encoder.release()
        prepared=[]
        for uid in tqdm(pending,desc='Prepare controlled guidance',unit='user'):
            seq=sequences[uid]
            latest=latest_behavior_view(test_cache[uid]['generated_behaviors'])
            recent3=recent3_behavior_views(test_cache[uid]['generated_behaviors'])
            suffix=latest_accepted_state_suffix(mapped.get(uid,[]))
            tr=tree.query(suffix,top_next=a.top_next,max_query_order=a.tree_max_query_order)
            predicted=tr.get('predicted_next_states',[])
            if not predicted and any(m in modes for m in ('tree','tree_original','recent3_tree',
                                                     'tree_stay','tree_original_stay','recent3_tree_stay')):
                raise ValueError(f'{uid}: Tree query yielded no continuation')
            cand_ids,targets=get_candidates_for_user(uid,seq,{},frozen,a.seed)
            if len(targets)!=1:
                raise ValueError(f'{uid}: frozen protocol expects one held-out test target')
            hist=[get_item_info(x,items) for x in as_str_list(seq.get('train'))[-a.history_size:]]
            cmeta=[get_item_info(x,items) for x in cand_ids]
            hows={}
            metadata={}
            if 'latest_how' in modes:
                hows['latest_how']=how_only(test_cache[uid]['generated_behaviors'][-1])
            if 'random_how' in modes:
                hows['random_how'],metadata['random_how']=_choose_random_how(train_how,uid,a.seed)
            if 'global_continuation' in modes:
                hows['global_continuation']=global_how
                metadata['global_continuation']=global_audit
            if any(m in modes for m in ('tree','recent3_tree','tree_stay','recent3_tree_stay')):
                hows['tree']=how_only(predicted[0])
                metadata['tree']={'state_id':predicted[0]['state_id'],
                                  'probability':predicted[0].get('probability'),
                                  'support_user_count':predicted[0].get('support_user_count')}
            if 'pmn' in modes:
                hows['pmn']=neighbours[uid]['how']
                metadata['pmn']={k:v for k,v in neighbours[uid].items() if k!='how'}
            if 'oracle' in modes:
                hows['oracle']=oracles[uid]
                metadata['oracle']={'future_aware':True,'diagnostic_only':True}
            stay = compute_order1_persistence(tree,mapped.get(uid,[])) if set(modes) & STAY_MODES else None
            rec={'user_id':uid,'history_items':hist,'candidate_items':cmeta,
                 'candidate_ids':cand_ids,'targets':targets,
                 'latest_behavior':latest,'recent3_behaviors':recent3,'hows':hows,
                 'guidance_metadata':metadata,'tree_result':tr,
                 'state_mappings':mapped.get(uid,[]),'tree_query_state_suffix':suffix,
                 'predicted_transition':predicted, 'last_state_persistence':stay}
            rec['prompts']={m:mode_prompt(m,rec) for m in modes}
            prepared.append(rec)
        # All prompts generated BEFORE any ranker inference. Their modalities are inspectable.
        selected=next((x for x in prepared if x['user_id']==a.prompt_example_user),prepared[0])
        text=['Dataset: '+a.dataset,'User: '+selected['user_id'],
              'System prompt: You are a recommendation ranking system. Return only the requested candidate-label ranking.']
        for mode,p in selected['prompts'].items():
            text.extend(['','='*18+f' {mode} '+'='*18,p])
        (run_dir/'prompt_example.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
        dump_json(run_dir/'prompt_example.json',{'dataset':a.dataset,'user_id':selected['user_id'],
               'candidate_label_map':{f'C{i:02d}':x for i,x in enumerate(selected['candidate_ids'],1)},
               'latest_behavior':selected['latest_behavior'],
               'guidance':selected['hows'],'last_state_persistence':selected['last_state_persistence'],
               'prompts':selected['prompts']})
        if a.save_all_prompts:
            with (run_dir/'prompts_all.jsonl').open('w',encoding='utf8') as fo:
                for rec in prepared:
                    fo.write(json.dumps({'user_id':rec['user_id'],'prompts':rec['prompts']},ensure_ascii=False)+'\n')
        if 'pmn' in modes:
            with (run_dir/'pmn_neighbors.jsonl').open('w',encoding='utf8') as fo:
                for rec in prepared:
                    fo.write(json.dumps({'user_id':rec['user_id'],
                           **rec['guidance_metadata']['pmn']},ensure_ascii=False)+'\n')
    finally:
        vocab.release_encoder()
    if a.prompts_only:
        print(f'[PROMPTS-ONLY] {run_dir}/prompt_example.txt; LLM reranking skipped')
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ranker=LocalQwenRanker(a.model,a.max_seq_length,a.load_in_4bit,
                           a.rank_max_new_tokens,a.temperature,a.seed)
    ranked_by_mode={}
    for mode in modes:
        print(f'[RANK] {mode}: {len(prepared)} users; batch={a.llm_batch_size}',flush=True)
        jobs=[{'prebuilt_prompt':rec['prompts'][mode],'candidate_items':rec['candidate_items']}
              for rec in prepared]
        ranked_by_mode[mode]=ranker.rank_jobs(jobs,batch_size=a.llm_batch_size,
                                              call_type=mode,max_retries=0)
    for i,rec in enumerate(tqdm(prepared,desc='Save ablation rankings',unit='user')):
        ranks={}
        for mode in modes:
            ordered,meta=ranked_by_mode[mode][i]
            ranks[mode]={'ranked_item_ids':ordered,'target_rank':target_rank(ordered,rec['targets'][0]),**meta}
        row={'schema_version':SCHEMA_VERSION,'rank_prompt_version':RANK_PROMPT_VERSION,
             'run_dir':str(run_dir),'user_id':rec['user_id'],
             'ground_truth_item_ids':rec['targets'],'candidate_item_ids':rec['candidate_ids'],
             'candidate_label_map':{f'C{k:02d}':iid for k,iid in enumerate(rec['candidate_ids'],1)},
             'history_items':rec['history_items'],
             'latest_observed_behavior':rec['latest_behavior'],
             'recent3_observed_behaviors':rec['recent3_behaviors'],
             'guidance':rec['hows'],'guidance_audit':rec['guidance_metadata'],
             'tree_result':rec['tree_result'],'state_mappings':rec['state_mappings'],
             'tree_query_state_suffix':rec['tree_query_state_suffix'],
             'last_state_persistence':rec['last_state_persistence'],'rankings':ranks}
        append_jsonl(path,row)

    rows=[r for r in read_jsonl(path) if r.get('schema_version')==SCHEMA_VERSION
          and r.get('rank_prompt_version')==RANK_PROMPT_VERSION]
    counts=Counter(str(r.get('user_id')) for r in rows)
    if any(v!=1 for v in counts.values()) or set(counts)!=set(users):
        raise ValueError(f'Unexpected completed cohort/multiple rows; users={len(counts)}, expected={len(users)}')
    metric_all={mode:aggregate_mode(rows,mode,False) for mode in modes}
    metric_complete={mode:aggregate_mode(rows,mode,True) for mode in modes}
    parse_quality={}
    for mode in modes:
        recs=[r.get('rankings',{}).get(mode) for r in rows]
        good=[r for r in recs if isinstance(r,dict)]
        parse_quality[mode]={'rows':len(good),'parse_complete_count':sum(bool(r.get('parse_complete')) for r in good),
                             'parse_complete_rate':sum(bool(r.get('parse_complete')) for r in good)/len(good) if good else None}
    deltas={mode:metric_gain(metric_all[mode],metric_all.get('latest_only'))
            for mode in modes if mode!='latest_only'} if 'latest_only' in modes else {}
    if 'pmn' in modes and 'tree' in modes:
        deltas['tree_minus_pmn']=metric_gain(metric_all['tree'],metric_all['pmn'])
    if 'tree_stay' in modes and 'tree' in modes:
        deltas['tree_stay_minus_tree']=metric_gain(metric_all['tree_stay'],metric_all['tree'])
    if 'tree_original_stay' in modes and 'tree_original' in modes:
        deltas['tree_original_stay_minus_tree_original']=metric_gain(metric_all['tree_original_stay'],metric_all['tree_original'])
    if 'recent3_tree_stay' in modes and 'recent3_tree' in modes:
        deltas['recent3_tree_stay_minus_recent3_tree']=metric_gain(metric_all['recent3_tree_stay'],metric_all['recent3_tree'])
    if 'recent3_tree' in modes and 'recent3' in modes:
        deltas['recent3_tree_minus_recent3']=metric_gain(metric_all['recent3_tree'],metric_all['recent3'])
    stay_rows=[r.get('last_state_persistence') for r in rows]
    stay_values=[float(x['probability_next_same_behavior_state']) for x in stay_rows
                 if isinstance(x,dict) and x.get('available')]
    stay_audit=({'order':1,'prompt_version':STAY_PROMPT_VERSION,
                 'available_count':len(stay_values),
                 'unavailable_count':len(rows)-len(stay_values),
                 'mean_probability':float(np.mean(stay_values)) if stay_values else None,
                 'median_probability':float(np.median(stay_values)) if stay_values else None,
                 'unavailable_reasons':dict(Counter(x.get('reason') for x in stay_rows
                     if isinstance(x,dict) and not x.get('available')))}
                if set(modes) & STAY_MODES else None)
    summary={'schema_version':SCHEMA_VERSION,'rank_prompt_version':RANK_PROMPT_VERSION,
             'dataset':a.dataset,'modes':modes,'num_users':len(users),
             'metrics':metric_all,'metrics_complete_only':metric_complete,
             'differences':deltas,'parse_quality':parse_quality,
             'tree_context_order_counts':dict(Counter(str(r.get('tree_result',{}).get('matched_order')) for r in rows)),
             'tree_root_fallback_count':sum(r.get('tree_result',{}).get('matched_order')==0 for r in rows),
             'ranking_llm':ranker.stats(), 'frozen_protocol':audit,'tree_provenance':prov,
             'last_state_persistence_audit':stay_audit,
             'pmnc':pmn_audit,'run_dir':str(run_dir),'config':vars(a),
             'warning':'ORACLE uses held-out test information; diagnostic only, not deployable. '
                       'Tree_original prompt differs from common controlled template; report separately.'}
    dump_json(run_dir/'summary.json',summary)
    print('[RESULTS]',json.dumps({'metrics':metric_all,'differences':deltas,
                                   'parse_quality':parse_quality},ensure_ascii=False,indent=2))
    print(f'Rankings: {path}\nSummary: {run_dir/"summary.json"}')


if __name__ == '__main__':
    main()
