from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from bt4rec.common.text import clean_text


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
