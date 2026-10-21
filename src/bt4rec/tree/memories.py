from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from bt4rec.common.text import clean_text
from bt4rec.taxonomy import (
    behavior_bucket_key,
    behavior_profile,
    choose_cluster_text,
    get_trajectory_signature,
)

REQUIRED_FIELDS = {"user_id", "window_index"}


DUAL_REQUIRED_FIELDS = {
    "recommendation_behavior",
    "mechanism",
    "scope",
    "direction",
}


DUAL_SCHEMA_NAME = "dual_view_semantic_plus_trajectory"


def read_user_id_set(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return {str(x) for x in obj}
        if isinstance(obj, dict):
            if "users" in obj and isinstance(obj["users"], list):
                return {str(x) for x in obj["users"]}
            return {str(x) for x in obj.keys()}
    except json.JSONDecodeError:
        pass
    return {line.strip() for line in text.splitlines() if line.strip()}


def make_memory_id(row: Dict[str, Any], row_idx: int) -> str:
    if row.get("memory_id") is not None:
        return str(row["memory_id"])
    if row.get("precompute_id") is not None:
        return f"M{row['precompute_id']}"
    if row.get("thought_id") is not None:
        return f"T{row['thought_id']}"
    return f"MROW{row_idx}"


def load_memories(
    path: str,
    cluster_text_field: str,
    include_users: Optional[Set[str]],
    exclude_users: Optional[Set[str]],
    skip_invalid: bool,
) -> List[Dict[str, Any]]:
    """
    Load already-precomputed dual-view memories.

    No LLM structuring is performed here.
    """
    rows: List[Dict[str, Any]] = []
    skipped_invalid_schema = 0
    skipped_parse_fail = 0

    with open(path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue

            try:
                x = json.loads(line)
            except Exception as e:
                if skip_invalid:
                    print(
                        f"[WARN] invalid JSON line {row_idx + 1}: {e}",
                        file=sys.stderr,
                    )
                    continue
                raise

            missing = REQUIRED_FIELDS - set(x)
            if missing:
                if skip_invalid:
                    print(
                        f"[WARN] line {row_idx + 1} missing "
                        f"{sorted(missing)}; skipped",
                        file=sys.stderr,
                    )
                    continue
                raise ValueError(
                    f"line {row_idx + 1} missing {sorted(missing)}"
                )

            if x.get("parse_ok") is False:
                skipped_parse_fail += 1
                if skip_invalid:
                    continue
                raise ValueError(
                    f"line {row_idx + 1}: parse_ok=False"
                )

            uid = str(x["user_id"])
            if include_users is not None and uid not in include_users:
                continue
            if exclude_users is not None and uid in exclude_users:
                continue

            # Accept the explicit trajectory field or its compatibility alias.
            trajectory_signature = get_trajectory_signature(x)

            missing_dual = [
                field
                for field in DUAL_REQUIRED_FIELDS
                if not clean_text(x.get(field))
            ]
            if not trajectory_signature:
                missing_dual.append("trajectory_signature/behavior_signature")

            if missing_dual:
                skipped_invalid_schema += 1
                msg = (
                    f"line {row_idx + 1} is not a valid dual-view memory; "
                    f"missing/empty={missing_dual}"
                )
                if skip_invalid:
                    print(f"[WARN] {msg}; skipped", file=sys.stderr)
                    continue
                raise ValueError(msg)

            y = dict(x)
            y["memory_id"] = make_memory_id(x, row_idx)
            y["user_id"] = uid
            y["window_index"] = int(x["window_index"])

            # Normalize dual fields without overwriting raw semantic evidence.
            y["trajectory_signature"] = trajectory_signature
            y["behavior_signature"] = trajectory_signature
            y["recommendation_behavior"] = clean_text(
                x.get("recommendation_behavior")
            )

            semantic_focus = x.get("semantic_focus")
            if semantic_focus is None:
                semantic_focus = x.get("keywords", [])
            if not isinstance(semantic_focus, list):
                semantic_focus = [str(semantic_focus)] if semantic_focus else []
            y["semantic_focus"] = [
                clean_text(v)
                for v in semantic_focus
                if clean_text(v)
            ][:12]

            prof = behavior_profile(y)
            y["mechanism"] = prof["mechanism"]
            y["scope"] = prof["scope"]
            y["direction"] = prof["direction"]
            y["behavior_profile"] = prof
            y["behavior_bucket_key"] = list(
                behavior_bucket_key(y)
            )

            cluster_text = choose_cluster_text(
                y,
                cluster_text_field,
            )
            if not cluster_text:
                msg = (
                    f"line {row_idx + 1}: empty cluster text "
                    f"for field={cluster_text_field}"
                )
                if skip_invalid:
                    print(f"[WARN] {msg}; skipped", file=sys.stderr)
                    continue
                raise ValueError(msg)

            y["cluster_text"] = cluster_text
            y["_row_idx"] = row_idx
            rows.append(y)

    if not rows:
        raise ValueError(
            "No valid dual-view memories found after filtering"
        )

    print(
        f"      skipped_parse_fail={skipped_parse_fail} "
        f"skipped_invalid_dual_schema={skipped_invalid_schema}"
    )

    return rows


def dual_schema_stats(
    memories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mechanism_counts = Counter(
        m["mechanism"] for m in memories
    )
    scope_counts = Counter(
        m["scope"] for m in memories
    )
    direction_counts = Counter(
        m["direction"] for m in memories
    )

    unknown_profile_count = sum(
        1
        for m in memories
        if "unknown" in behavior_bucket_key(m)
    )

    return {
        "schema": DUAL_SCHEMA_NAME,
        "num_memories": len(memories),
        "unknown_profile_count": int(unknown_profile_count),
        "unknown_profile_ratio": float(
            unknown_profile_count / len(memories)
        ),
        "mechanism_counts": dict(
            mechanism_counts.most_common()
        ),
        "scope_counts": dict(
            scope_counts.most_common()
        ),
        "direction_counts": dict(
            direction_counts.most_common()
        ),
    }
