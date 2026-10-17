from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from bt4rec.common.io import load_json
from bt4rec.common.text import clean_text
from bt4rec.data.behavior_cache import validate_behavior_cache_history
from bt4rec.data.items import as_str_list


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
