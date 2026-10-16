from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_ids(path: str | None, default: list[str]) -> list[str]:
    if not path:
        return default
    data = Path(path).read_text(encoding="utf-8").strip()
    try:
        value = json.loads(data)
        if isinstance(value, list):
            return list(dict.fromkeys(map(str, value)))
        if isinstance(value, dict):
            ids = value.get("users", list(value.keys()))
            if isinstance(ids, list):
                return list(dict.fromkeys(map(str, ids)))
    except json.JSONDecodeError:
        pass
    return list(dict.fromkeys(line.strip() for line in data.splitlines() if line.strip()))


def as_ids(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_info(items: dict, item_id: Any) -> dict[str, str] | None:
    key = str(item_id)
    info = items.get(key)
    if info is None:
        info = items.get(item_id)
    if not isinstance(info, dict):
        return None
    cat = info.get("main_cat") or info.get("category")
    if not cat:
        cats = info.get("categories")
        if isinstance(cats, list) and cats:
            cat = " > ".join(map(str, cats[0] if isinstance(cats[0], list) else cats))
    name = info.get("title")
    if not isinstance(name, str) or not name.strip():
        return None  # Avoid hallucinated "Item <id>" labels in Gemma evidence.
    return {
        "item_id": key,
        "item_name": name.strip(),
        "item_category": str(cat or "Unknown"),
        "action_type": "purchase",
    }


@dataclass
class Window:
    user_id: str
    items: list[dict[str, str]]
    positions: list[int]    # Absolute indices into concatenated train+val


def valid_history(items: dict, raw_ids: list[Any], start: int, end: int) -> list[tuple[int, dict]]:
    return [(i, info) for i in range(start, end)
            if (info := get_info(items, raw_ids[i])) is not None]


def chunk_windows(user_id: str, valid: list[tuple[int, dict]], size: int) -> list[Window]:
    """Match original training chunk/overlapping-last-window logic after fixing trailing missing metadata."""
    if not valid:
        return []
    slices = [valid[i:i+size] for i in range(0, len(valid) // size * size, size)]
    if len(valid) < size:
        slices = [valid]
    elif len(valid) % size:
        slices.append(valid[-size:])
    return [Window(user_id, [item for _, item in segment], [p for p, _ in segment])
            for segment in slices]


def make_train_windows(uid: str, train: list[Any], items: dict, cap: int, ws: int) -> list[Window]:
    first = max(0, len(train) - cap) if cap > 0 else 0
    return chunk_windows(uid, valid_history(items, train, first, len(train)), ws)


def window_key(w: Window) -> str:
    return json.dumps([w.user_id, w.items], sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def cache_key(uid: str, interaction_sequence: list[dict]) -> str:
    # Older output uses identical fields; avoid accidental reuse if metadata changed.
    return json.dumps([str(uid), interaction_sequence], sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def compact_test_window(w: Window) -> list[dict]:
    return [{"item_id": x["item_id"], "item": x["item_name"],
             "category": x["item_category"], "action": x["action_type"]}
            for x in w.items]


def make_prompt_summary(window: Window) -> list[dict]:
    return [{"item_id": x["item_id"], "item": x["item_name"],
             "category": x["item_category"], "action": x["action_type"]}
            for x in window.items]


def validate_selected_users(all_users: list[str], ids: list[str], sequences: dict) -> list[str]:
    ids = list(dict.fromkeys(ids))
    missing = [uid for uid in ids if uid not in sequences]
    if missing:
        raise ValueError(f"User IDs not found in sequences: {missing[:10]}")
    return ids
