from __future__ import annotations

from typing import Any, Dict, List


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


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    return [str(v) for v in x]
