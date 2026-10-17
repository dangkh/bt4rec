from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def item_ids(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for value_item in value:
            out.extend(item_ids(value_item, field))
        return out
    if isinstance(value, dict):
        for key in ('item_ids', 'items', 'item_id', 'item'):
            if key in value:
                return item_ids(value[key], field + '.' + key)
        raise ValueError(f'{field}: cannot interpret item ID dict {list(value)[:8]}')
    raise ValueError(f'{field}: unsupported item-ID type {type(value).__name__}')


def read_user_ids(path: Path) -> list[str]:
    content = path.read_text(encoding='utf-8').strip()
    if not content:
        return []
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return list(dict.fromkeys(line.strip() for line in content.splitlines() if line.strip()))
    if isinstance(obj, list):
        return list(dict.fromkeys(str(x) for x in obj))
    if isinstance(obj, dict):
        if isinstance(obj.get('users'), list):
            return list(dict.fromkeys(str(x) for x in obj['users']))
        return list(dict.fromkeys(str(x) for x in obj.keys()))
    raise ValueError(f'User file must be list/dict/lines: {path}')


def valid_title(iid: str, items: dict[str, Any]) -> bool:
    meta = items.get(iid)
    return isinstance(meta, dict) and isinstance(meta.get('title'), str) and bool(meta['title'].strip())


def valid_category(iid: str, items: dict[str, Any]) -> bool:
    meta = items.get(iid)
    if not isinstance(meta, dict):
        return False
    val = meta.get('main_cat') or meta.get('category') or meta.get('categories')
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (list, tuple)):
        return bool(val) and any(str(v).strip() for v in val if v is not None)
    return False


def get_val(seq: dict[str, Any], uid: str) -> list[str]:
    values = [item_ids(seq[k], f'{uid}.{k}') for k in ('val', 'valid', 'validation', 'validation_item_ids', 'val_item_ids') if k in seq]
    nonempty = [x for x in values if x]
    if len(nonempty) > 1 and any(x != nonempty[0] for x in nonempty[1:]):
        raise ValueError(f'{uid}: contradictory validation fields')
    return nonempty[0] if nonempty else []


def negative_ids(entry: Any, uid: str) -> list[str]:
    # user_negatives.json usually has {user_id: {"test_neg": [item_ids...]}}
    if isinstance(entry, dict):
        for key in ('test_neg', 'test_negatives', 'negatives', 'negative_items'):
            if key in entry:
                return item_ids(entry[key], f'negative:{uid}.{key}')
        raise ValueError(f'user_negatives[{uid}] has no test_neg/negatives: keys {list(entry)[:8]}')
    return item_ids(entry, f'negative:{uid}')


def seed_for_user(seed: int, uid: str) -> int:
    digest = hashlib.sha256(f'{seed}|{uid}|title-safe-v1'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def parse_sizes(text: str) -> list[int]:
    sizes = sorted(set(int(s.strip()) for s in text.split(',') if s.strip()))
    if not sizes or any(x < 1 for x in sizes):
        raise ValueError('--sizes should be comma separated positive integers')
    return sizes


def save_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite existing file: {path}; pass --overwrite to allow')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write('\n')
