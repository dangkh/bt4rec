from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm.auto import tqdm

from bt4rec.inference.constants import RANK_PROMPT_VERSION, SCHEMA_VERSION


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


def fingerprint(parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def jsonl_dump(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: str, rows: list[dict], overwrite: bool,
                       progress_label: str = "Write JSONL") -> None:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {p}; supply --overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=progress_label, unit="record",
                        dynamic_ncols=True, mininterval=0.5):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)
