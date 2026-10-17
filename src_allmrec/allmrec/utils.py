from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def stable_int(text: str, seed: int = 42) -> int:
    h = hashlib.sha256(f"{seed}::{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(obj: Any, path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(rows: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mapping_fingerprint(item2idx: dict[str, int], user_ids: list[str]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(item2idx.items(), key=lambda x: x[1]):
        h.update(f"I\t{k}\t{v}\n".encode())
    for i, u in enumerate(user_ids, 1):
        h.update(f"U\t{u}\t{i}\n".encode())
    return h.hexdigest()


def get_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}) but torch.cuda.is_available() is False")
    return torch.device(device)
