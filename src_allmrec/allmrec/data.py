from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import dump_json, mapping_fingerprint

USER_KEYS = ("user_id", "uid", "user", "reviewerID", "reviewer_id", "user_raw_id")
ITEM_KEYS = ("item_id", "iid", "item", "asin", "movie_id", "product_id", "id")
TRAIN_KEYS = ("train", "training", "train_items", "train_sequence", "train_history", "history")
TEST_KEYS = ("test", "test_item", "target_item", "target", "gt", "ground_truth", "positive_item")
NEG_KEYS = ("test_neg", "test_negs", "test_negative", "test_negatives", "negatives", "negative_items", "neg")
CAND_KEYS = ("candidates", "candidate_ids", "candidate_item_ids", "candidate_items", "test_candidates", "candidate_set")
CAND_TARGET_KEYS = ("target", "targets", "target_item_id", "target_item_ids", "gt", "ground_truth", "positive_item", "test")
HIST_KEYS = ("train_history_item_ids", "history_item_ids", "train_history", "history", "input_history", "history_ids")


def _read_json_or_jsonl(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if not stripped:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def load_any(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    name = p.name.lower()
    if name.endswith((".pkl", ".pickle", ".pkl.gz", ".pickle.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(p, "rb") as f:
            return pickle.load(f)
    return _read_json_or_jsonl(p)


def _to_id(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(int(x))
    return str(x)


def _as_id_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        for key in ("items", "item_ids", "sequence", "ids", "values"):
            if key in x:
                return _as_id_list(x[key])
        # dict of position -> item
        try:
            pairs = sorted(x.items(), key=lambda kv: int(kv[0]))
            return [_to_id(v) for _, v in pairs]
        except Exception:
            return []
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, dict):
                item = _first(v, ITEM_KEYS)
                if item is None:
                    for k in ("item_id", "asin", "id"):
                        if k in v:
                            item = v[k]
                            break
                if item is not None:
                    out.append(_to_id(item))
            else:
                out.append(_to_id(v))
        return [v for v in out if v != ""]
    return [_to_id(x)] if _to_id(x) else []


def _first(d: Any, keys: Iterable[str]) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _rows_by_user(obj: Any) -> tuple[list[str], dict[str, Any]]:
    """Return user order and a user->payload mapping while preserving source order."""
    order: list[str] = []
    out: dict[str, Any] = {}
    if isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            uid = _first(row, USER_KEYS)
            if uid is None:
                continue
            uid = _to_id(uid)
            if uid not in out:
                order.append(uid)
            out[uid] = row
        return order, out
    if isinstance(obj, dict):
        # Some files wrap rows in a top-level key.
        for wrap in ("users", "data", "records", "behaviors", "test_users"):
            if wrap in obj and isinstance(obj[wrap], (list, dict)):
                return _rows_by_user(obj[wrap])
        for k, v in obj.items():
            # If payload itself has an explicit user id, respect it; otherwise key is user id.
            uid = _first(v, USER_KEYS) if isinstance(v, dict) else None
            uid = _to_id(uid if uid is not None else k)
            order.append(uid)
            out[uid] = v
        return order, out
    raise ValueError(f"Unsupported user file object type: {type(obj)}")


def _normalize_items(obj: Any) -> tuple[list[str], dict[str, dict[str, str]]]:
    items: dict[str, dict[str, str]] = {}
    order: list[str] = []

    def add(iid: Any, payload: Any) -> None:
        iid_s = _to_id(iid)
        if not iid_s:
            return
        if iid_s not in items:
            order.append(iid_s)
        if isinstance(payload, dict):
            title = payload.get("title") or payload.get("name") or payload.get("item_title") or ""
            cat = (
                payload.get("category")
                or payload.get("categories")            
                or ""
            )
            if isinstance(cat, (list, tuple)):
                cat = " | ".join(map(str, cat))
            items[iid_s] = {"title": str(title or ""), "category": str(cat or "")}
        else:
            items[iid_s] = {"title": str(payload or ""), "category": ""}

    if isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict):
                iid = _first(row, ITEM_KEYS)
                if iid is not None:
                    add(iid, row)
        return order, items

    if not isinstance(obj, dict):
        raise ValueError("items file must decode to dict or list")

    # Original A-LLMRec-like {title:{id:title}, description:{...}} or custom mappings.
    if isinstance(obj.get("title"), dict):
        title_map = obj.get("title", {})
        cat_map = obj.get("category", obj.get("categories", obj.get("genre", obj.get("genres", {}))))
        if not isinstance(cat_map, dict):
            cat_map = {}
        for iid, title in title_map.items():
            add(iid, {"title": title, "category": cat_map.get(iid, cat_map.get(str(iid), ""))})
        return order, items

    for wrap in ("items", "data", "records", "metadata"):
        if wrap in obj and isinstance(obj[wrap], (list, dict)):
            return _normalize_items(obj[wrap])

    for iid, payload in obj.items():
        add(iid, payload)
    return order, items


def _extract_train(seq_payload: Any, behavior_payload: Any) -> list[str]:
    if isinstance(seq_payload, dict):
        v = _first(seq_payload, TRAIN_KEYS)
        if v is not None:
            return _as_id_list(v)
    elif isinstance(seq_payload, (list, tuple)):
        # Only use a raw list as train if no stronger history exists in test_behavior.
        b = _first(behavior_payload, HIST_KEYS) if isinstance(behavior_payload, dict) else None
        if b is None:
            return _as_id_list(seq_payload)
    if isinstance(behavior_payload, dict):
        v = _first(behavior_payload, HIST_KEYS)
        if v is not None:
            return _as_id_list(v)
    return []


def _extract_gt(seq_payload: Any, behavior_payload: Any, neg_payload: Any) -> str:
    for src in (seq_payload, behavior_payload, neg_payload):
        if isinstance(src, dict):
            v = _first(src, TEST_KEYS)
            vals = _as_id_list(v)
            if vals:
                return vals[-1]
    return ""


def _extract_neg(neg_payload: Any) -> list[str]:
    if isinstance(neg_payload, dict):
        v = _first(neg_payload, NEG_KEYS)
        if v is not None:
            return _as_id_list(v)
        # If payload is only a dict wrapper with one list, use it.
        list_vals = [v for v in neg_payload.values() if isinstance(v, list)]
        if len(list_vals) == 1:
            return _as_id_list(list_vals[0])
    return _as_id_list(neg_payload)


def _extract_candidate_row(payload: Any) -> tuple[list[str], str]:
    """Extract the exact ordered candidate list and optional GT from the frozen file.

    Supported rows include:
      {"user_id": ..., "candidates": [...], "target": ...}
      {"uid": ..., "candidate_item_ids": [...], "target_item_ids": [...]}
      user_id -> [candidate_1, ..., candidate_20]
    The returned candidate order is never shuffled or otherwise modified.
    """
    if isinstance(payload, dict):
        cand_v = _first(payload, CAND_KEYS)
        candidates = _as_id_list(cand_v) if cand_v is not None else []

        # Some candidate files wrap the list in a generic field.
        if not candidates:
            for key in ("items", "item_ids", "sequence", "ids", "values"):
                if key in payload:
                    candidates = _as_id_list(payload[key])
                    if candidates:
                        break

        target_v = _first(payload, CAND_TARGET_KEYS)
        target_vals = _as_id_list(target_v)
        target = target_vals[-1] if target_vals else ""
        return candidates, target

    return _as_id_list(payload), ""


@dataclass
class UserExample:
    user_id: str
    train_raw: list[str]
    gt_raw: str
    negatives_raw: list[str]
    candidates_raw: list[str]
    train: list[int]
    gt: int
    negatives: list[int]
    candidates: list[int]


@dataclass
class ProtocolData:
    dataset: str
    users: list[UserExample]
    item2idx: dict[str, int]
    idx2item: list[str]
    item_meta: dict[str, dict[str, str]]
    fingerprint: str

    @property
    def num_items(self) -> int:
        return len(self.idx2item) - 1

    @property
    def user_ids(self) -> list[str]:
        return [u.user_id for u in self.users]

    def item_text_raw(self, raw_id: str, max_chars: int = 180) -> str:
        meta = self.item_meta.get(raw_id, {})
        title = str(meta.get("title", "") or "").strip()
        cat = str(meta.get("category", "") or "").strip()
        if not title:
            title = f"Item {raw_id}"
        text = title if not cat else f"{title} | {cat}"
        return " ".join(text.split())[:max_chars]

    def title_raw(self, raw_id: str, max_chars: int = 160) -> str:
        meta = self.item_meta.get(raw_id, {})
        title = str(meta.get("title", "") or "").strip()
        if not title:
            title = f"Item {raw_id}"
        return " ".join(title.split())[:max_chars]

    def raw_item(self, idx: int) -> str:
        return self.idx2item[idx]


def load_protocol(
    *,
    dataset: str,
    items_path: str,
    sequence_path: str,
    negative_path: str,
    test_behavior_path: str,
    candidate_path: str,
    num_users: int = 300,
    history_size: int = 10,
    num_negatives: int = 19,
    seed: int = 42,
    strict: bool = True,
) -> ProtocolData:
    items_obj = load_any(items_path)
    seq_obj = load_any(sequence_path)
    neg_obj = load_any(negative_path)
    beh_obj = load_any(test_behavior_path)
    cand_obj = load_any(candidate_path)

    item_order, item_meta = _normalize_items(items_obj)
    _, seq_map = _rows_by_user(seq_obj)
    _, neg_map = _rows_by_user(neg_obj)
    behavior_order, beh_map = _rows_by_user(beh_obj)
    _, cand_map = _rows_by_user(cand_obj)

    if len(behavior_order) < num_users:
        raise ValueError(f"test_behavior has {len(behavior_order)} users, need {num_users}")
    selected = behavior_order[:num_users]
    if strict and len(selected) != num_users:
        raise ValueError(f"Expected exactly {num_users} selected users, got {len(selected)}")

    raw_rows: list[tuple[str, list[str], str, list[str], list[str]]] = []
    seen_items = list(item_order)
    seen_set = set(seen_items)

    for uid in selected:
        seq_payload = seq_map.get(uid)
        neg_payload = neg_map.get(uid)
        beh_payload = beh_map.get(uid, {})
        cand_payload = cand_map.get(uid)
        if seq_payload is None:
            raise KeyError(f"User {uid} from test_behavior missing in user_sequence")
        if neg_payload is None:
            raise KeyError(f"User {uid} from test_behavior missing in user_negative")
        if cand_payload is None:
            raise KeyError(f"User {uid} from test_behavior missing in candidate_file")

        train_all = _extract_train(seq_payload, beh_payload)
        # `history_size` is a MAXIMUM history cutoff, not an exact-length
        # requirement.  Keep the same frozen evaluation users even when a
        # user has fewer than 10 available training interactions; never pad
        # with validation/test interactions because that would leak data.
        train = train_all[-history_size:]
        if len(train) < 2:
            raise ValueError(
                f"User {uid}: only {len(train)} train interactions after applying "
                f"history_size={history_size}; A-LLMRec Stage 1/2 needs at least 2"
            )

        gt = _extract_gt(seq_payload, beh_payload, neg_payload)
        if not gt:
            raise ValueError(f"User {uid}: cannot find test GT in user_sequence/test_behavior")
        if gt in train:
            raise ValueError(
                f"User {uid}: test GT {gt} appears in training history; this is test leakage"
            )

        negs = _extract_neg(neg_payload)
        # Remove GT if a source accidentally includes it in negatives, but do not silently alter length in strict mode.
        if gt in negs:
            if strict:
                raise ValueError(f"User {uid}: GT appears in test negatives")
            negs = [x for x in negs if x != gt]
        if len(negs) < num_negatives:
            raise ValueError(f"User {uid}: only {len(negs)} negatives; need {num_negatives}")
        negs = negs[:num_negatives]
        if strict and len(negs) != num_negatives:
            raise ValueError(f"User {uid}: negatives len {len(negs)} != {num_negatives}")

        # IMPORTANT: candidate order is authoritative and comes exactly from
        # user_sequences_candidates_seed42 (or the supplied frozen candidate file).
        # Never reconstruct or reshuffle candidates here.
        candidates, candidate_gt = _extract_candidate_row(cand_payload)
        if not candidates:
            raise ValueError(f"User {uid}: empty candidate row in candidate_file")
        if len(candidates) != num_negatives + 1:
            raise ValueError(
                f"User {uid}: frozen candidate len={len(candidates)}, "
                f"expected {num_negatives + 1}"
            )
        if candidate_gt and candidate_gt != gt:
            raise ValueError(
                f"User {uid}: candidate_file GT={candidate_gt} differs from "
                f"user_sequence/test_behavior GT={gt}"
            )
        if candidates.count(gt) != 1:
            raise ValueError(
                f"User {uid}: GT {gt} appears {candidates.count(gt)} times "
                "in frozen candidate list; expected exactly once"
            )
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"User {uid}: frozen candidate list contains duplicates")

        # The frozen candidate file must contain exactly the same 1 GT + 19
        # negatives as user_negative.  Only the order comes from candidate_file.
        expected_set = set([gt] + negs)
        if set(candidates) != expected_set:
            missing = sorted(expected_set - set(candidates))
            extra = sorted(set(candidates) - expected_set)
            raise ValueError(
                f"User {uid}: frozen candidate set differs from GT + test negatives; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

        for iid in train + [gt] + negs + candidates:
            if iid not in seen_set:
                seen_items.append(iid)
                seen_set.add(iid)
                item_meta.setdefault(iid, {"title": "", "category": ""})
        raw_rows.append((uid, train, gt, negs, candidates))

    # Stable mapping: metadata/source order first, then any newly observed ids.
    item2idx = {iid: i + 1 for i, iid in enumerate(seen_items)}
    idx2item = ["<PAD>"] + seen_items

    users: list[UserExample] = []
    for uid, train, gt, negs, candidates in raw_rows:
        users.append(
            UserExample(
                user_id=uid,
                train_raw=train,
                gt_raw=gt,
                negatives_raw=negs,
                candidates_raw=candidates,
                train=[item2idx[x] for x in train],
                gt=item2idx[gt],
                negatives=[item2idx[x] for x in negs],
                candidates=[item2idx[x] for x in candidates],
            )
        )

    fp = mapping_fingerprint(item2idx, selected)
    return ProtocolData(dataset, users, item2idx, idx2item, item_meta, fp)


def save_protocol_snapshot(data: ProtocolData, path: str) -> None:
    rows = []
    for u in data.users:
        rows.append(
            {
                "user_id": u.user_id,
                "train": u.train_raw,
                "gt": u.gt_raw,
                "negatives": u.negatives_raw,
                "candidates": u.candidates_raw,
                "gt_position": u.candidates_raw.index(u.gt_raw) + 1,
            }
        )
    dump_json(
        {
            "dataset": data.dataset,
            "num_users": len(data.users),
            "num_items": data.num_items,
            "fingerprint": data.fingerprint,
            "users": rows,
        },
        path,
    )
