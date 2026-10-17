#!/usr/bin/env python3
"""
i²Agent adapted to the CoMemTree evaluation protocol with local Unsloth Qwen-2.5 and SCORE ranking.

WHAT IS PRESERVED FROM i²Agent
------------------------------
The original i²Agent pipeline contains:
  1) an individual-feedback preference probe,
  2) a profile generator,
  3) instruction-conditioned knowledge generation,
  4) a dynamic extractor producing dynamic interest/profile,
  5) an LLM reranker.

This adaptation keeps those conceptual components, but matches the current
CoMemTree controlled evaluation protocol.

STRICT COMEMTREE PROTOCOL
-------------------------
- Frozen train/eval user lists are explicit inputs and must be disjoint.
- Evaluation users come ONLY from eval_users_N.json; no behavior-cache inference.
- Same history: ONLY the last <= 10 TRAIN interactions of each evaluation user.
- No validation/test interaction is used as memory/profile/feedback.
- Same REQUIRED frozen candidate set/order (typically 20 = 1 GT + 19 negatives).
- Same target item(s).
- Same seed = 42.
- Same item evidence as CoMemTree: item_id + title + category only.
- Same LLM family/backbone:
      Qwen/Qwen2.5-7B-Instruct
- Metrics:
      Hit@1/3/5/10, NDCG@1/3/5/10, MRR.

IMPORTANT DIFFERENCE FROM ORIGINAL INSTRUCTREC
----------------------------------------------
CoMemTree data has no free-text user instruction or review text. To avoid
inventing target-dependent signals:
- every user receives the SAME neutral instruction;
- no review text is used;
- the last TRAIN interaction is used as observed positive feedback;
- the held-out test target is NEVER used in profile generation.

INITIAL RANKING
---------------
i²Agent is a reranker, so the strictest comparison is:
    Native Qwen ranking -> i²Agent reranking

Use --initial-ranking-file to pass the exact Native rankings already produced
by your CoMemTree inference script. If omitted, this script generates a Native
Qwen ranking using the CoMemTree-style native prompt before i²Agent reranking.

OUTPUT
------
<output-dir>/
  i2agent_rankings.jsonl
  i2agent_summary.json
  protocol_snapshot.json

Example (reuse an existing Native ranking):
-------------------------------------------
python i2agent_shared_ranking_qwen_score.py \
  --data-dir data/ML \
  --num-train-users 1000 --num-test-users 1000 \
  --history-size 30 --expected-candidates 20 \
  --initial-ranking-file results/native_qwen_list_1000.json \
  --no-instruction --llm-batch-size 4 --seed 42 \
  --output-dir results/i2agent_qwen_score_ML_1000

If --initial-ranking-file is absent, Stage 0 generates Native SCORE ranking
from TRAIN history + frozen candidates; Stage 4 always uses SCORE ranking.
To SKIP Stage 0 entirely and omit the Native ranking prior from the final
prompt, pass --no-initial-ranking (and do not pass --initial-ranking-file).
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
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# IMPORTANT: import Unsloth before transformers-like packages.
_startup_log = io.StringIO()
with redirect_stdout(_startup_log), redirect_stderr(_startup_log):
    import unsloth  # noqa: F401
    import torch
    from tqdm.auto import tqdm
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    


SCHEMA_VERSION = "i2agent_comemtree_qwen_score_v1"
SHARED_RANK_PROMPT_VERSION = "shared_next_item_scoring_clabel_v1"
DEFAULT_INSTRUCTION = (
    "Recommend items that best match this user's current preferences "
    "based on the user's recent interaction history."
)


# =============================================================================
# Generic I/O / reproducibility
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


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (str, int, float)):
        return [str(x)]
    if not isinstance(x, list):
        return [str(x)]
    return [str(v) for v in x]


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def extract_json_value(text: str) -> Any:
    """
    Robustly recover the first complete JSON object/array from model output.
    """
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts: List[Tuple[int, str, str]] = []
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
                return json.loads(s[start : i + 1])

    raise ValueError("Unbalanced JSON")


# =============================================================================
# Frozen user protocol
# =============================================================================

def load_user_ids_file(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Frozen user file is empty: {path}")

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            users = [str(x) for x in obj]
        elif isinstance(obj, dict):
            if isinstance(obj.get("users"), list):
                users = [str(x) for x in obj["users"]]
            else:
                users = [str(x) for x in obj.keys()]
        else:
            raise ValueError
    except json.JSONDecodeError:
        users = [x.strip() for x in text.splitlines() if x.strip()]

    if not users:
        raise ValueError(f"No users found in frozen user file: {path}")

    if len(users) != len(set(users)):
        raise ValueError(f"Duplicate user IDs in frozen user file: {path}")

    return users


def validate_frozen_users(
    users: Sequence[str],
    sequences: Dict[str, Dict[str, Any]],
    label: str,
) -> List[str]:
    users = [str(uid) for uid in users]
    missing = [uid for uid in users if uid not in sequences]
    if missing:
        raise ValueError(
            f"{label}: {len(missing)} frozen users are missing from "
            f"user_sequences.json. Examples: {missing[:10]}"
        )
    return users


def validate_disjoint_users(
    train_users: Sequence[str],
    eval_users: Sequence[str],
) -> None:
    overlap = sorted(set(train_users) & set(eval_users))
    if overlap:
        raise ValueError(
            f"Frozen train/eval user overlap detected: {len(overlap)} users. "
            f"Examples: {overlap[:10]}"
        )


# =============================================================================
# Candidate construction: same frozen candidate protocol as CoMemTree
# =============================================================================

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
        for row in read_jsonl(p):
            uid = row.get("user_id", row.get("uid"))
            if uid is not None:
                rows[str(uid)] = row
        return rows

    obj = load_json(p)
    if isinstance(obj, dict) and isinstance(obj.get("users"), dict):
        obj = obj["users"]

    if not isinstance(obj, dict):
        raise ValueError("Candidate file must be a JSON object or JSONL.")

    rows: Dict[str, Dict[str, Any]] = {}
    for uid, row in obj.items():
        if isinstance(row, list):
            rows[str(uid)] = {"candidates": row}
        elif isinstance(row, dict):
            rows[str(uid)] = row
        else:
            raise ValueError(f"Unsupported candidate row for user {uid}")
    return rows


def get_candidates_for_user(
    uid: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    candidate_rows: Optional[Dict[str, Dict[str, Any]]],
    seed: int,
) -> Tuple[List[str], List[str]]:
    # Under the frozen protocol every method MUST consume the shared candidate file.
    # `seed` is retained in the signature for shared-interface compatibility, but
    # no candidate generation is performed inside this method.
    del seed

    if candidate_rows is None:
        raise ValueError(
            "Frozen --candidate-file is required. Generate it once with "
            "prepare_frozen_protocol.py; methods must not rebuild candidates locally."
        )

    if uid not in candidate_rows:
        raise KeyError(f"user={uid}: missing from frozen candidate file")

    row = candidate_rows[uid]
    candidates = as_str_list(
        row.get(
            "candidates",
            row.get(
                "candidate_item_ids",
                row.get("candidate_ids", []),
            ),
        )
    )
    targets = as_str_list(
        row.get(
            "target",
            row.get(
                "targets",
                row.get(
                    "ground_truth",
                    row.get(
                        "ground_truth_item_ids",
                        row.get(
                            "target_item_ids",
                            user_data.get("test", []),
                        ),
                    ),
                ),
            ),
        )
    )

    if not candidates:
        raise ValueError(f"user={uid}: empty frozen candidate list")
    if not targets:
        raise ValueError(f"user={uid}: no target in frozen protocol")

    # Validate the frozen artifact against the two canonical raw inputs.
    expected_targets = as_str_list(user_data.get("test", []))
    expected_negatives = as_str_list(negative_data.get("test_neg", []))
    if not expected_targets:
        raise ValueError(f"user={uid}: no test item in user_sequences.json")
    if not expected_negatives:
        raise ValueError(f"user={uid}: no test_neg in user_negatives.json")

    expected_candidates = expected_targets + expected_negatives
    if targets != expected_targets:
        raise ValueError(
            f"user={uid}: frozen target differs from user_sequences.json test target"
        )
    if (
        len(candidates) != len(expected_candidates)
        or set(candidates) != set(expected_candidates)
    ):
        raise ValueError(
            f"user={uid}: frozen candidate membership differs from "
            "test + test_neg in the canonical raw inputs"
        )
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"user={uid}: duplicate IDs in frozen candidate list")

    return candidates, targets


# =============================================================================
# Item metadata: expose ONLY title + category, matching CoMemTree
# =============================================================================

ITEM_ID_KEYS = ("item_id", "item", "asin", "id", "product_id", "movie_id")
TITLE_KEYS = ("title", "name", "item_title", "movie_title")
CATEGORY_KEYS = (
    "main_cat",
    "category",
    "categories",
    "genre",
    "genres",
    "tags",
)


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, "", [], {}):
            return row[key]
    return None


def _category_text(v: Any) -> str:
    if v is None:
        return "Unknown"
    if isinstance(v, list):
        vals: List[str] = []
        for x in v:
            if isinstance(x, list):
                vals.extend(str(y) for y in x)
            else:
                vals.append(str(x))
        return " | ".join(clean_text(x) for x in vals if clean_text(x)) or "Unknown"
    if isinstance(v, dict):
        return " | ".join(f"{k}:{v[k]}" for k in sorted(v)) or "Unknown"
    return clean_text(v) or "Unknown"


def build_item_catalog(items_obj: Any) -> Dict[str, Dict[str, str]]:
    """
    Build alias -> {item_id,title,category}.  All output IDs remain raw string IDs.
    """
    catalog: Dict[str, Dict[str, str]] = {}

    if isinstance(items_obj, dict):
        iterator = list(items_obj.items())
    elif isinstance(items_obj, list):
        iterator = [(None, row) for row in items_obj]
    else:
        raise ValueError("Unsupported items.json structure.")

    for dict_key, raw in iterator:
        if isinstance(raw, str):
            row = {"title": raw}
        elif isinstance(raw, dict):
            row = dict(raw)
        else:
            continue

        aliases: List[str] = []
        if dict_key is not None:
            aliases.append(str(dict_key))
        for k in ITEM_ID_KEYS:
            if row.get(k) is not None:
                aliases.append(str(row[k]))

        aliases = list(dict.fromkeys(aliases))
        if not aliases:
            continue

        preferred = str(row.get("asin", aliases[0]))
        title = clean_text(_first_present(row, TITLE_KEYS))
        category = _category_text(_first_present(row, CATEGORY_KEYS))

        info = {
            "item_id": preferred,
            "title": title if title else f"Unknown Item {preferred}",
            "category": category,
        }

        for alias in aliases + [preferred]:
            # Preserve the queried raw ID in item_id at retrieval time.
            catalog[str(alias)] = dict(info)

    if not catalog:
        raise ValueError("No usable item metadata loaded.")

    return catalog


def get_item_info(item_id: str, catalog: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    iid = str(item_id)
    if iid in catalog:
        x = dict(catalog[iid])
        x["item_id"] = iid
        return x
    return {
        "item_id": iid,
        "title": f"Unknown Item {iid}",
        "category": "Unknown",
    }


def format_items(items: Sequence[Dict[str, str]]) -> str:
    return "\n".join(
        f"- ID={x['item_id']} | title={x['title']} | category={x['category']}"
        for x in items
    )


# =============================================================================
# Native ranking loader: reuse exact CoMemTree Native result when available
# =============================================================================

RANK_KEYS = (
    "reranked_item_ids",
    "ranked_item_ids",
    "ranking",
    "predictions",
    "prediction",
    "ranked_list",
)


def _extract_ranking_from_obj(
    row: Dict[str, Any],
    mode: str = "native",
) -> Optional[List[str]]:
    # Current CoMemTree tree output stores ranking under rankings[mode].
    rankings = row.get("rankings")
    if isinstance(rankings, dict):
        mode_obj = rankings.get(mode)
        if isinstance(mode_obj, dict):
            for key in RANK_KEYS:
                if isinstance(mode_obj.get(key), list):
                    return [str(x) for x in mode_obj[key]]
        elif isinstance(mode_obj, list):
            return [str(x) for x in mode_obj]

    # Other baseline outputs use direct ranking keys.
    for key in RANK_KEYS:
        if isinstance(row.get(key), list):
            return [str(x) for x in row[key]]

    # A few outputs may nest under "result".
    result = row.get("result")
    if isinstance(result, dict):
        for key in RANK_KEYS:
            if isinstance(result.get(key), list):
                return [str(x) for x in result[key]]

    return None


def load_initial_rankings(
    path: Optional[str],
    mode: str = "native",
) -> Optional[Dict[str, List[str]]]:
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    out: Dict[str, List[str]] = {}

    if p.suffix.lower() == ".jsonl":
        rows = read_jsonl(p)
    else:
        obj = load_json(p)
        if isinstance(obj, list):
            rows = [x for x in obj if isinstance(x, dict)]
        elif isinstance(obj, dict):
            if isinstance(obj.get("rows"), list):
                rows = [x for x in obj["rows"] if isinstance(x, dict)]
            elif isinstance(obj.get("users"), dict):
                rows = []
                for uid, row in obj["users"].items():
                    if isinstance(row, dict):
                        row = dict(row)
                        row.setdefault("user_id", uid)
                        rows.append(row)
            else:
                # Could be uid -> row/ranking.
                rows = []
                for uid, row in obj.items():
                    if isinstance(row, list):
                        rows.append({"user_id": uid, "ranking": row})
                    elif isinstance(row, dict):
                        row2 = dict(row)
                        row2.setdefault("user_id", uid)
                        rows.append(row2)
        else:
            raise ValueError("Unsupported initial ranking file.")

    for row in rows:
        uid = row.get("user_id", row.get("uid"))
        if uid is None:
            continue
        ranking = _extract_ranking_from_obj(row, mode=mode)
        if ranking:
            out[str(uid)] = ranking

    if not out:
        raise ValueError(
            f"No rankings found in {path}. "
            f"Tried mode={mode!r} and keys={RANK_KEYS}."
        )
    return out


def sanitize_ranking(
    predicted: Any,
    valid_ids: Sequence[str],
) -> Tuple[List[str], bool, int]:
    valid = [str(x) for x in valid_ids]
    valid_set = set(valid)

    if not isinstance(predicted, list):
        predicted = []

    out: List[str] = []
    seen: Set[str] = set()

    for x in predicted:
        iid = str(x)
        if iid in valid_set and iid not in seen:
            out.append(iid)
            seen.add(iid)

    model_returned = len(out)
    complete = model_returned == len(valid)

    # Deterministic fallback to the input candidate order.
    for iid in valid:
        if iid not in seen:
            out.append(iid)
            seen.add(iid)

    return out, complete, model_returned


# =============================================================================
# Metrics
# =============================================================================

def first_relevant_rank(
    ranking: Sequence[str],
    targets: Sequence[str],
) -> Optional[int]:
    target_set = {str(x) for x in targets}
    for idx, iid in enumerate(ranking):
        if str(iid) in target_set:
            return idx
    return None


def metrics_for_ranking(
    ranking: Sequence[str],
    targets: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    rank = first_relevant_rank(ranking, targets)
    out: Dict[str, float] = {}
    for k in ks:
        hit = 1.0 if rank is not None and rank < k else 0.0
        ndcg = (
            1.0 / math.log2(rank + 2.0)
            if rank is not None and rank < k
            else 0.0
        )
        out[f"Hit@{k}"] = hit
        out[f"NDCG@{k}"] = ndcg

    out["MRR"] = 1.0 / (rank + 1.0) if rank is not None else 0.0
    out["rank"] = float(rank + 1) if rank is not None else 0.0
    return out


def aggregate_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [
        "Hit@1", "NDCG@1",
        "Hit@3", "NDCG@3",
        "Hit@5", "NDCG@5",
        "Hit@10", "NDCG@10",
        "MRR",
    ]
    return {
        k: float(np.mean([float(row["metrics"][k]) for row in rows]))
        for k in keys
    }


# =============================================================================
# Local batched Qwen
# =============================================================================

class LocalQwen:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This script expects a CUDA GPU for Unsloth Qwen inference."
            )

        print(f"[INFO] Loading Qwen: {model_name}", flush=True)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=int(max_seq_length),
            load_in_4bit=bool(load_in_4bit),
            full_finetuning=False,
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-2.5",
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            FastLanguageModel.for_inference(self.model)
        except Exception:
            pass

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_seq_length = int(max_seq_length)

        self.generate_calls = 0
        self.samples = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_time_sec = 0.0
        self.stage_samples: Dict[str, int] = defaultdict(int)
        self.stage_calls: Dict[str, int] = defaultdict(int)

    def _render(self, prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.startswith("<bos>"):
            text = text[len("<bos>") :]
        return text

    def generate_batch(
        self,
        prompts: Sequence[str],
        batch_size: int,
        max_new_tokens: int,
        stage: str,
        system_prompt: str = "You are a careful recommendation assistant.",
    ) -> List[str]:
        if not prompts:
            return []

        outputs_all: List[str] = []
        batch_size = max(1, int(batch_size))

        starts = list(range(0, len(prompts), batch_size))
        progress = tqdm(
            starts,
            desc=f"[{stage}]",
            total=len(starts),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            file=sys.__stderr__,
        )

        for start in progress:
            chunk = prompts[start : start + batch_size]
            texts = [self._render(p, system_prompt) for p in chunk]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.device)

            input_width = int(inputs["input_ids"].shape[1])
            per_sample_inputs = [
                int(x)
                for x in inputs["attention_mask"].sum(dim=1).tolist()
            ]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            new_tokens = generated[:, input_width:]
            decoded = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )

            out_token_counts = [
                int((seq != self.tokenizer.pad_token_id).sum().item())
                for seq in new_tokens
            ]

            self.generate_calls += 1
            self.stage_calls[stage] += 1
            self.samples += len(chunk)
            self.stage_samples[stage] += len(chunk)
            self.input_tokens += sum(per_sample_inputs)
            self.output_tokens += sum(out_token_counts)
            self.total_time_sec += elapsed

            outputs_all.extend(x.strip() for x in decoded)

            done_samples = min(start + len(chunk), len(prompts))
            progress.set_postfix(
                {
                    "samples": f"{done_samples}/{len(prompts)}",
                    "sec/batch": f"{elapsed:.1f}",
                    "tok_out": sum(out_token_counts),
                },
                refresh=False,
            )

        progress.close()
        return outputs_all

    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int,
        stage: str,
        system_prompt: str = "You are a careful recommendation assistant.",
    ) -> str:
        return self.generate_batch(
            [prompt],
            batch_size=1,
            max_new_tokens=max_new_tokens,
            stage=stage,
            system_prompt=system_prompt,
        )[0]

    def stats(self) -> Dict[str, Any]:
        return {
            "generate_calls": int(self.generate_calls),
            "samples": int(self.samples),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "total_tokens": int(self.input_tokens + self.output_tokens),
            "total_time_sec": round(float(self.total_time_sec), 4),
            "stage_calls": dict(self.stage_calls),
            "stage_samples": dict(self.stage_samples),
            "avg_sample_time_sec": (
                round(self.total_time_sec / self.samples, 4)
                if self.samples
                else 0.0
            ),
        }


# =============================================================================
# Shared SCORE ranking, inlined to make this script fully standalone.
# Same history/candidate/method-context evidence as ranking_functions_qwen.py.
# Only OUTPUT REQUIREMENTS differ: scores instead of a full ranked list.
# =============================================================================


def build_candidate_label_maps(candidates):
    n = len(candidates)
    width = max(2, len(str(max(1, n))))
    to_label, to_id = {}, {}
    for idx, item in enumerate(candidates, start=1):
        iid = str(item["item_id"])
        label = f"C{idx:0{width}d}"
        if iid in to_label:
            raise ValueError(f"Duplicate candidate ID: {iid}")
        to_label[iid] = label
        to_id[label] = iid
    return to_label, to_id


def normalize_candidate_label(value, n):
    label = str(value).strip().upper().strip("[](){}<>")
    m = re.fullmatch(r"C0*(\d+)", label)
    if m is None or not (1 <= int(m.group(1)) <= n):
        return None
    return f"C{int(m.group(1)):0{max(2, len(str(n)))}d}"


def build_score_prompt(*, history_items, candidate_items, method_context=None):
    # Prompt text above OUTPUT REQUIREMENTS matches the shared list-ranking task.
    history = ("\n".join(
        f"{i}. title={clean_text(x.get('title', ''))} | "
        f"category={clean_text(x.get('category', 'Unknown')) or 'Unknown'}"
        for i, x in enumerate(history_items, start=1)
    ) if history_items else "None")
    id_to_label, _ = build_candidate_label_maps(candidate_items)
    labels = [id_to_label[str(x['item_id'])] for x in candidate_items]
    candidate_block = "\n".join(
        f"[{id_to_label[str(x['item_id'])]}] title={clean_text(x.get('title', ''))} | "
        f"category={clean_text(x.get('category', 'Unknown')) or 'Unknown'}"
        for x in candidate_items
    )
    if method_context in (None, "", [], {}):
        context = "Not provided."
    elif isinstance(method_context, str):
        context = clean_text(method_context)
    else:
        context = json.dumps(method_context, ensure_ascii=False, indent=2, default=str)
    n = len(candidate_items)
    return f"""TASK
Rank ALL candidate items for the target user's NEXT choice.

RECENT USER HISTORY
The interactions are ordered from older to newer.

{history}

ADDITIONAL METHOD CONTEXT
A method may provide extra evidence here. If it is not provided, use only the recent history.

{context}

CANDIDATE ITEMS
Candidate order is frozen/randomized for evaluation and has NO ranking meaning.

{candidate_block}

DECISION RULES
- Use only the evidence shown in RECENT USER HISTORY, ADDITIONAL METHOD CONTEXT, and the candidate title/category fields.
- Infer the user's current preference and rank candidates by compatibility with the user's NEXT choice.
- Treat ADDITIONAL METHOD CONTEXT as evidence, not as ground truth; ignore any part that conflicts with stronger observed user evidence.
- Do not use candidate input order as a preference signal.
- Score every candidate independently for relevance to the user's NEXT choice.

OUTPUT REQUIREMENTS
- Assign each candidate one relevance score in [0, 1]; larger means more relevant.
- Include ALL {n} candidate labels exactly once: {', '.join(labels)}.
- Do not invent, omit, or duplicate labels. Never output raw item IDs.
- Scores do not need to sum to 1; avoid assigning identical scores unless relevance is genuinely indistinguishable.
- Return ONLY a JSON object containing a scores dictionary. No explanation or markdown.
- Silently verify every required label appears exactly once before returning.

{{"scores":{{"C01":0.92,"C02":0.37}}}}
"""


def parse_score_response(raw, candidate_items):
    valid_ids = [str(x["item_id"]) for x in candidate_items]
    _, label_to_id = build_candidate_label_maps(candidate_items)
    scores = {}
    invalid_labels, duplicate_labels, invalid_scores = [], [], []
    json_ok = False
    parse_error = None
    try:
        obj = extract_json_value(raw)
        if not isinstance(obj, dict) or "scores" not in obj:
            raise ValueError('Expected JSON object with "scores" field')
        values = obj["scores"]
        if isinstance(values, dict):
            pairs = list(values.items())
        elif isinstance(values, list):
            pairs = []
            for entry in values:
                if not isinstance(entry, dict):
                    invalid_scores.append(str(entry))
                    continue
                pairs.append((entry.get("label", ""), entry.get("score")))
        else:
            raise ValueError('"scores" must be dictionary or list of {label,score}')
        json_ok = True
        for key, value in pairs:
            label = normalize_candidate_label(key, len(valid_ids))
            if label not in label_to_id:
                invalid_labels.append(str(key))
                continue
            if label in scores:
                duplicate_labels.append(label)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid_scores.append(f"{label}={value!r}")
                continue
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                invalid_scores.append(f"{label}={value!r}")
                continue
            scores[label] = float(value)
    except Exception as exc:
        parse_error = str(exc)
    missing = [label for label in label_to_id if label not in scores]
    # Stable ties follow the frozen candidate order (Python stable sort).
    ranked = sorted(scores, key=lambda label: -scores[label])
    ranked.extend(missing)  # Preserve valid model decisions; repair incompletes last.
    ranking = [label_to_id[label] for label in ranked]
    complete = json_ok and not missing and not duplicate_labels and not invalid_labels and not invalid_scores
    return ranking, {
        "prompt_version": SHARED_RANK_PROMPT_VERSION,
        "ranking_mode": "score",
        "parse_json_ok": json_ok,
        "parse_ok": json_ok,
        "parse_complete": bool(complete),
        "model_returned": len(scores),
        "model_returned_valid_labels": len(scores),
        "raw_valid_label_count": len(scores),
        "num_candidates": len(valid_ids),
        "predicted_labels": ranked[:len(scores)],
        "missing_labels": missing,
        "duplicate_labels": duplicate_labels,
        "invalid_labels": invalid_labels,
        "invalid_scores": invalid_scores,
        "scores_by_label": scores,
        "score_tie_count": len(scores) - len(set(scores.values())),
        "parse_error": parse_error,
        "repaired_missing_at_end": bool(missing),
        "raw_output": str(raw or ""),
        "reasoning": "",
    }


class SharedQwenScoreRanker:
    def __init__(self, model_name, max_seq_length=8192, load_in_4bit=True,
                 rank_max_new_tokens=1024, temperature=0.0, seed=42,
                 *, model=None, tokenizer=None, debug=False):
        if model is None or tokenizer is None:
            raise ValueError("Pass already loaded Qwen model and tokenizer")
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_seq_length = int(max_seq_length)
        self.max_new_tokens = int(rank_max_new_tokens)
        if self.max_seq_length <= self.max_new_tokens:
            raise ValueError("max_seq_length must exceed rank_max_new_tokens")
        self.debug = bool(debug)
        self.records = []
        self.generate_calls = 0
        self.samples = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_time_sec = 0.0

    def rank_jobs(self, jobs, batch_size=4, call_type="ranking", max_retries=0):
        if not jobs:
            return []
        batch_size = max(1, int(batch_size))
        budget = self.max_seq_length - self.max_new_tokens
        all_results = []
        progress = tqdm(range(0, len(jobs), batch_size), desc=f"[{call_type}-score]",
                        unit="batch", dynamic_ncols=True, file=sys.__stderr__)
        for start in progress:
            chunk = list(jobs[start:start+batch_size])
            prompts = [build_score_prompt(history_items=j.get("history_items", []),
                                          candidate_items=j["candidate_items"],
                                          method_context=j.get("method_context")) for j in chunk]
            texts = [self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a recommendation ranking system. Return only the requested valid JSON."},
                {"role": "user", "content": p},
            ], tokenize=False, add_generation_prompt=True) for p in prompts]
            lengths = []
            for t in texts:
                enc = self.tokenizer(t, add_special_tokens=False, truncation=False)
                token_ids = enc["input_ids"]
                if torch.is_tensor(token_ids):
                    lengths.append(int(token_ids.shape[-1]))
                elif len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
                    lengths.append(len(token_ids[0]))
                else:
                    lengths.append(len(token_ids))
            # Reserve max_new_tokens for score output. Log any input truncation.
            # Tokenizer decides the truncation side according to its settings.
            inputs = self.tokenizer(texts, add_special_tokens=False, return_tensors="pt",
                                    padding=True, truncation=True, max_length=budget).to(
                                        next(self.model.parameters()).device)
            kept = [int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()]
            width = inputs["input_ids"].shape[1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )[:,width:]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start_time
            raws = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            out_counts = []
            for tokens in generated:
                toks = [int(x) for x in tokens.tolist()]
                eos = self.tokenizer.eos_token_id
                if eos is not None and eos in toks:
                    out_counts.append(toks.index(eos)+1)
                else:
                    out_counts.append(len(toks))
            self.generate_calls += 1
            self.samples += len(chunk)
            self.input_tokens += sum(kept)
            self.output_tokens += sum(out_counts)
            self.total_time_sec += elapsed
            for job, raw, full_n, kept_n, out_n, tokens in zip(
                    chunk, raws, lengths, kept, out_counts, generated):
                ranking, meta = parse_score_response(raw, job["candidate_items"])
                has_eos = (self.tokenizer.eos_token_id is not None
                           and self.tokenizer.eos_token_id in tokens.tolist())
                dbg = {
                    "call_type": call_type,
                    "attempts": 1,
                    "full_input_tokens": full_n,
                    "input_tokens": kept_n,
                    "truncated_input_tokens": max(0, full_n-kept_n),
                    "input_truncated": full_n > kept_n,
                    "context_budget_risk": full_n > budget,
                    "output_tokens": out_n,
                    "output_limit_reached": (not has_eos and out_n >= self.max_new_tokens),
                    "output_has_eos": bool(has_eos),
                    "elapsed_sec": elapsed/max(1,len(chunk)),
                }
                meta.update(dbg)
                self.records.append(dict(user_id=str(job.get("user_id", "")), **{
                    k: v for k, v in meta.items() if k not in ("raw_output", "scores_by_label", "predicted_labels")
                }))
                all_results.append((ranking, meta))
                if self.debug:
                    print(f"[RANK-SCORE] {len(all_results)}/{len(jobs)} user={job.get('user_id')} "
                          f"valid={meta['parse_complete']} missing={meta['missing_labels']} "
                          f"ties={meta['score_tie_count']} in={kept_n}/{full_n} "
                          f"out={out_n}/{self.max_new_tokens}", flush=True)
        progress.close()
        return all_results

    def stats(self):
        n = self.samples
        return {
            "prompt_version": SHARED_RANK_PROMPT_VERSION,
            "ranking_mode": "score",
            "model": self.model_name,
            "debug_enabled": self.debug,
            "generate_calls": self.generate_calls,
            "samples": n,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_time_sec": round(self.total_time_sec,4),
            "avg_sample_time_sec": round(self.total_time_sec/n,4) if n else 0,
            "input_truncated_count": sum(bool(x['input_truncated']) for x in self.records),
            "context_budget_risk_count": sum(bool(x['context_budget_risk']) for x in self.records),
            "output_limit_reached_count": sum(bool(x['output_limit_reached']) for x in self.records),
            "parse_complete_count": sum(bool(x['parse_complete']) for x in self.records),
            "parse_complete_ratio": sum(bool(x['parse_complete']) for x in self.records)/n if n else 0,
            "users_with_score_ties": sum(x['score_tie_count'] > 0 for x in self.records),
            "call_type_counts": dict(__import__('collections').Counter(x['call_type'] for x in self.records)),
        }

    def tracking_records(self):
        return list(self.records)


# =============================================================================
# i²Agent prompts adapted to no-instruction/no-review CoMemTree data
# =============================================================================

def deterministic_feedback_negative(
    uid: str,
    catalog_ids: Sequence[str],
    forbidden: Set[str],
    seed: int,
) -> str:
    pool = [iid for iid in catalog_ids if iid not in forbidden]
    if not pool:
        raise ValueError(
            f"user={uid}: cannot sample a feedback negative outside "
            "history/candidates"
        )
    rng = random.Random(f"i2agent-feedback:{seed}:{uid}")
    return pool[rng.randrange(len(pool))]


def feedback_probe_prompt(
    previous_history: Sequence[Dict[str, str]],
    positive_item: Dict[str, str],
    negative_item: Dict[str, str],
    swap: bool,
) -> str:
    a = negative_item if swap else positive_item
    b = positive_item if swap else negative_item

    context = format_items(previous_history[-9:]) if previous_history else "None"
    return f"""This is an individual user preference probe used to update a personal recommendation agent.

Earlier observed history (oldest to newest):
{context}

Choose which ONE of the following two items is more consistent with this user's preferences.

A: ID={a['item_id']} | title={a['title']} | category={a['category']}
B: ID={b['item_id']} | title={b['title']} | category={b['category']}

Return JSON only:
{{"choice":"A","brief_reason":"one short sentence"}}

The choice must be exactly "A" or "B".
"""


def profile_prompt(
    previous_history: Sequence[Dict[str, str]],
    feedback_item: Dict[str, str],
    probe_choice: str,
) -> str:
    prev = format_items(previous_history) if previous_history else "None"
    return f"""Build an INDIVIDUAL user profile from observed history and feedback.

Earlier observed history:
{prev}

The agent's preference-probe choice was: {probe_choice}

Observed positive feedback:
The user then interacted with:
ID={feedback_item['item_id']} | title={feedback_item['title']} | category={feedback_item['category']}

Important:
- This interaction is part of TRAIN history, not the held-out target.
- Infer preferences only from the evidence above.
- Do not invent demographics, identities, ratings, reviews, or unseen items.
- Focus on content preferences, recurring themes/categories, continuity, breadth,
  and possible short-term tendencies.
- Keep the profile compact.

Return JSON only:
{{"profile":"2-4 concise sentences"}}
"""


def knowledge_prompt(instruction: str) -> str:
    return f"""Generate domain-general recommendation knowledge relevant to the following user instruction.

Instruction:
{instruction}

The knowledge should describe WHAT evidence is useful for matching candidate items
to a user's observed preferences. Do not recommend any specific item and do not
refer to a held-out target.

Return JSON only:
{{"knowledge":"2-4 concise sentences"}}
"""


def dynamic_extract_prompt(
    instruction: str,
    knowledge: str,
    history: Sequence[Dict[str, str]],
    profile: str,
) -> str:
    instruction_fields = (
        f"Instruction:\n{instruction}\n\nGeneral recommendation knowledge:\n{knowledge}\n\n"
        if instruction or knowledge else ""
    )
    return f"""Extract the user's CURRENT dynamic interest and dynamic profile for the next recommendation.

{instruction_fields}Static recent memory (oldest to newest):
{format_items(history)}

Static individual profile:
{profile}

Rules:
- Use only the observed history/profile.
- Emphasize recent evidence when it conflicts with older evidence.
- Do not use or guess the held-out target.
- Do not invent reviews, ratings, demographics, or other users.
- Dynamic interest should describe what content the user appears to want next.
- Dynamic profile should summarize the currently relevant part of the longer-term profile.

Return JSON only:
{{"dynamic_interest":"1-3 concise sentences","dynamic_profile":"1-3 concise sentences"}}
"""


# Final rerank prompt/labels are implemented in this standalone score-ranking file.


def parse_string_field(
    raw: str,
    field: str,
    fallback: str = "",
) -> Tuple[str, bool]:
    try:
        obj = extract_json_value(raw)
        if isinstance(obj, dict):
            val = clean_text(obj.get(field, ""))
            if val:
                return val, True
    except Exception:
        pass
    return fallback, False


def parse_feedback_choice(raw: str) -> Tuple[str, bool]:
    try:
        obj = extract_json_value(raw)
        if isinstance(obj, dict):
            choice = str(obj.get("choice", "")).strip().upper()
            if choice in {"A", "B"}:
                return choice, True
    except Exception:
        pass
    return "UNKNOWN", False


def parse_dynamic(
    raw: str,
) -> Tuple[str, str, bool]:
    try:
        obj = extract_json_value(raw)
        if isinstance(obj, dict):
            interest = clean_text(obj.get("dynamic_interest", ""))
            profile = clean_text(obj.get("dynamic_profile", ""))
            if interest and profile:
                return interest, profile, True
    except Exception:
        pass
    return "", "", False


# =============================================================================
# Prepare evaluation jobs
# =============================================================================

def prepare_jobs(
    users: Sequence[str],
    sequences: Dict[str, Dict[str, Any]],
    negatives: Dict[str, Dict[str, Any]],
    candidate_rows: Optional[Dict[str, Dict[str, Any]]],
    catalog: Dict[str, Dict[str, str]],
    history_size: int,
    expected_candidates: int,
    seed: int,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    catalog_ids = list(catalog.keys())

    for uid in users:
        user_data = sequences[uid]
        neg_data = negatives.get(uid, {})

        train_ids = as_str_list(user_data.get("train", []))
        history_ids = train_ids[-int(history_size) :]
        if not history_ids:
            raise ValueError(f"user={uid}: empty TRAIN history")

        candidate_ids, targets = get_candidates_for_user(
            uid=uid,
            user_data=user_data,
            negative_data=neg_data,
            candidate_rows=candidate_rows,
            seed=seed,
        )

        if expected_candidates > 0 and len(candidate_ids) != expected_candidates:
            raise ValueError(
                f"user={uid}: candidate_count={len(candidate_ids)} "
                f"!= expected={expected_candidates}"
            )

        missing_targets = [t for t in targets if t not in set(candidate_ids)]
        if missing_targets:
            raise ValueError(
                f"user={uid}: target(s) absent from candidate set: {missing_targets}"
            )

        history = [get_item_info(i, catalog) for i in history_ids]
        candidates = [get_item_info(i, catalog) for i in candidate_ids]

        # i²Agent feedback comes ONLY from TRAIN history.
        feedback_positive_id = history_ids[-1]
        previous_history_ids = history_ids[:-1]

        forbidden = set(history_ids) | set(candidate_ids)
        feedback_negative_id = deterministic_feedback_negative(
            uid=uid,
            catalog_ids=catalog_ids,
            forbidden=forbidden,
            seed=seed,
        )

        jobs.append(
            {
                "user_id": str(uid),
                "history_ids": history_ids,
                "history": history,
                "previous_history": [
                    get_item_info(i, catalog)
                    for i in previous_history_ids
                ],
                "feedback_positive": get_item_info(
                    feedback_positive_id, catalog
                ),
                "feedback_negative": get_item_info(
                    feedback_negative_id, catalog
                ),
                "candidate_item_ids": candidate_ids,
                "candidate_items": candidates,
                "targets": targets,
            }
        )

    return jobs


def _num_batches(n_samples: int, batch_size: int) -> int:
    if n_samples <= 0:
        return 0
    return math.ceil(n_samples / max(1, int(batch_size)))


# =============================================================================
# Pipeline
# =============================================================================

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "i2agent_rankings.jsonl"
    summary_path = output_dir / "i2agent_summary.json"
    snapshot_path = output_dir / "protocol_snapshot.json"

    print("[INFO] Loading CoMemTree raw data...", flush=True)
    items_obj = load_json(args.items)
    sequences_raw = load_json(args.sequences)
    negatives_raw = load_json(args.negatives)

    if not isinstance(sequences_raw, dict):
        raise ValueError("Sequences JSON must be a user-id -> row mapping.")
    if not isinstance(negatives_raw, dict):
        raise ValueError("Negatives JSON must be a user-id -> row mapping.")

    sequences = {str(k): v for k, v in sequences_raw.items()}
    negatives = {str(k): v for k, v in negatives_raw.items()}
    catalog = build_item_catalog(items_obj)
    candidate_rows = load_candidate_file(args.candidate_file)
    if candidate_rows is None:
        raise ValueError("--candidate-file is required by the frozen protocol")

    train_users = validate_frozen_users(
        load_user_ids_file(args.train_users_file),
        sequences=sequences,
        label="Training users",
    )
    eval_users_full = validate_frozen_users(
        load_user_ids_file(args.eval_users_file),
        sequences=sequences,
        label="Evaluation users",
    )
    # validate_disjoint_users(train_users, eval_users_full)

    if len(train_users) != int(args.num_train_users):
        raise ValueError(
            f"Expected train_users_{args.num_train_users}.json to contain exactly "
            f"{args.num_train_users} users, found {len(train_users)}"
        )
    if len(eval_users_full) != int(args.num_test_users):
        raise ValueError(
            f"Expected eval_users_{args.num_test_users}.json to contain exactly "
            f"{args.num_test_users} users, found {len(eval_users_full)}"
        )

    users = list(eval_users_full)
    if args.max_users > 0:
        # Debug/smoke-test cap only. Official runs should normally use 0 and let
        # eval_users_N.json define the exact evaluation size.
        users = users[: int(args.max_users)]

    missing_candidate_users = [uid for uid in users if uid not in candidate_rows]
    if missing_candidate_users:
        raise ValueError(
            f"Frozen candidate file misses {len(missing_candidate_users)} selected "
            f"evaluation users. Examples: {missing_candidate_users[:10]}"
        )

    jobs = prepare_jobs(
        users=users,
        sequences=sequences,
        negatives=negatives,
        candidate_rows=candidate_rows,
        catalog=catalog,
        history_size=args.history_size,
        expected_candidates=args.expected_candidates,
        seed=args.seed,
    )

    print(
        f"[INFO] Prepared {len(jobs)} users; "
        f"history<= {args.history_size}; "
        f"candidate_count={len(jobs[0]['candidate_item_ids'])}",
        flush=True,
    )

    if args.no_initial_ranking and args.initial_ranking_file:
        raise ValueError(
            "--no-initial-ranking cannot be combined with --initial-ranking-file. "
            "Choose one experimental setting."
        )
    initial_rankings = (
        None if args.no_initial_ranking else load_initial_rankings(
            args.initial_ranking_file,
            mode=args.initial_ranking_mode,
        )
    )

    # Snapshot before model execution.
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "method": "i2Agent adapted to CoMemTree protocol",
        "inputs": {
            "items": args.items,
            "sequences": args.sequences,
            "negatives": args.negatives,
            "candidate_file": args.candidate_file,
            "train_users_file": args.train_users_file,
            "eval_users_file": args.eval_users_file,
            "initial_ranking_file": args.initial_ranking_file,
            "initial_ranking_mode": args.initial_ranking_mode,
            "initial_ranking_disabled": bool(args.no_initial_ranking),
        },
        "protocol": {
            "num_users": len(jobs),
            "requested_train_users": int(args.num_train_users),
            "requested_test_users": int(args.num_test_users),
            "frozen_train_user_count": len(train_users),
            "frozen_eval_user_count": len(eval_users_full),
            "evaluated_user_count": len(users),
            "train_eval_user_overlap": len(set(train_users) & set(eval_users_full)),
            "train_users_role_for_i2agent": (
                "protocol/disjointness validation only; i2Agent has no cross-user "
                "training-memory stage in this adaptation"
            ),
            "history_source": "last <= history_size interactions from TRAIN only",
            "history_size": args.history_size,
            "uses_validation_in_memory": False,
            "uses_test_target_in_memory": False,
            "uses_review_text": False,
            "instruction": "" if args.no_instruction else args.instruction,
            "instruction_disabled": bool(args.no_instruction),
            "instruction_is_same_for_all_users": True,
            "candidate_source": "exact shared frozen candidate file",
            "expected_candidates": args.expected_candidates,
            "seed": args.seed,
            "metadata_exposed": ["item_id", "title", "category"],
            "rerank_output_interface": (
                "C-label numeric scores mapped to ranked raw item IDs"
            ),
            "rank_prompt_version": SHARED_RANK_PROMPT_VERSION,
            "ranking_mode": "score",
            "native_ranking_prior_enabled": not args.no_initial_ranking,
            "feedback_positive_source": "last TRAIN interaction only",
            "feedback_negative_excludes": [
                "all exposed TRAIN-history items",
                "all current candidate items including held-out target",
            ],
        },
        "model": {
            "name": args.model,
            "output_mode": "score",
            "load_in_4bit": args.load_in_4bit,
            "max_seq_length": args.max_seq_length,
            "llm_batch_size": args.llm_batch_size,
        },
        "users": [j["user_id"] for j in jobs],
    }
    dump_json(snapshot_path, snapshot)

    gemma = LocalQwen(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    # Reuse the same loaded Qwen weights for the controlled shared SCORE ranking task.
    shared_ranker = SharedQwenScoreRanker(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        rank_max_new_tokens=args.rank_max_new_tokens,
        temperature=0.0,
        seed=args.seed,
        model=gemma.model,
        tokenizer=gemma.tokenizer,
        debug=args.debug_ranking,
    )

    # -------------------------------------------------------------------------
    # Stage 0: initial ranking
    # -------------------------------------------------------------------------
    if args.no_initial_ranking:
        # Ablation: score the ORIGINAL frozen candidate order directly.
        # Do not generate or load Native ranking and do not provide any ranked prior.
        native_meta_by_user = {}
        missing_native = []
        for job in jobs:
            job["initial_ranking"] = None
            native_meta_by_user[job["user_id"]] = {
                "source": "disabled",
                "parse_complete": None,
                "model_returned": 0,
            }
        print("[INFO] Stage 0 skipped (--no-initial-ranking). Final ranking uses frozen candidates directly.", flush=True)
    else:
        native_meta_by_user: Dict[str, Dict[str, Any]] = {}

        missing_native: List[Dict[str, Any]] = []
        for job in jobs:
            uid = job["user_id"]
            candidate_ids = job["candidate_item_ids"]

            if initial_rankings is not None and uid in initial_rankings:
                ranking, complete, returned = sanitize_ranking(
                    initial_rankings[uid],
                    candidate_ids,
                )
                job["initial_ranking"] = ranking
                native_meta_by_user[uid] = {
                    "source": "initial_ranking_file",
                    "parse_complete": bool(complete),
                    "model_returned": int(returned),
                }
            else:
                missing_native.append(job)

        if missing_native:
            if initial_rankings is not None and not args.allow_missing_initial_rankings:
                missing_uids = [j["user_id"] for j in missing_native[:10]]
                raise ValueError(
                    "Initial ranking file does not cover all selected users. "
                    f"First missing users: {missing_uids}. "
                    "Pass --allow-missing-initial-rankings to generate missing "
                    "Native rankings locally."
                )

            print(
                f"[INFO] Generating Native Qwen SCORE ranking for "
                f"{len(missing_native)} users "
                f"({_num_batches(len(missing_native), args.llm_batch_size)} batches)...",
                flush=True,
            )
            native_jobs = [
                {
                    "user_id": j["user_id"],
                    "history_items": j["history"],
                    "candidate_items": j["candidate_items"],
                    "method_context": None,
                }
                for j in missing_native
            ]
            native_results = shared_ranker.rank_jobs(
                native_jobs,
                batch_size=args.llm_batch_size,
                call_type="native",
            )
            for job, (ranking, meta) in zip(missing_native, native_results):
                job["initial_ranking"] = ranking
                meta["source"] = "generated_shared_native"
                native_meta_by_user[job["user_id"]] = meta

    # -------------------------------------------------------------------------
    # Stage 1: individual feedback preference probe
    # -------------------------------------------------------------------------
    print(
        f"[INFO] i²Agent Stage 1/4: feedback preference probe "
        f"({len(jobs)} users, {_num_batches(len(jobs), args.llm_batch_size)} batches)...",
        flush=True,
    )

    probe_prompts: List[str] = []
    probe_swap: Dict[str, bool] = {}
    for job in jobs:
        # Deterministic A/B swap removes positional bias while preserving reproducibility.
        swap = (
            random.Random(f"i2agent-ab:{args.seed}:{job['user_id']}")
            .randrange(2)
            == 1
        )
        probe_swap[job["user_id"]] = swap
        probe_prompts.append(
            feedback_probe_prompt(
                previous_history=job["previous_history"],
                positive_item=job["feedback_positive"],
                negative_item=job["feedback_negative"],
                swap=swap,
            )
        )

    probe_raws = gemma.generate_batch(
        probe_prompts,
        batch_size=args.llm_batch_size,
        max_new_tokens=args.profile_max_new_tokens,
        stage="feedback_probe",
    )

    for job, raw in zip(jobs, probe_raws):
        choice, ok = parse_feedback_choice(raw)
        job["feedback_probe"] = {
            "choice": choice,
            "parse_ok": bool(ok),
            "raw": raw,
            "positive_item_id": job["feedback_positive"]["item_id"],
            "negative_item_id": job["feedback_negative"]["item_id"],
            "positive_position": (
                "B" if probe_swap[job["user_id"]] else "A"
            ),
        }

    # -------------------------------------------------------------------------
    # Stage 2: profile generator
    # -------------------------------------------------------------------------
    print(
        f"[INFO] i²Agent Stage 2/4: profile generator "
        f"({len(jobs)} users, {_num_batches(len(jobs), args.llm_batch_size)} batches)...",
        flush=True,
    )

    profile_prompts = [
        profile_prompt(
            previous_history=job["previous_history"],
            feedback_item=job["feedback_positive"],
            probe_choice=job["feedback_probe"]["choice"],
        )
        for job in jobs
    ]

    profile_raws = gemma.generate_batch(
        profile_prompts,
        batch_size=args.llm_batch_size,
        max_new_tokens=args.profile_max_new_tokens,
        stage="profile_generator",
    )

    for job, raw in zip(jobs, profile_raws):
        profile, ok = parse_string_field(
            raw,
            "profile",
            fallback=(
                "The user has shown preferences reflected by the recent "
                "interaction history."
            ),
        )
        job["static_profile"] = profile
        job["profile_parse_ok"] = bool(ok)
        job["profile_raw"] = raw

    # -------------------------------------------------------------------------
    # Stage 3a: instruction-conditioned knowledge
    # With the neutral instruction identical across users, this is generated ONCE.
    # -------------------------------------------------------------------------
    if args.no_instruction:
        print("[INFO] i²Agent Stage 3a/4: skipped (--no-instruction)", flush=True)
        knowledge, knowledge_ok, knowledge_raw = "", True, ""
    else:
        print("[INFO] i²Agent Stage 3a/4: instruction knowledge (1 shared call)...", flush=True)
        knowledge_raw = gemma.generate_one(
            knowledge_prompt(args.instruction),
            max_new_tokens=args.profile_max_new_tokens,
            stage="instruction_knowledge",
        )
        knowledge, knowledge_ok = parse_string_field(
            knowledge_raw,
            "knowledge",
            fallback=(
                "Use recent item semantics, recurring categories, and recency "
                "to judge candidate compatibility."
            ),
        )

    # -------------------------------------------------------------------------
    # Stage 3b: dynamic extractor
    # -------------------------------------------------------------------------
    print(
        f"[INFO] i²Agent Stage 3b/4: dynamic extractor "
        f"({len(jobs)} users, {_num_batches(len(jobs), args.llm_batch_size)} batches)...",
        flush=True,
    )

    dynamic_prompts = [
        dynamic_extract_prompt(
            instruction="" if args.no_instruction else args.instruction,
            knowledge="" if args.no_instruction else knowledge,
            history=job["history"],
            profile=job["static_profile"],
        )
        for job in jobs
    ]

    dynamic_raws = gemma.generate_batch(
        dynamic_prompts,
        batch_size=args.llm_batch_size,
        max_new_tokens=args.profile_max_new_tokens,
        stage="dynamic_extractor",
    )

    for job, raw in zip(jobs, dynamic_raws):
        interest, dyn_profile, ok = parse_dynamic(raw)
        if not interest:
            interest = "Use the user's most recent observed preferences."
        if not dyn_profile:
            dyn_profile = job["static_profile"]
        job["dynamic_interest"] = interest
        job["dynamic_profile"] = dyn_profile
        job["dynamic_parse_ok"] = bool(ok)
        job["dynamic_raw"] = raw

    # -------------------------------------------------------------------------
    # Stage 4: final reranker
    # -------------------------------------------------------------------------
    print(
        f"[INFO] i²Agent Stage 4/4: reranking "
        f"({len(jobs)} users, {_num_batches(len(jobs), args.llm_batch_size)} batches)...",
        flush=True,
    )

    rerank_jobs: List[Dict[str, Any]] = []
    for job in jobs:
        # Candidate labels are always tied to the ORIGINAL frozen candidate order.
        id_to_label, label_to_id = build_candidate_label_maps(job["candidate_items"])
        job["id_to_label"] = id_to_label
        job["label_to_id"] = label_to_id

        initial_ranking_labels = (
            [id_to_label[iid] for iid in job["initial_ranking"] if iid in id_to_label]
            if job["initial_ranking"] is not None else []
        )

        # i2Agent contributes ONLY its method-produced profile/dynamic evidence
        # (plus the base ranking prior required by the reranker formulation).
        method_context = {
            "context_type": "i2agent_individual_dynamic_context",
            **({} if args.no_instruction else {"instruction": args.instruction, "general_knowledge": knowledge}),
            "static_user_profile": job["static_profile"],
            "dynamic_interest": job["dynamic_interest"],
            "dynamic_user_profile": job["dynamic_profile"],
            **({} if args.no_initial_ranking else {
                "base_recommender_ranking_prior": initial_ranking_labels,
            }),
            "context_semantics": (
                "profile and dynamic fields summarize evidence inferred only from TRAIN history"
                if args.no_initial_ranking else
                "the base recommender ranking is only a prior; the profile and "
                "dynamic fields summarize evidence inferred from this user's TRAIN history"
            ),
        }

        rerank_jobs.append({
            "user_id": job["user_id"],
            "history_items": job["history"],
            "candidate_items": job["candidate_items"],
            "method_context": method_context,
        })

    rerank_results = shared_ranker.rank_jobs(
        rerank_jobs,
        batch_size=args.llm_batch_size,
        call_type="i2agent",
    )

    # Fresh output file for deterministic reruns.
    if output_jsonl.exists():
        output_jsonl.unlink()

    all_rows: List[Dict[str, Any]] = []
    for job, (ranking, rerank_meta) in zip(jobs, rerank_results):
        metrics = metrics_for_ranking(
            ranking,
            job["targets"],
        )
        initial_metrics = (
            metrics_for_ranking(job["initial_ranking"], job["targets"])
            if job["initial_ranking"] is not None else None
        )

        row = {
            "schema_version": SCHEMA_VERSION,
            "rank_prompt_version": SHARED_RANK_PROMPT_VERSION,
            "ranking_mode": "score",
            "user_id": job["user_id"],
            "rerank_scores_by_label": rerank_meta.get("scores_by_label", {}),
            "ground_truth_item_ids": job["targets"],
            "history_item_ids": job["history_ids"],
            "history": job["history"],
            "candidate_item_ids": job["candidate_item_ids"],
            "candidate_items": job["candidate_items"],
            "candidate_label_to_item_id": job["label_to_id"],
            "initial_ranking_item_ids": job["initial_ranking"],
            "initial_ranking_meta": native_meta_by_user[job["user_id"]],
            "feedback_memory": {
                "positive_train_item": job["feedback_positive"],
                "negative_probe_item": job["feedback_negative"],
                "probe": job["feedback_probe"],
            },
            "instruction": "" if args.no_instruction else args.instruction,
            "knowledge": knowledge,
            "knowledge_parse_ok": bool(knowledge_ok),
            "static_profile": job["static_profile"],
            "profile_parse_ok": job["profile_parse_ok"],
            "dynamic_interest": job["dynamic_interest"],
            "dynamic_profile": job["dynamic_profile"],
            "dynamic_parse_ok": job["dynamic_parse_ok"],
            "reranked_item_ids": ranking,
            "rerank_parse_json_ok": rerank_meta["parse_json_ok"],
            "rerank_parse_complete": rerank_meta["parse_complete"],
            "rerank_model_returned": rerank_meta["model_returned"],
            "ranking_reasoning": rerank_meta["reasoning"],
            "ranking_meta": rerank_meta,
            "metrics": metrics,
            "initial_metrics": initial_metrics,
        }
        all_rows.append(row)
        append_jsonl(output_jsonl, row)

    all_metrics = aggregate_metrics(all_rows)
    complete_rows = [
        r for r in all_rows
        if r["rerank_parse_complete"]
    ]
    complete_metrics = aggregate_metrics(complete_rows)

    initial_eval_rows = [
        {"metrics": r["initial_metrics"]}
        for r in all_rows if r["initial_metrics"] is not None
    ]
    initial_metrics = aggregate_metrics(initial_eval_rows) if initial_eval_rows else None

    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "i²Agent individual dynamic memory adapted to the "
            "CoMemTree frozen-candidate protocol"
        ),
        "num_users": len(all_rows),
        "model": args.model,
        "ranking_mode": "score",
        "protocol": snapshot["protocol"],
        "initial_ranking": {
            "enabled": not args.no_initial_ranking,
            "source_file": args.initial_ranking_file,
            "mode": args.initial_ranking_mode,
            "generated_locally_for_missing": len(missing_native),
            "metrics": initial_metrics,
        },
        "i2agent_metrics_all_rows": all_metrics,
        "i2agent_metrics_parse_complete_only": complete_metrics,
        "parse_quality": {
            "rerank_complete_count": int(len(complete_rows)),
            "rerank_complete_ratio": (
                float(len(complete_rows) / len(all_rows))
                if all_rows else 0.0
            ),
            "profile_parse_ok_ratio": (
                float(
                    sum(bool(r["profile_parse_ok"]) for r in all_rows)
                    / len(all_rows)
                )
                if all_rows else 0.0
            ),
            "dynamic_parse_ok_ratio": (
                float(
                    sum(bool(r["dynamic_parse_ok"]) for r in all_rows)
                    / len(all_rows)
                )
                if all_rows else 0.0
            ),
            "knowledge_parse_ok": bool(knowledge_ok),
        },
        "llm_stats": {
            "i2agent_context_generation": gemma.stats(),
            "shared_candidate_ranking": shared_ranker.stats(),
        },
        "adaptation_notes": [
            "No free-text InstructRec instruction exists; fixed neutral instruction is used unless --no-instruction.",
            "No review text is used.",
            "The latest TRAIN interaction is used as positive feedback for the individual profile generator.",
            "The feedback negative is sampled outside history and the entire current candidate set, preventing test-target leakage.",
            "Only title/category/item_id are exposed during method context construction; shared candidate ranking shows title/category under C-labels.",
            "Final ranking uses the original shared task/evidence layout, but requests numeric C-label scores, sorted descending; implemented within this file.",
            "The held-out test target is used only for evaluation.",
            "With --no-initial-ranking, Stage 0 is skipped entirely and no Native ranked prior is inserted into the final prompt.",
        ],
    }
    dump_json(summary_path, summary)
    if args.debug_ranking:
        ranking_debug_path = output_dir / "shared_ranking_debug.json"
        dump_json(
            ranking_debug_path,
            {
                "summary": shared_ranker.stats(),
                "records": shared_ranker.tracking_records(),
            },
        )
        print(f"[RANK-DEBUG] Saved: {ranking_debug_path}", flush=True)

    print("\nRESULTS", flush=True, file=sys.__stdout__)
    print(json.dumps(summary["i2agent_metrics_all_rows"], indent=2), flush=True, file=sys.__stdout__)
    print(
        "\nPARSE-COMPLETE ONLY",
        json.dumps(summary["i2agent_metrics_parse_complete_only"], indent=2),
        flush=True,
    )
    print(
        f"\nSaved:\n  {output_jsonl}\n  {summary_path}\n  {snapshot_path}",
        flush=True,
    )

    # Free GPU memory cleanly at script end.
    del gemma
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run i²Agent using SCORE ranking with the same frozen protocol and "
            "Unsloth Qwen backbone as CoMemTree."
        )
    )

    # Dataset/protocol selection.
    p.add_argument(
        "--data-dir",
        default="data/CDs",
        help=(
            "Dataset directory containing items.json, user_sequences.json, "
            "user_negatives.json and the frozen protocol artifacts."
        ),
    )
    p.add_argument(
        "--num-train-users",
        type=int,
        choices=[100, 300, 500, 1000],
        default=300,
        help=(
            "Frozen training-user list size. i2Agent loads this list only for "
            "protocol validation/disjointness; it has no cross-user training stage."
        ),
    )
    p.add_argument(
        "--num-test-users",
        type=int,
        choices=[100, 300, 500, 1000],
        default=300,
        help="Frozen evaluation-user list size.",
    )

    # Optional path overrides. If omitted, paths are resolved from --data-dir.
    p.add_argument("--items", default=None)
    p.add_argument("--sequences", default=None)
    p.add_argument("--negatives", default=None)
    p.add_argument("--train-users-file", default=None)
    p.add_argument(
        "--eval-users-file", "--user-ids-file",
        dest="eval_users_file",
        default=None,
    )
    p.add_argument(
        "--candidate-file",
        default=None,
        help=(
            "Optional override. Default: "
            "<data-dir>/user_candidates_testpool1000_seed<seed>.json"
        ),
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
        help=(
            "Optional smoke-test cap applied after loading eval_users_N.json. "
            "Use 0 for official runs."
        ),
    )

    # Initial ranking / recommender output.
    p.add_argument(
        "--initial-ranking-file",
        default=None,
        help=(
            "Recommended: exact Native Qwen ranking output from CoMemTree (list/score both readable). "
            "The script supports JSON/JSONL and common ranking keys."
        ),
    )
    p.add_argument(
        "--initial-ranking-mode",
        default="native",
        help=(
            "When the source file contains rankings by mode, read this mode. "
            "Default: native."
        ),
    )
    p.add_argument(
        "--allow-missing-initial-rankings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If the initial-ranking file misses users, generate those Native "
            "rankings locally. Disabled by default for strict reproducibility."
        ),
    )

    p.add_argument(
        "--no-initial-ranking", action="store_true",
        help=("Skip Stage 0 entirely and do not include a Native ranking prior "
              "in the final score-ranking prompt; use original frozen candidates."),
    )

    # Same experimental constraints as CoMemTree.
    p.add_argument(
        "--history-size",
        type=int,
        default=10,
    )
    p.add_argument(
        "--expected-candidates",
        type=int,
        default=20,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Same Qwen setup; all native-generated and final ranking outputs are scores.
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
    )
    p.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--llm-batch-size",
        type=int,
        default=4,
    )
    p.add_argument(
        "--profile-max-new-tokens",
        type=int,
        default=256,
    )
    p.add_argument(
        "--rank-max-new-tokens",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--debug-ranking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save shared-ranking diagnostics to file: prompt truncation, "
            "context-budget risk, output max-token limit, and parse completeness. "
            "This flag applies to the shared candidate-ranking calls; i2Agent's "
            "auxiliary profile/dynamic-generation stages are unchanged."
        ),
    )

    # No target-derived instruction.
    p.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help=(
            "Fixed neutral instruction used for ALL users. "
            "Do not derive this from the target/review."
        ),
    )

    p.add_argument(
        "--no-instruction", action="store_true",
        help="Disable fixed instruction and shared instruction-conditioned knowledge generation.",
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. If omitted, uses "
            "results/i2agent_<dataset>_tr<N>_te<M>."
        ),
    )

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if args.items is None:
        args.items = str(data_dir / "items.json")
    if args.sequences is None:
        args.sequences = str(data_dir / "user_sequences.json")
    if args.negatives is None:
        args.negatives = str(data_dir / "user_negatives.json")
    if args.train_users_file is None:
        args.train_users_file = str(
            data_dir / f"train_users_{args.num_train_users}.json"
        )
    if args.eval_users_file is None:
        args.eval_users_file = str(
            data_dir / f"eval_users_{args.num_test_users}.json"
        )
    if args.candidate_file is None:
        args.candidate_file = str(
            data_dir / f"user_candidates_testpool1000_seed{args.seed}.json"
        )
    if args.output_dir is None:
        dataset_name = data_dir.name or "dataset"
        args.output_dir = str(
            Path("results")
            / f"i2agent_qwen_score_{dataset_name}_tr{args.num_train_users}_te{args.num_test_users}"
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    runtime_dir = Path(args.output_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with (runtime_dir / "runtime.log").open("w", encoding="utf-8", buffering=1) as runtime_log:
        runtime_log.write(_startup_log.getvalue())
        with redirect_stdout(runtime_log), redirect_stderr(runtime_log):
            main(args)
