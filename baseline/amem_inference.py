#!/usr/bin/env python3
"""aMem4Rec inference under the CoMemTree frozen protocol.

* Users: data/<DS>/protocol/eval_users_1000.json (in that order).
* Candidates: the frozen 20-candidate file, read-only (never rebuilt or rewritten).
* History: last 20 TRAIN items (ML 30) from configs/datasets/<DS>.yaml.
* Memory: global pool from baseline/amem_build_memory.py (built from CoMemTree's
  train_shared.jsonl). The builder stores CENTERED centroids, so the query is
  centered with metadata.epsilon_net.mu before cosine search (--query-centering auto).
  The previous script searched with an uncentered query.
* Metrics: Hit/NDCG@1,3,5,10, MRR (all users and parse-complete only).
* Cost: <out>/cost_summary.json (bt4rec_cost_v1). online = 1 ranking call + 1 query
  embedding per user; offline = the build cost read from
  <global_memory stem>_cost_summary.json; memory = global pool size, k retrieved/user.

  python baseline/amem_inference.py --dataset CDs
  python baseline/amem_inference.py --dataset CDs --no_memory      # same prompt, no memory
  python baseline/amem_inference.py --dataset CDs --raw_memory     # same prompt, NO build step

--raw_memory skips baseline/amem_build_memory.py entirely: every local behaviour window
of precomputed/<DS>/train_shared.jsonl (train_users_1000 only) is one memory row, embedded
on the fly with the same text + embedding model as the builder. No epsilon-net, no
centering, no Gemma evolution/merge, no co-occurrence links. Retrieval (k, query text,
FAISS exact cosine) and the ranking prompt are unchanged.
"""
import argparse, json, os, random, sys, time, re
from unsloth import FastLanguageModel  # Import before torch/transformers.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bt4rec.eval.cost import CostTracker, memory_block  # noqa: E402
from bt4rec.eval.metrics import metrics_from_ranks, target_rank  # noqa: E402

RANK_PROMPT_VERSION = "amem_qwen_clabel_list_v1"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_ids(value):
    """Normalize a scalar ID or list without splitting strings into characters."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        if item is None or isinstance(item, (bool, dict, list, tuple)):
            raise ValueError(f"Invalid item ID: {item!r}")
        iid = str(item)
        if not iid:
            raise ValueError("Empty item ID")
        result.append(iid)
    return result


def normalize_items(items: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in items.items()}


def get_item_info(item_id: Any, items_meta: Dict[str, Any]) -> Dict[str, str]:
    iid = str(item_id)
    info = items_meta.get(iid, {})
    category = info.get("main_cat") or info.get("category") or info.get("categories") or "Unknown"
    if isinstance(category, list):
        category = " > ".join(map(str, category[:3]))
    return {
        "item_id": iid,
        "title": str(info.get("title") or f"Unknown Item {iid}"),
        "category": str(category),
    }


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        text = text[s:e]
    return json.loads(text)


class FlatMemoryRetriever:
    """Exact cosine retrieval using normalized vectors + FAISS IndexFlatIP."""

    def __init__(self, memory_path: str, embedding_model: str, device: str = "auto",
                 query_centering: str = "auto"):
        data = load_json(memory_path)
        self.memories = data.get("behavior_memories", [])
        meta = data.get("metadata", {}) or {}
        built_with = meta.get("embedding_model")
        if built_with and built_with != embedding_model:
            raise ValueError(f"Global memory was embedded with {built_with!r}, not {embedding_model!r}")
        mu = (meta.get("epsilon_net") or {}).get("mu")
        centered = bool((meta.get("run_config") or {}).get("center", mu is not None)) and mu is not None
        if query_centering == "on" and mu is None:
            raise ValueError("--query-centering on, but the memory file has no epsilon_net.mu")
        self.mu = (np.asarray(mu, dtype=np.float32)
                   if (query_centering == "on" or (query_centering == "auto" and centered)) else None)
        self.num_links = sum(len(m.get("links") or []) for m in self.memories)
        if not self.memories:
            raise RuntimeError("Global memory contains no behavior_memories")

        embs = np.asarray([m["embedding"] for m in self.memories], dtype=np.float32)
        if embs.ndim != 2:
            raise ValueError(f"Invalid embedding matrix shape: {embs.shape}")

        embs = np.ascontiguousarray(embs)
        faiss.normalize_L2(embs)

        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.search_calls = 0
        self.total_search_time = 0.0

        print(f"Loaded {len(self.memories)} memories; dim={self.dim}")
        print("Retrieval: FAISS IndexFlatIP (exact cosine)")

    def retrieve(self, profile_text: str, k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        if k <= 0:
            return [], []

        q = self.encoder.encode(
            profile_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        if self.mu is not None:
            # Memory rows are centroids in the centered space (builder default).
            q = q - self.mu[None, :]
        q = np.ascontiguousarray(q)
        if q.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {q.shape[1]} != memory dim {self.dim}. "
                "Use the same embedding model used to build global memory."
            )
        faiss.normalize_L2(q)

        k = min(k, len(self.memories))
        t0 = time.perf_counter()
        scores, idx = self.index.search(q, k)
        self.total_search_time += time.perf_counter() - t0
        self.search_calls += 1

        memories, sims = [], []
        for i, s in zip(idx[0], scores[0]):
            if i >= 0:
                memories.append(self.memories[int(i)])
                sims.append(float(s))
        return memories, sims

    def stats(self):
        return {
            "num_memories": len(self.memories),
            "query_centered": self.mu is not None,
            "search_calls": self.search_calls,
            "total_search_time_sec": round(self.total_search_time, 6),
            "avg_search_time_ms": (
                round(self.total_search_time / self.search_calls * 1000, 4)
                if self.search_calls else 0.0
            ),
        }


RAW_PRECOMPUTE_SCHEMA = "amem_local_memory_dual_behavior_v4"  # same contract as the builder


def raw_memory_text(rec: Dict[str, Any]) -> str:
    """Identical to amem_build_memory.canonical_text_from_record (not imported: that
    module loads Unsloth/Gemma at import time)."""
    kw = rec.get("keywords", [])
    if not isinstance(kw, list):
        kw = [str(kw)]
    return (f"{rec.get('behavior_explanation', '')} "
            f"{rec.get('pattern_description', '')} "
            f"{' '.join(str(x) for x in kw)}").strip()


def load_raw_local_memories(path: str, allowed_users: Optional[set]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            schema = rec.get("schema_version")
            if schema is not None and schema != RAW_PRECOMPUTE_SCHEMA:
                raise RuntimeError(f"{path}:{line_no}: schema_version={schema!r}, expected {RAW_PRECOMPUTE_SCHEMA!r}")
            if rec.get("parse_ok") is False:
                continue
            missing = [k for k in ("precompute_id", "behavior_explanation", "pattern_description", "keywords")
                       if k not in rec]
            if missing:
                raise RuntimeError(f"{path}:{line_no}: missing fields {missing}")
            uid = rec.get("user_id")
            if allowed_users is not None and str(uid) not in allowed_users:
                continue
            rows.append(rec)
    rows.sort(key=lambda r: int(r["precompute_id"]))
    ids = [int(r["precompute_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate precompute_id values in raw memory input")
    if not rows:
        raise RuntimeError(f"No raw local memories in {path} for the selected train users")
    return rows


class RawMemoryRetriever(FlatMemoryRetriever):
    """aMem4Rec without the build step: one memory per local behaviour window."""

    def __init__(self, precomputed_path: str, embedding_model: str, device: str = "auto",
                 allowed_users: Optional[set] = None, batch_size: int = 64):
        records = load_raw_local_memories(precomputed_path, allowed_users)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=device)
        texts = [raw_memory_text(r) for r in records]
        t0 = time.perf_counter()
        embs = self.encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=True)
        self.encode_time = time.perf_counter() - t0
        embs = np.ascontiguousarray(np.asarray(embs, dtype=np.float32))
        faiss.normalize_L2(embs)
        self.memories = [{
            "thought_id": int(r["precompute_id"]),
            "behavior_explanation": str(r.get("behavior_explanation", "")),
            "pattern_description": str(r.get("pattern_description", "")),
            "source_user_id": None if r.get("user_id") is None else str(r.get("user_id")),
            "source_window_index": r.get("window_index"),
        } for r in records]
        self.num_texts = len(texts)
        self.num_source_users = len({m["source_user_id"] for m in self.memories})
        self.mu = None  # raw rows are NOT centered; query is not centered either
        self.num_links = 0
        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)
        self.search_calls = 0
        self.total_search_time = 0.0
        print(f"Raw memory (no build): {len(self.memories)} local windows from "
              f"{self.num_source_users} users; dim={self.dim}; encode {self.encode_time:.1f}s")

    def stats(self):
        out = super().stats()
        out.update({"memory_source": "raw_local_windows_no_build",
                    "num_source_users": self.num_source_users,
                    "offline_encode_time_sec": round(self.encode_time, 4)})
        return out


class LocalRanker:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 1024,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Unsloth Qwen inference expects a CUDA GPU.")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        if self.model.config.model_type != "qwen2":
            raise ValueError("Use a Qwen2/Qwen2.5 Instruct checkpoint with this ranker")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = int(max_new_tokens)
        self.max_seq_length = int(max_seq_length)

        # calls = number of model.generate() invocations, not number of users.
        self.calls = 0
        self.samples = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_time = 0.0
        self.records = []

    @staticmethod
    def build_prompt(user_profile, candidates, memory_thoughts):
        memory_block = ""
        if memory_thoughts:
            memory_block = (
                "\nCollaborative memory (secondary evidence):\n"
                + json.dumps(memory_thoughts, indent=2, ensure_ascii=False)
            )

        # Expose C-labels only, never raw IDs or item descriptions.
        user_profile = [{"title": x.get("title", ""), "category": x.get("category", "")}
                        for x in user_profile]
        candidates = [{"item_id": f"C{i:02d}", "title": x.get("title", ""),
                       "category": x.get("category", "")} for i, x in enumerate(candidates, 1)]
        n = len(candidates)
        return f"""You are a recommendation ranking system.

Rank ALL candidate items for the target user.

Priority:
1. Match the user's recent preferences and behavioral pattern.
2. Use category and semantic compatibility.
3. Use retrieved collaborative memory as additional evidence when relevant.

User recent history (most recent last):
{json.dumps(user_profile, indent=2, ensure_ascii=False)}
{memory_block}

Candidate items:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Requirements:
- Rank ALL {n} candidates.
- Each candidate C-label (C01, C02, ...) must appear exactly once.
- Do not rank history items or return scores. Put ranked_item_ids before reasoning.
- Do not invent item IDs.
- Return ONLY valid JSON.

{{"ranked_item_ids": ["C02", "C01", "..."], "reasoning": "1 concise sentence"}}"""

    def _render_prompt(self, user_profile, candidates, memory_thoughts) -> str:
        prompt = self.build_prompt(user_profile, candidates, memory_thoughts)
        messages = [
            {"role": "system", "content": "You are a recommendation ranking system."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

    @staticmethod
    def _clean_prediction(response: str, candidates, user_id: str):
        valid_ids = [str(x["item_id"]) for x in candidates]
        if len(set(valid_ids)) != len(valid_ids):
            raise ValueError(f"Duplicate candidates for user {user_id}")
        mapping = {f"C{i:02d}": iid for i, iid in enumerate(valid_ids, 1)}
        reasoning = ""
        try:
            obj = parse_json_response(response)
            pred = obj.get("ranked_item_ids", obj.get("ranking", []))
            reasoning = str(obj.get("reasoning", ""))
            if not isinstance(pred, list):
                pred = []
        except (ValueError, TypeError, AttributeError):
            # Recover ONLY the ranking array, never labels inside reasoning.
            match = re.search(r'"(?:ranked_item_ids|ranking)"\s*:\s*\[([^\]]*)(?:\]|$)', response, re.S)
            pred = re.findall(r'"(C[0-9]+)"', match.group(1)) if match else []
        cleaned, seen = [], set()
        for label in pred:
            if isinstance(label, str) and label in mapping and label not in seen:
                cleaned.append(mapping[label])
                seen.add(label)
        ok = bool(cleaned)
        complete = len(cleaned) == len(mapping)
        cleaned.extend(iid for label, iid in mapping.items() if label not in seen)
        return cleaned, reasoning, ok, complete

    def _fit_prompt(self, job):
        history = list(job["user_profile"])
        memories = list(job.get("memory_thoughts") or [])
        budget = self.max_seq_length - self.max_new_tokens
        if budget <= 0:
            raise ValueError("max_seq_length must exceed max_new_tokens")
        dropped_history = dropped_memories = 0
        full_tokens = None
        while True:
            text = self._render_prompt(history, job["candidates"], memories)
            count = len(self.tokenizer.encode(text, add_special_tokens=False))
            if full_tokens is None:
                full_tokens = count
            if count <= budget:
                return text, {"full_input_tokens": full_tokens,
                              "truncated": bool(dropped_history or dropped_memories),
                              "dropped_history": dropped_history,
                              "dropped_memories": dropped_memories}
            # Preserve the entire candidate pool and output instructions.
            if memories:
                memories.pop(); dropped_memories += 1
            elif len(history) > 1:
                history.pop(0); dropped_history += 1
            else:
                raise ValueError(f"user={job['user_id']}: candidates + minimum history exceed context; increase max_seq_length")

    def rank_batch(self, jobs: List[Dict[str, Any]], batch_size: int = 8):
        """
        Rank independent users in GPU batches.

        Each job keeps its own prompt/history/candidates. Users are never merged
        into one prompt; batching only packs independent prompts into one
        model.generate() call.
        """
        if not jobs:
            return []

        batch_size = max(1, int(batch_size))
        all_results = []

        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start + batch_size]
            fitted = [self._fit_prompt(j) for j in chunk]
            texts = [x[0] for x in fitted]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(self.device)

            # With left padding, generated tokens begin after the common padded
            # input width for every sample in the batch.
            input_width = int(inputs["input_ids"].shape[1])
            per_sample_input_tokens = [
                int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()
            ]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            generated = outputs[:, input_width:]
            responses = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            self.calls += 1
            self.samples += len(chunk)
            self.total_time += elapsed
            self.input_tokens += sum(per_sample_input_tokens)

            for local_idx, (job, response) in enumerate(zip(chunk, responses)):
                # Generated tensors are padded to a common width. Count only
                # non-pad generated tokens for tracking.
                gen_row = generated[local_idx]
                if self.tokenizer.pad_token_id is None:
                    n_out = int(gen_row.numel())
                else:
                    n_out = int((gen_row != self.tokenizer.pad_token_id).sum().item())
                self.output_tokens += n_out

                pred, reasoning, parse_ok, complete = self._clean_prediction(
                    response.strip(),
                    job["candidates"],
                    job["user_id"],
                )

                self.records.append({
                    "user_id": str(job["user_id"]),
                    "batch_size": len(chunk),
                    "input_tokens": per_sample_input_tokens[local_idx],
                    "output_tokens": n_out,
                    "batch_inference_time_sec": round(elapsed, 4),
                    "approx_inference_time_per_sample_sec": round(elapsed / len(chunk), 4),
                    "parse_ok": bool(parse_ok),
                    "parse_complete": bool(complete),
                    "raw_output": response,
                    **fitted[local_idx][1],
                })

                all_results.append((pred, reasoning, self.records[-1]))

        return all_results

    def rank(self, user_id: str, user_profile, candidates, memory_thoughts):
        """Compatibility wrapper for batch_size=1."""
        return self.rank_batch(
            [{
                "user_id": str(user_id),
                "user_profile": user_profile,
                "candidates": candidates,
                "memory_thoughts": memory_thoughts,
            }],
            batch_size=1,
        )[0][:2]

    def stats(self):
        return {
            "generate_calls": self.calls,
            "samples": self.samples,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "total_inference_time_sec": round(self.total_time, 4),
            "avg_generate_call_time_sec": (
                round(self.total_time / self.calls, 4) if self.calls else 0.0
            ),
            "avg_sample_time_sec": (
                round(self.total_time / self.samples, 4) if self.samples else 0.0
            ),
        }

def prepare_one_user(
    user_id: str,
    user_data: Dict[str, Any],
    neg_data: Dict[str, Any],
    items_meta: Dict[str, Any],
    retriever: Optional[FlatMemoryRetriever],
    k_memories: int,
    history_size: int,
    seed: int,
    candidate_row: Optional[Dict[str, Any]] = None,
):
    """Prepare one independent ranking job without calling Qwen."""
    train_ids = normalize_ids(user_data.get('train', []))
    test_ids = normalize_ids(user_data.get('test', []))
    neg_ids = normalize_ids(neg_data.get('test_neg', []))

    if not test_ids:
        raise ValueError("No test item")

    history = [get_item_info(i, items_meta) for i in train_ids[-history_size:]]
    profile_text = " ".join(f"{x['title']} {x['category']}" for x in history)

    if candidate_row is not None:
        candidate_ids = normalize_ids(candidate_row.get('candidates', candidate_row.get('candidate_item_ids', [])))
        cached_target = normalize_ids(candidate_row.get('target', candidate_row.get('ground_truth', candidate_row.get('ground_truth_item_ids', []))))
        if cached_target and cached_target != test_ids:
            raise ValueError(f"user={user_id}: cached target={cached_target!r} differs from sequence test={test_ids!r}")
        if not candidate_ids:
            raise ValueError(f"Empty candidate cache row for user {user_id}")
    else:
        candidate_ids = test_ids + neg_ids
        rng = random.Random(f"{seed}:{user_id}")
        rng.shuffle(candidate_ids)

    candidates = [get_item_info(i, items_meta) for i in candidate_ids]

    memory_thoughts = []
    if retriever is not None and k_memories > 0:
        memories, scores = retriever.retrieve(profile_text, k_memories)
        for m, s in zip(memories, scores):
            memory_thoughts.append({
                "thought_id": m.get("thought_id"),
                "behavior_explanation": m.get("behavior_explanation", ""),
                "pattern": m.get("pattern_description", ""),
                "similarity": round(float(s), 4),
            })

    return {
        "user_id": str(user_id),
        "ground_truth_item_ids": test_ids,
        "candidate_item_ids": candidate_ids,
        "candidate_items": candidates,
        "history": history,
        "retrieved_memories": memory_thoughts,
    }


def finalize_one_user(
    prepared: Dict[str, Any],
    pred: List[str],
    reasoning: str,
    items_meta: Dict[str, Any],
    record: Optional[Dict[str, Any]] = None,
):
    record = record or {}
    return {
        "target_rank": target_rank(pred, prepared["ground_truth_item_ids"][0]),
        "parse_ok": record.get("parse_ok"),
        "parse_complete": record.get("parse_complete"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "num_retrieved_memories": len(prepared["retrieved_memories"]),
        "user_id": prepared["user_id"],
        "ground_truth_item_ids": prepared["ground_truth_item_ids"],
        "candidate_item_ids": prepared["candidate_item_ids"],
        "reranked_item_ids": pred,
        "candidate_items": prepared["candidate_items"],
        "reranked_items": [get_item_info(i, items_meta) for i in pred],
        "retrieved_memories": prepared["retrieved_memories"],
        "ranking_reasoning": reasoning,
        "ranking_model": prepared["ranking_model"],
        "rank_prompt_version": RANK_PROMPT_VERSION,
    }

def load_candidate_file(path: str) -> Dict[str, Dict[str, Any]]:
    obj = load_json(path)
    rows: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                rows[str(k)] = v
    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            uid = row.get("user_id", row.get("uid", row.get("user")))
            if uid is not None:
                rows[str(uid)] = row
    if not rows:
        raise ValueError(f"Could not parse candidate file: {path}")
    return rows


def load_frozen_candidates(
    path: str,
    users: List[str],
    sequences: Dict[str, Any],
    negatives: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Read the shared frozen candidate file and validate it; never writes it."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Frozen candidate file not found: {path}")
    rows = load_candidate_file(path)
    for uid in users:
        if uid not in rows:
            raise RuntimeError(f"user={uid} missing from frozen candidate file {path}")
        target = normalize_ids(sequences[uid].get("test", []))
        cached = normalize_ids(rows[uid].get("candidates", rows[uid].get("candidate_item_ids", [])))
        cached_target = normalize_ids(rows[uid].get("target", rows[uid].get(
            "ground_truth", rows[uid].get("ground_truth_item_ids", []))))
        if cached_target and cached_target != target:
            raise ValueError(f"user={uid}: frozen target differs from user_sequences test item")
        # The frozen file is authoritative for negatives (metadata-filtered,
        # possibly pooled back-fill), exactly as in CoMemTree inference.
        if len(cached) != len(set(cached)):
            raise ValueError(f"user={uid}: duplicate candidates in frozen file")
        if not target or cached.count(target[0]) != 1:
            raise ValueError(f"user={uid}: target must occur exactly once in frozen candidates")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, default=None, help="CDs | Video_Games | ML (sets all defaults)")
    p.add_argument("--global_memory", "--global-memory", dest="global_memory", type=str, default=None,
                   help="default results/baselines/<DS>/amem/global_memory.json")
    p.add_argument("--items", type=str, default=None)
    p.add_argument("--sequences", type=str, default=None)
    p.add_argument("--negatives", type=str, default=None)
    p.add_argument("--user_ids_file", "--user-ids-file", dest="user_ids_file", type=str, default=None,
                   help="default data/<DS>/protocol/eval_users_1000.json")
    p.add_argument("--output", type=str, default=None,
                   help="default results/baselines/<DS>/amem[_no_memory]/amem_rankings.json")
    p.add_argument("--candidate_file", "--candidate-file", dest="candidate_file", type=str, default=None,
                   help="frozen candidate file (read-only); default from configs")
    p.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embedding_device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--query_centering", "--query-centering", dest="query_centering",
                   choices=["auto", "on", "off"], default="auto",
                   help="auto: center the query with the builder's mu when the pool is centered")
    p.add_argument("--k_memories", type=int, default=3)
    p.add_argument("--history_size", "--history-size", dest="history_size", type=int, default=None,
                   help="default from configs (20; ML 30)")
    p.add_argument("--number_of_users", type=int, default=0, help="0 = all eval users")
    p.add_argument("--start_user", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--llm_batch_size", "--llm-batch-size", dest="llm_batch_size", type=int, default=8,
                   help="Independent user prompts per model.generate() call. Lower to 4/2 if GPU OOM.")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_memory", action="store_true", help="same prompt without retrieved memory")
    p.add_argument("--raw_memory", action="store_true",
                   help="skip amem_build_memory.py: retrieve directly from raw local behaviour windows")
    p.add_argument("--precomputed", type=str, default=None,
                   help="--raw_memory input; default precomputed/<DS>/train_shared.jsonl")
    p.add_argument("--train_users_file", "--train-users-file", dest="train_users_file", type=str, default=None,
                   help="--raw_memory source users; default data/<DS>/protocol/train_users_1000.json; 'none' = all")
    p.add_argument("--embedding_batch_size", type=int, default=64)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    cfg = {}
    if a.dataset:
        from bt4rec.config import load_dataset_config
        cfg = load_dataset_config(a.dataset)
        a.dataset = cfg["dataset"]
        data = Path("data") / a.dataset
        a.items = a.items or str(data / "items.json")
        a.sequences = a.sequences or str(data / "user_sequences.json")
        a.negatives = a.negatives or str(data / "user_negatives.json")
        a.user_ids_file = a.user_ids_file or cfg.get("eval_users")
        a.candidate_file = a.candidate_file or cfg.get("candidate_file")
        out_dir = Path("results/baselines") / a.dataset
        a.global_memory = a.global_memory or str(out_dir / "amem" / "global_memory.json")
        tag = "amem_no_memory" if a.no_memory else ("amem_raw_memory" if a.raw_memory else "amem")
        a.output = a.output or str(out_dir / tag / "amem_rankings.json")
        if a.precomputed is None:
            base = Path(cfg.get("precomputed_dir", f"precomputed/{a.dataset}"))
            cand = [base / "train_shared.jsonl", base / "qwen" / "train_shared.jsonl"]
            a.precomputed = str(next((c for c in cand if c.exists()), cand[0]))
        if a.train_users_file is None:
            a.train_users_file = cfg.get("train_users")
    if a.no_memory and a.raw_memory:
        raise SystemExit("--no_memory and --raw_memory are mutually exclusive")
    if a.train_users_file and a.train_users_file.lower() == "none":
        a.train_users_file = None
    if a.history_size is None:
        a.history_size = int(cfg.get("history_size", 20))
    for name in ("items", "sequences", "user_ids_file", "candidate_file", "output"):
        if not getattr(a, name):
            raise SystemExit(f"--{name} is required (or pass --dataset)")
    return a


def load_user_list(path: str) -> List[str]:
    obj = load_json(path)
    if isinstance(obj, dict):
        obj = obj.get("users", list(obj.keys()))
    users = [str(x) for x in obj]
    if len(users) != len(set(users)):
        raise ValueError(f"Duplicate users in {path}")
    return users


def build_cost(args, results, user_ids, retriever_stats):
    cost = CostTracker("amem4rec_no_memory" if args.no_memory else
                       ("amem4rec_raw_memory" if args.raw_memory else "amem4rec"))
    by_user = {str(r["user_id"]): r for r in results}
    for uid in user_ids:
        r = by_user.get(uid)
        if r is None:
            continue
        cost.llm("ranking", uid, int(r.get("input_tokens") or 0), int(r.get("output_tokens") or 0))
    if not args.no_memory:
        cost.embed("online", n_texts=len(by_user), n_calls=len(by_user))
    summary = cost.summary(user_ids)
    summary["online"]["token_counts_available"] = all(
        by_user[u].get("input_tokens") is not None for u in by_user)
    if args.no_memory:
        summary["memory"] = memory_block(0, {u: 0 for u in user_ids}, scope="none")
        return summary
    n = max(1, len(user_ids))
    if args.raw_memory:
        # No build: 0 LLM calls offline; only the one-off embedding of local windows.
        nm = int(retriever_stats.get("num_memories") or 0)
        summary["offline"] = {"llm_calls_total": 0, "input_tokens_total": 0, "output_tokens_total": 0,
                              "total_tokens_total": 0, "llm_calls_per_user": 0.0,
                              "embedding": {"calls": 1, "texts": nm},
                              "encode_time_sec": retriever_stats.get("offline_encode_time_sec"),
                              "note": "raw local windows embedded at inference start; no Gemma build"}
        summary["upstream_reused"] = {"behaviour_extraction": "precomputed train_shared.jsonl "
                                      "(1 LLM call / 3-item window, shared with CoMemTree)",
                                      "precomputed": args.precomputed}
        summary["memory"] = memory_block(nm, {}, scope="global pool of raw local windows (no build)",
                                         source_users=retriever_stats.get("num_source_users"))
    build_cost_path = (None if args.raw_memory else
                       Path(args.global_memory).with_name(Path(args.global_memory).stem + "_cost_summary.json"))
    if args.raw_memory:
        pass  # offline/memory already filled above
    elif build_cost_path.exists():
        build = load_json(str(build_cost_path))
        off = build.get("offline", {})
        for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens"):
            if f"{key}_total" in off:
                off[f"{key}_per_user"] = off[f"{key}_total"] / n
        summary["offline"] = off
        summary["upstream_reused"] = build.get("upstream_reused", {})
        summary["memory"] = build.get("memory", {})
        summary["build_cost_file"] = str(build_cost_path)
    else:
        summary["offline"] = {"note": f"build cost file not found: {build_cost_path}"}
        summary["memory"] = {"memories_created_total": retriever_stats.get("num_memories"),
                             "scope": "global pool shared by all users"}
    summary["memory"]["retrieved_per_user"] = float(np.mean(
        [by_user[u].get("num_retrieved_memories", 0) for u in by_user])) if by_user else 0.0
    online, offline = summary["online"], summary["offline"]
    summary["end_to_end_per_eval_user"] = {
        "llm_calls": (online["llm_calls_total"] + offline.get("llm_calls_total", 0)) / n,
        "input_tokens": (online["input_tokens_total"] + offline.get("input_tokens_total", 0)) / n,
        "output_tokens": (online["output_tokens_total"] + offline.get("output_tokens_total", 0)) / n,
        "note": "online + offline (amortised). Upstream behaviour extraction listed separately.",
    }
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    for p in [args.items, args.sequences, args.user_ids_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if args.raw_memory:
        if not args.precomputed or not os.path.exists(args.precomputed):
            raise FileNotFoundError(f"--raw_memory needs --precomputed train_shared.jsonl: {args.precomputed}")
    elif not args.no_memory:
        if not args.global_memory:
            raise ValueError("--global_memory is required unless --no_memory is used")
        if not os.path.exists(args.global_memory):
            raise FileNotFoundError(args.global_memory)

    items_meta = normalize_items(load_json(args.items))
    sequences = {str(k): v for k, v in load_json(args.sequences).items()}
    negatives = ({str(k): v for k, v in load_json(args.negatives).items()}
                 if args.negatives and os.path.exists(args.negatives) else {})

    # Frozen evaluation cohort, in file order.
    user_ids = load_user_list(args.user_ids_file)[args.start_user:]
    if args.number_of_users > 0:
        user_ids = user_ids[:args.number_of_users]
    missing = [u for u in user_ids if u not in sequences]
    if missing:
        raise ValueError(f"{len(missing)} eval users missing from sequences, e.g. {missing[:5]}")

    candidate_rows = load_frozen_candidates(args.candidate_file, user_ids, sequences, negatives)

    retriever = None
    if args.raw_memory:
        allowed = set(load_user_list(args.train_users_file)) if args.train_users_file else None
        retriever = RawMemoryRetriever(args.precomputed, args.embedding_model, args.embedding_device,
                                       allowed_users=allowed, batch_size=args.embedding_batch_size)
    elif not args.no_memory:
        retriever = FlatMemoryRetriever(
            args.global_memory,
            args.embedding_model,
            args.embedding_device,
            args.query_centering,
        )

    ranker = LocalRanker(
        args.model_name,
        args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    completed = set()

    if args.resume and output.exists():
        try:
            results = load_json(str(output))
            if any(x.get("ranking_model") != args.model_name or
                   x.get("rank_prompt_version") != RANK_PROMPT_VERSION or
                   "target_rank" not in x for x in results):
                raise RuntimeError("Existing output uses another model/prompt/format. "
                                   "Use a new --output file or --no-resume.")
            completed = {str(x["user_id"]) for x in results}
            print(f"Resume: {len(completed)} users already completed")
        except RuntimeError:
            raise
        except Exception as e:
            print(f"Could not resume old output: {e}")
            results = []
            completed = set()

    pending = [u for u in user_ids if str(u) not in completed]
    print(f"Users: total={len(user_ids)}, pending={len(pending)}, history={args.history_size}")

    t0 = time.perf_counter()
    batch_size = max(1, int(args.llm_batch_size))
    print(f"LLM batch size: {batch_size}")
    pbar = tqdm(total=len(pending), desc="Inference", unit="user")
    processed_since_save = 0

    for batch_start in range(0, len(pending), batch_size):
        batch_uids = pending[batch_start:batch_start + batch_size]
        prepared_batch = []
        # A preparation failure is a protocol error: never silently drop a user.
        for uid in batch_uids:
            prepared = prepare_one_user(
                user_id=str(uid),
                user_data=sequences[uid],
                neg_data=negatives.get(uid, {}),
                items_meta=items_meta,
                retriever=retriever,
                k_memories=args.k_memories,
                history_size=args.history_size,
                seed=args.seed,
                candidate_row=candidate_rows[str(uid)],
            )
            prepared["ranking_model"] = args.model_name
            prepared_batch.append(prepared)

        jobs = [
            {
                "user_id": x["user_id"],
                "user_profile": x["history"],
                "candidates": x["candidate_items"],
                "memory_thoughts": x["retrieved_memories"] or None,
            }
            for x in prepared_batch
        ]
        try:
            ranked_batch = ranker.rank_batch(jobs, batch_size=batch_size)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"CUDA OOM with --llm_batch_size {batch_size}. Retry with --llm_batch_size 4, 2, or 1."
            )
        for prepared, (pred, reasoning, record) in zip(prepared_batch, ranked_batch):
            results.append(finalize_one_user(prepared=prepared, pred=pred, reasoning=reasoning,
                                             items_meta=items_meta, record=record))
            processed_since_save += 1

        pbar.update(len(batch_uids))
        if args.save_every > 0 and processed_since_save >= args.save_every:
            save_json_atomic(results, output)
            processed_since_save = 0

    pbar.close()
    save_json_atomic(results, output)

    selected = set(user_ids)
    rows = [r for r in results if str(r["user_id"]) in selected]
    ranks = [int(r["target_rank"]) for r in rows]
    ranks_complete = [int(r["target_rank"]) for r in rows if r.get("parse_complete")]
    retrieval_stats = retriever.stats() if retriever else {}
    cost = build_cost(args, rows, user_ids, retrieval_stats)
    cost_path = output.with_name("cost_summary.json")
    save_json_atomic(cost, cost_path)

    stats = {
        "dataset": args.dataset,
        "num_results": len(rows),
        "metrics": metrics_from_ranks(ranks),
        "metrics_complete_only": metrics_from_ranks(ranks_complete),
        "parse_quality": {
            "parse_ok_rate": float(np.mean([bool(r.get("parse_ok")) for r in rows])) if rows else None,
            "parse_complete_rate": float(np.mean([bool(r.get("parse_complete")) for r in rows])) if rows else None,
        },
        "elapsed_sec_this_run": round(time.perf_counter() - t0, 4),
        "use_memory": not args.no_memory,
        "memory_source": ("none" if args.no_memory else
                          "raw_local_windows_no_build" if args.raw_memory else "built_global_memory"),
        "k_memories": args.k_memories,
        "history_size": args.history_size,
        "llm_batch_size": args.llm_batch_size,
        "user_ids_file": args.user_ids_file,
        "candidate_file": args.candidate_file,
        "ranking_model": args.model_name,
        "embedding_model": args.embedding_model,
        "retrieval": retrieval_stats or None,
        "llm": ranker.stats(),
        "cost": {k: cost.get(k) for k in ("online", "end_to_end_per_eval_user", "memory")},
    }

    stats_path = output.with_name(output.stem + "_inference_stats.json")
    tracking_path = output.with_name(output.stem + "_llm_tracking.json")
    save_json_atomic(stats, stats_path)
    save_json_atomic({"summary": ranker.stats(), "records": ranker.records}, tracking_path)

    print("\nInference complete")
    print(f"Results : {output}")
    print(f"Stats   : {stats_path}")
    print(f"Cost    : {cost_path}")
    print(json.dumps({"metrics": stats["metrics"], "parse_quality": stats["parse_quality"]}, indent=2))


if __name__ == "__main__":
    main()
