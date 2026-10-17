#!/usr/bin/env python3
import argparse, json, os, random, time, re
from unsloth import FastLanguageModel  # Import before torch/transformers.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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

    def __init__(self, memory_path: str, embedding_model: str, device: str = "auto"):
        data = load_json(memory_path)
        self.memories = data.get("behavior_memories", [])
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
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(1, -1))
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
            "search_calls": self.search_calls,
            "total_search_time_sec": round(self.total_search_time, 6),
            "avg_search_time_ms": (
                round(self.total_search_time / self.search_calls * 1000, 4)
                if self.search_calls else 0.0
            ),
        }


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
        cleaned.extend(iid for label, iid in mapping.items() if label not in seen)
        return cleaned, reasoning, ok

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

                pred, reasoning, parse_ok = self._clean_prediction(
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
                    "raw_output": response,
                    **fitted[local_idx][1],
                })

                all_results.append((pred, reasoning))

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
        )[0]

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
):
    return {
        "user_id": prepared["user_id"],
        "ground_truth_item_ids": prepared["ground_truth_item_ids"],
        "candidate_item_ids": prepared["candidate_item_ids"],
        "reranked_item_ids": pred,
        "candidate_items": prepared["candidate_items"],
        "reranked_items": [get_item_info(i, items_meta) for i in pred],
        "retrieved_memories": prepared["retrieved_memories"],
        "ranking_reasoning": reasoning,
        "ranking_model": prepared["ranking_model"],
        "rank_prompt_version": "amem_qwen_clabel_list_v1",
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


def resolve_candidate_file_path(
    candidate_file: Optional[str],
    sequences_path: str,
    seed: int,
) -> str:
    if candidate_file:
        return str(candidate_file)
    seq = Path(sequences_path)
    return str(seq.with_name(f"{seq.stem}_candidates_seed{int(seed)}.json"))


def ensure_candidate_cache(
    path: str,
    users: List[str],
    sequences: Dict[str, Any],
    negatives: Dict[str, Any],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Persist canonical candidates once and reuse the exact same order later."""
    p = Path(path)
    rows: Dict[str, Dict[str, Any]] = load_candidate_file(str(p)) if p.exists() else {}
    changed = False

    for uid in users:
        suid = str(uid)
        user_data = sequences[suid]
        neg_data = negatives.get(suid, {})
        target = normalize_ids(user_data.get('test', []))
        neg_ids = normalize_ids(neg_data.get('test_neg', []))
        if not target:
            raise ValueError(f"user={suid}: no test item for candidate cache")

        expected = target + neg_ids
        if suid in rows:
            cached = normalize_ids(rows[suid].get('candidates', rows[suid].get('candidate_item_ids', [])))
            cached_target = normalize_ids(rows[suid].get('target', rows[suid].get('ground_truth', rows[suid].get('ground_truth_item_ids', []))))
            if not cached or sorted(cached) != sorted(expected):
                raise ValueError(
                    f"user={suid}: existing candidate cache does not match "
                    "current test + test_neg. Check that all input files belong to the same frozen protocol; "
                    "do not rebuild shared candidates just to bypass this check."
                )
            if cached_target and cached_target != target:
                raise ValueError(
                    f"user={suid}: cached target={cached_target!r} differs from current test target={target!r}. "
                    "Check that sequences, negatives and candidate_file belong to the same frozen protocol."
                )
            continue

        candidates = list(expected)
        rng = random.Random(f"{seed}:{suid}")
        rng.shuffle(candidates)
        rows[suid] = {
            "user_id": suid,
            "candidates": candidates,
            "target": target,
        }
        changed = True

    if changed or not p.exists():
        save_json_atomic(rows, p)
        print(f"Candidate cache saved: {p} ({len(rows)} users)")
    else:
        print(f"Candidate cache loaded: {p} ({len(rows)} users)")

    missing = [str(u) for u in users if str(u) not in rows]
    if missing:
        raise RuntimeError(
            f"Candidate cache missing {len(missing)} selected users; examples={missing[:10]}"
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--global_memory",
        type=str,
        default=None,
        help="Required only when memory retrieval is enabled.",
    )
    p.add_argument("--items", type=str, required=True)
    p.add_argument("--sequences", type=str, required=True)
    p.add_argument("--negatives", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--candidate_file", "--candidate-file",
        dest="candidate_file",
        type=str,
        default=None,
        help=(
            "Shared candidate cache. If omitted, a deterministic cache is created "
            "next to --sequences and reused on later runs."
        ),
    )

    p.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    p.add_argument(
        "--embedding_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    p.add_argument("--k_memories", type=int, default=3)
    p.add_argument("--history_size", type=int, default=10)
    p.add_argument("--number_of_users", type=int, default=0)
    p.add_argument("--start_user", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument(
        "--llm_batch_size", "--llm-batch-size",
        dest="llm_batch_size",
        type=int,
        default=8,
        help="Independent user prompts per model.generate() call. Lower to 4/2 if GPU OOM.",
    )
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--no_memory",
        action="store_true",
        help="Native-LLM baseline: skip memory retrieval",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    for p in [args.items, args.sequences, args.negatives]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not args.no_memory:
        if not args.global_memory:
            raise ValueError("--global_memory is required unless --no_memory is used")
        if not os.path.exists(args.global_memory):
            raise FileNotFoundError(args.global_memory)

    items_meta = normalize_items(load_json(args.items))
    sequences = load_json(args.sequences)
    negatives = load_json(args.negatives)

    # Preserve the exact sequence-file order before applying start/count.
    user_ids = [str(u) for u in list(sequences.keys())[args.start_user:]]
    if args.number_of_users > 0:
        user_ids = user_ids[:args.number_of_users]

    candidate_file_path = resolve_candidate_file_path(
        args.candidate_file,
        args.sequences,
        args.seed,
    )
    candidate_rows = ensure_candidate_cache(
        path=candidate_file_path,
        users=user_ids,
        sequences=sequences,
        negatives=negatives,
        seed=args.seed,
    )
    args.candidate_file = candidate_file_path

    retriever = None
    if not args.no_memory:
        retriever = FlatMemoryRetriever(
            args.global_memory,
            args.embedding_model,
            args.embedding_device,
        )

    ranker = LocalRanker(
        args.model_name,
        args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    output = Path(args.output)
    results = []
    completed = set()

    if args.resume and output.exists():
        try:
            results = load_json(str(output))
            if any(x.get("ranking_model") != args.model_name or
                   x.get("rank_prompt_version") != "amem_qwen_clabel_list_v1" for x in results):
                raise RuntimeError("Existing output uses another model/prompt. Use a new --output file or --no-resume.")
            completed = {str(x["user_id"]) for x in results}
            print(f"Resume: {len(completed)} users already completed")
        except RuntimeError:
            raise
        except Exception as e:
            print(f"Could not resume old output: {e}")
            results = []
            completed = set()

    pending = [u for u in user_ids if str(u) not in completed]
    print(f"Users: total={len(user_ids)}, pending={len(pending)}")

    t0 = time.perf_counter()

    batch_size = max(1, int(args.llm_batch_size))
    print(f"LLM batch size: {batch_size}")
    pbar = tqdm(total=len(pending), desc="Inference", unit="user")
    processed_since_save = 0

    for batch_start in range(0, len(pending), batch_size):
        batch_uids = pending[batch_start:batch_start + batch_size]
        prepared_batch = []

        # Preparation/retrieval remains per-user; only Qwen ranking is batched.
        for uid in batch_uids:
            try:
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
            except Exception as e:
                print(f"\nWarning: user {uid} failed during preparation: {e}")

        if prepared_batch:
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
                ranked_batch = ranker.rank_batch(
                    jobs,
                    batch_size=batch_size,
                )
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    f"CUDA OOM with --llm_batch_size {batch_size}. "
                    "Retry with --llm_batch_size 4, 2, or 1."
                )

            for prepared, (pred, reasoning) in zip(prepared_batch, ranked_batch):
                results.append(
                    finalize_one_user(
                        prepared=prepared,
                        pred=pred,
                        reasoning=reasoning,
                        items_meta=items_meta,
                    )
                )
                processed_since_save += 1

        pbar.update(len(batch_uids))

        if args.save_every > 0 and processed_since_save >= args.save_every:
            save_json_atomic(results, output)
            processed_since_save = 0

    pbar.close()

    save_json_atomic(results, output)

    stats = {
        "num_results": len(results),
        "selected_user_ids": user_ids,
        "elapsed_sec_this_run": round(time.perf_counter() - t0, 4),
        "use_memory": not args.no_memory,
        "k_memories": args.k_memories,
        "history_size": args.history_size,
        "llm_batch_size": args.llm_batch_size,
        "candidate_file": args.candidate_file,
        "ranking_model": args.model_name,
        "embedding_model": args.embedding_model,
        "retrieval": retriever.stats() if retriever else None,
        "llm": ranker.stats(),
    }

    stats_path = output.with_name(output.stem + "_inference_stats.json")
    tracking_path = output.with_name(output.stem + "_llm_tracking.json")

    save_json_atomic(stats, stats_path)
    save_json_atomic(
        {"summary": ranker.stats(), "records": ranker.records},
        tracking_path,
    )

    print("\nInference complete")
    print(f"Results : {output}")
    print(f"Stats   : {stats_path}")
    print(f"Tracking: {tracking_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
