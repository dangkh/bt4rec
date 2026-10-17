#!/usr/bin/env python3
"""
Build an aMem4Rec global memory pool from the SAME text-only precomputed
behavior JSONL used by build_global_memory.py, but use the newer epsilon-net
packing/covering construction with post-hoc cross-user co-occurrence links.

COMEMTREE PROTOCOL (this repo)
------------------------------
The local memories are the SAME behaviour records CoMemTree builds its Tree from:
    precomputed/<DS>/train_shared.jsonl   (bt4rec.cli.precompute, schema v4,
                                           3-item TRAIN windows of train_users_1000)
so aMem4Rec and CoMemTree start from identical per-window LLM behaviours and the
behaviour-extraction cost is shared (reported as "upstream_reused").
    python baseline/amem_build_memory.py --dataset CDs
writes results/baselines/<DS>/amem/global_memory.json (+ _build_stats.json,
+ _cost_summary.json). Only records of --train-users-file users are used.

COST REPORTING
--------------
<output stem>_cost_summary.json (bt4rec_cost_v1, phase=offline): Gemma calls and
tokens for memory evolution / merge (attributed to the train user whose window
triggered them; consolidation merges are shared), embedding calls, number of local
memories consumed, global memories created / absorbed / merged / evolved, links.

INPUT CONTRACT (same as build_global_memory.py)
------------------------------------------------
JSONL produced by precompute_memory_create_gemma_text.py (or bt4rec.cli.precompute
train_shared.jsonl, which carries the same fields). Each record contains:
    precompute_id
    user_id / user_order / window_index
    interaction_sequence
    behavior_explanation
    pattern_description
    keywords
    parse_ok / raw_response (optional here)
and intentionally contains NO embedding.

OUTPUT CONTRACT (same outer + per-memory schema as build_global_memory.py)
-------------------------------------------------------------------------
{
  "behavior_memories": [
    {
      "thought_id": ...,
      "interaction_sequence": [...],
      "behavior_explanation": ...,
      "pattern_description": ...,
      "keywords": [...],
      "embedding": [...],
      "links": [...],
      "timestamp": ...,
      "evolution_count": ...,
      "evolution_history": [...],
      "max_evolutions": ...,
      "last_evolved_timestamp": ...,
      "source_user_id": ...,
      "source_user_order": ...,
      "source_window_index": ...,
      "source_precompute_id": ...
    }, ...
  ],
  "user_interaction_history": [],
  "next_thought_id": ...,
  "metadata": {...}
}

NEW CONSTRUCTION
----------------
1. Load precomputed behavior records in precompute_id order.
2. Compose exactly the same canonical behavior text as the old builder.
3. Batch-encode local behaviors with the same embedding model.
4. Calibrate sigma_cover / sigma_sep from behavior-embedding NN similarities
   (unless explicit --sigma_cover/--sigma_sep are supplied).
5. Sequential epsilon-net replay:
       sim < sigma_cover  -> SPAWN a new global prototype
       sim >= sigma_cover -> ABSORB into nearest global prototype
   The row embedding is the normalized running centroid of member behaviors.
6. Periodically consolidate prototype pairs whose centroid cosine >= sigma_sep.
7. Optional bounded Gemma-3 text evolution on absorbed rows; geometry never moves
   because of text edits -- it is determined by member behavior embeddings.
8. After construction, build memory-to-memory links from DISTINCT-USER
   co-occurrence: A-B is linked when enough users triggered both rows.

LLM
---
Uses Unsloth Gemma-3-4B exactly like the previous build_global_memory.py for
text evolution / merge synthesis. No per-observation LLM semantic-link call is
used in the new method.

Important for retrieval
-----------------------
When centering is enabled (default), the saved memory "embedding" values are
CENTERED global centroids. A downstream retriever must center its query with
metadata["epsilon_net"]["mu"] before cosine search. Use --no_center if an old
retriever must consume the output unchanged and cannot center queries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sys

import numpy as np
import torch

# IMPORTANT: import Unsloth before libraries that import Transformers.
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EXPECTED_PRECOMPUTE_SCHEMA = "amem_local_memory_dual_behavior_v4"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bt4rec.eval.cost import CostTracker, memory_block  # noqa: E402


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data classes -- intentionally identical to old build_global_memory.py
# =============================================================================

@dataclass
class UserInteraction:
    item_id: str
    item_name: str
    item_category: str
    action_type: str
    rating: Optional[float] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BehaviorMemory:
    thought_id: int
    interaction_sequence: List[UserInteraction]
    behavior_explanation: str
    pattern_description: str
    keywords: List[str]
    embedding: np.ndarray

    links: List[int] = field(default_factory=list)
    timestamp: Optional[str] = None
    evolution_count: int = 0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    max_evolutions: Optional[int] = None
    last_evolved_timestamp: Optional[str] = None

    source_user_id: Optional[str] = None
    source_user_order: Optional[int] = None
    source_window_index: Optional[int] = None
    source_precompute_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        self.embedding = np.asarray(self.embedding, dtype=np.float32)

    def can_evolve(self) -> bool:
        if self.max_evolutions is None:
            return True
        return self.evolution_count < self.max_evolutions

    def record_evolution(
        self,
        update_type: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        reasoning: str,
    ) -> None:
        self.evolution_count += 1
        self.last_evolved_timestamp = datetime.now().isoformat()
        self.evolution_history.append({
            "evolution_number": self.evolution_count,
            "timestamp": self.last_evolved_timestamp,
            "update_type": update_type,
            "old_values": old_values,
            "new_values": new_values,
            "reasoning": reasoning,
        })

    def to_dict(self) -> Dict[str, Any]:
        x = asdict(self)
        x["embedding"] = self.embedding.astype(np.float32).tolist()
        x["interaction_sequence"] = [asdict(v) for v in self.interaction_sequence]
        return x

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorMemory":
        d = dict(data)
        d["embedding"] = np.asarray(d["embedding"], dtype=np.float32)
        d["interaction_sequence"] = [
            v if isinstance(v, UserInteraction) else UserInteraction(**v)
            for v in d.get("interaction_sequence", [])
        ]
        return cls(**d)


@dataclass
class LLMCallRecord:
    call_id: int
    call_type: str
    input_tokens: int
    output_tokens: int
    inference_time_sec: float
    timestamp: str


@dataclass
class LLMTracker:
    records: List[LLMCallRecord] = field(default_factory=list)
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_inference_time: float = 0.0

    def record(
        self,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
        elapsed: float,
    ) -> None:
        self.total_calls += 1
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)
        self.total_inference_time += float(elapsed)
        self.records.append(LLMCallRecord(
            call_id=self.total_calls,
            call_type=call_type,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            inference_time_sec=round(float(elapsed), 4),
            timestamp=datetime.now().isoformat(),
        ))

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            x = by_type.setdefault(r.call_type, {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "inference_time_sec": 0.0,
            })
            x["calls"] += 1
            x["input_tokens"] += r.input_tokens
            x["output_tokens"] += r.output_tokens
            x["inference_time_sec"] = round(
                x["inference_time_sec"] + r.inference_time_sec, 4
            )
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_inference_time_sec": round(self.total_inference_time, 4),
            "avg_inference_time_sec": (
                round(self.total_inference_time / self.total_calls, 4)
                if self.total_calls else 0.0
            ),
            "by_call_type": by_type,
        }


# =============================================================================
# Canonical text + JSON helpers
# =============================================================================

def compose_memory_text(
    behavior_explanation: str,
    pattern_description: str,
    keywords: List[str],
) -> str:
    """Exactly the same semantic combination used by old build_global_memory.py."""
    return (
        f"{behavior_explanation} "
        f"{pattern_description} "
        f"{' '.join(str(x) for x in keywords)}"
    ).strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


def safe_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-10:
        return v
    return (v / n).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-10)).astype(np.float32)


# =============================================================================
# Input -- same text-only JSONL contract as old builder
# =============================================================================

def load_precomputed_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc

            if "embedding" in rec:
                raise RuntimeError(
                    f"Input line {line_no} contains `embedding`. "
                    "This builder expects the same text-only precompute as build_global_memory.py."
                )

            schema = rec.get("schema_version")
            if schema is not None and schema != EXPECTED_PRECOMPUTE_SCHEMA:
                raise RuntimeError(
                    f"Unexpected schema_version={schema!r} at line {line_no}; "
                    f"expected {EXPECTED_PRECOMPUTE_SCHEMA!r}."
                )

            required = [
                "precompute_id",
                "behavior_explanation",
                "pattern_description",
                "keywords",
                "interaction_sequence",
            ]
            missing = [k for k in required if k not in rec]
            if missing:
                raise RuntimeError(f"Missing fields {missing} at JSONL line {line_no}")
            records.append(rec)

    records.sort(key=lambda r: int(r["precompute_id"]))
    ids = [int(r["precompute_id"]) for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate precompute_id values detected.")
    return records


def canonical_text_from_record(rec: Dict[str, Any]) -> str:
    keywords = rec.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = [str(keywords)]
    return compose_memory_text(
        behavior_explanation=str(rec.get("behavior_explanation", "")),
        pattern_description=str(rec.get("pattern_description", "")),
        keywords=[str(x) for x in keywords],
    )


def record_to_local_memory(
    rec: Dict[str, Any],
    embedding: np.ndarray,
) -> BehaviorMemory:
    interactions: List[UserInteraction] = []
    for x in rec.get("interaction_sequence", []):
        interactions.append(UserInteraction(
            item_id=str(x.get("item_id", "")),
            item_name=str(x.get("item_name", "")),
            item_category=str(x.get("item_category", "Unknown")),
            action_type=str(x.get("action_type", "purchase")),
            rating=x.get("rating"),
            timestamp=x.get("timestamp"),
            metadata=x.get("metadata", {}),
        ))

    keywords = rec.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = [str(keywords)]
    pid = int(rec["precompute_id"])

    return BehaviorMemory(
        thought_id=pid,
        interaction_sequence=interactions,
        behavior_explanation=str(rec.get("behavior_explanation", "")),
        pattern_description=str(rec.get("pattern_description", "")),
        keywords=[str(x) for x in keywords],
        embedding=np.asarray(embedding, dtype=np.float32),
        source_user_id=(
            str(rec.get("user_id")) if rec.get("user_id") is not None else None
        ),
        source_user_order=(
            int(rec["user_order"]) if rec.get("user_order") is not None else None
        ),
        source_window_index=(
            int(rec["window_index"]) if rec.get("window_index") is not None else None
        ),
        source_precompute_id=pid,
    )


# =============================================================================
# Gemma prompts
# =============================================================================

def build_evolution_prompt(
    new_behavior: str,
    new_pattern: str,
    memory: BehaviorMemory,
) -> str:
    return f"""You are a collaborative memory evolution expert.
Your goal is to refine one shared cross-user behavioral prototype with minimal,
concise changes when a newly absorbed behavior adds useful evidence.

New behavior:
- Behavior: {new_behavior}
- Pattern: {new_pattern}

Current global memory:
- thought_id: {memory.thought_id}
- Behavior: {memory.behavior_explanation}
- Pattern: {memory.pattern_description}
- Evolution count: {memory.evolution_count}

Rules:
- Update only if the new behavior meaningfully strengthens or refines the shared pattern.
- Keep text concise and general enough to represent multiple users.
- Preserve useful existing evidence; do not overfit to one item title.
- If no meaningful improvement is needed, return should_evolve=false.

Return ONLY valid JSON:
{{"should_evolve": true/false, "behavior_explanation": "updated text or null", "pattern_description": "updated text or null", "reasoning": "1 sentence"}}"""


def build_merge_prompt(a: BehaviorMemory, b: BehaviorMemory) -> str:
    return f"""You are a memory consolidation expert.
Two global behavioral prototypes have become near-duplicates in embedding space
and must be represented as one concise memory.

Memory A:
- Behavior: {a.behavior_explanation}
- Pattern: {a.pattern_description}
- Keywords: {a.keywords}

Memory B:
- Behavior: {b.behavior_explanation}
- Pattern: {b.pattern_description}
- Keywords: {b.keywords}

Synthesize one shared memory. Preserve common and complementary evidence, but do
not simply concatenate. Keep it concise and cross-user general.

Return ONLY valid JSON:
{{"behavior_explanation": "...", "pattern_description": "...", "keywords": ["kw1", "kw2", "..."]}}"""


# =============================================================================
# Main builder
# =============================================================================

class EpsilonNetGlobalBuilder:
    def __init__(
        self,
        model_name: str,
        embedding_model_name: str,
        max_new_tokens: int,
        llm_max_seq_length: int,
        load_in_4bit: bool,
        embedding_device: str,
    ) -> None:
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.max_new_tokens = int(max_new_tokens)

        print(f"Loading Gemma evolution/merge LLM: {model_name}")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=llm_max_seq_length,
            load_in_4bit=load_in_4bit,
            full_finetuning=False,
        )
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma3")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            FastModel.for_inference(self.model)
        except Exception:
            pass
        self.model.eval()
        self.llm_device = next(self.model.parameters()).device
        print(f"Gemma device: {self.llm_device}")

        if embedding_device == "auto":
            emb_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            emb_device = embedding_device
        print(f"Loading embedding model: {embedding_model_name} (device={emb_device})")
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=emb_device,
        )
        self.embedding_device = emb_device

        self.behavior_memories: List[BehaviorMemory] = []
        self.sum_vecs: List[np.ndarray] = []
        self.counts: List[int] = []
        self.user_ids: List[Set[str]] = []
        self.members: List[List[int]] = []
        self.created_step: List[int] = []
        self.last_evolved_step: List[int] = []
        self._id2idx: Dict[int, int] = {}

        self.mu: Optional[np.ndarray] = None
        self.sigma_cover: Optional[float] = None
        self.sigma_sep: Optional[float] = None
        self.step: int = 0

        self.llm_tracker = LLMTracker()
        self.event_counts: Counter = Counter()
        self.cost = CostTracker("amem4rec_build")
        self._attrib_user: Optional[str] = None
        self.num_consolidation_merges: int = 0
        self.processed_local_memories: int = 0

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def gemma_generate(self, prompt: str, role_prompt: str, call_type: str) -> str:
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.llm_device)
        input_width = int(inputs["input_ids"].shape[1])
        input_token_count = int(inputs["attention_mask"].sum().item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        generated = outputs[0][input_width:]
        self.llm_tracker.record(
            call_type=call_type,
            input_tokens=input_token_count,
            output_tokens=int(len(generated)),
            elapsed=elapsed,
        )
        eos = self.tokenizer.eos_token_id
        toks = [int(t) for t in generated.tolist()]
        n_out = toks.index(eos) + 1 if eos is not None and eos in toks else len(toks)
        self.cost.llm(call_type, self._attrib_user, input_token_count, n_out, phase="offline")
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self.cost.embed("offline", n_texts=len(texts),
                        n_calls=(len(texts) + max(1, batch_size) - 1) // max(1, batch_size))
        e = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(e, dtype=np.float32)

    def center_one(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float32)
        if self.mu is None:
            return safe_normalize(e)
        return safe_normalize(e - self.mu)

    def center_many(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float32)
        if self.mu is not None:
            e = e - self.mu[None, :]
        return normalize_rows(e)

    # ------------------------------------------------------------------
    # Pool state helpers
    # ------------------------------------------------------------------

    def _rebuild_id2idx(self) -> None:
        self._id2idx = {
            m.thought_id: i for i, m in enumerate(self.behavior_memories)
        }

    def nearest(self, e: np.ndarray) -> Tuple[float, int]:
        if not self.behavior_memories:
            return -1.0, -1
        c = np.stack([m.embedding for m in self.behavior_memories], axis=0)
        sims = c @ e
        j = int(np.argmax(sims))
        return float(sims[j]), j

    def spawn(self, local: BehaviorMemory) -> int:
        mem = BehaviorMemory(
            thought_id=int(local.thought_id),
            interaction_sequence=list(local.interaction_sequence),
            behavior_explanation=local.behavior_explanation,
            pattern_description=local.pattern_description,
            keywords=list(local.keywords),
            embedding=safe_normalize(local.embedding),
            links=[],
            timestamp=local.timestamp,
            evolution_count=0,
            evolution_history=[],
            max_evolutions=local.max_evolutions,
            last_evolved_timestamp=None,
            source_user_id=local.source_user_id,
            source_user_order=local.source_user_order,
            source_window_index=local.source_window_index,
            source_precompute_id=local.source_precompute_id,
        )
        if mem.thought_id in self._id2idx:
            raise RuntimeError(f"Duplicate spawned thought_id={mem.thought_id}")
        self.behavior_memories.append(mem)
        self.sum_vecs.append(np.asarray(local.embedding, dtype=np.float32).copy())
        self.counts.append(1)
        self.user_ids.append(
            {str(local.source_user_id)} if local.source_user_id is not None else set()
        )
        self.members.append([int(local.source_precompute_id or local.thought_id)])
        self.created_step.append(self.step)
        self.last_evolved_step.append(self.step)
        self._id2idx[mem.thought_id] = len(self.behavior_memories) - 1
        return mem.thought_id

    def absorb(self, j: int, local: BehaviorMemory) -> None:
        self.sum_vecs[j] = self.sum_vecs[j] + np.asarray(local.embedding, dtype=np.float32)
        self.counts[j] += 1
        self.behavior_memories[j].embedding = safe_normalize(self.sum_vecs[j])
        if local.source_user_id is not None:
            self.user_ids[j].add(str(local.source_user_id))
        self.members[j].append(int(local.source_precompute_id or local.thought_id))

    def _remove_row(self, j: int) -> None:
        del self.behavior_memories[j]
        del self.sum_vecs[j]
        del self.counts[j]
        del self.user_ids[j]
        del self.members[j]
        del self.created_step[j]
        del self.last_evolved_step[j]
        self._rebuild_id2idx()

    # ------------------------------------------------------------------
    # Text evolution / consolidation text merge
    # ------------------------------------------------------------------

    def evolve_row(
        self,
        j: int,
        local: BehaviorMemory,
        max_evolutions_per_memory: int,
    ) -> None:
        mem = self.behavior_memories[j]
        mem.max_evolutions = max_evolutions_per_memory
        if not mem.can_evolve():
            return
        prompt = build_evolution_prompt(
            local.behavior_explanation,
            local.pattern_description,
            mem,
        )
        try:
            response = self.gemma_generate(
                prompt,
                role_prompt=(
                    "You are a collaborative behavioral memory evolution system. "
                    "Return only valid JSON."
                ),
                call_type="memory_evolution",
            )
            result = parse_json_response(response)
            if not result.get("should_evolve", False):
                return

            placeholders = {
                "", "null", "none", "...", "updated text",
                "updated text or null", "<updated text>",
            }
            old = {
                "behavior_explanation": mem.behavior_explanation,
                "pattern_description": mem.pattern_description,
            }
            update_type: List[str] = []

            new_expl = result.get("behavior_explanation")
            if (
                new_expl is not None
                and str(new_expl).strip().lower() not in placeholders
                and str(new_expl).strip() != mem.behavior_explanation.strip()
            ):
                mem.behavior_explanation = str(new_expl).strip()
                update_type.append("behavior_explanation")

            new_pat = result.get("pattern_description")
            if (
                new_pat is not None
                and str(new_pat).strip().lower() not in placeholders
                and str(new_pat).strip() != mem.pattern_description.strip()
            ):
                mem.pattern_description = str(new_pat).strip()
                update_type.append("pattern")

            if update_type:
                new = {
                    "behavior_explanation": mem.behavior_explanation,
                    "pattern_description": mem.pattern_description,
                }
                mem.record_evolution(
                    update_type=", ".join(update_type),
                    old_values=old,
                    new_values=new,
                    reasoning=str(result.get("reasoning", "")),
                )
                self.last_evolved_step[j] = self.step
        except Exception as exc:
            print(f"  ! Evolution error [{type(exc).__name__}]: {exc!r}")

    def merge_texts(self, a: BehaviorMemory, b: BehaviorMemory) -> Dict[str, Any]:
        prompt = build_merge_prompt(a, b)
        try:
            response = self.gemma_generate(
                prompt,
                role_prompt=(
                    "You are a behavioral memory consolidation system. "
                    "Return only valid JSON."
                ),
                call_type="memory_merge",
            )
            r = parse_json_response(response)
            kws = r.get("keywords", a.keywords)
            if not isinstance(kws, list):
                kws = [str(kws)]
            return {
                "behavior_explanation": str(
                    r.get("behavior_explanation", a.behavior_explanation)
                ).strip(),
                "pattern_description": str(
                    r.get("pattern_description", a.pattern_description)
                ).strip(),
                "keywords": [str(x) for x in kws][:10],
            }
        except Exception as exc:
            print(f"  ! Merge-text error [{type(exc).__name__}]: {exc!r}")
            return {
                "behavior_explanation": (
                    f"{a.behavior_explanation} {b.behavior_explanation}"
                ).strip()[:700],
                "pattern_description": (
                    f"{a.pattern_description} {b.pattern_description}"
                ).strip()[:700],
                "keywords": list(dict.fromkeys(a.keywords + b.keywords))[:10],
            }

    def merge_rows(self, i: int, j: int) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        a = self.behavior_memories[i]
        b = self.behavior_memories[j]
        merged_text = self.merge_texts(a, b)

        self.sum_vecs[i] = self.sum_vecs[i] + self.sum_vecs[j]
        self.counts[i] += self.counts[j]
        self.user_ids[i] |= self.user_ids[j]
        self.members[i].extend(self.members[j])
        self.behavior_memories[i].embedding = safe_normalize(self.sum_vecs[i])
        self.behavior_memories[i].behavior_explanation = merged_text["behavior_explanation"]
        self.behavior_memories[i].pattern_description = merged_text["pattern_description"]
        self.behavior_memories[i].keywords = merged_text["keywords"]

        # Preserve the old-schema evolution bookkeeping rather than dropping it.
        a.evolution_history.extend(b.evolution_history)
        a.evolution_count = int(a.evolution_count) + int(b.evolution_count)
        a.last_evolved_timestamp = max(
            [x for x in [a.last_evolved_timestamp, b.last_evolved_timestamp] if x is not None],
            default=None,
        )
        a.links = []  # rebuilt from co-occurrence after construction

        self._remove_row(j)
        self.num_consolidation_merges += 1

    def consolidate(
        self,
        sigma_sep: float,
        max_absorb_count: int,
        debug_invariants: bool = False,
    ) -> None:
        previous_user, self._attrib_user = self._attrib_user, None  # merges are shared
        try:
            self._consolidate(sigma_sep, max_absorb_count, debug_invariants)
        finally:
            self._attrib_user = previous_user

    def _consolidate(
        self,
        sigma_sep: float,
        max_absorb_count: int,
        debug_invariants: bool = False,
    ) -> None:
        while len(self.behavior_memories) >= 2:
            c = np.stack([m.embedding for m in self.behavior_memories], axis=0)
            g = c @ c.T
            np.fill_diagonal(g, -1.0)
            if max_absorb_count > 0:
                cnt = np.asarray(self.counts, dtype=np.int64)
                g[(cnt[:, None] + cnt[None, :]) > max_absorb_count] = -1.0

            flat = int(np.argmax(g))
            i, j = divmod(flat, g.shape[1])
            if float(g[i, j]) < sigma_sep:
                break
            self.merge_rows(i, j)

        if debug_invariants and len(self.behavior_memories) > 1:
            c = np.stack([m.embedding for m in self.behavior_memories], axis=0)
            g = c @ c.T
            np.fill_diagonal(g, -1.0)
            if max_absorb_count > 0:
                cnt = np.asarray(self.counts, dtype=np.int64)
                g[(cnt[:, None] + cnt[None, :]) > max_absorb_count] = -1.0
            max_offdiag = float(g.max())
            if max_offdiag >= sigma_sep:
                raise AssertionError(
                    f"I1 violated after consolidation: max cosine "
                    f"{max_offdiag:.6f} >= sigma_sep {sigma_sep:.6f}"
                )

    # ------------------------------------------------------------------
    # Process one local behavior
    # ------------------------------------------------------------------

    def process_local_memory(
        self,
        local: BehaviorMemory,
        sigma_cover: float,
        sigma_sep: float,
        evolve_every_T: int,
        consolidate_every_P: int,
        max_absorb_count: int,
        max_evolutions_per_memory: int,
        wo_evolving: bool,
        debug_invariants: bool,
    ) -> str:
        self.processed_local_memories += 1
        self._attrib_user = local.source_user_id

        # Keep the old ablation semantics: every local memory remains separate.
        if wo_evolving:
            self.spawn(local)
            self.step += 1
            self.event_counts["spawn_wo_evolving"] += 1
            return "spawn_wo_evolving"

        s_max, j = self.nearest(local.embedding)
        capped = (
            j >= 0
            and max_absorb_count > 0
            and self.counts[j] >= max_absorb_count
        )

        if j < 0 or s_max < sigma_cover or capped:
            self.spawn(local)
            event = "spawn_capped" if capped and s_max >= sigma_cover else "spawn"
        else:
            self.absorb(j, local)
            event = "absorb" if s_max >= sigma_sep else "related_absorb"
            if evolve_every_T > 0 and self.counts[j] % evolve_every_T == 0:
                self.evolve_row(
                    j,
                    local,
                    max_evolutions_per_memory=max_evolutions_per_memory,
                )

        self.step += 1
        self.event_counts[event] += 1

        if consolidate_every_P > 0 and self.step % consolidate_every_P == 0:
            self.consolidate(
                sigma_sep=sigma_sep,
                max_absorb_count=max_absorb_count,
                debug_invariants=debug_invariants,
            )
        return event

    # ------------------------------------------------------------------
    # Post-hoc collaborative co-occurrence graph
    # ------------------------------------------------------------------

    def build_cooccurrence_links(self, min_shared: int, top_k: int) -> Dict[str, Any]:
        for m in self.behavior_memories:
            m.links = []

        user_to_rows: Dict[str, Set[int]] = defaultdict(set)
        for row, users in enumerate(self.user_ids):
            for u in users:
                user_to_rows[str(u)].add(row)

        cooc: Dict[Tuple[int, int], int] = defaultdict(int)
        for rows in user_to_rows.values():
            ordered = sorted(rows)
            for a in range(len(ordered)):
                for b in range(a + 1, len(ordered)):
                    cooc[(ordered[a], ordered[b])] += 1

        per_row: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (i, j), count in cooc.items():
            if count >= min_shared:
                per_row[i].append((count, j))
                per_row[j].append((count, i))

        edge_pairs: Set[Tuple[int, int]] = set()
        for row, values in per_row.items():
            # deterministic: shared-user count desc, row index asc
            values.sort(key=lambda x: (-x[0], x[1]))
            ids: List[int] = []
            for _, other in values[:top_k]:
                ids.append(int(self.behavior_memories[other].thought_id))
                a = int(self.behavior_memories[row].thought_id)
                b = int(self.behavior_memories[other].thought_id)
                edge_pairs.add((min(a, b), max(a, b)))
            self.behavior_memories[row].links = ids

        return {
            "num_users_with_memory_assignments": len(user_to_rows),
            "num_candidate_pairs": len(cooc),
            "num_linked_undirected_pairs": len(edge_pairs),
            "min_shared": int(min_shared),
            "top_k": int(top_k),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pool_metrics(self, sigma_sep: float) -> Dict[str, Any]:
        k = len(self.behavior_memories)
        if k == 0:
            return {"K": 0}
        counts = np.asarray(self.counts, dtype=np.int64)
        total = int(counts.sum())
        if k > 1:
            c = np.stack([m.embedding for m in self.behavior_memories], axis=0)
            g = c @ c.T
            off = g[~np.eye(k, dtype=bool)]
            mean_off = float(off.mean())
            max_off = float(off.max())
        else:
            mean_off = 0.0
            max_off = 0.0
        sorted_counts = np.sort(counts)[::-1]
        return {
            "K": k,
            "mean_offdiag_cos": mean_off,
            "max_offdiag_cos": max_off,
            "invariant_I1_holds": bool(k <= 1 or max_off < sigma_sep),
            "max_count": int(sorted_counts[0]),
            "max_count_share": float(sorted_counts[0] / total) if total else 0.0,
            "top5_count_share": (
                float(sorted_counts[:5].sum() / total) if total else 0.0
            ),
            "median_count": float(np.median(sorted_counts)),
            "n_singletons": int((sorted_counts == 1).sum()),
        }

    # ------------------------------------------------------------------
    # Save / load. Top-level + BehaviorMemory schema stays old-builder compatible.
    # ------------------------------------------------------------------

    def _state_dict(self) -> Dict[str, Any]:
        return {
            "sum_vecs": [np.asarray(v, dtype=np.float32).tolist() for v in self.sum_vecs],
            "counts": [int(x) for x in self.counts],
            "user_ids": [sorted(s) for s in self.user_ids],
            "members": self.members,
            "created_step": self.created_step,
            "last_evolved_step": self.last_evolved_step,
            "step": int(self.step),
            "mu": self.mu.tolist() if self.mu is not None else None,
            "sigma_cover": self.sigma_cover,
            "sigma_sep": self.sigma_sep,
            "event_counts": dict(self.event_counts),
            "num_consolidation_merges": int(self.num_consolidation_merges),
        }

    def save_memory(
        self,
        filepath: str,
        processed_precompute_id: int,
        source_precomputed: str,
        embedding_model_name: str,
        run_config: Dict[str, Any],
        completed: bool,
        cooc_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "save_timestamp": datetime.now().isoformat(),
            "completed": bool(completed),
            "processed_precompute_id": int(processed_precompute_id),
            "processed_local_memories": int(self.processed_local_memories),
            "num_memories": len(self.behavior_memories),
            "source_precomputed": source_precomputed,
            "precompute_schema": EXPECTED_PRECOMPUTE_SCHEMA,
            "embedding_model": embedding_model_name,
            "construction_method": "epsilon_net_packing_covering_cooccurrence",
            "llm_stats": self.llm_tracker.summary(),
            "run_config": run_config,
            "epsilon_net": self._state_dict(),
            "cooccurrence": cooc_stats or {},
        }

        next_thought_id = max(
            (m.thought_id for m in self.behavior_memories),
            default=-1,
        ) + 1

        data = {
            "behavior_memories": [m.to_dict() for m in self.behavior_memories],
            "user_interaction_history": [],
            "next_thought_id": next_thought_id,
            "metadata": metadata,
        }

        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp, path)

    def load_memory(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.behavior_memories = [
            BehaviorMemory.from_dict(x)
            for x in data.get("behavior_memories", [])
        ]
        metadata = data.get("metadata", {})
        state = metadata.get("epsilon_net")
        if not state:
            raise RuntimeError(
                "Existing output has no metadata['epsilon_net'] state. "
                "It was probably produced by the old build_global_memory.py; "
                "use a new --output for the epsilon-net builder."
            )

        self.sum_vecs = [
            np.asarray(v, dtype=np.float32) for v in state.get("sum_vecs", [])
        ]
        self.counts = [int(x) for x in state.get("counts", [])]
        self.user_ids = [set(map(str, s)) for s in state.get("user_ids", [])]
        self.members = [list(map(int, xs)) for xs in state.get("members", [])]
        self.created_step = [int(x) for x in state.get("created_step", [])]
        self.last_evolved_step = [
            int(x) for x in state.get("last_evolved_step", [])
        ]
        self.step = int(state.get("step", 0))
        self.mu = (
            np.asarray(state["mu"], dtype=np.float32)
            if state.get("mu") is not None else None
        )
        self.sigma_cover = (
            float(state["sigma_cover"])
            if state.get("sigma_cover") is not None else None
        )
        self.sigma_sep = (
            float(state["sigma_sep"])
            if state.get("sigma_sep") is not None else None
        )
        self.event_counts = Counter(state.get("event_counts", {}))
        self.num_consolidation_merges = int(
            state.get("num_consolidation_merges", 0)
        )
        self.processed_local_memories = int(
            metadata.get("processed_local_memories", self.step)
        )
        self._rebuild_id2idx()

        n = len(self.behavior_memories)
        lengths = {
            "sum_vecs": len(self.sum_vecs),
            "counts": len(self.counts),
            "user_ids": len(self.user_ids),
            "members": len(self.members),
            "created_step": len(self.created_step),
            "last_evolved_step": len(self.last_evolved_step),
        }
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise RuntimeError(
                f"Corrupt epsilon-net checkpoint: num memories={n}, state lengths={bad}"
            )
        return metadata


# =============================================================================
# Calibration
# =============================================================================

def calibrate_thresholds_from_embeddings(
    raw_embeddings: np.ndarray,
    q_cover: float,
    q_sep: float,
    sample_size: int,
    seed: int,
    center: bool,
) -> Tuple[float, float, Optional[np.ndarray], Dict[str, Any]]:
    n = len(raw_embeddings)
    if n < 2:
        raise RuntimeError("Need at least 2 precomputed behaviors for threshold calibration.")
    if not (0.0 <= q_cover < q_sep <= 1.0):
        raise ValueError("Require 0 <= q_cover < q_sep <= 1")

    rng = np.random.default_rng(seed)
    m = min(int(sample_size), n) if sample_size > 0 else n
    idx = np.sort(rng.choice(n, size=m, replace=False)) if m < n else np.arange(n)
    sample = np.asarray(raw_embeddings[idx], dtype=np.float32)

    mu: Optional[np.ndarray] = None
    raw_g = sample @ sample.T
    iu = np.triu_indices(m, 1)
    raw_mean_cos = float(raw_g[iu].mean()) if len(iu[0]) else 0.0

    if center:
        mu = sample.mean(axis=0).astype(np.float32)
        sample = normalize_rows(sample - mu[None, :])
    else:
        sample = normalize_rows(sample)

    g = sample @ sample.T
    np.fill_diagonal(g, -1.0)
    nn = g.max(axis=1)
    sigma_cover = float(np.quantile(nn, q_cover))
    sigma_sep = float(np.quantile(nn, q_sep))
    if sigma_sep <= sigma_cover:
        raise RuntimeError(
            f"Calibration failed: sigma_sep={sigma_sep:.6f} <= "
            f"sigma_cover={sigma_cover:.6f}. Adjust quantiles."
        )

    info = {
        "mode": "quantile_nn_similarity",
        "n_total_embeddings": int(n),
        "n_calibration": int(m),
        "q_cover": float(q_cover),
        "q_sep": float(q_sep),
        "sigma_cover": sigma_cover,
        "sigma_sep": sigma_sep,
        "centered": bool(center),
        "raw_mean_pairwise_cos": raw_mean_cos,
        "centered_mean_pairwise_cos": (
            float(g[iu].mean()) if len(iu[0]) else 0.0
        ),
        "mu_norm": float(np.linalg.norm(mu)) if mu is not None else None,
    }
    return sigma_cover, sigma_sep, mu, info


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build old-schema aMem4Rec global memory from text-only precomputed "
            "behaviors using epsilon-net packing/covering + co-occurrence links."
        )
    )
    p.add_argument("--dataset", type=str, default=None,
                   help="CDs | Video_Games | ML: sets --precomputed/--output/--train-users-file defaults")
    p.add_argument("--precomputed", type=str, default=None,
                   help="default precomputed/<DS>/train_shared.jsonl (CoMemTree train behaviours)")
    p.add_argument("--output", type=str, default=None,
                   help="default results/baselines/<DS>/amem/global_memory.json")
    p.add_argument("--train-users-file", "--train_users_file", dest="train_users_file",
                   type=str, default=None,
                   help="default data/<DS>/protocol/train_users_1000.json; 'none' = use all records")

    p.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    )
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
    p.add_argument("--embedding_batch_size", type=int, default=64)

    # Either explicitly supply BOTH sigmas, or let quantile calibration derive them.
    p.add_argument("--sigma_cover", type=float, default=None)
    p.add_argument("--sigma_sep", type=float, default=None)
    p.add_argument("--q_cover", type=float, default=0.50)
    p.add_argument("--q_sep", type=float, default=0.90)
    p.add_argument("--calib_sample_size", type=int, default=2000)
    p.add_argument("--no_center", action="store_true", default=False)

    p.add_argument("--evolve_every_T", type=int, default=3)
    p.add_argument("--consolidate_every_P", type=int, default=200)
    p.add_argument("--max_absorb_count", type=int, default=0)
    p.add_argument("--max_evolutions_per_memory", type=int, default=10)
    p.add_argument("--debug_invariants", action="store_true", default=False)

    p.add_argument("--cooc_min_shared", type=int, default=2)
    p.add_argument("--top_k_links", type=int, default=5)
    # Alias for launch scripts that previously used --link_size.
    p.add_argument("--link_size", type=int, default=None)

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--llm_max_seq_length", type=int, default=8192)
    p.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--wo_evolving", action="store_true", default=False)
    p.add_argument(
        "--wo_link",
        action="store_true",
        default=False,
        help="Disable post-hoc cross-user co-occurrence links.",
    )
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--quiet", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.dataset:
        from bt4rec.config import load_dataset_config
        cfg = load_dataset_config(a.dataset)
        a.dataset = cfg["dataset"]
        if a.precomputed is None:
            base = Path(cfg.get("precomputed_dir", f"precomputed/{a.dataset}"))
            cand = [base / "train_shared.jsonl", base / "qwen" / "train_shared.jsonl"]
            a.precomputed = str(next((c for c in cand if c.exists()), cand[0]))
        if a.output is None:
            a.output = str(Path("results/baselines") / a.dataset / "amem" / "global_memory.json")
        if a.train_users_file is None:
            a.train_users_file = cfg.get("train_users")
    if a.precomputed is None or a.output is None:
        raise SystemExit("Pass --dataset, or both --precomputed and --output")
    if a.train_users_file and a.train_users_file.lower() == "none":
        a.train_users_file = None
    return a


def load_user_list(path: str) -> List[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        obj = obj.get("users", list(obj.keys()))
    users = [str(x) for x in obj]
    if len(users) != len(set(users)):
        raise ValueError(f"Duplicate users in {path}")
    return users


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    precomputed_path = Path(args.precomputed)
    output_path = Path(args.output)
    if not precomputed_path.exists():
        raise FileNotFoundError(f"Precomputed JSONL not found: {precomputed_path}")

    records = load_precomputed_jsonl(str(precomputed_path))
    if not records:
        raise RuntimeError("No precomputed local behaviors found.")
    train_users: Optional[List[str]] = None
    if args.train_users_file:
        train_users = load_user_list(args.train_users_file)
        allowed = set(train_users)
        before = len(records)
        records = [r for r in records if str(r.get("user_id")) in allowed]
        covered = {str(r.get("user_id")) for r in records}
        print(f"Train-user filter      : {len(records)}/{before} records, "
              f"{len(covered)}/{len(allowed)} users covered ({args.train_users_file})")
        if not records:
            raise RuntimeError("No precomputed records belong to --train-users-file users.")
    else:
        train_users = sorted({str(r.get("user_id")) for r in records})

    top_k_links = int(args.link_size) if args.link_size is not None else int(args.top_k_links)
    center = not args.no_center

    if (args.sigma_cover is None) != (args.sigma_sep is None):
        raise ValueError("Provide both --sigma_cover and --sigma_sep, or neither.")
    if args.sigma_cover is not None and args.sigma_cover >= args.sigma_sep:
        raise ValueError("Require sigma_cover < sigma_sep.")

    run_config = {
        "model_name": args.model_name,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "sigma_cover_override": args.sigma_cover,
        "sigma_sep_override": args.sigma_sep,
        "q_cover": args.q_cover,
        "q_sep": args.q_sep,
        "calib_sample_size": args.calib_sample_size,
        "center": center,
        "evolve_every_T": args.evolve_every_T,
        "consolidate_every_P": args.consolidate_every_P,
        "max_absorb_count": args.max_absorb_count,
        "max_evolutions_per_memory": args.max_evolutions_per_memory,
        "cooc_min_shared": args.cooc_min_shared,
        "top_k_links": top_k_links,
        "wo_evolving": args.wo_evolving,
        "wo_link": args.wo_link,
        "seed": args.seed,
        "train_users_file": args.train_users_file,
    }

    print("=" * 80)
    print("AMEM4REC GLOBAL MEMORY: EPSILON-NET + CROSS-USER CO-OCCURRENCE")
    print("=" * 80)
    print(f"Input behaviors       : {len(records)}")
    print(f"Input                 : {precomputed_path}")
    print(f"Output                : {output_path}")
    print(f"Gemma                 : {args.model_name}")
    print(f"Embedding model       : {args.embedding_model}")
    print(f"Centering             : {center}")

    builder = EpsilonNetGlobalBuilder(
        model_name=args.model_name,
        embedding_model_name=args.embedding_model,
        max_new_tokens=args.max_new_tokens,
        llm_max_seq_length=args.llm_max_seq_length,
        load_in_4bit=args.load_in_4bit,
        embedding_device=args.embedding_device,
    )

    last_processed_id = -1
    calibration_info: Dict[str, Any] = {}

    if args.resume and output_path.exists():
        print("\nResume enabled: loading epsilon-net checkpoint...")
        metadata = builder.load_memory(str(output_path))
        old_source = metadata.get("source_precomputed")
        old_embedding = metadata.get("embedding_model")
        if old_source and Path(old_source).resolve() != precomputed_path.resolve():
            raise RuntimeError(
                "Existing output was built from a different precomputed file. "
                "Use a new --output."
            )
        if old_embedding and old_embedding != args.embedding_model:
            raise RuntimeError(
                "Existing output used a different embedding model. Use a new --output."
            )

        old_cfg = metadata.get("run_config", {})
        guarded_keys = [
            "center", "sigma_cover_override", "sigma_sep_override",
            "q_cover", "q_sep", "calib_sample_size",
            "evolve_every_T", "consolidate_every_P",
            "max_absorb_count", "max_evolutions_per_memory",
            "wo_evolving", "seed",
        ]
        for key in guarded_keys:
            if key in old_cfg and old_cfg[key] != run_config[key]:
                raise RuntimeError(
                    f"Cannot resume: {key} changed from {old_cfg[key]!r} "
                    f"to {run_config[key]!r}. Use a new --output."
                )

        last_processed_id = int(metadata.get("processed_precompute_id", -1))
        calibration_info = metadata.get("run_config", {}).get("calibration", {})
        print(f"Resume after precompute_id={last_processed_id}")
        print(
            f"Loaded pool K={len(builder.behavior_memories)}, "
            f"sigma_cover={builder.sigma_cover}, sigma_sep={builder.sigma_sep}"
        )

    resumed_from_checkpoint = last_processed_id >= 0
    pending_records = [
        rec for rec in records
        if int(rec["precompute_id"]) > last_processed_id
    ]

    # Fresh run: encode ALL once, calibrate in exactly the same behavior space,
    # then sequentially replay those already-computed embeddings.
    if last_processed_id < 0:
        print(f"\nBatch encoding {len(records)} behavior memories...")
        texts = [canonical_text_from_record(r) for r in records]
        t0 = time.perf_counter()
        raw_embeddings = builder.encode_texts(
            texts,
            batch_size=args.embedding_batch_size,
            show_progress_bar=True,
        )
        embedding_time = time.perf_counter() - t0
        if len(raw_embeddings) != len(records):
            raise RuntimeError("Embedding count does not match input record count.")

        if args.wo_evolving:
            # Thresholds are not used when every input is stored separately.
            builder.mu = None if args.no_center else np.zeros(raw_embeddings.shape[1], dtype=np.float32)
            builder.sigma_cover = 0.0
            builder.sigma_sep = 1.0
            calibration_info = {"skipped": True, "reason": "wo_evolving"}
            local_embeddings = normalize_rows(raw_embeddings)
        elif args.sigma_cover is not None:
            # If explicit sigmas are supplied, compute mu from the deterministic calibration
            # subset when centering is enabled so construction still has a defined query transform.
            if center:
                n = len(raw_embeddings)
                rng = np.random.default_rng(args.seed)
                m = min(args.calib_sample_size, n) if args.calib_sample_size > 0 else n
                idx = (
                    np.sort(rng.choice(n, size=m, replace=False))
                    if m < n else np.arange(n)
                )
                builder.mu = raw_embeddings[idx].mean(axis=0).astype(np.float32)
            else:
                builder.mu = None
            builder.sigma_cover = float(args.sigma_cover)
            builder.sigma_sep = float(args.sigma_sep)
            calibration_info = {
                "mode": "explicit",
                "sigma_cover": builder.sigma_cover,
                "sigma_sep": builder.sigma_sep,
                "centered": center,
                "mu_norm": (
                    float(np.linalg.norm(builder.mu)) if builder.mu is not None else None
                ),
            }
            local_embeddings = builder.center_many(raw_embeddings)
        else:
            (
                builder.sigma_cover,
                builder.sigma_sep,
                builder.mu,
                calibration_info,
            ) = calibrate_thresholds_from_embeddings(
                raw_embeddings=raw_embeddings,
                q_cover=args.q_cover,
                q_sep=args.q_sep,
                sample_size=args.calib_sample_size,
                seed=args.seed,
                center=center,
            )
            local_embeddings = builder.center_many(raw_embeddings)

        print(
            f"Calibration: sigma_cover={builder.sigma_cover:.6f}, "
            f"sigma_sep={builder.sigma_sep:.6f}"
        )
        if builder.mu is not None:
            print(f"Centering |mu|={np.linalg.norm(builder.mu):.6f}")
        print(f"Embedding time: {embedding_time:.2f}s")

        pending_embeddings = local_embeddings
    else:
        if not pending_records:
            print("\nNothing new to process; rebuilding final co-occurrence links/stats only.")
            pending_embeddings = np.empty((0, 0), dtype=np.float32)
            embedding_time = 0.0
        else:
            print(f"\nBatch encoding {len(pending_records)} pending behavior memories...")
            texts = [canonical_text_from_record(r) for r in pending_records]
            t0 = time.perf_counter()
            raw_pending = builder.encode_texts(
                texts,
                batch_size=args.embedding_batch_size,
                show_progress_bar=True,
            )
            embedding_time = time.perf_counter() - t0
            pending_embeddings = builder.center_many(raw_pending)

    if builder.sigma_cover is None or builder.sigma_sep is None:
        raise RuntimeError("Missing epsilon-net thresholds after initialization/resume.")

    build_start = time.perf_counter()
    progress = tqdm(
        range(len(pending_records)),
        desc="Building epsilon-net global memory",
    )

    for local_idx in progress:
        rec = pending_records[local_idx]
        emb = pending_embeddings[local_idx]
        local = record_to_local_memory(rec, emb)
        local.max_evolutions = args.max_evolutions_per_memory

        event = builder.process_local_memory(
            local=local,
            sigma_cover=float(builder.sigma_cover),
            sigma_sep=float(builder.sigma_sep),
            evolve_every_T=args.evolve_every_T,
            consolidate_every_P=(0 if args.wo_evolving else args.consolidate_every_P),
            max_absorb_count=args.max_absorb_count,
            max_evolutions_per_memory=args.max_evolutions_per_memory,
            wo_evolving=args.wo_evolving,
            debug_invariants=args.debug_invariants,
        )
        last_processed_id = int(rec["precompute_id"])

        progress.set_postfix({
            "pool": len(builder.behavior_memories),
            "event": event,
        })

        processed_this_run = local_idx + 1
        if args.save_every > 0 and processed_this_run % args.save_every == 0:
            # Checkpoint is deliberately link-free; co-occurrence links are rebuilt
            # deterministically at the end from the stored user sets.
            for m in builder.behavior_memories:
                m.links = []
            builder.save_memory(
                filepath=str(output_path),
                processed_precompute_id=last_processed_id,
                source_precomputed=str(precomputed_path.resolve()),
                embedding_model_name=args.embedding_model,
                run_config=run_config,
                completed=False,
                cooc_stats={},
            )

    if not args.wo_evolving:
        builder.consolidate(
            sigma_sep=float(builder.sigma_sep),
            max_absorb_count=args.max_absorb_count,
            debug_invariants=args.debug_invariants,
        )

    if args.wo_link:
        for m in builder.behavior_memories:
            m.links = []
        cooc_stats = {"disabled": True}
    else:
        cooc_stats = builder.build_cooccurrence_links(
            min_shared=args.cooc_min_shared,
            top_k=top_k_links,
        )

    build_time = time.perf_counter() - build_start
    pool_metrics = builder.pool_metrics(float(builder.sigma_sep))

    # Put calibration summary into run_config-compatible metadata without changing
    # the old top-level memory schema.
    final_run_config = dict(run_config)
    final_run_config["calibration"] = calibration_info

    builder.save_memory(
        filepath=str(output_path),
        processed_precompute_id=(
            last_processed_id
            if last_processed_id >= 0
            else int(records[-1]["precompute_id"])
        ),
        source_precomputed=str(precomputed_path.resolve()),
        embedding_model_name=args.embedding_model,
        run_config=final_run_config,
        completed=True,
        cooc_stats=cooc_stats,
    )

    stats = {
        "completed": True,
        "precomputed_file": str(precomputed_path.resolve()),
        "output_file": str(output_path.resolve()),
        "num_precomputed_records": len(records),
        "num_processed_this_run": len(pending_records),
        "embedding_time_sec": round(embedding_time, 4),
        "global_build_time_sec": round(build_time, 4),
        "num_global_memories": len(builder.behavior_memories),
        "sigma_cover": builder.sigma_cover,
        "sigma_sep": builder.sigma_sep,
        "centered": builder.mu is not None,
        "mu_norm": (
            float(np.linalg.norm(builder.mu)) if builder.mu is not None else None
        ),
        "event_counts": dict(builder.event_counts),
        "num_consolidation_merges": builder.num_consolidation_merges,
        "pool_metrics": pool_metrics,
        "cooccurrence": cooc_stats,
        "llm": builder.llm_tracker.summary(),
        "run_config": final_run_config,
    }
    # ---------------- cost / memory accounting (bt4rec_cost_v1, offline) ----
    windows_per_user = Counter(str(r.get("user_id")) for r in records)
    rows_per_user = Counter(u for users in builder.user_ids for u in users)
    evolutions = int(sum(int(m.evolution_count) for m in builder.behavior_memories))
    builder.cost.set_upstream(
        behavior_extraction_llm_calls=len(records),
        behavior_extraction_calls_per_train_user=len(records) / max(1, len(train_users)),
        source=str(precomputed_path),
        note="Local memories = CoMemTree train behaviours (1 LLM call per 3-item TRAIN window), "
             "precomputed once and shared with CoMemTree; not re-generated here.")
    cost_memory = memory_block(
        total=len(builder.behavior_memories),
        per_user={u: int(windows_per_user.get(u, 0)) for u in train_users},
        kinds={
            "local_memories_consumed": len(records),
            "global_memories_final": len(builder.behavior_memories),
            "spawn_events": int(sum(v for k, v in builder.event_counts.items() if k.startswith("spawn"))),
            "absorb_events": int(sum(v for k, v in builder.event_counts.items() if "absorb" in k)),
            "consolidation_merges": int(builder.num_consolidation_merges),
            "text_evolutions_applied": evolutions,
            "cooccurrence_links": int(cooc_stats.get("num_linked_undirected_pairs", 0) or 0),
        },
        scope="global pool shared by all users",
        per_user_meaning="local behaviour memories (windows) contributed by each train user",
        global_rows_touched_per_user=(
            {"mean": float(np.mean([rows_per_user.get(u, 0) for u in train_users])),
             "max": int(max([rows_per_user.get(u, 0) for u in train_users] or [0]))}),
        resumed_run=bool(resumed_from_checkpoint),
        resumed_note=("cost covers only the calls made in this run" if resumed_from_checkpoint else None),
    )
    cost_path = output_path.with_name(output_path.stem + "_cost_summary.json")
    builder.cost.write(cost_path, train_users, memory=cost_memory,
                       extra={"amortised_over": "train users (== eval users in this protocol)"})
    print(f"Cost summary       : {cost_path}")

    stats_path = output_path.with_name(output_path.stem + "_build_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Global memory      : {output_path}")
    print(f"Build stats        : {stats_path}")
    print(f"Global pool size   : {len(builder.behavior_memories)}")
    print(f"Event counts       : {dict(builder.event_counts)}")
    print(f"Consolidation merge: {builder.num_consolidation_merges}")
    print(f"Co-occurrence      : {cooc_stats}")
    print(f"Pool metrics       : {json.dumps(pool_metrics, indent=2)}")
    print(f"LLM stats          : {json.dumps(builder.llm_tracker.summary(), indent=2)}")


if __name__ == "__main__":
    main()
