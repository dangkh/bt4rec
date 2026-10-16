#!/usr/bin/env python3
"""Qwen2.5-7B CoMemTree behavior precompute: SAME users, NO validation.

Expected user_sequences.json per user:
    {"train": [...observed item IDs...], "test": [held-out item ID]}

Behavior extraction uses ONLY `train`. `test` is NEVER used to create
behavior windows, prompts, or the test-side observed history. The two outputs
use the SAME user IDs and exactly the SAME cached behavior windows:

    precomputed/<dataset>/train_shared.jsonl   window-level memory for Tree
    precomputed/<dataset>/test_shared.jsonl    user-level memory for inference

Qwen processes each distinct observed train window ONCE, globally batched,
with durable resumable cache. Both outputs are packed from the same cache;
there is NO separate test-time Gemma extraction and NO validation window.

Recommended (run from project root):

  python precompute_train_test_shared_qwen.py --dataset CDs \
    --user-ids-file data/CDs/protocol/eval_users_1000.json \
    --history-size 20 --window-size 3 --plan-only

  CUDA_VISIBLE_DEVICES=0 python precompute_train_test_shared_qwen.py --dataset CDs \
    --user-ids-file data/CDs/protocol/eval_users_1000.json \
    --history-size 20 --window-size 3 --llm-batch-size 4

Legacy compatible output schemas: amem_local_memory_dual_behavior_v4 and
amem_test_behavior_precompute_dual_v3. Only compatible Qwen caches are reused; Gemma caches remain untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import math
import re
import oracle_probe_qwen as oracle_tax  # same HOW taxonomy as Oracle-Qwen; keep both scripts together
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm.auto import tqdm

SCHEMA_TRAIN = "amem_local_memory_dual_behavior_v4"
SCHEMA_TEST = "amem_test_behavior_precompute_dual_v3"
PROMPT_VERSION = "dual_view_discriminative_behavior_qwen_v1_strict"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    positions: list[int]    # Absolute indices within train-only interaction sequence


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


def generation_config(args: argparse.Namespace) -> dict:
    """Fields that must match before new checkpoint behavior reuse."""
    return {"model_name": args.model_name, "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens, "load_in_4bit": args.load_in_4bit,
            "dtype": args.dtype, "sample": False, "prompt_version": PROMPT_VERSION}


def read_cache(path: str | None, expected_config: dict | None = None,
               allow_config_mismatch: bool = False) -> tuple[dict[str, dict], Counter]:
    cached: dict[str, dict] = {}
    counts = Counter()
    if path is None:
        return cached, counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            if rec.get("schema_version") != SCHEMA_TRAIN or rec.get("behavior_prompt_version") != PROMPT_VERSION:
                counts["incompatible_schema_or_prompt"] += 1
                continue
            if rec.get("parse_ok") is not True:
                counts["prior_failed_or_missing_parse_ok"] += 1
                continue
            if not rec.get("recommendation_behavior") or not rec.get("trajectory_signature"):
                counts["missing_behavior"] += 1
                continue
            try:
                validated = validate_behavior(rec, rec.get("interaction_sequence", []))
            except ValueError:
                counts["invalid_cached_behavior"] += 1
                continue
            config = rec.get("generation_config")
            if config is not None and expected_config is not None and config != expected_config:
                counts["generation_config_mismatch"] += 1
                if not allow_config_mismatch:
                    continue
            key = cache_key(rec["user_id"], rec.get("interaction_sequence", []))
            cached[key] = rec
            counts["valid_records"] += 1
    return cached, counts


def to_train_record(w: Window, rec: dict, user_order: int, index: int, pcid: int) -> dict:
    output = dict(rec)
    output.update({
        "schema_version": SCHEMA_TRAIN, "behavior_prompt_version": PROMPT_VERSION,
        "precompute_id": pcid, "user_id": w.user_id,
        "user_order": user_order, "window_index": index,
        "interaction_sequence": w.items,
    })
    return output


def to_test_behavior(rec: dict, uid: str, i: int) -> dict:
    keys = ["behavior_explanation", "recommendation_behavior", "semantic_focus",
            "trajectory_signature", "behavior_signature", "mechanism", "scope",
            "direction", "confidence", "consistency_adjustments", "structural_evidence",
            "pattern_description", "keywords"]
    out = {k: rec.get(k) for k in keys}
    out.update(task_id=f"{uid}_W{i}", window_index=i,
               behavior_prompt_version=PROMPT_VERSION)
    return out


def compact_test_window(w: Window) -> list[dict]:
    return [{"item_id": x["item_id"], "item": x["item_name"],
             "category": x["item_category"], "action": x["action_type"]}
            for x in w.items]


def to_test_row(uid: str, obs: list[Any], windows: list[Window], cache: dict,
                recent_count: int, size: int, cap: int, mode: str) -> dict:
    b = [to_test_behavior(cache[window_key(w)], uid, i) for i, w in enumerate(windows)]
    evidence_fields = ["window_index", "recommendation_behavior", "semantic_focus",
                       "trajectory_signature", "behavior_signature", "mechanism", "scope",
                       "direction", "confidence"]
    evidence = [{k: bh.get(k) for k in evidence_fields} for bh in (b[-recent_count:] if recent_count > 0 else [])]
    return {
        "schema_version": SCHEMA_TEST,
        "behavior_prompt_version": PROMPT_VERSION,
        "behavior_schema": "dual_view_semantic_plus_trajectory",
        "window_policy": mode,
        "precompute_ok": True,
        "user_id": uid,
        "train_history_item_ids": [str(x) for x in obs],
        "train_history_hash": hashlib.sha256(json.dumps([str(x) for x in obs], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "window_size": size,
        "max_train_interactions": cap,
        "behavior_windows": [compact_test_window(w) for w in windows],
        "generated_behaviors": b,
        "recent_behavior_evidence": evidence,
    }


def make_prompt_summary(window: Window) -> list[dict]:
    return [{"item_id": x["item_id"], "item": x["item_name"],
             "category": x["item_category"], "action": x["action_type"]}
            for x in window.items]




# -----------------------------------------------------------------------------
# Built-in canonical Gemma prompt and parse/consistency helpers.
# Extracted directly from precompute_memory_create_dual_behavior.py, so the
# behavior prompt and taxonomy rules match existing v4 training cache.
# -----------------------------------------------------------------------------

def build_behavior_extraction_prompt(
    interaction_summary: List[Dict[str, Any]]
) -> str:
    """
    Create TWO complementary behavior views from the SAME observed window:

    1) recommendation_behavior / semantic_focus:
       content-aware and discriminative enough to help downstream item ranking.

    2) trajectory_signature + mechanism/scope/direction:
       domain-agnostic abstraction used for cross-user clustering and Tree learning.

    IMPORTANT: this stage only sees OBSERVED interactions. It never sees a future
    test/ground-truth item, so there is no oracle leakage.
    """
    return f"""You are creating a behavioral memory from a SHORT OBSERVED purchase window.

OBSERVED INTERACTIONS:
{json.dumps(interaction_summary, indent=2, ensure_ascii=False)}

Produce TWO COMPLEMENTARY representations.

A) RECOMMENDATION BEHAVIOR — semantic and discriminative
Describe the concrete preference represented by this observed window.
Preserve useful evidence when supported, such as:
- genre/subgenre or product subcategory;
- style, mood, theme, or use case;
- soundtrack / film / game orientation;
- regional or cultural direction;
- live / compilation / collection characteristics;
- creator continuity when it is actually visible;
- other concrete semantic properties that can distinguish relevant future items
  from clearly unrelated items.

This view should be specific enough to help a downstream recommender rank candidate
items, but it must remain grounded ONLY in the observed interactions.

B) TRAJECTORY ABSTRACTION — domain-agnostic and collaborative
Describe HOW the preference evolves within this observed window, independently of
the concrete content. This view is used to cluster behaviors across users and learn
behavior transitions.

Allowed mechanism labels:
[
  "repetition",
  "persistence",
  "collection expansion",
  "deepening",
  "narrowing",
  "broadening",
  "shifting",
  "returning",
  "adjacent exploration",
  "cross-category exploration",
  "refinement",
  "unknown"
]

Allowed scope labels:
[
  "same item",
  "same creator",
  "same collection/series",
  "same subcategory",
  "same broad category",
  "related category",
  "cross-category",
  "mixed",
  "unknown"
]

Allowed direction labels:
[
  "stable",
  "repeat",
  "deepen",
  "narrow",
  "broaden",
  "shift",
  "return",
  "mixed",
  "unknown"
]


TAXONOMY — USE THESE DEFINITIONS STRICTLY

MECHANISM:
- repetition:
  Literal repetition of the EXACT SAME item_id inside the observed window.
  NEVER use repetition merely because different items are similar.
- persistence:
  Different item_ids maintain the same broad preference without a clear move
  toward a narrower/more specific preference.
- collection expansion:
  Different item_ids expand an identifiable creator/series/collection/family.
- deepening:
  Different item_ids move further into the same focused subcategory/style/theme.
- narrowing:
  Preference becomes more selective/specific than earlier interactions.
- broadening:
  Preference expands to a wider range while retaining a recognizable anchor.
- shifting:
  A clear move from one preference/category/style toward another.
- returning:
  The user comes back to an earlier preference after an intervening different one.
- adjacent exploration:
  Movement into a clearly related neighboring category/style.
- cross-category exploration:
  Movement across substantially different categories/styles.
- refinement:
  Different item_ids explore variants inside a very narrow preference.
- unknown:
  Only when the observed evidence is genuinely insufficient.

SCOPE:
- same item:
  ONLY when the exact same item_id occurs more than once.
- same creator:
  Different item_ids but clear creator/artist continuity is visible.
- same collection/series:
  Different item_ids within an identifiable series/collection/family.
- same subcategory:
  Different item_ids sharing a focused genre/subgenre/product subtype.
- same broad category:
  Different item_ids sharing only a broad category.
- related category:
  Different but clearly neighboring categories.
- cross-category:
  Clearly different categories.
- mixed:
  No single relational scope dominates.
- unknown:
  Evidence is insufficient.

DIRECTION:
- repeat:
  ONLY for literal repeat of an exact item_id.
- stable:
  Different item_ids maintain a broadly stable preference.
- deepen:
  Move further into a focused preference.
- narrow:
  Move toward a more specific subset.
- broaden:
  Expand the preference range.
- shift:
  Move toward a different preference.
- return:
  Revisit an earlier preference after leaving it.
- mixed:
  Multiple directions coexist.
- unknown:
  Evidence is insufficient.

CRITICAL DISAMBIGUATION:
- Different item_ids that are semantically similar are NOT "repetition".
- Different albums/products from the same creator are NOT "same item".
- Different items in the same focused style should usually be persistence,
  deepening, refinement, or collection expansion depending on the evidence.
- Use "repetition + same item + repeat" only when an exact item_id repeats.

STRICT RULES:
1. Use ONLY evidence present in the observed interactions.
2. Do NOT predict or invent the user's future item.
3. recommendation_behavior MUST preserve concrete semantic evidence when available.
4. Avoid generic phrases such as "exploring related products", "expanding interests",
   "showing varied preferences", or "continuing preferences" when a more specific
   semantic description is supported.
5. semantic_focus should contain short reusable semantic phrases, not full sentences.
6. trajectory_signature MUST describe HOW behavior evolves, not WHAT content is preferred.
7. Do NOT put artist/item/product titles into trajectory_signature.
8. mechanism/scope/direction must be domain-agnostic.
9. Prefer a specific supported label; use "unknown" only when evidence is genuinely weak.
10. BEFORE choosing repetition/same item/repeat, explicitly check whether an exact
    item_id occurs more than once. If all item_ids are distinct, those labels are forbidden.
11. When item_ids are distinct, distinguish persistence vs collection expansion vs
    deepening/refinement vs broadening/shifting from the semantic relations in the window.
12. confidence reflects confidence in the trajectory abstraction, from 0.0 to 1.0.
13. trajectory_signature must be a RELATIONAL 12-35 word description of the
    structural transition: what is preserved vs changed, and whether the
    sequence stabilizes, deepens, shifts or returns. Never merely repeat the
    taxonomy labels, and never insert actual titles/creators/genres into HOW.
14. Do not use "repeat" or "same item" if all IDs are distinct. A repeated
    artist or similar album is a continuity pattern, NOT repetition of an item.
15. Return JSON only. No markdown.

EXAMPLE OF HOW PRECISION (abstract only):
- Too generic: "The user changes their preferences."
- Better: "The user moves from a loosely related choice to a closely connected
  follow-up, preserving a recognizable preference anchor while narrowing the
  variation across consecutive choices."
This is an example of wording, not a default answer. Infer the actual relationship
from the observed IDs, item titles and categories.

Return exactly:
{{
  "behavior_explanation": "1-2 concise sentences grounded in the observed interactions",
  "recommendation_behavior": "one concise, concrete, semantically discriminative description of the observed preference",
  "semantic_focus": ["3-8", "short", "semantic", "phrases"],
  "trajectory_signature": "short domain-agnostic phrase describing HOW preference evolves",
  "mechanism": "one allowed mechanism label",
  "scope": "one allowed scope label",
  "direction": "one allowed direction label",
  "confidence": 0.0
}}
"""


def enforce_structural_consistency(
    mechanism: str,
    scope: str,
    direction: str,
    interaction_sequence: List[Dict[str, Any]],
) -> Tuple[str, str, str, List[str], Dict[str, Any]]:
    """
    Deterministic guardrail for labels with literal structural meaning.

    We do NOT try to infer deepening/collection-expansion automatically.
    We only prevent impossible repetition/same-item/repeat labels when no
    exact item_id repeats. This keeps semantic interpretation with Gemma while
    protecting the Tree from taxonomy collapse.
    """
    item_ids = [
        str(x.get("item_id", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_id", "")).strip()
    ]
    unique_ids = set(item_ids)
    has_exact_repeat = (
        len(item_ids) >= 2
        and len(unique_ids) < len(item_ids)
    )
    all_same_item = (
        len(item_ids) >= 2
        and len(unique_ids) == 1
    )

    categories = [
        str(x.get("item_category", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_category", "")).strip()
        and str(x.get("item_category", "")).strip().lower() != "unknown"
    ]
    same_known_category = (
        len(categories) >= 2
        and len(set(categories)) == 1
    )

    adjustments: List[str] = []

    mechanism = str(mechanism or "unknown").strip().lower()
    scope = str(scope or "unknown").strip().lower()
    direction = str(direction or "unknown").strip().lower()

    if not has_exact_repeat:
        if mechanism == "repetition":
            old = mechanism
            # Do not guess persistence/deepening here. The item category can be
            # too broad (e.g., all CDs), so an impossible literal repetition is
            # converted to unknown and exposed for audit.
            mechanism = "unknown"
            adjustments.append(
                f"mechanism:{old}->{mechanism}:no_exact_item_repeat"
            )

        if scope == "same item":
            old = scope
            scope = "mixed"
            adjustments.append(
                f"scope:{old}->{scope}:no_exact_item_repeat"
            )

        if direction == "repeat":
            old = direction
            direction = "unknown"
            adjustments.append(
                f"direction:{old}->{direction}:no_exact_item_repeat"
            )

    # If every observed interaction is literally the same item, these labels
    # have unambiguous structural support.
    if all_same_item:
        if mechanism in {"unknown", "persistence"}:
            adjustments.append(
                f"mechanism:{mechanism}->repetition:all_same_item"
            )
            mechanism = "repetition"
        if scope != "same item":
            adjustments.append(
                f"scope:{scope}->same item:all_same_item"
            )
            scope = "same item"
        if direction in {"unknown", "stable"}:
            adjustments.append(
                f"direction:{direction}->repeat:all_same_item"
            )
            direction = "repeat"

    evidence = {
        "num_items": len(item_ids),
        "num_unique_item_ids": len(unique_ids),
        "has_exact_item_repeat": bool(has_exact_repeat),
        "all_same_item": bool(all_same_item),
        "same_known_category": bool(same_known_category),
    }

    return mechanism, scope, direction, adjustments, evidence


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


class BatchedQwenExtractor:
    """Match the existing Qwen2.5 Unsloth ranking backend; load only for inference."""
    def __init__(self, model_name: str, max_seq_length: int, load_in_4bit: bool, dtype: str) -> None:
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen extraction requires CUDA; use --plan-only for CPU")
        self.torch=torch
        self.max_seq_length=max_seq_length
        kw={"model_name":model_name,"max_seq_length":max_seq_length,"load_in_4bit":load_in_4bit}
        if dtype in ("fp16","bf16"):
            kw["dtype"]={"fp16":torch.float16,"bf16":torch.bfloat16}[dtype]
        self.model,self.tokenizer=FastLanguageModel.from_pretrained(**kw)
        self.tokenizer=get_chat_template(self.tokenizer,chat_template="qwen-2.5")
        self.tokenizer.padding_side="left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        self.device=next(self.model.parameters()).device

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        if not prompts:return []
        rendered=[]
        for prompt in prompts:
            txt=self.tokenizer.apply_chat_template(
                [{"role":"system","content":"Extract user behavior evidence from observed items only. Return JSON only."},
                 {"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
            if txt.startswith("<bos>"):txt=txt[len("<bos>"):]
            rendered.append(txt)
        enc=self.tokenizer(rendered,return_tensors="pt",padding=True,
                            truncation=False,add_special_tokens=False)
        width=int(enc["input_ids"].shape[1])
        if width+max_new_tokens>self.max_seq_length:
            raise ValueError(f"Input {width}+generation {max_new_tokens} exceeds max_seq_length={self.max_seq_length}")
        enc=enc.to(self.device)
        with self.torch.inference_mode():
            outputs=self.model.generate(**enc,max_new_tokens=max_new_tokens,
                do_sample=False,use_cache=True,pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        return [t.strip() for t in self.tokenizer.batch_decode(outputs[:,width:],skip_special_tokens=True)]


def validate_behavior(obj: dict, interaction_sequence: list[dict]) -> dict:
    """Reject invalid HOW rather than hiding it behind unknown labels or stale text."""
    if not isinstance(obj,dict):raise ValueError("Behavior response must be JSON object")
    rb=str(obj.get("recommendation_behavior") or "").strip()
    ts=str(obj.get("trajectory_signature") or "").strip()
    if len(rb.split())<7:raise ValueError("recommendation_behavior too short/missing")
    if not 8<=len(ts.split())<=70:
        raise ValueError("trajectory_signature must be 8-70 words with meaningful relational detail")
    m=oracle_tax.normalize_choice(obj.get("mechanism"),oracle_tax.MECHANISMS,"mechanism")
    sc=oracle_tax.normalize_choice(obj.get("scope"),oracle_tax.SCOPES,"scope")
    d=oracle_tax.normalize_choice(obj.get("direction"),oracle_tax.DIRECTIONS,"direction")
    ids=[str(x.get("item_id")) for x in interaction_sequence if x.get("item_id") is not None]
    if len(ids)>=2 and len(set(ids))==len(ids) and (m=="repetition" or sc=="same item" or d=="repeat"):
        raise ValueError("Impossible exact-item repetition for distinct item IDs")
    if ts.lower() in (f"{m}; {sc}; {d}", f"{m} {sc} {d}"):
        raise ValueError("trajectory_signature only repeats 3 taxonomy labels")
    for it in interaction_sequence:
        if str(it.get("item_id", "")).lower() in ts.lower():
            raise ValueError("trajectory_signature copies item ID")
        title=str(it.get("item_name", "")).strip()
        if len(title)>=12 and title.lower() in ts.lower():
            raise ValueError("trajectory_signature copies an item title")
    sf=obj.get("semantic_focus")
    if not isinstance(sf,list):raise ValueError("semantic_focus must be a JSON list")
    sf=[str(x).strip() for x in sf if str(x).strip()][:8]
    if not sf:raise ValueError("semantic_focus is empty")
    conf=obj.get("confidence",0.0)
    try:conf=float(conf)
    except (TypeError,ValueError):conf=0.0
    if not math.isfinite(conf):conf=0.0
    return {"recommendation_behavior":rb,"trajectory_signature":ts,
            "semantic_focus":sf,"mechanism":m,"scope":sc,"direction":d,
            "confidence":max(0.,min(1.,conf)),
            "behavior_explanation":str(obj.get("behavior_explanation") or "").strip()}


def infer_missing(tasks:dict[str,Window],cache:dict[str,dict],args,cache_output:Path)->dict:
    """Retry only invalid windows; checkpoint success; STOP if some windows never validate."""
    if not tasks:
        print("[INFER] All Qwen windows validated in cache. No GPU inference needed.",flush=True)
        return {"new_windows":0,"calls":0,"failed":0}
    extractor=BatchedQwenExtractor(args.model_name,args.max_seq_length,args.load_in_4bit,args.dtype)
    pending=dict(tasks)
    errors={}; previous_outputs={}; calls=0
    cache_output.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,args.max_attempts+1):
        if not pending:break
        print(f"[INFER] Qwen attempt {attempt}/{args.max_attempts}: {len(pending)} windows",flush=True)
        next_pending={}
        with tqdm(total=len(pending),desc=f"Qwen behavior attempt {attempt}",unit="window",
                  dynamic_ncols=True,mininterval=0.5) as bar:
            keys=list(pending)
            for offset in range(0,len(keys),args.llm_batch_size):
                chunk=keys[offset:offset+args.llm_batch_size]
                prompts=[]
                for key in chunk:
                    prompt=build_behavior_extraction_prompt(make_prompt_summary(pending[key]))
                    if key in errors:
                        prompt+=("\nREPAIR: Previous extraction invalid: "+errors[key]+
                          "\nPrevious output: "+previous_outputs.get(key,"")[:500]+
                          "\nReturn a corrected FULL JSON with concrete WHAT, relational HOW, "
                          "exact allowed HOW labels and no unsupported literal repetition.")
                    prompts.append(prompt)
                responses=extractor.generate_batch(prompts,args.max_new_tokens)
                if len(responses)!=len(chunk):raise RuntimeError("Qwen batch/output size mismatch")
                calls+=1
                rows=[]
                for key,raw in zip(chunk,responses):
                    win=pending[key]
                    try:
                        extracted=validate_behavior(parse_json_response(raw),win.items)
                        adjustments=[];evidence={"num_items":len(win.items),
                            "num_unique_item_ids":len({x["item_id"] for x in win.items})}
                        row={"schema_version":SCHEMA_TRAIN,"behavior_prompt_version":PROMPT_VERSION,
                             "user_id":win.user_id,"interaction_sequence":win.items,
                             "generation_config":generation_config(args),
                             "behavior_explanation":extracted["behavior_explanation"],
                             "recommendation_behavior":extracted["recommendation_behavior"],
                             "semantic_focus":extracted["semantic_focus"],
                             "trajectory_signature":extracted["trajectory_signature"],
                             "behavior_signature":extracted["trajectory_signature"],
                             "mechanism":extracted["mechanism"],"scope":extracted["scope"],
                             "direction":extracted["direction"],"confidence":extracted["confidence"],
                             "consistency_adjustments":adjustments,"structural_evidence":evidence,
                             "pattern_description":extracted["recommendation_behavior"],
                             "keywords":extracted["semantic_focus"],
                             "parse_ok":True,"parse_error":None,"raw_response":raw}
                        cache[key]=row
                    except Exception as ex:
                        errors[key]=f"{type(ex).__name__}: {ex}"
                        previous_outputs[key]=raw
                        next_pending[key]=win
                        row={"schema_version":SCHEMA_TRAIN,"behavior_prompt_version":PROMPT_VERSION,
                             "user_id":win.user_id,"interaction_sequence":win.items,
                             "generation_config":generation_config(args),
                             "parse_ok":False,"parse_error":errors[key],"raw_response":raw,
                             "attempt":attempt}
                    rows.append(row)
                with cache_output.open("a",encoding="utf8") as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
                    f.flush();os.fsync(f.fileno())
                bar.update(len(chunk))
        pending=next_pending
    if pending:
        counts=Counter(errors[k] for k in pending)
        print("[FAIL] Remaining invalid windows:",len(pending),dict(counts.most_common(10)))
        print("[FAIL] Examples:",[(tasks[k].user_id,errors[k]) for k in list(pending)[:8]])
        raise RuntimeError(f"{len(pending)} windows did not produce valid Qwen behaviors. "
                           "Valid windows are checkpointed; rerun the same command to retry.")
    return {"new_windows":len(tasks),"calls":calls,"failed":0}


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="Dataset name, e.g. CDs, Video_Games, Books, ML")
    p.add_argument("--data-root", default="data", help="Root folder containing dataset subfolders")
    p.add_argument("--data-dir", help="Optional direct dataset folder; alternative to --dataset")
    p.add_argument("--sequences", help="Optional input override; defaults to <data-dir>/user_sequences.json")
    p.add_argument("--items", help="Optional input override; defaults to <data-dir>/items.json")
    p.add_argument("--sequences-file", default="user_sequences.json", help="Filename inside data-dir")
    p.add_argument("--items-file", default="items.json", help="Filename inside data-dir")
    p.add_argument("--user-ids-file", help="Common protocol user list for both outputs; auto-uses protocol/eval_users_1000.json if available")
    p.add_argument("--number-of-users", type=int, default=0, help="Select first N users for BOTH outputs; 0 = all")
    p.add_argument("--output-root", default="precomputed", help="Root for automatically named output files")
    p.add_argument("--output-dir", help="Optional output folder override")
    p.add_argument("--existing-train", help="Old training memories (optional; auto-detect known filenames if omitted)")
    p.add_argument("--no-auto-existing-train", action="store_true", help="Disable auto-discovery of old training memories")
    p.add_argument("--cache-output", help="Durable incremental cache; auto-detected on restart")
    p.add_argument("--train-output", help="Optional output override (default: <output-dir>/train_shared.jsonl)")
    p.add_argument("--test-output", help="Optional output override (default: <output-dir>/test_shared.jsonl)")
    p.add_argument("--history-size", type=int, default=20, help="0 = unlimited; cap raw items BEFORE metadata filtering")
    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--recent-behaviors", type=int, default=5)
    p.add_argument("--plan-only", action="store_true", help="Audit input and cache reuse; no Qwen/GPU needed")
    p.add_argument("--max-attempts",type=int,default=3,help="Retry malformed/invalid Qwen extractions")
    p.add_argument("--overwrite", action="store_true", help="Atomically replace existing finished train/test outputs")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--llm-batch-size", type=int, default=4)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    p.add_argument("--allow-cache-config-mismatch", action="store_true", help="Explicitly allow cache model/config mismatch (not recommended)")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if args.dataset and data_dir.name != args.dataset:
            raise ValueError("--dataset does not match --data-dir basename")
        dataset = args.dataset or data_dir.name
    elif args.dataset:
        dataset = args.dataset
        data_dir = Path(args.data_root) / dataset
    else:
        raise ValueError("Set --dataset CDs (recommended) or --data-dir PATH")

    args.dataset = dataset
    args.data_dir = str(data_dir)
    args.sequences = args.sequences or str(data_dir / args.sequences_file)
    args.items = args.items or str(data_dir / args.items_file)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / dataset / "qwen"
    args.output_dir = str(output_dir)
    args.train_output = args.train_output or str(output_dir / "train_shared.jsonl")
    args.test_output = args.test_output or str(output_dir / "test_shared.jsonl")
    args.cache_output = args.cache_output or str(output_dir / "shared_windows.cache.jsonl")

    # Prefer the frozen protocol cohort over arbitrary first-N source users.
    if not args.user_ids_file:
        protocol_users = data_dir / "protocol" / "eval_users_1000.json"
        if protocol_users.is_file():
            args.user_ids_file = str(protocol_users)
            print(f"[AUTO] Using protocol user IDs: {protocol_users}", flush=True)

    if not args.existing_train and not args.no_auto_existing_train:
        # Reuse only from same Qwen folder / same schema and prompt.
        for candidate in (output_dir / "local_memories_qwen.jsonl",):
            if candidate.is_file():
                args.existing_train = str(candidate)
                print(f"[AUTO] Found previous train cache: {candidate}", flush=True)
                break

    for field in ("sequences", "items"):
        if not Path(getattr(args, field)).is_file():
            raise FileNotFoundError(f"Missing {field} file: {getattr(args, field)}")


def validate_selected_users(all_users: list[str], ids: list[str], sequences: dict) -> list[str]:
    ids = list(dict.fromkeys(ids))
    missing = [uid for uid in ids if uid not in sequences]
    if missing:
        raise ValueError(f"User IDs not found in sequences: {missing[:10]}")
    return ids


def verify_train_cache_metadata(path: str | None, args) -> None:
    if not path:
        return
    meta = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    if not meta.is_file():
        print(f"[WARN] Missing cache metadata {meta}: model/config could not be verified.", flush=True)
        return
    obj = load_json(meta)
    mismatches = []
    checks = {"model_name": args.model_name, "max_new_tokens": args.max_new_tokens,
              "max_seq_length": args.max_seq_length, "sample": False}
    for field, expected in checks.items():
        if field in obj and obj[field] != expected:
            mismatches.append(f"{field}: previous={obj[field]!r}, requested={expected!r}")
    if mismatches:
        msg = "; ".join(mismatches)
        if not args.allow_cache_config_mismatch:
            raise ValueError("Training cache model configuration differs: " + msg +
                             ". Use --allow-cache-config-mismatch only if intentional.")
        print("[WARN] Reusing cache with mismatched model configuration: " + msg, flush=True)


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    if args.window_size <= 0 or args.history_size < 0 or args.llm_batch_size <= 0 or args.max_attempts<1:
        raise ValueError("window-size and batch-size must be positive, history-size non-negative")
    if args.number_of_users < 0:
        raise ValueError("number-of-users must be >= 0 (0 means all)")
    if Path(args.train_output).resolve() == Path(args.test_output).resolve():
        raise ValueError("Train and test outputs must have different paths")
    if args.existing_train and Path(args.existing_train).resolve() == Path(args.train_output).resolve():
        raise ValueError("Do not overwrite --existing-train in-place; use a different --train-output")
    if Path(args.cache_output).resolve() in (Path(args.train_output).resolve(), Path(args.test_output).resolve()):
        raise ValueError("Cache output cannot be a final train/test output")

    sequences = {str(k): v for k, v in load_json(args.sequences).items()}
    items = load_json(args.items)
    all_users = list(sequences)
    users = validate_selected_users(all_users, read_ids(args.user_ids_file, all_users), sequences)
    if args.number_of_users > 0:
        users = users[:args.number_of_users]
    if not users:
        raise ValueError("No users selected")

    verify_train_cache_metadata(args.existing_train, args)
    expected_config = generation_config(args)
    old_cache, read_counts = read_cache(args.existing_train, expected_config,
                                        args.allow_cache_config_mismatch)
    checkpoint_path = Path(args.cache_output)
    if args.existing_train and checkpoint_path.resolve() == Path(args.existing_train).resolve():
        raise ValueError("--cache-output must differ from --existing-train")
    checkpoint_cache, checkpoint_counts = read_cache(
        str(checkpoint_path) if checkpoint_path.is_file() else None,
        expected_config, args.allow_cache_config_mismatch)
    cache = dict(old_cache)
    cache.update(checkpoint_cache)

    # The SAME observed train windows power both output schemas.
    # NEVER read the held-out `test` items when constructing behavior.
    train_by_user: dict[str, list[Window]] = {}
    test_by_user: dict[str, list[Window]] = {}
    raw_train_items = 0
    valid_train_items = 0
    missing_train_metadata = 0
    ignored_validation_items = 0
    missing_test_field = 0
    for uid in tqdm(users, desc="Build shared train-only windows", unit="user",
                    dynamic_ncols=True, mininterval=0.5):
        user_rec = sequences[uid]
        if not isinstance(user_rec, dict):
            raise TypeError(f"User {uid!r} does not have a dict with train/test fields")
        if "train" not in user_rec:
            raise ValueError(f"User {uid!r} does not have a train history")
        train = as_ids(user_rec.get("train"))
        if not train:
            raise ValueError(f"User {uid!r} has empty train history")
        if not as_ids(user_rec.get("test")):
            missing_test_field += 1  # Not used for extraction; audit only.
        ignored_validation_items += len(as_ids(user_rec.get("val"))) + len(as_ids(user_rec.get("validation"))) + len(as_ids(user_rec.get("valid")))
        raw_train_items += len(train)
        capped_train = train[-args.history_size:] if args.history_size > 0 else train
        valid_train_items += sum(get_info(items, item) is not None for item in capped_train)
        missing_train_metadata += sum(get_info(items, item) is None for item in capped_train)
        train_windows = make_train_windows(uid, train, items, args.history_size, args.window_size)
        if not train_windows:
            raise ValueError(f"User {uid!r} has no titled observed train items")
        train_by_user[uid] = train_windows
        test_by_user[uid] = train_windows  # identical history/windows, no validation

    if ignored_validation_items:
        print(f"[WARN] Ignoring {ignored_validation_items} validation interactions present in input; this protocol uses train only.", flush=True)
    if missing_test_field:
        print(f"[WARN] {missing_test_field} users have no test item; target labels were never used in precompute.", flush=True)

    train_all = [w for uid in users for w in train_by_user[uid]]
    test_all = [w for uid in users for w in test_by_user[uid]]
    needed_windows: dict[str, Window] = {}
    for w in train_all + test_all:
        needed_windows.setdefault(window_key(w), w)
    in_train = {window_key(w) for w in train_all}
    in_test = {window_key(w) for w in test_all}
    if in_train != in_test or len(needed_windows) != len(in_train):
        raise AssertionError("Train/test behavior windows differ in no-validation mode")
    missing = {key: win for key, win in needed_windows.items() if key not in cache}

    info = {
        "dataset": args.dataset, "sequences": args.sequences, "items": args.items,
        "train_output": args.train_output, "test_output": args.test_output,
        "users": len(users), "same_users_train_test": True,
        "same_observed_train_history": True, "validation_used": False,
        "test_ground_truth_used_in_extraction": False,
        "ignored_validation_items_present_in_source": ignored_validation_items,
        "users_without_test_target": missing_test_field,
        "user_ids_file": args.user_ids_file,
        "raw_train_items": raw_train_items,
        "valid_train_items_after_history_cap": valid_train_items,
        "skipped_train_items_without_title_after_cap": missing_train_metadata,
        "history_size": args.history_size, "window_size": args.window_size,
        "test_window_policy": "same_as_train_no_validation",
        "train_windows": len(train_all), "test_windows": len(test_all),
        "shared_train_test_window_keys": len(in_train & in_test),
        "test_only_window_keys": len(in_test - in_train),
        "train_only_window_keys": len(in_train - in_test),
        "total_unique_windows": len(needed_windows),
        "valid_existing_train_cache_rows": read_counts.get("valid_records", 0),
        "valid_checkpoint_cache_rows": checkpoint_counts.get("valid_records", 0),
        "checkpoint_config_mismatch_skipped": checkpoint_counts.get("generation_config_mismatch", 0),
        "windows_reused_from_cache": len(needed_windows) - len(missing),
        "qwen_windows_pending": len(missing),
        "qwen_generate_calls_pending_at_batch_size": (len(missing) + args.llm_batch_size - 1) // args.llm_batch_size,
        "cache_checkpoint_path": str(checkpoint_path),
        "max_new_tokens": args.max_new_tokens,
    }
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    for output in [args.train_output, args.test_output]:
        if Path(output).exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite or another filename")

    stats = infer_missing(missing, cache, args, checkpoint_path)
    records: list[dict] = []
    for user_order, uid in enumerate(
            tqdm(users, desc="Pack train records", unit="user", dynamic_ncols=True,
                 mininterval=0.5)):
        for i, w in enumerate(train_by_user[uid]):
            records.append(to_train_record(w, cache[window_key(w)], user_order, i, len(records)))
    write_jsonl_atomic(args.train_output, records, args.overwrite,
                       progress_label="Write train JSONL")
    print(f"[WRITE] train records={len(records)} users={len(users)} path={args.train_output}", flush=True)

    test_results = []
    for uid in tqdm(users, desc="Pack test records", unit="user",
                    dynamic_ncols=True, mininterval=0.5):
        observed = as_ids(sequences[uid].get("train"))  # train ONLY; never val/test
        if args.history_size > 0:
            observed = observed[-args.history_size:]
        test_results.append(to_test_row(uid, observed, test_by_user[uid], cache,
                                        args.recent_behaviors, args.window_size,
                                        args.history_size, "same_as_train_no_validation"))
    write_jsonl_atomic(args.test_output, test_results, args.overwrite,
                       progress_label="Write test JSONL")
    print(f"[WRITE] test users={len(test_results)} path={args.test_output}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
