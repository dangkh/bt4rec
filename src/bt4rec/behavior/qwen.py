"""Qwen2.5-7B behaviour-extraction backend (strict prompt, validated + retried)."""
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

from bt4rec.behavior.parsing import parse_json_response
from bt4rec.behavior.records import SCHEMA_TRAIN
from bt4rec.behavior.windows import Window, cache_key, make_prompt_summary

try:  # Prefer the original taxonomy module when it is on PYTHONPATH.
    import oracle_probe_qwen as oracle_tax  # type: ignore
except ImportError:  # Fallback: identical label lists from the extraction prompt.
    from bt4rec import taxonomy as oracle_tax
PROMPT_VERSION = "dual_view_discriminative_behavior_qwen_v1_strict"


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


# -----------------------------------------------------------------------------
# Backend interface consumed by bt4rec.cli.precompute
# -----------------------------------------------------------------------------
from bt4rec.behavior import records as _records  # noqa: E402


def generation_config(args) -> dict:
    return _records.generation_config(args, PROMPT_VERSION)


NAME = "qwen"
OUTPUT_SUBDIR = "qwen"  # precomputed/<dataset>/qwen/
LEGACY_TRAIN_NAMES = ("local_memories_qwen.jsonl",)
DEFAULTS = {"model_name": "Qwen/Qwen2.5-7B-Instruct", "max_seq_length": 8192,
            "max_new_tokens": 512, "llm_batch_size": 4, "max_attempts": 3}
