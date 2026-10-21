"""Gemma 3 behaviour-extraction backend (prompt v2 taxonomy)."""
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

from bt4rec.behavior.parsing import enforce_structural_consistency, parse_json_response
from bt4rec.behavior.records import SCHEMA_TRAIN
from bt4rec.behavior.windows import Window, cache_key, make_prompt_summary

PROMPT_VERSION = "dual_view_discriminative_behavior_v2_taxonomy"
BASE_PROMPT_VERSION = PROMPT_VERSION
# precompute --behavior-domain books: book-specific body (view A evidence + taxonomy
# definitions worded for genres/topics/series/authors). Same labels and JSON schema,
# so clustering / Tree / inference code is unchanged. PROMPT_VERSION gets a suffix
# so caches of different prompts never mix.
DOMAIN = "generic"
BOOKS_BODY = 'Produce TWO COMPLEMENTARY representations.\n\nA) RECOMMENDATION BEHAVIOR — semantic and discriminative\nDescribe the concrete reading preference represented by this observed window.\nPreserve useful evidence when supported, such as:\n- genre/subgenre or topic/subject;\n- style, mood, theme, setting, or reading use case;\n- fiction / non-fiction or audience orientation;\n- regional, cultural, or historical direction;\n- series / collection / edition characteristics;\n- author/creator continuity when it is actually visible;\n- other concrete semantic properties that can distinguish relevant future books\n  from clearly unrelated books.\n\nThis view should be specific enough to help a downstream recommender rank candidate\nitems, but it must remain grounded ONLY in the observed interactions.\n\nB) TRAJECTORY ABSTRACTION — domain-agnostic and collaborative\nDescribe HOW the preference evolves within this observed window, independently of\nthe concrete content. This view is used to cluster behaviors across users and learn\nbehavior transitions.\n\nAllowed mechanism labels:\n[\n  "repetition",\n  "persistence",\n  "collection expansion",\n  "deepening",\n  "narrowing",\n  "broadening",\n  "shifting",\n  "returning",\n  "adjacent exploration",\n  "cross-category exploration",\n  "refinement",\n  "unknown"\n]\n\nAllowed scope labels:\n[\n  "same item",\n  "same creator",\n  "same collection/series",\n  "same subcategory",\n  "same broad category",\n  "related category",\n  "cross-category",\n  "mixed",\n  "unknown"\n]\n\nAllowed direction labels:\n[\n  "stable",\n  "repeat",\n  "deepen",\n  "narrow",\n  "broaden",\n  "shift",\n  "return",\n  "mixed",\n  "unknown"\n]\n\n\nTAXONOMY — USE THESE DEFINITIONS STRICTLY\n\nMECHANISM:\n- repetition:\n  Literal repetition of the EXACT SAME item_id inside the observed window.\n  NEVER use repetition merely because different books are similar.\n- persistence:\n  Different item_ids maintain the same recognizable genre/topic preference\n  without a clear directional change.\n- collection expansion:\n  Different item_ids expand an identifiable series/collection/family.\n- deepening:\n  Different item_ids move progressively further into an already focused\n  preference, with evidence of greater depth or specialization.\n- narrowing:\n  Preference becomes more selective/specific than earlier interactions.\n- broadening:\n  Preference expands to a wider range while retaining a recognizable anchor.\n- shifting:\n  A clear move from one genre/topic/theme preference toward another.\n- returning:\n  The user comes back to an earlier preference after an intervening different one.\n- adjacent exploration:\n  Movement into a clearly related neighboring genre/topic/theme.\n- cross-category exploration:\n  Movement across substantially different genres/topics/themes.\n- refinement:\n  Different item_ids explore close variants inside an already narrow preference.\n- unknown:\n  Only when the observed evidence is genuinely insufficient.\n\nSCOPE:\n- same item:\n  ONLY when the exact same item_id occurs more than once.\n- same creator:\n  Different item_ids but clear author/creator continuity is visible.\n- same collection/series:\n  Different item_ids within an identifiable series/collection/family.\n- same subcategory:\n  Different item_ids sharing a focused genre/subgenre/topic.\n- same broad category:\n  Different item_ids sharing a meaningful broad semantic category.\n  "Books" or "reading" alone is NOT sufficient.\n- related category:\n  Different but clearly neighboring genres/topics.\n- cross-category:\n  Clearly different genres/topics.\n- mixed:\n  No single relational scope dominates.\n- unknown:\n  Evidence is insufficient.\n\nDIRECTION:\n- repeat:\n  ONLY for literal repeat of an exact item_id.\n- stable:\n  Different item_ids maintain a broadly stable preference.\n- deepen:\n  Move progressively further into an already focused preference.\n- narrow:\n  Move toward a more specific subset.\n- broaden:\n  Expand the preference range.\n- shift:\n  Move toward a different preference.\n- return:\n  Revisit an earlier preference after leaving it.\n- mixed:\n  Multiple directions coexist.\n- unknown:\n  Evidence is insufficient.\n\nCRITICAL DISAMBIGUATION:\n- Different item_ids that are semantically similar are NOT "repetition".\n- Different books from the same author are NOT "same item".\n- Different books in the same genre/topic without directional progression should\n  usually be "persistence", not "deepening".\n- Different installments in an identifiable series/collection should usually be\n  "collection expansion".\n- Use "deepening" only when interaction order shows progression toward greater\n  depth or specialization.\n- Use "refinement" for close variants within an already narrow preference when\n  there is no clear directional progression.\n- Use "repetition + same item + repeat" only when an exact item_id repeats.\n\nSTRICT RULES:\n1. Use ONLY evidence present in the observed interactions.\n2. Do NOT predict or invent the user\'s future item.\n3. recommendation_behavior MUST preserve concrete semantic evidence when available.\n4. Avoid generic phrases such as "exploring related books", "expanding interests",\n   "showing varied preferences", or "continuing preferences" when a more specific\n   semantic description is supported.\n5. semantic_focus should contain short reusable semantic phrases, not full sentences.\n6. trajectory_signature MUST describe HOW behavior evolves, not WHAT content is preferred.\n7. Do NOT put author/item/series/genre/topic names into trajectory_signature.\n8. mechanism/scope/direction must be domain-agnostic.\n9. Prefer a specific supported label; use "unknown" only when evidence is genuinely weak.\n10. BEFORE choosing repetition/same item/repeat, explicitly check whether an exact\n    item_id occurs more than once. If all item_ids are distinct, those labels are forbidden.\n11. When item_ids are distinct, distinguish persistence vs collection expansion vs\n    deepening/refinement vs broadening/shifting from BOTH semantic relations and order.\n12. Do NOT use "deepening" solely because later books are semantically similar.\n13. confidence reflects confidence in the trajectory abstraction, from 0.0 to 1.0.\n14. Return JSON only. No markdown.\n\nReturn exactly:\n{\n  "behavior_explanation": "1-2 concise sentences grounded in the observed interactions",\n  "recommendation_behavior": "one concise, concrete, semantically discriminative description of the observed preference",\n  "semantic_focus": ["3-8", "short", "semantic", "phrases"],\n  "trajectory_signature": "short domain-agnostic phrase describing HOW preference evolves",\n  "mechanism": "one allowed mechanism label",\n  "scope": "one allowed scope label",\n  "direction": "one allowed direction label",\n  "confidence": 0.0\n}\n'

# v2: stricter "collection expansion / same collection/series" (explicit series
# evidence only) and "deepening" (progression, not mere similarity).
BOOKS_BODY_V2 = 'Produce TWO COMPLEMENTARY representations.\n\nA) RECOMMENDATION BEHAVIOR — semantic and discriminative\nDescribe the concrete reading preference represented by this observed window.\nPreserve useful evidence when supported, such as:\n- genre/subgenre or topic/subject;\n- style, mood, theme, setting, or reading use case;\n- fiction / non-fiction or audience orientation;\n- regional, cultural, or historical direction;\n- series / collection characteristics;\n- author/creator continuity when it is actually visible;\n- other concrete semantic properties that can distinguish relevant future books\n  from clearly unrelated books.\n\nThis view should be specific enough to help a downstream recommender rank candidate\nitems, but it must remain grounded ONLY in the observed interactions.\n\nB) TRAJECTORY ABSTRACTION — domain-agnostic and collaborative\nDescribe HOW the preference evolves within this observed window, independently of\nthe concrete content. This view is used to cluster behaviors across users and learn\nbehavior transitions.\n\nAllowed mechanism labels:\n[\n  "repetition",\n  "persistence",\n  "collection expansion",\n  "deepening",\n  "narrowing",\n  "broadening",\n  "shifting",\n  "returning",\n  "adjacent exploration",\n  "cross-category exploration",\n  "refinement",\n  "unknown"\n]\n\nAllowed scope labels:\n[\n  "same item",\n  "same creator",\n  "same collection/series",\n  "same subcategory",\n  "same broad category",\n  "related category",\n  "cross-category",\n  "mixed",\n  "unknown"\n]\n\nAllowed direction labels:\n[\n  "stable",\n  "repeat",\n  "deepen",\n  "narrow",\n  "broaden",\n  "shift",\n  "return",\n  "mixed",\n  "unknown"\n]\n\n\nTAXONOMY — USE THESE DEFINITIONS STRICTLY\n\nMECHANISM:\n- repetition:\n  Literal repetition of the EXACT SAME item_id inside the observed window.\n  NEVER use repetition merely because different books are similar.\n- persistence:\n  Different item_ids maintain the same recognizable genre/topic preference\n  without a clear directional change.\n- collection expansion:\n  Different item_ids expand an EXPLICITLY IDENTIFIABLE series/collection/family.\n  Similar genre, topic, theme, title wording, or author continuity alone is NOT\n  sufficient evidence of collection expansion.\n- deepening:\n  Different item_ids move progressively further into an already focused\n  preference, with evidence of greater depth or specialization.\n  Mere semantic similarity or continued interest is NOT sufficient.\n- narrowing:\n  Preference becomes more selective/specific than earlier interactions.\n- broadening:\n  Preference expands to a wider range while retaining a recognizable anchor.\n- shifting:\n  A clear move from one genre/topic/theme preference toward another.\n- returning:\n  The user comes back to an earlier preference after an intervening different one.\n- adjacent exploration:\n  Movement into a clearly related neighboring genre/topic/theme.\n- cross-category exploration:\n  Movement across substantially different genres/topics/themes.\n- refinement:\n  Different item_ids explore close variants inside an already narrow preference.\n- unknown:\n  Only when the observed evidence is genuinely insufficient.\n\nSCOPE:\n- same item:\n  ONLY when the exact same item_id occurs more than once.\n- same creator:\n  Different item_ids but clear author/creator continuity is visible.\n- same collection/series:\n  Different item_ids within an EXPLICITLY IDENTIFIABLE series/collection/family.\n  Similar genre, topic, theme, title wording, or same-author evidence alone is\n  NOT sufficient.\n- same subcategory:\n  Different item_ids sharing a focused genre/subgenre/topic.\n- same broad category:\n  Different item_ids sharing a meaningful broad semantic category.\n  "Books", "reading", or "novels" alone is NOT sufficient.\n- related category:\n  Different but clearly neighboring genres/topics.\n- cross-category:\n  Clearly different genres/topics.\n- mixed:\n  No single relational scope dominates.\n- unknown:\n  Evidence is insufficient.\n\nDIRECTION:\n- repeat:\n  ONLY for literal repeat of an exact item_id.\n- stable:\n  Different item_ids maintain a broadly stable preference.\n- deepen:\n  Move progressively further into an already focused preference.\n- narrow:\n  Move toward a more specific subset.\n- broaden:\n  Expand the preference range.\n- shift:\n  Move toward a different preference.\n- return:\n  Revisit an earlier preference after leaving it.\n- mixed:\n  Multiple directions coexist.\n- unknown:\n  Evidence is insufficient.\n\nCRITICAL DISAMBIGUATION:\n- Different item_ids that are semantically similar are NOT "repetition".\n- Different books from the same author are NOT "same item".\n- Different books in the same genre/topic without directional progression should\n  usually be "persistence", not "deepening".\n- Use "collection expansion" ONLY when a shared series/collection/family is\n  explicitly identifiable from the observed evidence.\n- Same author, similar titles, shared genre, or shared topic alone must NOT be\n  interpreted as the same series/collection.\n- When an identifiable series/collection is expanded without another clear\n  directional change, use "collection expansion + same collection/series + stable".\n- Use "deepening" only when interaction order shows progression toward greater\n  depth or specialization.\n- Use "refinement" for close variants within an already narrow preference when\n  there is no clear directional progression.\n- Use "repetition + same item + repeat" only when an exact item_id repeats.\n\nSTRICT RULES:\n1. Use ONLY evidence present in the observed interactions.\n2. Do NOT predict or invent the user\'s future item.\n3. recommendation_behavior MUST preserve concrete semantic evidence when available.\n4. Avoid generic phrases such as "exploring related books", "expanding interests",\n   "showing varied preferences", "continuing preferences", or "deepening within books"\n   when a more specific semantic description is supported.\n5. semantic_focus should contain short reusable semantic phrases, not full sentences.\n6. trajectory_signature MUST describe HOW behavior evolves, not WHAT content is preferred.\n7. Do NOT put author/item/series/genre/topic names into trajectory_signature.\n8. mechanism/scope/direction must be domain-agnostic.\n9. Prefer a specific supported label; use "unknown" only when evidence is genuinely weak.\n10. BEFORE choosing repetition/same item/repeat, explicitly check whether an exact\n    item_id occurs more than once. If all item_ids are distinct, those labels are forbidden.\n11. When item_ids are distinct, distinguish persistence vs collection expansion vs\n    deepening/refinement vs broadening/shifting from BOTH semantic relations and order.\n12. Do NOT infer collection expansion solely from semantic similarity, shared authorship,\n    or similar title wording.\n13. Do NOT infer deepening solely because later books are semantically similar.\n14. confidence reflects confidence in the trajectory abstraction, from 0.0 to 1.0.\n15. Return JSON only. No markdown.\n\nReturn exactly:\n{\n  "behavior_explanation": "1-2 concise sentences grounded in the observed interactions",\n  "recommendation_behavior": "one concise, concrete, semantically discriminative description of the observed preference",\n  "semantic_focus": ["3-8", "short", "semantic", "phrases"],\n  "trajectory_signature": "short domain-agnostic phrase describing HOW preference evolves",\n  "mechanism": "one allowed mechanism label",\n  "scope": "one allowed scope label",\n  "direction": "one allowed direction label",\n  "confidence": 0.0\n}\n'
DOMAIN_BODIES = {"books": BOOKS_BODY, "books_v2": BOOKS_BODY_V2}
DOMAIN_VERSIONS = {"books": "_books_v1", "books_v2": "_books_v2"}


def set_domain(domain: str) -> None:
    global DOMAIN, PROMPT_VERSION
    if domain != "generic" and domain not in DOMAIN_BODIES:
        raise ValueError(f"Unknown behavior domain {domain!r}")
    DOMAIN = domain
    PROMPT_VERSION = BASE_PROMPT_VERSION + DOMAIN_VERSIONS.get(domain, "")



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
    if DOMAIN in DOMAIN_BODIES:
        return ("You are creating a behavioral memory from a SHORT OBSERVED reading/purchase window.\n\n"
                "OBSERVED INTERACTIONS:\n"
                + json.dumps(interaction_summary, indent=2, ensure_ascii=False) + "\n\n"
                + DOMAIN_BODIES[DOMAIN])
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
13. Return JSON only. No markdown.

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


class BatchedGemmaExtractor:
    """Lazy Unsloth loader. --plan-only works without torch/unsloth/CUDA."""
    def __init__(self, model_name: str, max_seq_length: int,
                 load_in_4bit: bool, dtype: str) -> None:
        import torch
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        if not torch.cuda.is_available():
            raise RuntimeError("Gemma precompute requires CUDA; use --plan-only for CPU audit")
        self.torch = torch
        self.max_seq_length = max_seq_length
        t_dtype = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        kwargs = {"model_name": model_name, "max_seq_length": max_seq_length,
                  "load_in_4bit": load_in_4bit, "full_finetuning": False}
        if t_dtype is not None:
            kwargs["dtype"] = t_dtype
        self.model, self.tokenizer = FastModel.from_pretrained(**kwargs)
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma3")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            FastModel.for_inference(self.model)
        except Exception:
            pass
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content":
             "You are a behavioral memory modeling system for recommender systems. "
             "Create both a recommendation-oriented semantic view and a "
             "domain-agnostic trajectory view. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]
        return text

    def generate_batch(self, prompts: list[str], max_new_tokens: int,
                       temperature: float, top_p: float, top_k: int,
                       do_sample: bool) -> tuple[list[str], dict]:
        rendered = [self._format_prompt(p) for p in prompts]
        encoded = self.tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_seq_length).to(self.device)
        input_width = int(encoded["input_ids"].shape[1])
        kwargs = {"max_new_tokens": max_new_tokens, "use_cache": True,
                  "do_sample": do_sample,
                  "pad_token_id": self.tokenizer.pad_token_id,
                  "eos_token_id": self.tokenizer.eos_token_id}
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
        with self.torch.inference_mode():
            outputs = self.model.generate(**encoded, **kwargs)
        responses = self.tokenizer.batch_decode(
            outputs[:, input_width:], skip_special_tokens=True)
        return responses, {"batch_size": len(prompts)}


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
            if not rec.get("recommendation_behavior") or not rec.get("trajectory_signature"):
                counts["missing_behavior"] += 1
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


def infer_missing(tasks: dict[str, Window], cache: dict[str, dict], args, cache_output: Path) -> dict:
    """Globally batch uncached windows; show tqdm progress and checkpoint each batch."""
    if not tasks:
        print("[INFER] All required windows found in cache; no Gemma calls needed.", flush=True)
        return {"new_windows": 0, "calls": 0, "parse_failures": 0}
    extractor = BatchedGemmaExtractor(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit, dtype=args.dtype,
    )
    keys = list(tasks)
    failures = 0
    calls = 0
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(keys), desc="Gemma inference", unit="window",
              dynamic_ncols=True, mininterval=0.5, leave=True) as pbar:
        for offset in range(0, len(keys), args.llm_batch_size):
            batch_keys = keys[offset:offset + args.llm_batch_size]
            prompts = [build_behavior_extraction_prompt(make_prompt_summary(tasks[k]))
                       for k in batch_keys]
            responses, _ = extractor.generate_batch(
                prompts=prompts, max_new_tokens=args.max_new_tokens,
                temperature=1.0, top_p=0.95, top_k=64, do_sample=False,
            )
            if len(responses) != len(batch_keys):
                raise RuntimeError(
                    f"Gemma returned {len(responses)} responses for {len(batch_keys)} windows"
                )
            calls += 1
            for key, response in zip(batch_keys, responses):
                w = tasks[key]
                try:
                    p = parse_json_response(response)
                    rb = str(p.get("recommendation_behavior") or "").strip()
                    ts = str(p.get("trajectory_signature") or "").strip()
                    if not rb or not ts:
                        raise ValueError("Missing recommendation_behavior / trajectory_signature")
                    sf = p.get("semantic_focus") or []
                    if not isinstance(sf, list):
                        sf = [sf]
                    sf = [str(x).strip() for x in sf if str(x).strip()][:8]
                    m, s, d, adj, ev = enforce_structural_consistency(
                        str(p.get("mechanism") or "unknown").replace("_", " "),
                        str(p.get("scope") or "unknown").replace("_", " "),
                        str(p.get("direction") or "unknown").replace("_", " "),
                        w.items,
                    )
                    confidence = max(0., min(1., float(p.get("confidence") or 0.)))
                    explanation = str(p.get("behavior_explanation") or "").strip()
                    ok, error = True, None
                except Exception as ex:
                    failures += 1
                    cat = list(dict.fromkeys(x.get("item_category", "Unknown") for x in w.items))
                    rb = "Observed preference is concentrated around: " + ", ".join(cat[:4]) + "."
                    ts = "uncertain observed behavior"
                    explanation = f"User purchased {len(w.items)} observed items."
                    sf, m, s, d, adj = cat[:8], "unknown", "unknown", "unknown", []
                    _, _, _, _, ev = enforce_structural_consistency(m, s, d, w.items)
                    confidence = 0.0
                    ok, error = False, repr(ex)
                cache[key] = {
                    "schema_version": SCHEMA_TRAIN,
                    "behavior_prompt_version": PROMPT_VERSION,
                    "user_id": w.user_id, "interaction_sequence": w.items,
                    "generation_config": generation_config(args),
                    "behavior_explanation": explanation,
                    "recommendation_behavior": rb,
                    "semantic_focus": sf, "trajectory_signature": ts,
                    "behavior_signature": ts,
                    "mechanism": m, "scope": s, "direction": d,
                    "confidence": confidence,
                    "consistency_adjustments": adj, "structural_evidence": ev,
                    "pattern_description": rb, "keywords": sf,
                    "parse_ok": ok, "parse_error": error, "raw_response": response,
                }
            # Durable per-batch cache: interruption does not repeat completed batches.
            with cache_output.open("a", encoding="utf-8") as f:
                for key in batch_keys:
                    f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            # Progress advances only AFTER checkpoint has been safely written.
            pbar.update(len(batch_keys))
            pbar.set_postfix(batch=f"{calls}/{(len(keys) + args.llm_batch_size - 1) // args.llm_batch_size}",
                             parse_fail=failures, refresh=False)
    return {"new_windows": len(keys), "calls": calls, "parse_failures": failures}


# -----------------------------------------------------------------------------
# Backend interface consumed by bt4rec.cli.precompute
# -----------------------------------------------------------------------------
from bt4rec.behavior import records as _records  # noqa: E402


def generation_config(args) -> dict:
    return _records.generation_config(args, PROMPT_VERSION)


NAME = "gemma"
OUTPUT_SUBDIR = ""  # precomputed/<dataset>/
LEGACY_TRAIN_NAMES = ("local_memories_gemma_text.jsonl", "local_memories_gemma.jsonl")
DEFAULTS = {"model_name": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit", "max_seq_length": 2048,
            "max_new_tokens": 256, "llm_batch_size": 8, "max_attempts": 1}
