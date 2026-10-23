from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from bt4rec.ranking.parse import sanitize_ranking
from bt4rec.ranking.prompts import build_rank_prompt


@dataclass
class CallStat:
    call_type: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    parse_ok: bool
    complete_ranking: bool


class LocalQwenRanker:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        rank_max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import (
            get_chat_template,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local Unsloth Qwen ranking expects CUDA."
            )

        print(
            f"[INFO] Loading ranking model: "
            f"{model_name}",
            flush=True,
        )
        print(
            "[INFO] Loading ranking model weights/tokenizer; "
            "ranking starts immediately after this step.",
            flush=True,
        )

        from bt4rec.llm.openai_chat import gemma_stop_ids, is_gemma
        self.model_name = model_name
        self.is_gemma = is_gemma(model_name)
        if self.is_gemma:
            # Gemma-3 (e.g. unsloth/gemma-3-4b-it-unsloth-bnb-4bit): same loader/template as
            # the behaviour extractor (bt4rec.behavior.gemma).
            from unsloth import FastModel
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                full_finetuning=False,
            )
            self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma3")
        else:
            self.model, self.tokenizer = (
                FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                )
            )
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="qwen-2.5",
            )
        self.tokenizer.padding_side = (
            "left"
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )
        self.stop_ids = gemma_stop_ids(self.tokenizer) if self.is_gemma else [self.tokenizer.eos_token_id]

        if self.is_gemma:
            try:
                from unsloth import FastModel
                FastModel.for_inference(self.model)
            except Exception:
                pass
        else:
            FastLanguageModel.for_inference(self.model)
        self.model.eval()
        self.device = next(
            self.model.parameters()
        ).device

        print(
            f"[INFO] Ranker loaded on {self.device}. "
            "Starting ranking batches.",
            flush=True,
        )
        self.max_seq_length = int(
            max_seq_length
        )
        self.rank_max_new_tokens = int(
            rank_max_new_tokens
        )
        self.temperature = float(
            temperature
        )
        if not 0 < self.rank_max_new_tokens < self.max_seq_length:
            raise ValueError("rank-max-new-tokens must be positive and smaller than max-seq-length")
        self.calls: List[CallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

    def _render(
        self,
        prompt: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": getattr(
                    self,
                    "system_prompt",
                    "You are a recommendation ranking system. "
                    "Return only the requested candidate-label ranking.",
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        # Tokenization below uses add_special_tokens=False, so Gemma's <bos> must stay
        # in the rendered text (Qwen has no <bos>; legacy behaviour unchanged).
        if text.startswith("<bos>") and not getattr(self, "is_gemma", False):
            text = text[len("<bos>"):]

        return text

    def _generate_batch_raw(
        self,
        prompts: List[str],
        call_type: str,
    ) -> List[str]:
        if not prompts:
            return []

        rendered = [
            self._render(p)
            for p in prompts
        ]

        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length - self.rank_max_new_tokens,
            add_special_tokens=False,
        ).to(self.device)

        input_width = int(
            batch["input_ids"].shape[1]
        )

        kwargs: Dict[str, Any] = {
            "max_new_tokens": (
                self.rank_max_new_tokens
            ),
            "use_cache": True,
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
            "eos_token_id": (
                self.tokenizer.eos_token_id
                if not getattr(self, "is_gemma", False)
                else self.stop_ids
            ),
        }

        if self.temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": (
                    self.temperature
                ),
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                **kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (
            time.perf_counter()
            - t0
        )

        generated = outputs[
            :,
            input_width:,
        ]

        raws = (
            self.tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        # Detailed parse/complete stats are added later.
        per_sample_time = (
            elapsed
            / max(
                1,
                len(prompts),
            )
        )

        # Exact per-sample counts: real prompt tokens (no left padding) and
        # generated tokens up to and including EOS. Used for cost reporting.
        prompt_tokens = [int(x) for x in batch["attention_mask"].sum(dim=1).tolist()]
        stops = [x for x in getattr(self, "stop_ids", [self.tokenizer.eos_token_id]) if x is not None]
        self._last_token_counts = []
        for gen_row, n_in in zip(generated, prompt_tokens):
            toks = [int(t) for t in gen_row.tolist()]
            hits = [i for i, t in enumerate(toks) if t in stops]
            n_out = hits[0] + 1 if hits else len(toks)
            self._last_token_counts.append((n_in, n_out))
            self.calls.append(
                CallStat(
                    call_type=call_type,
                    input_tokens=n_in,
                    output_tokens=n_out,
                    elapsed_sec=float(
                        per_sample_time
                    ),
                    parse_ok=False,
                    complete_ranking=False,
                )
            )

        return [
            str(x).strip()
            for x in raws
        ]

    def set_rank_format(self, rank_format: str = "list") -> None:
        """list (default): C-label list + list repair. score: i2Agent-style scores
        (system prompt + parser/repair of baseline/i2agent.py). The prompt itself must
        already use the matching OUTPUT REQUIREMENTS (bt4rec.ranking.score_format)."""
        if rank_format not in ("list", "score"):
            raise ValueError(rank_format)
        self.rank_format = rank_format
        if rank_format == "score":
            from bt4rec.ranking.score_format import SCORE_SYSTEM_PROMPT
            self.system_prompt = SCORE_SYSTEM_PROMPT
        else:
            self.system_prompt = RANK_SYSTEM_PROMPT

    def _parse_rank(
        self,
        raw: str,
        valid_ids: List[str],
    ) -> Tuple[
        List[str],
        Dict[str, Any],
    ]:
        if getattr(self, "rank_format", "list") == "score":
            from bt4rec.ranking.score_format import parse_score_ranking
            return parse_score_ranking(raw, valid_ids)
        (
            ranking,
            complete,
            returned_n,
            parse_ok,
            parse_reason,
            parsed_order,
        ) = sanitize_ranking(
            raw,
            valid_ids,
        )

        return ranking, {
            "parse_ok": bool(
                parse_ok
            ),
            "parse_complete": bool(
                complete
            ),
            "parse_reason": (
                parse_reason
            ),
            "model_returned_valid_ids": int(
                returned_n
            ),
            "num_candidates": len(
                valid_ids
            ),
            "parsed_candidate_order_1based": (
                parsed_order
            ),
            "parsed_candidate_labels": [
                f"C{x:02d}"
                for x in parsed_order
            ],
            "repair_used": bool(
                parse_ok
                and not complete
            ),
            "fallback_used": bool(
                not parse_ok
            ),
            "reasoning": "",
            "raw_output": raw,
        }

    def rank_jobs(
        self,
        jobs: List[Dict[str, Any]],
        batch_size: int,
        call_type: str,
        max_retries: int,
    ) -> List[
        Tuple[
            List[str],
            Dict[str, Any],
        ]
    ]:
        if not jobs:
            return []

        final: List[
            Optional[
                Tuple[
                    List[str],
                    Dict[str, Any],
                ]
            ]
        ] = [
            None
            for _ in jobs
        ]

        batch_size = max(
            1,
            int(batch_size),
        )

        batch_starts = list(
            range(
                0,
                len(jobs),
                batch_size,
            )
        )

        pbar = tqdm(
            batch_starts,
            desc=call_type,
            unit="batch",
            dynamic_ncols=True,
        )

        for start in pbar:
            end = min(
                start + batch_size,
                len(jobs),
            )
            pbar.set_postfix(
                users=f"{start + 1}-{end}/{len(jobs)}",
                refresh=False,
            )
            chunk = jobs[
                start:end
            ]

            prompts = [
                j["prebuilt_prompt"] if "prebuilt_prompt" in j else build_rank_prompt(
                    history_items=j[
                        "history_items"
                    ],
                    candidate_items=j[
                        "candidate_items"
                    ],
                    latest_behavior=j.get(
                        "latest_behavior"
                    ),
                    predicted_transition=j.get(
                        "predicted_transition"
                    ),
                )
                for j in chunk
            ]

            try:
                raws = (
                    self._generate_batch_raw(
                        prompts,
                        call_type,
                    )
                )
            except Exception as e:
                raise RuntimeError(
                    f"{call_type} generation batch {start}:{end} failed; "
                    "not substituting frozen candidate order for a runtime failure"
                ) from e

            for local_idx, (
                job,
                prompt,
                raw,
            ) in enumerate(
                zip(
                    chunk,
                    prompts,
                    raws,
                )
            ):
                global_idx = (
                    start
                    + local_idx
                )
                valid_ids = [
                    str(
                        x["item_id"]
                    )
                    for x in job[
                        "candidate_items"
                    ]
                ]

                ranking, meta = (
                    self._parse_rank(
                        raw,
                        valid_ids,
                    )
                )

                # No individual retry: keep the first batched generation result.
                # Duplicate Cxx labels keep their first occurrence; missing labels
                # are appended in frozen candidate order. Only a true Cxx parse
                # failure falls back to the full frozen candidate order.
                attempts = 1
                meta["attempts"] = attempts
                counts = getattr(self, "_last_token_counts", None)
                if counts and local_idx < len(counts):
                    meta["input_tokens"], meta["output_tokens"] = counts[local_idx]
                meta["call_type"] = (
                    call_type
                )
                final[
                    global_idx
                ] = (
                    ranking,
                    meta,
                )

        pbar.close()

        if any(
            x is None
            for x in final
        ):
            raise RuntimeError(
                "Internal ranking result missing"
            )

        return [
            x
            for x in final
            if x is not None
        ]

    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {
                "model": None,
                "generation_records": 0,
            }

        by_type = Counter(
            x.call_type
            for x in self.calls
        )

        return {
            "model": str(
                getattr(self, "model_name", None)
                or getattr(
                    self.model,
                    "name_or_path",
                    "local_qwen",
                )
            ),
            "backend": getattr(self, "backend", "local"),
            "generation_records": len(
                self.calls
            ),
            "token_count_method": "exact per sample (attention mask; output up to EOS)",
            "input_tokens_approx": int(
                sum(
                    x.input_tokens
                    for x in self.calls
                )
            ),
            "output_tokens_approx": int(
                sum(
                    x.output_tokens
                    for x in self.calls
                )
            ),
            "elapsed_sec": float(
                sum(
                    x.elapsed_sec
                    for x in self.calls
                )
            ),
            "call_type_counts": dict(
                by_type
            ),
        }


RANK_SYSTEM_PROMPT = (
    "You are a recommendation ranking system. "
    "Return only the requested candidate-label ranking."
)


class OpenAIRanker(LocalQwenRanker):
    """Same prompts, system prompt, parser and repair policy as LocalQwenRanker, but the
    generation is done by the OpenAI chat API (e.g. gpt-4o-mini). Token counts = API usage."""

    def __init__(self, model_name: str = "gpt-4o-mini", rank_max_new_tokens: int = 1024,
                 temperature: float = 0.0, seed: int = 42, *, workers: int = 8,
                 api_base: Optional[str] = None, cache_path: Optional[str] = None) -> None:
        from bt4rec.llm.openai_chat import OpenAIChat
        self.backend = "openai"
        self.model_name = model_name
        self.is_gemma = False
        self.rank_max_new_tokens = int(rank_max_new_tokens)
        self.temperature = float(temperature)
        self.calls: List[CallStat] = []
        self.client = OpenAIChat(model_name, temperature=temperature, seed=seed, workers=workers,
                                 api_base=api_base, cache_path=cache_path)

    def _generate_batch_raw(self, prompts: List[str], call_type: str) -> List[str]:
        if not prompts:
            return []
        res = self.client.generate(prompts, getattr(self, "system_prompt", RANK_SYSTEM_PROMPT),
                                   self.rank_max_new_tokens)
        self._last_token_counts = []
        for r in res:
            n_in, n_out = int(r["input_tokens"]), int(r["output_tokens"])
            self._last_token_counts.append((n_in, n_out))
            self.calls.append(CallStat(call_type=call_type, input_tokens=n_in, output_tokens=n_out,
                                       elapsed_sec=float(r.get("elapsed_sec", 0.0)),
                                       parse_ok=False, complete_ranking=False))
        return [str(r["text"]).strip() for r in res]

    def stats(self) -> Dict[str, Any]:
        out = super().stats() if self.calls else {"model": self.model_name, "generation_records": 0}
        out["token_count_method"] = "OpenAI API usage (prompt_tokens / completion_tokens)"
        out.update(self.client.stats())
        return out


def build_ranker(a, cache_dir=None):
    """--ranker local (Qwen or Gemma, inferred from --model) | openai."""
    if getattr(a, "ranker", "local") == "openai":
        cache = getattr(a, "api_cache", None)
        if not cache and cache_dir is not None:
            from pathlib import Path
            cache = str(Path(cache_dir) / "openai_cache.jsonl")
        return OpenAIRanker(a.model, a.rank_max_new_tokens, a.temperature, a.seed,
                            workers=getattr(a, "api_workers", 8), api_base=getattr(a, "api_base", None),
                            cache_path=cache)
    return LocalQwenRanker(a.model, a.max_seq_length, a.load_in_4bit,
                           a.rank_max_new_tokens, a.temperature, a.seed)
