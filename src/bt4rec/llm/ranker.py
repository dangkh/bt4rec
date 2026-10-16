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
            "[INFO] Loading Qwen model weights/tokenizer; "
            "ranking starts immediately after this step.",
            flush=True,
        )

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

        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        self.device = next(
            self.model.parameters()
        ).device

        print(
            f"[INFO] Qwen loaded on {self.device}. "
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
                "content": (
                    "You are a recommendation ranking system. "
                    "Return only the requested candidate-label ranking."
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

        if text.startswith("<bos>"):
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

        for gen_row in generated:
            self.calls.append(
                CallStat(
                    call_type=call_type,
                    input_tokens=input_width,
                    output_tokens=int(
                        gen_row.shape[0]
                    ),
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

    def _parse_rank(
        self,
        raw: str,
        valid_ids: List[str],
    ) -> Tuple[
        List[str],
        Dict[str, Any],
    ]:
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
                getattr(
                    self.model,
                    "name_or_path",
                    "local_qwen",
                )
            ),
            "generation_records": len(
                self.calls
            ),
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
