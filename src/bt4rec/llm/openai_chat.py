"""Thin, cached, parallel OpenAI chat-completions client (used for the LLM ablation).

* API key: read ONLY from the environment (OPENAI_API_KEY); never stored in files.
* Optional base URL: --api-base or OPENAI_BASE_URL (Azure/proxy/compatible servers).
* Deterministic settings: temperature (default 0), fixed seed, max_tokens.
* Response cache (jsonl, keyed by sha256(model|system|prompt|max_tokens|temperature|seed)):
  re-running a crashed/finished run never pays twice for the same prompt.
* Token counts come from the API ``usage`` field (prompt_tokens / completion_tokens);
  cached responses keep the counts recorded when they were first generated.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence


class OpenAIChat:
    def __init__(self, model: str = "gpt-4o-mini", *, temperature: float = 0.0, seed: int = 42,
                 workers: int = 8, api_base: Optional[str] = None, cache_path: Optional[str] = None,
                 max_retries: int = 8, timeout: float = 120.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install openai  (needed for --ranker openai / --llm openai)") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (export OPENAI_API_KEY=... before running)")
        base = api_base or os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(base_url=base, timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.workers = max(1, int(workers))
        self.max_retries = int(max_retries)
        self.cache_path = Path(cache_path) if cache_path else None
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = {}
        self.api_calls = 0
        self.cache_hits = 0
        self.retries = 0
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            self._cache[r["key"]] = r
                        except Exception:
                            pass
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] OpenAI model={model} base={base or 'default'} workers={self.workers} "
              f"cache={self.cache_path} ({len(self._cache)} cached)", flush=True)

    def _key(self, system: str, prompt: str, max_tokens: int) -> str:
        s = json.dumps([self.model, system, prompt, int(max_tokens), self.temperature, self.seed],
                       ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _one(self, system: str, prompt: str, max_tokens: int) -> dict:
        key = self._key(system, prompt, max_tokens)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            with self._lock:
                self.cache_hits += 1
            return {**hit, "cached": True, "elapsed_sec": 0.0}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    max_tokens=int(max_tokens), seed=self.seed)
                elapsed = time.perf_counter() - t0
                u = resp.usage
                rec = {"key": key, "model": self.model, "text": resp.choices[0].message.content or "",
                       "finish_reason": resp.choices[0].finish_reason,
                       "input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                       "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                       "system_fingerprint": getattr(resp, "system_fingerprint", None)}
                with self._lock:
                    self.api_calls += 1
                    self._cache[key] = rec
                    if self.cache_path:
                        with self.cache_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return {**rec, "cached": False, "elapsed_sec": elapsed}
            except Exception as e:
                name = type(e).__name__
                fatal = name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                                 "BadRequestError")
                if fatal or attempt >= self.max_retries:
                    raise
                with self._lock:
                    self.retries += 1
                print(f"[WARN] OpenAI {name}: {str(e)[:160]} -> retry in {delay:.0f}s", flush=True)
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def generate(self, prompts: Sequence[str], system: str, max_tokens: int) -> List[dict]:
        """Returns one dict per prompt (same order): text, input_tokens, output_tokens, ..."""
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(prompts))) as ex:
            return list(ex.map(lambda p: self._one(system, p, max_tokens), prompts))

    def stats(self) -> dict:
        return {"api_model": self.model, "api_calls_paid": self.api_calls,
                "cache_hits": self.cache_hits, "retries": self.retries,
                "cache_path": str(self.cache_path) if self.cache_path else None}


def is_gemma(model_name: str) -> bool:
    return "gemma" in str(model_name).lower()


def gemma_stop_ids(tokenizer) -> List[int]:
    """Gemma-3 instruct ends a turn with <end_of_turn>; stop on it as well as on <eos>."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    try:
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0 and eot != getattr(tokenizer, "unk_token_id", None):
            ids.append(int(eot))
    except Exception:
        pass
    return sorted(set(ids))
