from __future__ import annotations

import gc
from typing import List

import numpy as np


class QwenEncoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
    ) -> None:
        import torch
        import unsloth  # noqa: F401
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.model_name = str(model_name)
        self.device = str(device)
        self.batch_size = int(batch_size)

        print(
            f"[INFO] Loading state encoder: "
            f"{self.model_name} on {self.device}"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, 1),
                dtype=np.float32,
            )
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(
            emb,
            dtype=np.float32,
        )

    def release(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
