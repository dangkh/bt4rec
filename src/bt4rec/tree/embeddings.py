from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from bt4rec.common.vectors import l2_normalize


def encode_qwen(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Need sentence-transformers for --encoder qwen. "
            "Install: pip install -U sentence-transformers transformers accelerate"
        ) from e

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"[INFO] embedding model={model_name} device={device}"
    )

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(emb, dtype=np.float32)


def encode_tfidf(
    texts: List[str],
    max_features: int,
) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        raise RuntimeError(
            "Need scikit-learn for --encoder tfidf"
        ) from e

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
    )

    x = vec.fit_transform(texts).toarray().astype(np.float32)
    return l2_normalize(x)


def create_embeddings(
    args: argparse.Namespace,
    memories: List[Dict[str, Any]],
) -> np.ndarray:
    if args.embeddings_npy:
        x = np.load(args.embeddings_npy)
        if len(x) != len(memories):
            raise ValueError(
                f"--embeddings-npy has {len(x)} rows but "
                f"filtered input has {len(memories)} memories"
            )
        return l2_normalize(
            np.asarray(x, dtype=np.float32)
        )

    texts = [m["cluster_text"] for m in memories]

    if args.encoder == "qwen":
        return encode_qwen(
            texts,
            args.embedding_model,
            args.batch_size,
            args.device,
        )

    if args.encoder == "tfidf":
        return encode_tfidf(
            texts,
            args.tfidf_max_features,
        )

    raise ValueError(args.encoder)
