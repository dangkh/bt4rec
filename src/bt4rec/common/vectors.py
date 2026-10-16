"""Vector helpers shared by the Tree builder and inference."""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left unchanged. Accepts 1-D or 2-D."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / n if n > eps else arr
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(n > eps, n, 1.0)
