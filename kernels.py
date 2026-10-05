from __future__ import annotations

import numpy as np


def center_gram(K: np.ndarray) -> np.ndarray:
    """Double-center a square Gram matrix and symmetrize roundoff."""
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    centered = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    return 0.5 * (centered + centered.T)


def linear_gram(x: np.ndarray) -> np.ndarray:
    """Centered linear Gram matrix for one- or multi-column observations."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must be one- or two-dimensional")
    return center_gram(x @ x.T)


def _median_gamma(x: np.ndarray) -> float:
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)
    positive = d2[np.triu_indices_from(d2, k=1)]
    positive = positive[positive > 0]
    if positive.size == 0:
        return 1.0
    return 1.0 / (2.0 * float(np.median(positive)))


def rbf_gram(x: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Centered RBF Gram matrix using a deterministic median heuristic by default."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must be one- or two-dimensional")
    if gamma is None:
        gamma = _median_gamma(x)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)
    return center_gram(np.exp(-float(gamma) * d2))
