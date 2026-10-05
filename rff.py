from __future__ import annotations

import numpy as np

from .kernels import _median_gamma, center_gram


def rff_rbf_gram(
    x: np.ndarray,
    n_components: int,
    *,
    gamma: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Centered RBF Gram approximation from random Fourier features."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must be one- or two-dimensional")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if gamma is None:
        gamma = _median_gamma(x)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=np.sqrt(2.0 * gamma), size=(x.shape[1], n_components))
    offsets = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    z = np.sqrt(2.0 / n_components) * np.cos(x @ weights + offsets)
    return center_gram(z @ z.T)
