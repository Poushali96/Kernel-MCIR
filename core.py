from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np


DEFAULT_RANK_TOL = 1e-10
DEFAULT_STRENGTH_TOL = 1e-12


def _as_vectors(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.empty((0, 0), dtype=float)
    shape = np.asarray(matrices[0]).shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("kernel matrices must be square")
    vectors = []
    for matrix in matrices:
        array = np.asarray(matrix, dtype=float)
        if array.shape != shape:
            raise ValueError("all kernel matrices must have the same shape")
        vectors.append(array.reshape(-1))
    return np.column_stack(vectors)


class _ProjectionCache:
    """Rank-revealing orthogonal projector for vectorized Gram matrices."""

    def __init__(self, matrices: Sequence[np.ndarray], rank_tol: float = DEFAULT_RANK_TOL):
        self.rank_tol = float(rank_tol)
        if self.rank_tol < 0:
            raise ValueError("rank_tol must be non-negative")
        vectors = _as_vectors(matrices)
        if vectors.size == 0:
            self.basis = np.empty((0, 0), dtype=float)
            return
        U, singular_values, _ = np.linalg.svd(vectors, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] == 0:
            rank = 0
        else:
            rank = int(np.sum(singular_values > self.rank_tol * singular_values[0]))
        self.basis = U[:, :rank]

    def project_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if self.basis.shape[0] == 0:
            return np.zeros_like(vector)
        if vector.size != self.basis.shape[0]:
            raise ValueError("projection target has incompatible shape")
        return self.basis @ (self.basis.T @ vector)

    def project_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        return self.project_vector(matrix).reshape(matrix.shape)


@dataclass(frozen=True)
class ContextScore:
    feature: int
    predecessors: tuple[int, ...]
    signed_unique: float
    signed_redundant: float
    unique: float
    redundant: float
    total: float
    score: float | None
    extended_score: float
    informative: bool
    signed_total_error: float


@dataclass(frozen=True)
class SymmetrizedScores:
    bar_U: np.ndarray
    bar_R: np.ndarray
    bar_T: np.ndarray
    bar_s: np.ndarray
    activity_score: np.ndarray
    informative_fraction: np.ndarray
    orders: np.ndarray


def context_score(
    feature_kernels: Sequence[np.ndarray],
    output_kernel: np.ndarray,
    feature: int,
    predecessors: Iterable[int],
    *,
    rank_tol: float = DEFAULT_RANK_TOL,
    strength_tol: float = DEFAULT_STRENGTH_TOL,
) -> ContextScore:
    """Compute the signed decomposition and magnitude score in one context."""
    kernels = [np.asarray(K, dtype=float) for K in feature_kernels]
    output = np.asarray(output_kernel, dtype=float)
    if not kernels:
        raise ValueError("at least one feature kernel is required")
    if not 0 <= feature < len(kernels):
        raise IndexError("feature index out of range")
    if any(K.shape != output.shape for K in kernels):
        raise ValueError("feature and output kernels must have matching shapes")
    pred = tuple(int(j) for j in predecessors)
    if feature in pred or any(j < 0 or j >= len(kernels) for j in pred):
        raise ValueError("predecessors must be valid feature indices excluding the candidate")

    current = _ProjectionCache([kernels[j] for j in pred], rank_tol=rank_tol)
    expanded = _ProjectionCache([kernels[j] for j in pred] + [kernels[feature]], rank_tol=rank_tol)
    L_phi = current.project_matrix(output)
    L_expanded = expanded.project_matrix(output)
    delta = L_expanded - L_phi
    candidate_projection = current.project_matrix(kernels[feature])

    signed_unique = float(np.vdot(kernels[feature], delta).real)
    signed_redundant = float(np.vdot(candidate_projection, L_phi).real)
    unique = abs(signed_unique)
    redundant = abs(signed_redundant)
    total = unique + redundant
    informative = bool(total > strength_tol)
    score = unique / total if informative else None
    marginal = float(np.vdot(kernels[feature], output).real)
    signed_total_error = abs((signed_unique + signed_redundant) - marginal)
    return ContextScore(
        feature=feature,
        predecessors=pred,
        signed_unique=signed_unique,
        signed_redundant=signed_redundant,
        unique=unique,
        redundant=redundant,
        total=total,
        score=score,
        extended_score=float(score) if score is not None else 0.0,
        informative=informative,
        signed_total_error=signed_total_error,
    )


def fixed_order_scores(
    feature_kernels: Sequence[np.ndarray],
    output_kernel: np.ndarray,
    order: Sequence[int],
    **kwargs,
) -> list[ContextScore]:
    """Return scores in feature-index order for one sequential ordering."""
    d = len(feature_kernels)
    order_array = np.asarray(order, dtype=int)
    if order_array.shape != (d,) or set(order_array.tolist()) != set(range(d)):
        raise ValueError("order must be a permutation of all feature indices")
    out: list[ContextScore | None] = [None] * d
    predecessors: list[int] = []
    for feature in order_array:
        out[int(feature)] = context_score(
            feature_kernels, output_kernel, int(feature), predecessors, **kwargs
        )
        predecessors.append(int(feature))
    return [score for score in out if score is not None]


def symmetrized_scores(
    feature_kernels: Sequence[np.ndarray],
    output_kernel: np.ndarray,
    *,
    n_orders: int | None = None,
    seed: int = 0,
    orders: np.ndarray | None = None,
    **kwargs,
) -> SymmetrizedScores:
    """Aggregate unique and redundant strengths over feature permutations."""
    d = len(feature_kernels)
    if orders is None:
        if n_orders is None or n_orders <= 0:
            raise ValueError("provide positive n_orders when orders is omitted")
        rng = np.random.default_rng(seed)
        order_array = np.vstack([rng.permutation(d) for _ in range(n_orders)])
    else:
        order_array = np.asarray(orders, dtype=int)
        if order_array.ndim == 1:
            order_array = order_array[None, :]
    if order_array.ndim != 2 or order_array.shape[1] != d or order_array.shape[0] == 0:
        raise ValueError("orders must have shape (B, number_of_features)")
    expected = set(range(d))
    if any(set(row.tolist()) != expected for row in order_array):
        raise ValueError("every row of orders must be a feature permutation")

    unique = np.zeros((len(order_array), d), dtype=float)
    redundant = np.zeros_like(unique)
    informative = np.zeros_like(unique, dtype=float)
    for b, order in enumerate(order_array):
        scores = fixed_order_scores(feature_kernels, output_kernel, order, **kwargs)
        unique[b] = [score.unique for score in scores]
        redundant[b] = [score.redundant for score in scores]
        informative[b] = [float(score.informative) for score in scores]
    bar_U = unique.mean(axis=0)
    bar_R = redundant.mean(axis=0)
    bar_T = bar_U + bar_R
    bar_s = np.divide(bar_U, bar_T, out=np.zeros_like(bar_U), where=bar_T > 0)
    maximum = float(np.max(bar_T)) if bar_T.size else 0.0
    activity = bar_s * (bar_T / maximum) if maximum > 0 else np.zeros_like(bar_s)
    return SymmetrizedScores(
        bar_U=bar_U,
        bar_R=bar_R,
        bar_T=bar_T,
        bar_s=bar_s,
        activity_score=activity,
        informative_fraction=informative.mean(axis=0),
        orders=order_array.copy(),
    )


def exhaustive_symmetrized_scores(
    feature_kernels: Sequence[np.ndarray], output_kernel: np.ndarray, **kwargs
) -> SymmetrizedScores:
    """Exactly average over all feature orders; intended for small d."""
    d = len(feature_kernels)
    orders = np.asarray(list(permutations(range(d))), dtype=int)
    return symmetrized_scores(feature_kernels, output_kernel, orders=orders, **kwargs)
