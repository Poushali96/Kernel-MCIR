import numpy as np

from kernel_mcir import (
    context_score,
    exhaustive_symmetrized_scores,
    fixed_order_scores,
    linear_gram,
)


def _rank1(v):
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    return np.outer(v, v)


def test_zero_context_extended_score_is_zero():
    z = np.zeros(4)
    K = linear_gram(z)
    L = linear_gram(z)
    cs = context_score([K], L, 0, [])
    assert cs.score is None
    assert cs.extended_score == 0.0
    assert not cs.informative


def test_boundedness_and_signed_total_identity():
    x = np.array([-2, -1, 0, 1, 2.0])
    z = np.array([1, -1, 1, -1, 0.0])
    K0 = linear_gram(x)
    K1 = linear_gram(z)
    L = linear_gram(x + 0.2 * z)
    for cs in fixed_order_scores([K0, K1], L, order=[0, 1]):
        if cs.informative:
            assert 0.0 <= cs.extended_score <= 1.0
        assert cs.signed_total_error < 1e-8


def test_context_dependence_constructive_example():
    A = _rank1([1, -1, 1, -1])
    B = _rank1([1, 1, -1, -1])
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    Ki = A + B
    L = A + B
    K = [Ki, A]
    s0 = context_score(K, L, 0, []).extended_score
    sA = context_score(K, L, 0, [1]).extended_score
    assert abs(s0 - 1.0) < 1e-8
    assert abs(sA - 0.5) < 1e-8


def test_exact_duplicate_symmetry_under_exhaustive_permutations():
    x = np.array([-2, -1, 0, 1, 2.0])
    dup = x.copy()
    z = np.array([1, 0, -1, 0, 1.0])
    K = [linear_gram(x), linear_gram(dup), linear_gram(z)]
    L = linear_gram(1.3 * x + 0.4 * z)
    out = exhaustive_symmetrized_scores(K, L)
    assert abs(out.bar_s[0] - out.bar_s[1]) < 1e-10
