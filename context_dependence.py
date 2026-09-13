from __future__ import annotations

import numpy as np

from kernel_mcir import context_score
from common import save_json


def centered_rank1(v):
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    v = v / np.linalg.norm(v)
    return np.outer(v, v)


def main():
    # Construct centered orthogonal rank-one PSD directions A and B.
    a = np.array([1.0, -1.0, 1.0, -1.0])
    b = np.array([1.0, 1.0, -1.0, -1.0])
    A = centered_rank1(a)
    B = centered_rank1(b)
    K_i = A + B
    L = A + B
    K_feats = [K_i, A]

    empty = context_score(K_feats, L, feature=0, predecessors=[])
    with_A = context_score(K_feats, L, feature=0, predecessors=[1])

    payload = {
        "s_empty": empty.extended_score,
        "s_condition_on_A": with_A.extended_score,
        "U_empty": empty.unique,
        "R_empty": empty.redundant,
        "U_condition_on_A": with_A.unique,
        "R_condition_on_A": with_A.redundant,
        "signed_total_error_empty": empty.signed_total_error,
        "signed_total_error_condition_on_A": with_A.signed_total_error,
    }
    save_json("context_dependence.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
