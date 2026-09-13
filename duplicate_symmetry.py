from __future__ import annotations

from kernel_mcir import exhaustive_symmetrized_scores
from common import trained_controlled_problem, save_json


def main():
    _, _, names, K_feats, L, _, _ = trained_controlled_problem(seed=42, explain_n=220)
    # Five features -> 120 permutations, so exact averaging is inexpensive.
    out = exhaustive_symmetrized_scores(K_feats, L)
    gap = abs(float(out.bar_s[0] - out.bar_s[1]))
    payload = {
        "feature_names": names,
        "bar_s": dict(zip(names, out.bar_s.tolist())),
        "duplicate_gap": gap,
        "n_orders": int(out.orders.shape[0]),
    }
    save_json("duplicate_symmetry.json", payload)
    print(payload)
    if gap > 1e-8:
        raise SystemExit("duplicate symmetry check failed")


if __name__ == "__main__":
    main()
