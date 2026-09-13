from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kernel_mcir import symmetrized_scores
from common import trained_controlled_problem, RESULTS, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-orders", type=int, default=1000)
    ap.add_argument("--explain-n", type=int, default=300)
    args = ap.parse_args()

    _, _, names, K_feats, L, _, _ = trained_controlled_problem(
        seed=args.seed, explain_n=args.explain_n
    )
    rng = np.random.default_rng(args.seed + 123)
    ref_orders = np.vstack([rng.permutation(len(names)) for _ in range(args.reference_orders)])
    ref = symmetrized_scores(K_feats, L, orders=ref_orders)

    B_grid = [1, 5, 10, 25, 50, 100, 200, 500, args.reference_orders]
    B_grid = sorted(set(b for b in B_grid if b <= args.reference_orders))
    rows = []
    for B in B_grid:
        cur = symmetrized_scores(K_feats, L, orders=ref_orders[:B])
        rho = float(spearmanr(cur.bar_s, ref.bar_s).statistic)
        maxerr = float(np.max(np.abs(cur.bar_s - ref.bar_s)))
        rows.append({"orders_B": B, "spearman_vs_reference": rho, "max_abs_score_error": maxerr})

    df = pd.DataFrame(rows)
    out_csv = RESULTS / "controlled_order_convergence.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "feature_names": names,
        "reference_orders": args.reference_orders,
        "bar_s": dict(zip(names, ref.bar_s.tolist())),
        "bar_U": dict(zip(names, ref.bar_U.tolist())),
        "bar_R": dict(zip(names, ref.bar_R.tolist())),
        "bar_T": dict(zip(names, ref.bar_T.tolist())),
        "activity_score": dict(zip(names, ref.activity_score.tolist())),
        "informative_fraction": dict(zip(names, ref.informative_fraction.tolist())),
        "exact_duplicate_gap": float(abs(ref.bar_s[0] - ref.bar_s[1])),
    }
    out_json = save_json("controlled_order_summary.json", summary)
    print(df.to_string(index=False))
    print("\nSymmetrized scores:")
    for n, s, a, t in zip(names, ref.bar_s, ref.activity_score, ref.bar_T):
        print(f"  {n:10s}  novelty={s:.4f}  activity={a:.4f}  total={t:.6g}")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
