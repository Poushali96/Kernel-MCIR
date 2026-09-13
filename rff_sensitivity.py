from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kernel_mcir import rbf_gram, rff_rbf_gram, symmetrized_scores
from common import controlled_dataset, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--orders", type=int, default=100)
    args = ap.parse_args()

    X, y, names = controlled_dataset(seed=args.seed, n=max(args.n, 400))
    X = X[:args.n]
    y_cont = 1.5 * X[:, 0] + 1.1 * np.sin(1.8 * X[:, 2]) + 0.2 * X[:, 3]

    exact_K = [rbf_gram(X[:, j]) for j in range(X.shape[1])]
    exact_L = rbf_gram(y_cont)
    rng = np.random.default_rng(args.seed + 55)
    orders = np.vstack([rng.permutation(X.shape[1]) for _ in range(args.orders)])
    exact = symmetrized_scores(exact_K, exact_L, orders=orders)

    rows = []
    for D in [64, 128, 256, 512, 1024]:
        t0 = time.perf_counter()
        Kd = [rff_rbf_gram(X[:, j], D, seed=args.seed + 1000 * j + D) for j in range(X.shape[1])]
        Ld = rff_rbf_gram(y_cont, D, seed=args.seed + 9999 + D)
        approx = symmetrized_scores(Kd, Ld, orders=orders)
        elapsed = time.perf_counter() - t0
        rows.append({
            "D": D,
            "time_seconds": elapsed,
            "spearman_vs_exact": float(spearmanr(approx.bar_s, exact.bar_s).statistic),
            "max_abs_score_error": float(np.max(np.abs(approx.bar_s - exact.bar_s))),
            "mean_score": float(np.mean(approx.bar_s)),
            "std_score": float(np.std(approx.bar_s)),
        })
    df = pd.DataFrame(rows)
    out = RESULTS / "rff_sensitivity.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
