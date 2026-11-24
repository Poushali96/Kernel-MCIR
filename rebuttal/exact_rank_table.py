"""Exact-permutation version of the synthetic duplicate rank table (tab:q2ranks): d = 5 (X4 = exact copy of X0), so all
5! = 120 orders are enumerated and there is no Monte Carlo asymmetry between an original and its copy.
Same data as kmcir_extra_analyses.synthetic (seeds 0-4, n = 300). Scores are rounded to 1e-10 before ranking so exact
ties receive average tied ranks; |score(X0) - score(X4)| is reported.
python exact_rank_table.py -> local_runs/exact_rank_table.json and the LaTeX table on stdout"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kmcir_extra_analyses import all_scores  # noqa: E402

NAMES = [("s", r"$\bar s$"), ("seqRatio", r"(a) seq.\ HSIC gain / HSIC"), ("Eshare", r"(b) energy share"),
         ("U", r"$\bar U$"), ("pdCov", "partial dCov")]
rows = []
for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    X = rng.normal(size=(300, 4))
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2]
    Xa = np.column_stack([X, X[:, 0]])
    sc = all_scores(Xa, y, [np.array(p) for p in itertools.permutations(range(5))])
    row = {"seed": seed}
    for k, _ in NAMES:
        raw = np.nan_to_num(sc[k], nan=-1e9); scale = np.max(np.abs(raw[raw > -1e8]))
        v = np.round(raw / scale, 10)                                  # relative rounding keeps exact ties tied
        row["rank_" + k] = rankdata(-v).tolist(); row["gap_" + k] = float(abs(sc[k][0] - sc[k][4]) / scale)
    rows.append(row)
json.dump(rows, open(os.path.join(HERE, "local_runs", "exact_rank_table.json"), "w"), indent=1)
L = [r"\begin{table}[htbp]\centering\scriptsize",
     r"\caption{\rev{Mean rank per feature (1 = highest; ties receive the average rank) on the synthetic duplicate test, averaging over all $5!=120$ orders (5 seeds, $n=300$). The last column is the largest relative score difference $|\mathrm{score}_0-\mathrm{score}_4|/\max_j|\mathrm{score}_j|$ between $X_0$ and its exact copy $X_4$ over seeds (floating-point level).}}\label{tab:q2ranks}",
     r"\begin{tabular}{lcccccc}\toprule",
     r"Score & $X_0$ & $X_1$ & $X_2$ & $X_3$ (noise) & $X_4$ (copy of $X_0$) & rel.\ gap \\\midrule"]
for k, lab in NAMES:
    R = np.array([r["rank_" + k] for r in rows]).mean(0); g = max(r["gap_" + k] for r in rows)
    e = int(np.floor(np.log10(g))) if g > 0 else None
    gs = "$0$" if e is None else f"$<10^{{{e + 1}}}$"
    L.append(f"{lab} & " + " & ".join(f"{x:.1f}" for x in R) + f" & {gs} \\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
print("\n".join(L))
