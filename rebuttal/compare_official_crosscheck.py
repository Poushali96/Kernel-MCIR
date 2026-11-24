"""Compare the sweep's oracle DFI/DIP/LOCO (local_runs/dependence_sweep.json) with the official fdfi/dipd estimates
(local_runs/official_crosscheck.json). Writes local_runs/official_crosscheck_table.tex and prints a summary.
python compare_official_crosscheck.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
orc = {(r["regime"], r["rho"]): r["oracle"] for r in json.load(open(os.path.join(HERE, "local_runs", "dependence_sweep.json")))}
off = json.load(open(os.path.join(HERE, "local_runs", "official_crosscheck.json")))
NAME = {"duplicate": "duplicate", "proxy": "proxy", "complementary": "suppressor", "additive": "distinct roles",
        "interaction": "interaction"}
L = [r"""\begin{table}[htbp]\centering\scriptsize
\caption{\rev{Cross-check of the population references in Table~\ref{tab:dep_sweep} against the authors' implementations, for $X_2$: DFI from \texttt{fdfi} 0.1.0 (Gaussian optimal-transport explainer) and the DIP decomposition from \texttt{dipd} 0.0.3 (\texttt{get\_loo}, EBM learner), %d seeds, $n=%d$ (DIP: 70/30 train/test split; DFI: $2\times10^4$ background points). Entries: ours / official. $^\ddagger$At $\rho=1$ the covariance is singular and DFI is undefined; our value is the limit $\rho\to1$ of Proposition~\ref{prop:nonequiv}(a), while \texttt{fdfi} regularises the whitening map.}}\label{tab:official}
\begin{tabular}{lc|cccc}\toprule
Regime & $\rho$ & DFI$_2$ & DIP stand.$_2$ & DIP dep.$_2$ & LOCO$_2$ (DIP) \\\midrule""" % (off[0]["seeds"], off[0]["n"])]
diffs = {"dfi": [], "stand": [], "dep": [], "loco": []}
for r in off:
    o = orc[(r["regime"], r["rho"])]
    pairs = [("dfi", o["dfi"][1], r["dfi"][1]), ("stand", o["v_single"][1], r["dip_standalone"][1]),
             ("dep", o["dip_dependence"][1], r["dip_dependence"][1]), ("loco", o["loco"][1], r["dip_loco"][1])]
    cells = []
    for k, a, b in pairs:
        if not (k == "dfi" and r["regime"] == "duplicate"):
            diffs[k].append(abs(a - b))
        cells.append(f"{a + 0.0:.2f} / {b + 0.0:.2f}".replace("-0.00", "0.00") + ("$^\\ddagger$" if k == "dfi" and r["regime"] == "duplicate" else ""))
    L.append(f"{NAME[r['regime']]} & {r['rho']:.2f} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule\end{tabular}\end{table}")
summ = {k: (float(np.max(v)), float(np.mean(v))) for k, v in diffs.items()}
L.append("% max / mean absolute difference: " + ", ".join(f"{k} {m:.3f}/{a:.3f}" for k, (m, a) in summ.items()))
open(os.path.join(HERE, "local_runs", "official_crosscheck_table.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
