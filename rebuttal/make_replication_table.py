"""LaTeX table for the representation-perturbation benchmark (local_runs/replication_invariance.json).
python make_replication_table.py -> local_runs/replication_table.tex"""
import json, os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.DataFrame(json.load(open(os.path.join(HERE, "local_runs", "replication_invariance.json"))))
lab = {("copies", 0.0): "exact copies", ("noisy", 0.05): r"noisy, $\sigma{=}0.05$", ("noisy", 0.1): r"noisy, $\sigma{=}0.1$",
       ("noisy", 0.2): r"noisy, $\sigma{=}0.2$", ("irrelevant", 0.0): "irrelevant", ("rescale", 3.0): r"rescale $\times3$"}


def f(v):
    if v < 1e-6:
        return r"$<10^{%d}$" % (int(np.floor(np.log10(v))) + 1)
    return f"${v:.3f}$"


T = [r"""\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{\rev{Representation-perturbation benchmark (means over %d seeds, $n=400$, $B=1500$). AD: attribution drift of $\bar s$ over the unaffected features $X_1,\dots,X_5$; CE: group conservation error of $\bar U$ for $X_0$ and its copies; dil.: $\bar U_0^{\mathrm{post}}/\bar U_0^{\mathrm{pre}}$ (theory under block orders $1/(k+1)$, in expectation). Uniform and block orders are sampled independently before and after the perturbation, so rescaling and irrelevant features show the Monte Carlo floor; coupled block orders derive each post-perturbation order from a pre-perturbation order (Proposition~\ref{prop:replication}), so for exact copies AD and CE vanish to floating-point precision. Noisy copies are not replicas: with $\sigma\le0.1$ they are still grouped (CKA $\ge0.95$) and the results measure empirical stability; with $\sigma=0.2$ they are not grouped by the CKA rule (the coupled column groups them by construction).}}\label{tab:replication}
\begin{tabular}{lc|ccc|ccc|cc}\toprule
 & & \multicolumn{3}{c|}{uniform (paper)} & \multicolumn{3}{c|}{block} & \multicolumn{2}{c}{coupled block} \\
Perturbation & $k$ & AD & CE & dil. & AD & CE & dil. & AD & CE \\\midrule""" % df.seed.nunique()]
for (kind, k, sig), g in df.groupby(["kind", "k", "sigma"], sort=False):
    c = []
    for mode in ("uniform", "block"):
        h = g[g["mode"] == mode]
        c += [f(h.AD_s.mean()), f(h.CE.mean()), f"${h.dil.mean():.3f}$"]
    h = g[g["mode"] == "coupled"]
    c += [f(h.AD_s.mean()), f(h.CE.mean())] if len(h) else ["--", "--"]
    T.append(f"{lab[(kind, sig)]} & {k if kind != 'rescale' else '--'} & " + " & ".join(c) + r" \\")
T.append(r"\bottomrule\end{tabular}\end{table}")
open(os.path.join(HERE, "local_runs", "replication_table.tex"), "w").write("\n".join(T) + "\n")
print("\n".join(T))
