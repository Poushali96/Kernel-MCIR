"""Build the LaTeX for the revised appendix duplicate-dilution section from rebuttal/local_runs/*.json.
Prints the section; `python make_appendix_tables.py > /tmp/x.tex`. No numbers are typed by hand."""
import json
import os

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_runs")
J = lambda f: pd.DataFrame(json.load(open(os.path.join(R, f))))
ms = lambda v: f"${np.mean(v):.3f}\\pm{np.std(v):.3f}$"

main, fulln = J("dil_main.json"), J("dil_fulln.json")
sage = pd.concat([J("dil_sage.json"), J("dil_sage_s3_9.json")], ignore_index=True)  # seeds 0-2 and 3-9
df200, lr500 = J("dil_df200.json"), J("dil_lr500.json")
extra = J("extra_20ng.json")
syn = json.load(open(os.path.join(R, "extra_synthetic.json")))
bconv, rsym = J("b_convergence.json"), J("rff_copy_symmetry.json")

out = []
P = out.append
P(r"""\subsection{Duplicate dilution on 20 Newsgroups: protocol and full results}\label{app:fulln}
\rev{This subsection replaces the within-run duplicate/original ratio of the submitted version, which equals one for exact copies by Proposition~\ref{prop:duplicate} and is therefore uninformative.}
\paragraph{Protocol.} TF-IDF (20{,}000 terms, sublinear term frequency) is fitted on the training split; the top 200 terms by maximum absolute logistic-regression coefficient (fitted on the training split only) are kept. The explained model is a one-hidden-layer MLP (256 units, early stopping, at most 60 epochs). For each seed we duplicate the $m=10$ terms with the largest first-layer weight norm times feature standard deviation, a choice that does not use Kernel-MCIR. Copies are exact ($p=0$) or have a fraction $p$ of their non-zero entries set to zero. Feature kernels are RBF with the median heuristic per term; the output kernel is RBF on the predicted class-probability vector. Unless stated otherwise, attributions use 500 random test documents and $B=200$ uniformly sampled orders shared by the pre- and post-duplication runs. The dilution ratio is $\bar U^{\mathrm{post}}_i/\bar U^{\mathrm{pre}}_i$ averaged over the duplicated terms; the AUROC scores how well the relative score change $(\mathrm{post}-\mathrm{pre})/\mathrm{pre}$ identifies the duplicated terms among all 200. ``Fixed'' keeps the trained model (copies are ignored by the model); ``retrained'' refits the MLP on the augmented input.

\paragraph{All test documents.} Storing 200 centred Gram matrices for all $n=7{,}532$ test documents would require about 90\,GB. Since Kernel-MCIR needs only $(M,b)$ (Proposition~\ref{prop:metagram}), we stream over blocks of rows using
$\Fro{\tilde K_j,\tilde K_k}=\Fro{K_j,K_k}-2n\,\bar r_j^\top\bar r_k+n^2\bar m_j\bar m_k$, with $\bar r_j$ the row means and $\bar m_j$ the grand mean of $K_j$; this gives the exact centred meta-Gram in one pass (relative error $<10^{-12}$ against the dense computation) and about 4\,min per seed on a laptop. For exact copies every post-duplication meta-Gram is an index expansion of the same $M$.
""")

# ---- Table: full grid at n=500
g = main[~main.method.str.startswith("_")]
P(r"""\begin{table}[htbp]\centering\scriptsize
\caption{\rev{20 Newsgroups duplicate dilution $\bar U^{\mathrm{post}}/\bar U^{\mathrm{pre}}$ and duplicate-detection AUROC (mean$\pm$sd over 10 seeds; 500 test documents; $B=200$). Theory for exact copies: $1/(k{+}1)$.}}\label{tab:dil_grid}
\begin{tabular}{llr|cc|cc|cc}\toprule
 & & & \multicolumn{2}{c|}{K-MCIR $\bar U$} & \multicolumn{2}{c|}{Permutation imp.} & \multicolumn{2}{c}{HSIC} \\
Model & $k$ & $p$ & dilution & AUROC & dilution & AUROC & dilution & AUROC \\\midrule""")
for st in ("fixed", "retrain"):
    for k in (1, 2, 3):
        for p in (0.0, 0.1, 0.3):
            cells = []
            for m in ("KMCIR_U", "PermImp", "HSIC"):
                s = g[(g.setting == st) & (g.k == k) & (g.p == p) & (g.method == m)]
                cells += [ms(s.dilution), ms(s.auroc)]
            P(f"{st} & {k} & {p:.1f} & " + " & ".join(cells) + r" \\")
    if st == "fixed":
        P(r"\midrule")
P(r"\bottomrule\end{tabular}\end{table}")

# ---- Table: k=1, p=0 robustness and relatives
rows = []
def add(label, df, setting="retrain", method="KMCIR_U", k=1):
    s = df[(df.setting == setting) & (df.k == k) & (df.p == 0) & (df.method == method)]
    rows.append((label, s.seed.nunique(), ms(s.dilution), ms(s.auroc)))
add("K-MCIR $\\bar U$, LR top-200, 500 docs", main)
add("K-MCIR $\\bar U$, DF top-200 (unsupervised)", df200)
add("K-MCIR $\\bar U$, LR top-500", lr500)
add("K-MCIR $\\bar U$, all 7{,}532 docs", fulln)
add("K-MCIR $\\bar U$, all 7{,}532 docs, fixed model", fulln, "fixed")
add("SAGE (200 docs)", sage, method="SAGE")
mt = J("matched_sage.json")   # Kernel-MCIR re-scored on exactly the 200 documents used by SAGE (matched_sage_rerun.py)
rows.append(("K-MCIR $\\bar U$, SAGE's 200 docs (matched)", len(mt), ms(mt.kmcir200_dil), ms(mt.kmcir200_auroc)))
rows.append(("Permutation imp., SAGE's 200 docs (matched)", len(mt), ms(mt.perm200_dil), ms(mt.perm200_auroc)))
for lab, m in (("Energy (kernel LMG)", "E"), ("Partial dCov", "pdCov"), ("Sequential HSIC gain", "seqG")):
    rows.append((lab, len(extra), ms(extra[f"dil_{m}"]), ms(extra[f"auroc_{m}"])))
P(r"""\begin{table}[htbp]\centering\scriptsize
\caption{\rev{Duplicate dilution for $k=1$ exact copies, MLP retrained unless stated: screening rules, all test documents, SAGE and kernel relatives of Kernel-MCIR on the same sampled orders.}}\label{tab:dil_k1}
\begin{tabular}{lccc}\toprule
Method / setting & seeds & dilution & AUROC \\\midrule""")
for r in rows:
    P(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\")
P(r"\bottomrule\end{tabular}\end{table}")

fx = fulln[(fulln.setting == "fixed") & (fulln.method == "KMCIR_U")].groupby("k").dilution.mean()
P(r"""\rev{With all test documents and a fixed model the ratios are %.3f, %.3f and %.3f for $k=1,2,3$, closer to $1/(k{+}1)$ than with 500 documents, as expected from Proposition~\ref{prop:consistency}. Once the model is retrained on the copies, model-based allocations (permutation importance, SAGE) also split credit, as Shapley symmetry predicts; Kernel-MCIR additionally reports the overlap for a fixed model, where model-based scores are unchanged, and returns the shared fraction $1-\bar s$. Projection energy and partial dCov dilute less and detect duplicates less reliably than $\bar U$; sequential HSIC gain with product kernels does not detect them.}
""" % tuple(fx.values))

# ---- reviewer Q2: ranks and rank agreement
sy = pd.DataFrame(syn); one = sy[sy.k == 1]
P(r"""\paragraph{Relation to sequential HSIC and projection energy.} On the synthetic duplicate test ($Y=\sin 2X_0+0.5X_1^2+0.3X_2$, $X_3$ noise, $X_4$ an exact copy of $X_0$; $n=300$, $B=200$, 5 seeds) we compare, on the same orders, $\bar s$ with (a) the order-averaged sequential HSIC gain of the product kernel divided by marginal HSIC and (b) the normalised projection energy $\bar E_i/\sum_j\bar E_j$, $\bar E_i=\mathbb E_\pi[u_i^2/\|Q_i\|_F^2]$.
\begin{table}[htbp]\centering\scriptsize
\caption{\rev{Mean rank per feature (1 = highest) on the synthetic duplicate test.}}\label{tab:q2ranks}
\begin{tabular}{lccccc}\toprule
Score & $X_0$ & $X_1$ & $X_2$ & $X_3$ (noise) & $X_4$ (copy of $X_0$) \\\midrule""")
for name, lab in (("s", "$\\bar s$"), ("seqRatio", "(a) seq.\\ HSIC gain / HSIC"), ("Eshare", "(b) energy share"), ("U", "$\\bar U$"), ("pdCov", "partial dCov")):
    Rk = np.array(one[f"rank_{name}"].tolist()).mean(0)
    P(lab + " & " + " & ".join(f"{v:.1f}" for v in Rk) + r" \\")
P(r"\bottomrule\end{tabular}\end{table}")
dils = sy.groupby("k")[["dil_U", "dil_E", "dil_seqG", "dil_pdCov"]].mean()
P(r"""\rev{Only $\bar s$ ranks the duplicated pair $X_0,X_4$ last (it is the only score that measures novelty rather than relevance); (a) and (b) rank the pair first, and the $\bar U$ ranking coincides with the energy ranking. Dilution of the original for $k=1,2,3$: $\bar U$ %s; energy %s; sequential HSIC %s; partial dCov %s. On 20 Newsgroups (5 seeds), Spearman correlations with $\bar s$ are $%.2f$ for (a) and $%.2f$ for (b); with $\bar U$ they are $%.2f$ for sequential HSIC gain, $%.2f$ for energy and $%.2f$ for partial dCov.}
""" % ("/".join(f"{v:.3f}" for v in dils.dil_U), "/".join(f"{v:.3f}" for v in dils.dil_E),
       "/".join(f"{v:.3f}" for v in dils.dil_seqG), "/".join(f"{v:.3f}" for v in dils.dil_pdCov),
       extra.spearman_s_seqRatio.mean(), extra.spearman_s_Eshare.mean(), extra.spearman_U_seqG.mean(),
       extra.spearman_U_E.mean(), extra.spearman_U_pdCov.mean()))

# ---- B convergence and RFF copy symmetry
bc = bconv.groupby("B")[["max_s", "rho_U", "top20"]].mean()
rs = rsym.groupby("D").gap.agg(["mean", "std"])
P(r"""\paragraph{Number of orders and RFF symmetry.} \rev{Against a $B=2000$ reference on 20 Newsgroups (3 seeds, 3 repetitions), $B=50,100,200,400$ give $\max_i|\bar s_i-\bar s_i^{\mathrm{ref}}|=%s$, Spearman correlation of the $\bar U$ ranking $\ge%.3f$, and top-20 overlap $%s$. With independent random frequencies for an exact copy (synthetic, 5 seeds, $B=4000$), $|\bar s_{\mathrm{orig}}-\bar s_{\mathrm{copy}}|=%s$ for $D=32,128,512,1024$, reaching the Monte Carlo floor of exact kernels ($%.3f$).}
""" % ("/".join(f"{v:.3f}" for v in bc.max_s), bc.rho_U.min(), "/".join(f"{v:.2f}" for v in bc.top20),
       "/".join(f"{rs.loc[D, 'mean']:.3f}" for D in (32, 128, 512, 1024)), rs.loc[0, "mean"]))
print("\n".join(out))
