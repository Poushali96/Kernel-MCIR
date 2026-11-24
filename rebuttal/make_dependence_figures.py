"""Figures + LaTeX table for the dependence sweep (local_runs/dependence_sweep.json). No numbers typed by hand.
python make_dependence_figures.py  ->  figures/dependence_sweep.png, figures/joint_map.png, local_runs/dependence_sweep_tables.tex
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "local_runs", "dependence_sweep.json")))
C = {"s": "#2a78d6", "cka": "#eb6834", "loco": "#1baf7a", "dfi": "#eda100"}   # reference palette slots 1-4
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def est(r, key, i, fn=np.nanmean):
    return fn([e[key][i] for e in r["est"]])


def get(regime):
    return sorted([r for r in rows if r["regime"] == regime], key=lambda r: r["rho"])


# ---------------- Figure: sweep for X2 (the feature whose role changes across regimes)
regs = [("proxy", "Unused proxy: $f=X_1$"), ("complementary", "Suppressor: $f\\propto X_1-X_2$"),
        ("additive", "Distinct roles: $f=\\sin 2X_1+X_2^2$"), ("interaction", "Interaction: $f=X_1X_2$")]
fig, axs = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
for ax, (reg, title) in zip(axs, regs):
    R = get(reg); x = [r["rho"] for r in R]
    for key, lab, mk, ls, src in (("s", "K-MCIR $\\bar s_2$", "o", "-", "est"), ("cka", "CKA$_2$ (marginal alignment)", "s", "-", "est"),
                                  ("loco", "LOCO$_2$ (oracle)", "^", "--", "orc"), ("dfi", "DFI$_2$ (oracle)", "D", ":", "orc")):
        if src == "est":
            m = [est(r, key, 1) for r in R]; sd = [est(r, key, 1, np.nanstd) for r in R]
            ax.fill_between(x, np.subtract(m, sd), np.add(m, sd), color=C[key], alpha=0.15, lw=0)
        else:
            m = [r["oracle"][key][1] for r in R]
        ax.plot(x, m, ls, color=C[key], marker=mk, ms=4, lw=1.6, label=lab)
    ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("correlation $\\rho$")
    ax.set_ylim(-0.03, 1.05); ax.grid(axis="y", color=GRID, lw=0.6)
axs[0].set_ylabel("score for $X_2$ (all on [0,1])")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "dependence_sweep.png"), dpi=220, bbox_inches="tight")

# ---------------- Figure: joint geometric-functional map at rho = 0.9 (+ duplicate), y = LOCO and y = DFI
noise_cka = np.mean([est(r, "cka", 2) for r in rows])
mk = {"duplicate": "P", "proxy": "o", "complementary": "s", "additive": "^", "interaction": "D"}
cols = {"duplicate": "#6250d6", "proxy": "#2a78d6", "complementary": "#eb6834", "additive": "#1baf7a", "interaction": "#eda100"}
# label offsets (points) so coincident markers stay readable
off = {("duplicate", 0): (-38, 6), ("proxy", 1): (6, -11), ("interaction", 1): (6, -11), ("additive", 0): (6, -4)}
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4), sharex=True)
pts = []
for ax, key, ylab in ((axs[0], "loco", "LOCO$_i$ (oracle)"), (axs[1], "dfi", "DFI$_i$ (oracle)")):
    for r in rows:
        if not (r["rho"] == 0.9 or r["regime"] == "duplicate"):
            continue
        for i in (0, 1):
            s, c, F = est(r, "s", i), est(r, "cka", i), r["oracle"][key][i]
            active = c >= 0.05
            jit = 0.012 if r["regime"] == "duplicate" else 0.0
            ax.scatter(s - jit, F, marker=mk[r["regime"]], s=48, facecolor=cols[r["regime"]] if active else "white",
                       edgecolor=cols[r["regime"]], lw=1.5, zorder=3 + (r["regime"] == "duplicate"),
                       label=r["regime"] if (i == 0 and key == "loco") else None)
            lab = {("duplicate", 0): "$X_1{=}X_2$", ("duplicate", 1): None, ("interaction", 0): None,
                   ("interaction", 1): "$X_1,X_2$"}.get((r["regime"], i), f"$X_{i+1}$")
            if lab:
                ax.annotate(lab, (s - jit, F), xytext=off.get((r["regime"], i), (5, 3)), textcoords="offset points",
                            fontsize=7, color=MUTED)
            if key == "loco":
                pts.append((r["regime"], i + 1, s, c, F))
    ax.set_xlabel("geometric novelty $\\bar s_i$ (Kernel-MCIR)"); ax.set_ylabel("functional importance " + ylab)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.05, 1.08); ax.grid(color=GRID, lw=0.6)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
fig.text(0.5, -0.02, "$\\rho=0.9$ (duplicate: $\\rho=1$), 20 seeds; hollow marker = weak marginal alignment (CKA$_i<0.05$): "
         "$\\bar s_i$ is then a ratio of small numbers and is not interpreted", ha="center", fontsize=7, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "joint_map.png"), dpi=220, bbox_inches="tight")

# ---------------- Table + agreement summary
T = []
P = T.append
P(r"""\begin{table}[htbp]\centering\scriptsize
\caption{\rev{Dependence sweep for $X_2$ ($n=400$, %d seeds, exact average over all orders; mean$\pm$sd). LOCO, DFI and DIP are population (oracle) values for the known $f$ and Gaussian $X$, normalised by $\mathrm{Var} f$; DIP: LOCO$_2$ = standalone $+$ interaction $-$ dependence. CKA$_3$ of the independent noise feature is $%.3f$ on average.}}\label{tab:dep_sweep}
\begin{tabular}{lc|ccc|ccccc}\toprule
Regime & $\rho$ & $\bar s_2$ & CKA$_2$ & energy share$_2$ & LOCO$_2$ & DFI$_2$ & DIP stand. & DIP int. & DIP dep. \\\midrule""" % (rows[0]["seeds"], noise_cka))
for r in rows:
    if r["rho"] not in (0.6, 0.9, 0.99, 1.0):
        continue
    o = r["oracle"]
    pm = lambda k: f"${est(r, k, 1):.2f}\\pm{est(r, k, 1, np.nanstd):.2f}$"
    P(f"{r['regime']} & {r['rho']:.2f} & {pm('s')} & {pm('cka')} & {pm('E_share')} & {o['loco'][1]:.2f} & {o['dfi'][1]:.2f} & "
      f"{o['v_single'][1]:.2f} & {o['dip_interaction'][1]:.2f} & {o['dip_dependence'][1]:.2f} \\\\")
P(r"\bottomrule\end{tabular}\end{table}")
# rank agreement with functional estimands over all (regime, rho, feature in {X1, X2}) cells
cells = [(est(r, k, i), r["oracle"]["loco"][i], r["oracle"]["dfi"][i])
         for r in rows for i in (0, 1) for k in ("s",)]
agree = {}
for k in ("s", "U", "cka", "hsic", "E_share", "pdcov"):
    v = np.array([est(r, k, i) for r in rows for i in (0, 1)])
    lo = np.array([r["oracle"]["loco"][i] for r in rows for i in (0, 1)])
    df = np.array([r["oracle"]["dfi"][i] for r in rows for i in (0, 1)])
    agree[k] = (spearmanr(v, lo).correlation, spearmanr(v, df).correlation)
sh = [1 - est(r, "s", i) for r in rows for i in (0, 1) if r["rho"] > 0]
dp = [r["oracle"]["dip_dependence"][i] for r in rows for i in (0, 1) if r["rho"] > 0]
P("%% Spearman(1 - s, DIP dependence term), rho>0 cells: %+.2f" % spearmanr(sh, dp).correlation)
P("% Spearman over all (regime, rho, X1/X2) cells: score vs LOCO, vs DFI")
for k, (a, b) in agree.items():
    P(f"% {k:8s} LOCO {a:+.2f}  DFI {b:+.2f}")
open(os.path.join(HERE, "local_runs", "dependence_sweep_tables.tex"), "w").write("\n".join(T) + "\n")
print("\n".join(T))
print("joint-map points (regime, feature, s, cka, LOCO):")
for p in pts:
    print("  %-13s X%d  s=%.2f  cka=%.3f  LOCO=%.2f" % p)
print("noise CKA floor %.4f" % noise_cka)

# ---------------- Figure: joint geometric-functional attribution map (quadrants) with rho trajectories
XT, YT = 0.75, 0.5   # midpoint between full sharing within a pair (s=0.5) and full novelty (s=1); half of Var f
fig, ax = plt.subplots(figsize=(6.4, 4.6))
quad = [((0.40, XT), (YT, 1.08), "#f6efe4", "Shared geometry, essential function", "needed jointly with a correlated feature"),
        ((XT, 1.03), (YT, 1.08), "#e8f4ee", "Novel geometry, essential function", "distinct contribution"),
        ((0.40, XT), (-0.05, YT), "#efefec", "Shared geometry, inessential", "duplicate or redundant proxy"),
        ((XT, 1.03), (-0.05, YT), "#e9eff8", "Novel geometry, inessential", "small or recoverable role: audit")]
for (x0, x1), (y0, y1), fc, t1, t2 in quad:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor="none", zorder=0))
    tx = x0 + 0.012                                   # all labels left-aligned inside their quadrant
    ty = {True: y1 - 0.015, False: YT - 0.025}[y0 >= YT] if x0 < XT or y0 >= YT else y0 + 0.13
    ax.text(tx, ty, t1, ha="left", va="top", fontsize=7.5, color=INK, weight="bold", zorder=1)
    ax.text(tx, ty - 0.045, t2, ha="left", va="top", fontsize=7, color=MUTED, zorder=1)
ax.axvline(XT, color=MUTED, ls="--", lw=0.8, zorder=1); ax.axhline(YT, color=MUTED, ls="--", lw=0.8, zorder=1)
tracks = [("proxy", 0, "#2a78d6", "--", "$f=X_1$: used feature $X_1$"),
          ("proxy", 1, "#2a78d6", "-", "$f=X_1$: unused proxy $X_2$"),
          ("complementary", 1, "#eb6834", "-", "suppressor $X_2$"),
          ("additive", 1, "#1baf7a", "-", "distinct roles $X_2$"),
          ("interaction", 1, "#eda100", "-", "interaction $X_2$")]
RH = (0.0, 0.6, 0.9, 0.99)
for reg, i, col, ls, lab in tracks:
    R = [r for r in get(reg) if r["rho"] in RH]
    xs = [est(r, "s", i) for r in R]; ys = [r["oracle"]["loco"][i] for r in R]; cs = [est(r, "cka", i) for r in R]
    ax.plot(xs, ys, ls, color=col, lw=1.3, zorder=2)
    ax.plot([], [], ls, color=col, lw=1.3, marker=mk[reg], ms=5, label=lab)
    for a, b in zip(range(len(xs) - 1), range(1, len(xs))):
        ax.annotate("", (xs[b], ys[b]), (xs[a], ys[a]), arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=9,
                    shrinkA=0, shrinkB=5), zorder=2)
    for x, y, c, rr in zip(xs, ys, cs, R):
        ax.scatter(x, y, marker=mk[reg], s=14 + 34 * rr["rho"], facecolor=col if c >= 0.05 else "white", edgecolor=col,
                   lw=1.3, zorder=3)
D = get("duplicate")[0]
ax.scatter(est(D, "s", 0) - 0.012, D["oracle"]["loco"][0], marker=mk["duplicate"], s=60, color=cols["duplicate"], zorder=4,
           edgecolor="white", lw=0.8, label="exact duplicate ($X_2=X_1$)")
ax.set_xlim(0.40, 1.03); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("geometric novelty $\\bar s_i$ (Kernel-MCIR, estimated)"); ax.set_ylabel("functional importance: oracle LOCO$_i$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=7)
fig.text(0.5, -0.035, "Arrows follow increasing $\\rho\\in\\{0,0.6,0.9,0.99\\}$ (marker size grows with $\\rho$); hollow = CKA$_i<0.05$, "
         "where $\\bar s_i$ is not interpreted. Thresholds are illustrative.", ha="center", fontsize=6.8, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "attribution_map.png"), dpi=220, bbox_inches="tight")
print("wrote figures/attribution_map.png")
