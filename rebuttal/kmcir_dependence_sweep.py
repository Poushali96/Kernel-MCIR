"""Controlled dependence sweep for reviewer questions on correlated features: Kernel-MCIR vs. functional-importance oracles.

Features: X1, X2 standard Gaussian with correlation rho, X3 independent noise (d = 3). The explained model is the
oracle function f(X) itself (no estimation error), so every functional quantity has an exact population value.

Regimes (f):
  duplicate      X2 = X1 exactly,                 f = X1
  proxy          X2 correlated but unused,         f = X1
  complementary  suppressor,                       f = (X1 - X2) / sqrt(2 (1 - rho))   (unit variance)
  additive       distinct functional roles,        f = sin(2 X1) + X2^2
  interaction    joint information,                f = X1 * X2

Population (oracle) functional references, all divided by Var f:
  LOCO_i   = E Var(f | X_-i)                                   (nested Monte Carlo)
  DFI_i    = sum_j (Sigma^{1/2})_{ij}^2 E Var(f(Sigma^{1/2} Z) | Z_-j)   (Du, Roeder & Wasserman, Eq. 4, Gaussian
             Bures-Wasserstein map Z = Sigma^{-1/2} X; FDFI estimates the same estimand with a learned flow)
  DIP_i    = (standalone v(i), interaction Var h*, dependence Dep) with LOCO_i = v(i) + Var h* - Dep
             (Koenig, Guenther & von Luxburg 2025, Thm 6; h* = f - best additive fit g(X_i) + g(X_-i))
Estimated on n samples per seed (median-heuristic RBF kernels, exact average over all 3! orders):
  Kernel-MCIR U, R, T, s, activity a = U / max T; HSIC; CKA; kernel LMG energy share; partial dCov.

Usage:  python kmcir_dependence_sweep.py --seeds 20 --n 400 --out local_runs/dependence_sweep.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from numpy.polynomial.hermite_e import hermevander

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmcir_rebuttal_runs as K  # noqa: E402
from kmcir_extra_analyses import kmcir_energy, pdcov_all  # noqa: E402

RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
REGIMES = ["duplicate", "proxy", "complementary", "additive", "interaction"]


def model(regime, rho):
    if regime in ("duplicate", "proxy"):
        return lambda X: X[..., 0]
    if regime == "complementary":
        c = 1.0 / np.sqrt(2 * (1 - rho))
        return lambda X: c * (X[..., 0] - X[..., 1])
    if regime == "additive":
        return lambda X: np.sin(2 * X[..., 0]) + X[..., 1] ** 2
    if regime == "interaction":
        return lambda X: X[..., 0] * X[..., 1]
    raise ValueError(regime)


def sigma(rho):
    S = np.eye(3); S[0, 1] = S[1, 0] = rho
    return S


def sample_X(n, rho, rng, duplicate=False):
    x1 = rng.normal(size=n)
    x2 = x1.copy() if duplicate else rho * x1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return np.column_stack([x1, x2, rng.normal(size=n)])


def cond_sample(X, i, rho, rng, k, duplicate):
    """k draws of X_i | X_-i for each row of X (Gaussian; X3 independent). Returns array (N, k, 3)."""
    N = len(X); Xr = np.repeat(X[:, None, :], k, axis=1)
    if i == 2:
        Xr[:, :, 2] = rng.normal(size=(N, k))
    else:
        j = 1 - i
        if duplicate:
            Xr[:, :, i] = X[:, j][:, None]
        else:
            Xr[:, :, i] = rho * X[:, j][:, None] + np.sqrt(1 - rho ** 2) * rng.normal(size=(N, k))
    return Xr


def oracle(regime, rho, rng, N=20000, k=200, Nadd=200000):
    dup = regime == "duplicate"
    f = model(regime, rho)
    X = sample_X(Nadd, rho, rng, dup); F = f(X); V = F.var()
    out = {"var_f": float(V)}
    # LOCO_i = E Var(f | X_-i) ; v(i) = Var E[f | X_i]
    Xs = X[:N]
    loco, v1 = [], []
    for i in range(3):
        loco.append(float(f(cond_sample(Xs, i, rho, rng, k, dup)).var(axis=1).mean() / V))
        # E[f | X_i]: resample everything except X_i
        Xr = np.repeat(Xs[:, None, :], k, axis=1)
        if i == 2:
            Xr[:, :, :2] = sample_X(N * k, rho, rng, dup)[:, :2].reshape(N, k, 2)
        else:
            j = 1 - i
            Xr[:, :, j] = Xs[:, i][:, None] if dup else rho * Xs[:, i][:, None] + np.sqrt(1 - rho ** 2) * rng.normal(size=(N, k))
            Xr[:, :, 2] = rng.normal(size=(N, k))
        v1.append(float(f(Xr).mean(axis=1).var() / V))
    out["loco"], out["v_single"] = loco, v1
    # DFI with the Gaussian whitening map (population)
    w, Q = np.linalg.eigh(sigma(1.0 if dup else rho))
    Lh = Q @ np.diag(np.sqrt(np.clip(w, 0, None))) @ Q.T
    Z = rng.normal(size=(N, 3))
    phiZ = []
    for j in range(3):
        Zr = np.repeat(Z[:, None, :], k, axis=1); Zr[:, :, j] = rng.normal(size=(N, k))
        phiZ.append(float(f(Zr @ Lh.T).var(axis=1).mean() / V))
    out["dfi_latent"] = phiZ
    out["dfi"] = [float(sum(Lh[l, j] ** 2 * phiZ[j] for j in range(3))) for l in range(3)]
    # DIP: additive projection f ~ g_J(X_i) + g_Jbar(X_-i); X3 is independent and unused, so g_Jbar depends on X_j only
    def basis(x):
        return np.column_stack([hermevander(x, 15)[:, 1:], np.sin(2 * x), np.cos(2 * x)])
    inter, dep = [], []
    for i in range(3):
        if i == 2:
            A = np.column_stack([np.ones(Nadd), basis(X[:, 2]), basis(X[:, 0]), basis(X[:, 1]),
                                 (X[:, 0] * X[:, 1])])      # X_-i = (X1, X2) is a group: allow its own interaction
        else:
            A = np.column_stack([np.ones(Nadd), basis(X[:, i]), basis(X[:, 1 - i])])
        coef, *_ = np.linalg.lstsq(A, F, rcond=None)
        h = F - A @ coef
        vh = float(h.var() / V)
        inter.append(vh); dep.append(float(v1[i] + vh - loco[i]))
    out["dip_interaction"], out["dip_dependence"] = inter, dep
    return out


def estimated(regime, rho, n, seed):
    rng = np.random.default_rng(10_000 + seed)
    dup = regime == "duplicate"
    X = sample_X(n, rho, rng, dup); Y = model(regime, rho)(X)
    Kc = [K.feature_gram(X[:, j]) for j in range(3)]
    Lc = K.output_gram(Y)
    M, b = K.meta_gram_exact(Kc, Lc)
    r = kmcir_energy(M, b, [np.array(p) for p in itertools.permutations(range(3))])
    T = r["T"]
    return dict(U=r["U"].tolist(), R=r["R"].tolist(), T=T.tolist(), s=r["s"].tolist(),
                a=(r["U"] / T.max()).tolist(), E_share=(r["E"] / r["E"].sum()).tolist(),
                hsic=K.hsic_scores(Kc, Lc).tolist(), cka=K.cka_scores(Kc, Lc).tolist(),
                pdcov=pdcov_all(X, Y).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="local_runs/dependence_sweep.json")
    a = ap.parse_args()
    rows = []
    t0 = time.time()
    for regime in REGIMES:
        for rho in ([1.0] if regime == "duplicate" else RHOS):
            orc = oracle(regime, rho, np.random.default_rng(7))
            est = [estimated(regime, rho, a.n, s) for s in range(a.seeds)]
            rows.append(dict(regime=regime, rho=rho, n=a.n, seeds=a.seeds, oracle=orc, est=est))
            m = lambda key, i: np.nanmean([e[key][i] for e in est])
            print(f"{regime:13s} rho={rho:4.2f} | s={m('s',0):.2f}/{m('s',1):.2f}/{m('s',2):.2f} "
                  f"a={m('a',0):.2f}/{m('a',1):.2f}/{m('a',2):.2f} | LOCO={orc['loco'][0]:.2f}/{orc['loco'][1]:.2f} "
                  f"DFI={orc['dfi'][0]:.2f}/{orc['dfi'][1]:.2f} | DIP2 v/int/dep={orc['v_single'][1]:.2f}/"
                  f"{orc['dip_interaction'][1]:.2f}/{orc['dip_dependence'][1]:.2f} | {time.time()-t0:.0f}s", flush=True)
            json.dump(rows, open(a.out, "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
