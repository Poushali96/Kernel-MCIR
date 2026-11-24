"""Cross-check of the dependence-sweep oracle values against the authors' official implementations.

DFI : fdfi 0.1.0 (github.com/jinhongdu-lab/FDFI), OTExplainer = Gaussian optimal-transport DFI (Du, Roeder & Wasserman).
      Unnormalised output phi_X is divided by Var f, as in our table.
DIP : dipd 0.0.3 (github.com/gcskoenig/dipd), DIP.get_loo(feature) with the EBM learner (Koenig, Guenther & von Luxburg
      2025). Mapping to our columns: standalone = v1, interaction = pure_interactions,
      dependence = -(main_effect_cross_predictability + main_effect_cov); LOCO = v1 + interaction - dependence.
      dipd fits learners to data, so its values are estimates (sampling + learner error), not population values.

Requires Python >= 3.11 (dipd). Run in the isolated venv, e.g.
  <venv>/bin/python crosscheck_official_dfi_dip.py --seeds 3 --n 4000 --out local_runs/official_crosscheck.json
Same regimes and model definitions as kmcir_dependence_sweep.py.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


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


def sample_X(n, rho, rng, duplicate=False):
    if duplicate:
        z = rng.standard_normal((n, 2))
        return np.column_stack([z[:, 0], z[:, 0], z[:, 1]])
    S = np.eye(3); S[0, 1] = S[1, 0] = rho
    return rng.multivariate_normal(np.zeros(3), S, size=n)


def dfi_official(f, rho, dup, rng, n_bg, n_test):
    from fdfi.explainers import OTExplainer
    Xbg, Xte = sample_X(n_bg, rho, rng, dup), sample_X(n_test, rho, rng, dup)
    ex = OTExplainer(f, data=Xbg, nsamples=50, random_state=int(rng.integers(1 << 30)), compute_diagnostics=False)
    res = ex(Xte)
    return (res["phi_X"] / np.var(f(Xte))).tolist()


def dip_official(f, rho, dup, rng, n, features):
    from dipd import DIP, EBM
    X = sample_X(n, rho, rng, dup)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"]); df["y"] = f(X)
    dip = DIP(df, "y", EBM, test_size=0.3, random_state=int(rng.integers(1 << 30)))
    out = {}
    for i in features:
        r = dip.get_loo(f"x{i + 1}").scores
        dep = -(r["main_effect_cross_predictability"] + r["main_effect_cov"])
        out[i] = dict(standalone=float(r["v1"]), interaction=float(r["pure_interactions"]), dependence=float(dep),
                      loco=float(r["v1"] + r["pure_interactions"] - dep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.6, 0.9, 0.99])
    ap.add_argument("--out", default=os.path.join(HERE, "local_runs", "official_crosscheck.json"))
    a = ap.parse_args()
    cells = [("duplicate", 1.0)] + [(g, r) for g in ("proxy", "complementary", "additive", "interaction") for r in a.rhos]
    rows = []
    for reg, rho in cells:
        t0 = time.time(); dup = reg == "duplicate"; f = model(reg, min(rho, 0.999))
        dfi, dip = [], []
        for s in range(a.seeds):
            rng = np.random.default_rng(1000 + s)
            dfi.append(dfi_official(f, rho, dup, rng, 20000, a.n))
            dip.append(dip_official(f, rho, dup, rng, a.n, (0, 1)))
        row = dict(regime=reg, rho=rho, seeds=a.seeds, n=a.n,
                   dfi=np.mean(dfi, 0).tolist(), dfi_sd=np.std(dfi, 0).tolist())
        for key in ("standalone", "interaction", "dependence", "loco"):
            v = np.array([[d[i][key] for i in (0, 1)] for d in dip])
            row["dip_" + key], row["dip_" + key + "_sd"] = v.mean(0).tolist(), v.std(0).tolist()
        rows.append(row)
        print(f"{reg:13s} rho={rho:.2f}  DFI2={row['dfi'][1]:.3f}  DIP2 stand={row['dip_standalone'][1]:.3f} "
              f"int={row['dip_interaction'][1]:.3f} dep={row['dip_dependence'][1]:.3f} loco={row['dip_loco'][1]:.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
