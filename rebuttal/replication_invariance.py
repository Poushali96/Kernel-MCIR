"""Representation-perturbation benchmark: does adding copies / noisy copies / irrelevant features change Kernel-MCIR
scores of the UNAFFECTED features, and is the unique strength of a replicated feature conserved at group level?

Compares Kernel-MCIR with uniform orders over all features (the paper's estimator) and with block orders, in which
features detected as replicas (meta-Gram CKA >= kappa) form one block: blocks are permuted uniformly and members are
shuffled within their block (Proposition prop:replication).

Base data: d = 6 Gaussian features (X0, X1 correlated 0.5; X2, X3 correlated 0.5; X4, X5 independent),
f = sin(2 X0) + X1^2 + X2 X3 + 0.5 X4  (X5 irrelevant). The explained model is f itself.
Perturbations (all applied to the same n points, same output kernel):
  copies k in {1,2,3,5}       exact copies of X0
  noisy copies sigma in {.05,.1,.2} (k = 2)   X0 + sigma * N(0,1)
  irrelevant m = 3            independent N(0,1) features
  rescale                     all features multiplied by 3 (median-heuristic bandwidths rescale with them)
Metrics over the unaffected features A = {X1..X5}:
  AD_s = mean_{j in A} |s_j^post - s_j^pre|,  AD_U = mean_{j in A} |U_j^post - U_j^pre| / U_j^pre
  CE   = |sum_{g in G} U_g^post - U_0^pre| / U_0^pre   (G = X0 and its copies)
  dil  = U_0^post / U_0^pre
python replication_invariance.py --seeds 20 --n 400 --B 600 --out local_runs/replication_invariance.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmcir_rebuttal_runs as K  # noqa: E402
from kmcir_extra_analyses import kmcir_energy  # noqa: E402


def base_data(n, rng):
    X = rng.standard_normal((n, 6))
    X[:, 1] = 0.5 * X[:, 0] + np.sqrt(0.75) * X[:, 1]
    X[:, 3] = 0.5 * X[:, 2] + np.sqrt(0.75) * X[:, 3]
    f = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + X[:, 2] * X[:, 3] + 0.5 * X[:, 4]
    return X, f


def meta(X, f):
    grams = [K.feature_gram(X[:, j]) for j in range(X.shape[1])]
    L = K.output_gram(f[:, None])
    return K.meta_gram_exact(grams, L)


def groups_from_meta(M, kappa):
    d = len(M); cka = M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
    lab = -np.ones(d, int); g = 0
    for i in range(d):
        if lab[i] < 0:
            members = [j for j in range(d) if lab[j] < 0 and cka[i, j] >= kappa]
            lab[members] = g; g += 1
    return [list(np.where(lab == k)[0]) for k in range(g)]


def orders(d, B, rng, groups=None):
    if groups is None:
        return [rng.permutation(d) for _ in range(B)]
    out = []
    for _ in range(B):
        out.append(np.concatenate([rng.permutation(groups[k]) for k in rng.permutation(len(groups))]))
    return out


def scores(M, b, B, rng, mode, kappa):
    gr = None if mode == "uniform" else groups_from_meta(M, kappa)
    return kmcir_energy(M, b, orders(len(b), B, rng, gr)), gr


def coupled(M0, b0, M, b, B, rng, G):
    """Coupled block orders: each post order replaces feature 0 of a uniform pre order by a uniformly shuffled block G."""
    pre_o = [rng.permutation(len(b0)) for _ in range(B)]
    post_o = []
    for p in pre_o:
        q = []
        for j in p:
            q.extend(rng.permutation(G) if j == 0 else [j])
        post_o.append(np.array(q))
    return kmcir_energy(M0, b0, pre_o), kmcir_energy(M, b, post_o)


def run(args):
    perts = [("copies", k, 0.0) for k in (1, 2, 3, 5)] + [("noisy", 2, s) for s in (0.05, 0.1, 0.2)] + [("irrelevant", 3, 0.0), ("rescale", 0, 3.0)]
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        X, f = base_data(args.n, rng)
        M0, b0 = meta(X, f)
        for mode in ("uniform", "block"):
            pre, _ = scores(M0, b0, args.B, np.random.default_rng(10_000 + seed), mode, args.kappa)
            for kind, k, sig in perts:
                r2 = np.random.default_rng(20_000 + seed)
                if kind == "rescale":
                    Xp = X * sig; G = [0]
                elif kind == "irrelevant":
                    Xp = np.column_stack([X, r2.standard_normal((args.n, k))]); G = [0]
                else:
                    cp = [X[:, 0] + sig * r2.standard_normal(args.n) for _ in range(k)]
                    Xp = np.column_stack([X] + cp); G = [0] + list(range(6, 6 + k))
                M, b = meta(Xp, f)
                post, gr = scores(M, b, args.B, np.random.default_rng(30_000 + seed), mode, args.kappa)
                A = [1, 2, 3, 4, 5]
                rows.append(dict(seed=seed, mode=mode, kind=kind, k=k, sigma=sig,
                                 AD_s=float(np.mean(np.abs(post["s"][A] - pre["s"][A]))),
                                 AD_U=float(np.mean(np.abs(post["U"][A] - pre["U"][A]) / pre["U"][A])),
                                 CE=float(abs(post["U"][G].sum() - pre["U"][0]) / pre["U"][0]),
                                 dil=float(post["U"][0] / pre["U"][0]),
                                 grouped=bool(gr is not None and any(set(G) <= set(g) for g in gr))))
        for kind, k, sig in perts:                                   # coupled block orders (replica perturbations only)
            if kind not in ("copies", "noisy"):
                continue
            r2 = np.random.default_rng(20_000 + seed)
            cp = [X[:, 0] + sig * r2.standard_normal(args.n) for _ in range(k)]
            Xp = np.column_stack([X] + cp); G = [0] + list(range(6, 6 + k))
            M, b = meta(Xp, f)
            pre, post = coupled(M0, b0, M, b, args.B, np.random.default_rng(40_000 + seed), G)
            A = [1, 2, 3, 4, 5]
            rows.append(dict(seed=seed, mode="coupled", kind=kind, k=k, sigma=sig,
                             AD_s=float(np.mean(np.abs(post["s"][A] - pre["s"][A]))),
                             AD_U=float(np.mean(np.abs(post["U"][A] - pre["U"][A]) / pre["U"][A])),
                             CE=float(abs(post["U"][G].sum() - pre["U"][0]) / pre["U"][0]),
                             dil=float(post["U"][0] / pre["U"][0]), grouped=True))
        print(f"seed {seed} done", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.2e}")
    print(df.groupby(["kind", "k", "sigma", "mode"])[["AD_s", "AD_U", "CE", "dil"]].mean().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--B", type=int, default=600)
    ap.add_argument("--kappa", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_runs", "replication_invariance.json"))
    run(ap.parse_args())
