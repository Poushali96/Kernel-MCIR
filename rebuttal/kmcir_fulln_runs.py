"""
Kernel-MCIR 20NG duplicate dilution on ALL 7,532 test documents (no evaluation subsample).

Storing 200 centred n x n Gram matrices at n = 7,532 would need ~90 GB, so the meta-Gram
M_jk = <K~_j, K~_k>_F and b_j = <K~_j, L~>_F is streamed over row blocks:

    <K~_j, K~_k>_F = <K_j, K_k>_F - 2 n r_j.r_k + n^2 m_j m_k        (r = row means, m = grand mean)

so one pass over the n^2 kernel entries gives the exact centred meta-Gram (float64).

With exact copies (p = 0) a copy's Gram equals its original's, so every post-duplication meta-Gram
is an index expansion of the same M. One streamed pass per seed (features + one output kernel per
model) covers k = 1, 2, 3 in both settings (fixed model, MLP retrained on the augmented input).
Rows use the same format as kmcir_rebuttal_runs.py, so its `report` command works on the json.

  python kmcir_fulln_runs.py check                 # streamed M == exact M at n = 600
  python kmcir_fulln_runs.py run --seeds 3 --data-home DATA --out local_runs/dil_fulln.json
"""
import argparse
import json
import os
import sys
import time

import numexpr as ne
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmcir_rebuttal_runs as K  # noqa: E402


def streamed_meta(X, outs, nb=24, verbose=True):
    """X: n x d (one scalar feature per column); outs: list of n x c output matrices.
    Per-feature RBF kernels and RBF output kernels, median-heuristic bandwidths exactly as in
    kmcir_rebuttal_runs.feature_gram / output_gram. Returns the centred meta-Gram over the
    d + len(outs) kernels (features first, then outputs)."""
    n, d = X.shape
    gam = np.array([1.0 / (2 * K._median_bw(X[:, j]) ** 2) for j in range(d)])[:, None, None]
    ogam = [1.0 / (2 * K._median_bw(P) ** 2) for P in outs]
    osq = [(P ** 2).sum(1) for P in outs]
    m = d + len(outs)
    S = np.zeros((m, m)); R = np.zeros((m, n))
    XT = np.ascontiguousarray(X.T)[:, None, :]                       # d x 1 x n
    t0 = time.time()
    for a0 in range(0, n, nb):
        a1 = min(n, a0 + nb); w = a1 - a0
        G = np.empty((m, w, n))
        xa = np.ascontiguousarray(X[a0:a1].T)[:, :, None]            # d x w x 1
        ne.evaluate("exp(-gam * (xa - XT) ** 2)", out=G[:d])
        for o, P in enumerate(outs):
            d2 = np.maximum(osq[o][a0:a1, None] + osq[o][None, :] - 2 * P[a0:a1] @ P.T, 0)
            g = ogam[o]
            G[d + o] = ne.evaluate("exp(-g * d2)")
        R[:, a0:a1] = G.mean(2)
        G2 = G.reshape(m, w * n)
        S += G2 @ G2.T
        if verbose and (a0 // nb) % 50 == 0:
            el = time.time() - t0
            print(f"   rows {a1}/{n}  {el:.0f}s elapsed, ~{el / a1 * (n - a1):.0f}s left", flush=True)
    mm = R.mean(1)
    return S - 2 * n * (R @ R.T) + n ** 2 * np.outer(mm, mm)


def check(n=600):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 6)); X[:, 1] = np.where(rng.random(n) < 0.8, 0, X[:, 1])  # sparse-ish column
    P = np.abs(rng.normal(size=(n, 4))); P /= P.sum(1, keepdims=True)
    Ms = streamed_meta(X, [P], nb=7, verbose=False)
    grams = [K.feature_gram(X[:, j]) for j in range(6)]
    Me, be = K.meta_gram_exact(grams, K.output_gram(P))
    print("max rel err M:", np.abs(Ms[:6, :6] - Me).max() / np.abs(Me).max())
    print("max rel err b:", np.abs(Ms[:6, 6] - be).max() / np.abs(be).max())
    print("max rel err ||L||^2:", abs(Ms[6, 6] - np.sum(K.output_gram(P) ** 2)) / Ms[6, 6])


def run(args):
    data = K.load_20ng(args.data_home)
    Xtr_s, ytr, Xte_s, yte, _ = data
    rows, done = [], set()
    if os.path.exists(args.out) and not args.fresh:
        rows = json.load(open(args.out)); done = {r["seed"] for r in rows}
        print(f"[resume] seeds done: {sorted(done)}", flush=True)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if seed in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng(1000 + seed)
        idx = K.screen(Xtr_s, ytr, args.screen, seed)
        Xtr = np.asarray(Xtr_s[:, idx].todense()); Xe = np.asarray(Xte_s[:, idx].todense())
        n, d = Xe.shape
        model = K.train_model(Xtr, ytr, seed)
        dup = K.pick_terms_to_duplicate(model, Xtr, args.m)
        models, outs = {0: model}, [model.predict_proba(Xe)]
        for k in args.k:
            Xtr_a = np.column_stack([Xtr] + [Xtr[:, dup]] * k)
            models[k] = K.train_model(Xtr_a, ytr, seed)
            outs.append(models[k].predict_proba(np.column_stack([Xe] + [Xe[:, dup]] * k)))
        print(f"seed {seed}: n={n}, d={d}, models trained ({time.time() - t0:.0f}s); streaming meta-Gram", flush=True)
        M = streamed_meta(Xe, outs)
        Mf = M[:d, :d]
        b0 = M[:d, d]
        pre = K.kmcir_from_meta(Mf, b0, args.B, np.random.default_rng(seed))
        hs0 = b0 / (n - 1) ** 2
        ck0 = b0 / (np.sqrt(np.diag(Mf)) * np.sqrt(M[d, d]) + 1e-300)
        pi0 = K.perm_importance(model, Xe, rng) if args.perm else None
        is_dup = np.isin(np.arange(d), dup).astype(int)
        for ki, k in enumerate(args.k):
            ix = np.concatenate([np.arange(d), np.tile(dup, k)])
            M1 = Mf[np.ix_(ix, ix)]
            for setting in args.settings:
                o = d if setting == "fixed" else d + 1 + ki
                b1 = M[ix, o]
                post = K.kmcir_from_meta(M1, b1, args.B, np.random.default_rng(seed))
                hs1 = b1 / (n - 1) ** 2
                ck1 = b1 / (np.sqrt(np.diag(M1)) * np.sqrt(M[o, o]) + 1e-300)

                def summarize(name, pre_v, post_v):
                    rel = (post_v[:d] - pre_v[:d]) / np.maximum(np.abs(pre_v[:d]), 1e-12)
                    rows.append(dict(seed=seed, k=k, p=0.0, setting=setting, method=name, n_eval=n,
                                     dilution=float(np.nanmean(post_v[dup] / np.maximum(pre_v[dup], 1e-12))),
                                     auroc=float(roc_auc_score(is_dup, -np.nan_to_num(rel, nan=0.0)))))

                summarize("KMCIR_U", pre["U"], post["U"])
                summarize("KMCIR_s", pre["s"], post["s"])
                summarize("HSIC", hs0, hs1)
                summarize("CKA", ck0, ck1)
                if args.perm:
                    if setting == "retrain":
                        Xe_a = np.column_stack([Xe] + [Xe[:, dup]] * k)
                        pi1 = K.perm_importance(models[k], Xe_a, rng)
                    else:
                        pi1 = np.concatenate([pi0, np.zeros(len(ix) - d)])
                    summarize("PermImp", pi0, pi1)
                rows.append(dict(seed=seed, k=k, p=0.0, setting=setting, method="_s_orig_pre_post", n_eval=n,
                                 dilution=float(np.nanmean(post["s"][dup])), auroc=float(np.nanmean(pre["s"][dup]))))
        tmp = args.out + ".tmp"
        json.dump(rows, open(tmp, "w"), indent=1); os.replace(tmp, args.out)
        print(f"seed {seed} done in {time.time() - t0:.0f}s -> saved {args.out}", flush=True)
    K.report(rows, argparse.Namespace(screen=args.screen + " | n_eval=ALL", seeds=args.seeds, B=args.B,
                                      n_eval=rows[0]["n_eval"] if rows else "?", out=args.out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--seed-start", dest="seed_start", type=int, default=0)
    r.add_argument("--B", type=int, default=200)
    r.add_argument("--m", type=int, default=10)
    r.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument("--screen", default="lr200")
    r.add_argument("--settings", nargs="+", default=["fixed", "retrain"])
    r.add_argument("--no-perm", dest="perm", action="store_false")
    r.add_argument("--data-home", default=None)
    r.add_argument("--out", default="dil_fulln.json")
    r.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    check() if a.cmd == "check" else run(a)


if __name__ == "__main__":
    main()
