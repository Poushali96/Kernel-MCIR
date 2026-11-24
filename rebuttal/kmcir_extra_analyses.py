"""
Extra rebuttal analyses for Kernel-MCIR: relatives of Kernel-MCIR on the same sampled orders.

  energy  (kernel LMG / Shapley of v(S)=||Pi_S L~||^2):  E_i = E_pi[ u_i^2 / ||Q_i||^2 ]   (reviewer Q2(b), reviewer Q2)
  seqHSIC (sequential HSIC gain / marginal HSIC):        G_i = E_pi[ HSIC(K_{Phi+i}) - HSIC(K_Phi) ] / HSIC(K_i)
          with the joint kernel of a set = product of the per-feature RBF kernels (HSIC forward selection) (reviewer Q2(a))
  pdCov   (partial distance covariance, Szekely & Rizzo 2014): pdCov(X_i, Y ; X_{-i}), U-centred estimator

  python kmcir_extra_analyses.py synthetic --out local_runs/extra_synthetic.json
  python kmcir_extra_analyses.py text --seeds 5 --data-home DATA --out local_runs/extra_20ng.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmcir_rebuttal_runs as K  # noqa: E402


# ------------------------------------------------------------------ estimators on shared orders

def kmcir_energy(M, b, orders, tau=1e-8):
    """Kernel-MCIR (U, R, s) and order-averaged projection energy u^2/q, on the given orders."""
    d = len(b); U = np.zeros(d); R = np.zeros(d); E = np.zeros(d)
    for p in orders:
        Lc = np.zeros((d, d)); beta = np.zeros(d); basis = []
        for i in p:
            r_ = len(basis)
            if r_:
                c = solve_triangular(Lc[:r_, :r_], M[basis, i], lower=True)
                r = float(c @ beta[:r_]); q = M[i, i] - c @ c
            else:
                c = np.zeros(0); r = 0.0; q = M[i, i]
            u = b[i] - r
            U[i] += abs(u); R[i] += abs(r)
            if q > tau * max(M[i, i], 1e-300):
                E[i] += u * u / q
                sq = np.sqrt(q); Lc[r_, :r_] = c; Lc[r_, r_] = sq
                beta[r_] = (b[i] - (c @ beta[:r_] if r_ else 0.0)) / sq
                basis.append(i)
    B = len(orders); U /= B; R /= B; E /= B
    T = U + R
    s = np.divide(U, T, out=np.full(d, np.nan), where=T > 0)
    return dict(U=U, R=R, T=T, s=s, E=E)


def seq_hsic_gain(Kraw, Lc, orders):
    """Order-averaged gain of HSIC(product kernel of Phi, L) when i is added. <P, L~> = <HPH, L~> as L~ is centred."""
    d = len(Kraw); n = Lc.shape[0]; G = np.zeros(d)
    P = np.empty((n, n))
    for p in orders:
        P.fill(1.0); prev = 0.0
        for i in p:
            P *= Kraw[i]
            h = float(np.vdot(P, Lc))
            G[i] += h - prev; prev = h
    return G / len(orders)


def _ucenter(D):
    n = D.shape[0]
    A = D - D.sum(0, keepdims=True) / (n - 2) - D.sum(1, keepdims=True) / (n - 2) + D.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(A, 0.0)
    return A


def _uin(A, B):
    n = A.shape[0]
    return float(np.vdot(A, B)) / (n * (n - 3))


def _dist(x):
    x = np.asarray(x, float).reshape(len(x), -1)
    sq = (x ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0))


def pdcov_all(X, Y):
    """pdCov(X_i, Y ; X_{-i}) for every column i (U-centred estimator, Szekely & Rizzo 2014)."""
    n, d = X.shape
    Ay = _ucenter(_dist(Y))
    xs = np.asarray(X, float)
    sq = (xs ** 2).sum(1); D2all = np.maximum(sq[:, None] + sq[None, :] - 2 * xs @ xs.T, 0)
    out = np.zeros(d)
    for i in range(d):
        di = xs[:, i][:, None] - xs[:, i][None, :]
        Ax = _ucenter(np.abs(di))
        Az = _ucenter(np.sqrt(np.maximum(D2all - di ** 2, 0)))
        zz = _uin(Az, Az)
        if zz <= 0:
            out[i] = _uin(Ax, Ay); continue
        Px = Ax - _uin(Ax, Az) / zz * Az
        Py = Ay - _uin(Ay, Az) / zz * Az
        out[i] = _uin(Px, Py)
    return out


def all_scores(X, Y, orders):
    """Feature RBF grams (median heuristic), RBF output gram, then every score on the same orders."""
    Kraw = [K.rbf_gram(X[:, j]) for j in range(X.shape[1])]
    Lc = K.center_gram(K.rbf_gram(Y))
    M, b = K.meta_gram_exact([K.center_gram(Kj) for Kj in Kraw], Lc)
    out = kmcir_energy(M, b, orders)
    out["Eshare"] = out["E"] / max(out["E"].sum(), 1e-300)
    out["HSIC"] = b.copy()
    out["seqG"] = seq_hsic_gain(Kraw, Lc, orders)
    out["seqRatio"] = out["seqG"] / np.where(np.abs(b) > 0, b, np.nan)
    out["pdCov"] = np.abs(pdcov_all(X, Y))
    return out


METHODS = ["U", "E", "seqG", "pdCov", "HSIC"]


# ------------------------------------------------------------------ synthetic duplicate test (E3 generator)

def synthetic(args):
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(100 + seed)
        X = rng.normal(size=(args.n, 4))
        y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2]
        pre_orders = [rng.permutation(4) for _ in range(args.B)]
        pre = all_scores(X, y, pre_orders)
        for k in (1, 2, 3):
            Xa = np.column_stack([X] + [X[:, 0]] * k)
            post = all_scores(Xa, y, [rng.permutation(4 + k) for _ in range(args.B)])
            row = dict(seed=seed, k=k)
            for m in METHODS:
                row[f"dil_{m}"] = float(post[m][0] / pre[m][0])
            if k == 1:   # reviewer Q2: mean ranks per feature (1 = highest) of s, (a) seqRatio, (b) energy share, plus U
                for name in ("s", "seqRatio", "Eshare", "U", "pdCov"):
                    row[f"rank_{name}"] = rankdata(-np.nan_to_num(post[name], nan=-1e9)).tolist()
                    row[f"val_{name}"] = np.nan_to_num(post[name]).tolist()
            rows.append(row)
        print(f"seed {seed} done", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== synthetic duplicate test (n=%d, B=%d, %d seeds): dilution of the ORIGINAL, post/pre ===" % (args.n, args.B, args.seeds))
    print(df.groupby("k")[[f"dil_{m}" for m in METHODS]].agg(["mean", "std"]).round(3).to_string())
    print("\nmean rank per feature, k=1 (features: X0 generator, X1 generator, X2 weak generator, X3 noise, X4 = copy of X0)")
    one = df[df.k == 1]
    for name in ("s", "seqRatio", "Eshare", "U", "pdCov"):
        R = np.array(one[f"rank_{name}"].tolist()); V = np.array(one[f"val_{name}"].tolist())
        print(f"  {name:9s} ranks {np.round(R.mean(0), 2).tolist()}   values {np.round(V.mean(0), 3).tolist()}")


# ------------------------------------------------------------------ 20NG

def text(args):
    Xtr_s, ytr, Xte_s, yte, _ = K.load_20ng(args.data_home)
    rows = []
    if os.path.exists(args.out) and not args.fresh:
        rows = json.load(open(args.out))
    done = {r["seed"] for r in rows}
    for seed in range(args.seeds):
        if seed in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng(1000 + seed)
        idx = K.screen(Xtr_s, ytr, "lr200", seed)
        Xtr = np.asarray(Xtr_s[:, idx].todense()); Xte = np.asarray(Xte_s[:, idx].todense())
        model = K.train_model(Xtr, ytr, seed)
        ev = rng.choice(len(Xte), size=min(args.n_eval, len(Xte)), replace=False)
        Xe = Xte[ev]; d = Xe.shape[1]
        dup = K.pick_terms_to_duplicate(model, Xtr, 10)
        orng = np.random.default_rng(seed)
        pre = all_scores(Xe, model.predict_proba(Xe), [orng.permutation(d) for _ in range(args.B)])
        row = dict(seed=seed)
        ok = np.isfinite(pre["seqRatio"]) & np.isfinite(pre["s"])
        row["spearman_s_seqRatio"] = float(spearmanr(pre["s"][ok], pre["seqRatio"][ok]).correlation)
        row["spearman_s_Eshare"] = float(spearmanr(pre["s"][ok], pre["Eshare"][ok]).correlation)
        row["n_s_defined"] = int(ok.sum())
        row["spearman_U_seqG"] = float(spearmanr(pre["U"], pre["seqG"]).correlation)
        row["spearman_U_E"] = float(spearmanr(pre["U"], pre["E"]).correlation)
        row["spearman_U_pdCov"] = float(spearmanr(pre["U"], pre["pdCov"]).correlation)
        # duplicate dilution, k=1, p=0, MLP retrained on the augmented input
        Xtr_a = np.column_stack([Xtr, Xtr[:, dup]]); Xe_a = np.column_stack([Xe, Xe[:, dup]])
        mdl = K.train_model(Xtr_a, ytr, seed)
        post = all_scores(Xe_a, mdl.predict_proba(Xe_a), [orng.permutation(d + len(dup)) for _ in range(args.B)])
        is_dup = np.isin(np.arange(d), dup).astype(int)
        for m in METHODS:
            row[f"dil_{m}"] = float(np.mean(post[m][dup] / np.maximum(np.abs(pre[m][dup]), 1e-300)))
            rel = (post[m][:d] - pre[m][:d]) / np.maximum(np.abs(pre[m][:d]), 1e-300)
            row[f"auroc_{m}"] = float(roc_auc_score(is_dup, -np.nan_to_num(rel)))
        rows.append(row)
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n=== 20NG (LR-200, n_eval={args.n_eval}, B={args.B}, {len(df)} seeds) ===")
    print(df.drop(columns="seed").agg(["mean", "std"]).T.round(3).to_string())


def deletion(args):
    """Re-run of the appendix 20NG text table under the revised estimator, every row labelled:
    deletion AUC (mean-fill, predicted-class probability, 10 steps; lower = more faithful), runtime,
    Spearman and top-30 overlap with the K-MCIR scalar ranking a = U/max T."""
    from sklearn.feature_selection import mutual_info_classif
    Xtr_s, ytr, Xte_s, yte, _ = K.load_20ng(args.data_home)
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        idx = K.screen(Xtr_s, ytr, "lr200", seed)
        Xtr = np.asarray(Xtr_s[:, idx].todense()); Xte = np.asarray(Xte_s[:, idx].todense())
        model = K.train_model(Xtr, ytr, seed)
        acc = float((model.predict(Xte) == yte).mean())
        Xe = Xte[rng.choice(len(Xte), size=min(args.n_eval, len(Xte)), replace=False)]
        P = model.predict_proba(Xe); fill = Xtr.mean(0)
        t0 = time.time()
        grams = [K.feature_gram(Xe[:, j]) for j in range(Xe.shape[1])]; L = K.output_gram(P)
        t_gram = time.time() - t0
        t0 = time.time(); M, b = K.meta_gram_exact(grams, L)
        km = K.kmcir_from_meta(M, b, args.B, np.random.default_rng(seed)); t_km = time.time() - t0 + t_gram
        scores, times = {}, {}
        scores["K-MCIR $a=\\bar U/\\max\\bar T$"] = km["a"]; times["K-MCIR $a=\\bar U/\\max\\bar T$"] = t_km
        scores["K-MCIR $\\bar s$"] = np.nan_to_num(km["s"], nan=-1.0); times["K-MCIR $\\bar s$"] = t_km
        scores["K-MCIR $\\bar R$"] = km["R"]; times["K-MCIR $\\bar R$"] = t_km
        t0 = time.time(); scores["HSIC"] = K.hsic_scores(grams, L); times["HSIC"] = time.time() - t0 + t_gram
        t0 = time.time(); scores["CKA"] = K.cka_scores(grams, L); times["CKA"] = time.time() - t0 + t_gram
        t0 = time.time(); scores["Permutation"] = K.perm_importance(model, Xe, rng); times["Permutation"] = time.time() - t0
        t0 = time.time()
        scores["Mutual info"] = mutual_info_classif(Xe, P.argmax(1), random_state=seed); times["Mutual info"] = time.time() - t0
        scores["Random"] = rng.random(Xe.shape[1]); times["Random"] = 0.0
        ref = scores["K-MCIR $a=\\bar U/\\max\\bar T$"]; top_ref = set(np.argsort(-ref)[:30])
        for name, s in scores.items():
            order = np.argsort(-s)
            rows.append(dict(seed=seed, method=name, mlp_acc=acc, runtime=times[name],
                             del_auc=K.deletion_auc(model, Xe, order, Xfill=fill),
                             spearman=float(spearmanr(s, ref).correlation),
                             top30=len(top_ref & set(order[:30])) / 30))
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"seed {seed} done (MLP acc {acc:.3f})", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby("method")[["del_auc", "runtime", "spearman", "top30"]].agg(["mean", "std"]).round(3)
          .sort_values(("del_auc", "mean")).to_string())
    print("MLP test accuracy:", round(df.groupby("seed").mlp_acc.first().mean(), 3), "+-", round(df.groupby("seed").mlp_acc.first().std(ddof=0), 3))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("deletion")
    dl.add_argument("--seeds", type=int, default=10); dl.add_argument("--B", type=int, default=200)
    dl.add_argument("--n-eval", dest="n_eval", type=int, default=500)
    dl.add_argument("--data-home", default=None); dl.add_argument("--out", default="text_deletion.json")
    s = sub.add_parser("synthetic")
    s.add_argument("--n", type=int, default=300); s.add_argument("--B", type=int, default=200)
    s.add_argument("--seeds", type=int, default=5); s.add_argument("--out", default="extra_synthetic.json")
    t = sub.add_parser("text")
    t.add_argument("--seeds", type=int, default=5); t.add_argument("--B", type=int, default=200)
    t.add_argument("--n-eval", dest="n_eval", type=int, default=500)
    t.add_argument("--data-home", default=None); t.add_argument("--out", default="extra_20ng.json")
    t.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    {"synthetic": synthetic, "text": text, "deletion": deletion}[a.cmd](a)


if __name__ == "__main__":
    main()
