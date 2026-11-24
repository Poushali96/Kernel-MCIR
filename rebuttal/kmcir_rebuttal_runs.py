"""
Kernel-MCIR rebuttal runs: produce the numbers for the [FILL] slots.

  python kmcir_rebuttal_runs.py dilution --seeds 10 --B 200            # 20NG duplicate dilution (P0)
  python kmcir_rebuttal_runs.py dilution --screen df200                # reviewer screening check (also lr500)
  python kmcir_rebuttal_runs.py rff --datasets synthetic 20ng --airquality AirQualityUCI.csv   # Table 8 (P0)
  python kmcir_rebuttal_runs.py selftest                               # offline smoke test, no downloads

Requirements: numpy, scipy, scikit-learn. Optional: `sage-importance` (SAGE baseline).
20 Newsgroups is fetched by sklearn (needs internet once, or pass --data-home to an existing cache).

The kernel and model choices below are generic (RBF + median heuristic per feature, RBF on the
model's predicted-probability vector, sklearn MLP). If your paper's pipeline differs, swap in your
own `feature_gram`, `output_gram` and `train_model`; everything else stays the same.

Core identity (Prop. A): u_i(Phi) = b_i - M_{i,Phi} M_{Phi,Phi}^+ b_Phi, r_i(Phi) = b_i - u_i(Phi),
with M_jk = <K~_j, K~_k>_F and b_j = <K~_j, L~>_F. Permutations are evaluated with an incremental
Cholesky factor of M_{Phi,Phi}: O(r^2) per step.
"""
import argparse
import json
import time

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.metrics import roc_auc_score

# ============================================================== kernels


def _median_bw(x):
    x = np.asarray(x, float).reshape(len(x), -1)
    sub = x[: min(len(x), 1000)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    pos = d2[d2 > 0]
    return float(np.sqrt(np.median(pos) / 2)) if pos.size else 1.0


def rbf_gram(x, bw=None):
    x = np.asarray(x, float).reshape(len(x), -1)
    bw = _median_bw(x) if bw is None else bw
    sq = (x ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0)
    return np.exp(-d2 / (2 * bw ** 2))


def center_gram(K):
    K = K - K.mean(0, keepdims=True)
    return K - K.mean(1, keepdims=True)


def feature_gram(col):
    return center_gram(rbf_gram(col))


def output_gram(Y):
    return center_gram(rbf_gram(Y))


def rff_factor(x, D, bw, rng):
    x = np.asarray(x, float).reshape(len(x), -1)
    W = rng.normal(0, 1.0 / bw, size=(x.shape[1], D))
    c = rng.uniform(0, 2 * np.pi, D)
    Z = np.sqrt(2.0 / D) * np.cos(x @ W + c)
    return Z - Z.mean(0, keepdims=True)                     # column-centred


# ============================================================== meta-Gram


def meta_gram_exact(grams, L):
    """grams: list of centred n x n matrices; L: centred output Gram."""
    G = np.stack([K.ravel() for K in grams])                 # d x n^2
    return G @ G.T, G @ L.ravel()


def meta_gram_rff(factors, Zy):
    """Exact Gram-space inner products of RFF kernels (Eq. 54).
    Uses min(n, D) route: ||Z_j^T Z_k||_F^2 = <Z_j Z_j^T, Z_k Z_k^T>_F."""
    n, D = Zy.shape
    if D >= n:   # cheaper to form the n x n approximate Grams
        grams = [Z @ Z.T for Z in factors]
        return meta_gram_exact(grams, Zy @ Zy.T)
    d = len(factors)
    M = np.empty((d, d))
    for j in range(d):
        for k in range(j, d):
            M[j, k] = M[k, j] = np.linalg.norm(factors[j].T @ factors[k]) ** 2
    b = np.array([np.linalg.norm(Z.T @ Zy) ** 2 for Z in factors])
    return M, b


def kmcir_from_meta(M, b, B, rng, tau=1e-8, return_orders=False):
    """Permutation-symmetrised Kernel-MCIR via incremental Cholesky. Returns Ubar, Rbar, Tbar, sbar."""
    d = len(b)
    U = np.zeros(d); R = np.zeros(d)
    orders = [rng.permutation(d) for _ in range(B)]
    for p in orders:
        Lc = np.zeros((d, d)); beta = np.zeros(d); basis = []
        for i in p:
            r_ = len(basis)
            if r_:
                c = solve_triangular(Lc[:r_, :r_], M[basis, i], lower=True)
                r = float(c @ beta[:r_])
                q = M[i, i] - c @ c
            else:
                c = np.zeros(0); r = 0.0; q = M[i, i]
            u = b[i] - r
            U[i] += abs(u); R[i] += abs(r)
            if q > tau * max(M[i, i], 1e-300):
                sq = np.sqrt(q)
                Lc[r_, :r_] = c; Lc[r_, r_] = sq
                beta[r_] = (b[i] - (c @ beta[:r_] if r_ else 0.0)) / sq
                basis.append(i)
    U /= B; R /= B
    T = U + R
    s = np.divide(U, T, out=np.full(d, np.nan), where=T > 0)
    out = dict(U=U, R=R, T=T, s=s, a=U / T.max() if T.max() > 0 else U * 0)
    if return_orders:
        out["orders"] = orders
    return out


# ============================================================== baselines


def hsic_scores(grams, L):
    return np.array([np.sum(K * L) for K in grams]) / (L.shape[0] - 1) ** 2


def cka_scores(grams, L):
    nl = np.linalg.norm(L)
    return np.array([np.sum(K * L) / (np.linalg.norm(K) * nl + 1e-300) for K in grams])


def perm_importance(model, X, rng, n_repeats=3):
    P0 = model.predict_proba(X); yhat = P0.argmax(1)
    base = P0[np.arange(len(X)), yhat].mean()
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = Xp[rng.permutation(len(X)), j]
            drops.append(base - model.predict_proba(Xp)[np.arange(len(X)), yhat].mean())
        imp[j] = np.mean(drops)
    return np.abs(imp)


def sage_scores(model, Xbg, X, Ylab, thresh=0.1):
    try:
        import sage
    except ImportError:
        return None
    imputer = sage.MarginalImputer(model, Xbg[:64])
    est = sage.PermutationEstimator(imputer, "cross entropy")
    return np.abs(est(X, Ylab, thresh=thresh, verbose=False, bar=False).values)


# ============================================================== 20NG pipeline


def load_20ng(data_home=None, max_features=20000):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), data_home=data_home)
    te = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), data_home=data_home)
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    Xtr = vec.fit_transform(tr.data); Xte = vec.transform(te.data)
    return Xtr, tr.target, Xte, te.target, np.array(vec.get_feature_names_out())


def screen(Xtr, ytr, rule, seed):
    from sklearn.linear_model import LogisticRegression
    if rule.startswith("df"):
        k = int(rule[2:])
        df = np.asarray((Xtr > 0).sum(0)).ravel()
        return np.argsort(-df)[:k]
    k = int(rule[2:])
    lr = LogisticRegression(max_iter=300, C=1.0, random_state=seed).fit(Xtr, ytr)
    return np.argsort(-np.abs(lr.coef_).max(0))[:k]


def train_model(X, y, seed):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                         random_state=seed).fit(X, y)


def make_copy(col, p, rng):
    c = col.copy()
    nz = np.flatnonzero(c)
    if p > 0 and nz.size:
        c[rng.choice(nz, size=int(round(p * nz.size)), replace=False)] = 0.0
    return c


def pick_terms_to_duplicate(model, X, m):
    """Model-based, Kernel-MCIR-independent choice: first-layer weight norm x feature std."""
    W = model.coefs_[0]                                        # n_features x hidden
    return np.argsort(-(np.linalg.norm(W, axis=1) * X.std(0)))[:m]


def run_dilution(args, data=None):
    rng0 = np.random.default_rng(0)
    if data is None:
        data = load_20ng(args.data_home)
    Xtr_s, ytr, Xte_s, yte, _ = data
    rows, done = [], set()
    import os
    if not getattr(args, "fresh", False) and os.path.exists(args.out):
        try:
            rows = json.load(open(args.out))
            done = {r["seed"] for r in rows}
            print(f"[resume] {args.out}: seeds already finished = {sorted(done)}", flush=True)
        except Exception as e:
            print(f"[resume] could not read {args.out} ({e}); starting fresh", flush=True)
            rows, done = [], set()
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if seed in done:
            print(f"seed {seed} already done, skipping", flush=True)
            continue
        rng = np.random.default_rng(1000 + seed)
        idx = screen(Xtr_s, ytr, args.screen, seed)
        Xtr = np.asarray(Xtr_s[:, idx].todense()) if hasattr(Xtr_s, "todense") else Xtr_s[:, idx]
        Xte = np.asarray(Xte_s[:, idx].todense()) if hasattr(Xte_s, "todense") else Xte_s[:, idx]
        model = train_model(Xtr, ytr, seed)
        ev = rng.choice(len(Xte), size=min(args.n_eval, len(Xte)), replace=False)
        Xe = Xte[ev]
        dup = pick_terms_to_duplicate(model, Xtr, args.m)
        d0 = Xe.shape[1]
        grams0 = [feature_gram(Xe[:, j]) for j in range(d0)]
        L0 = output_gram(model.predict_proba(Xe))
        M0, b0 = meta_gram_exact(grams0, L0)
        pre = kmcir_from_meta(M0, b0, args.B, np.random.default_rng(seed))
        hs0, ck0 = hsic_scores(grams0, L0), cka_scores(grams0, L0)
        pi0 = perm_importance(model, Xe, rng) if args.perm else None
        ns = min(args.sage_n, len(Xe))
        sg0 = sage_scores(model, Xtr, Xe[:ns], yte[ev][:ns], args.sage_thresh) if args.sage else None
        for k in args.k:
            for p in args.p:
                copies_tr, copies_e = [], []
                for j in dup:
                    for _ in range(k):
                        mask_rng = np.random.default_rng(rng.integers(1 << 31))
                        copies_tr.append(make_copy(Xtr[:, j], p, mask_rng))
                        copies_e.append(make_copy(Xe[:, j], p, np.random.default_rng(rng.integers(1 << 31))))
                Xtr_a = np.column_stack([Xtr] + copies_tr); Xe_a = np.column_stack([Xe] + copies_e)
                for setting in args.settings:
                    mdl = model if setting == "fixed" else train_model(Xtr_a, ytr, seed)
                    L1 = output_gram(mdl.predict_proba(Xe if setting == "fixed" else Xe_a))
                    grams1 = grams0 + [feature_gram(c) for c in copies_e]
                    M1, b1 = meta_gram_exact(grams1, L1)
                    post = kmcir_from_meta(M1, b1, args.B, np.random.default_rng(seed))
                    is_dup = np.isin(np.arange(d0), dup).astype(int)

                    def summarize(name, pre_v, post_v):
                        if pre_v is None:
                            return
                        rel = (post_v[:d0] - pre_v[:d0]) / np.maximum(np.abs(pre_v[:d0]), 1e-12)
                        rows.append(dict(seed=seed, k=k, p=p, setting=setting, method=name,
                                         dilution=float(np.nanmean(post_v[dup] / np.maximum(pre_v[dup], 1e-12))),
                                         auroc=float(roc_auc_score(is_dup, -np.nan_to_num(rel, nan=0.0)))))

                    summarize("KMCIR_U", pre["U"], post["U"])
                    summarize("KMCIR_s", pre["s"], post["s"])
                    summarize("HSIC", hs0, hsic_scores(grams1, L1))
                    summarize("CKA", ck0, cka_scores(grams1, L1))
                    if args.perm:
                        summarize("PermImp", pi0, perm_importance(mdl, Xe if setting == "fixed" else Xe_a, rng)
                                  if setting == "retrain" else np.concatenate([pi0, np.zeros(len(copies_e))]))
                    if args.sage and sg0 is not None and setting == "retrain" and k == 1 and p == 0:
                        summarize("SAGE", sg0, sage_scores(mdl, Xtr_a, Xe_a[:ns], yte[ev][:ns], args.sage_thresh))
                    rows.append(dict(seed=seed, k=k, p=p, setting=setting, method="_s_orig_pre_post",
                                     dilution=float(np.nanmean(post["s"][dup])), auroc=float(np.nanmean(pre["s"][dup]))))
        tmp = args.out + ".tmp"                      # write-then-rename: a disconnect never corrupts the file
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, args.out)
        print(f"seed {seed} done -> saved {args.out}", flush=True)
    report(rows, args)
    return rows


def report(rows, args):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["setting"], r["k"], r["p"], r["method"])].append((r["dilution"], r["auroc"]))
    print(f"\n=== 20NG duplicate dilution | screen={args.screen} | seeds={args.seeds} | B={args.B} | n_eval={args.n_eval} ===")
    print("setting  k   p     method        dilution (post/pre)    AUROC(dup-detect)   theory 1/(k+1)")
    for key in sorted(g):
        s, k, p, m = key
        v = np.array(g[key])
        if m == "_s_orig_pre_post":
            print(f"{s:8s} {k}  {p:.1f}   s_orig pre->post: {np.mean(v[:,1]):.3f} -> {np.mean(v[:,0]):.3f}")
            continue
        print(f"{s:8s} {k}  {p:.1f}   {m:12s}  {v[:,0].mean():.3f} ± {v[:,0].std():.3f}       "
              f"{v[:,1].mean():.3f} ± {v[:,1].std():.3f}      {1/(k+1):.3f}")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"raw rows -> {args.out}")


# ============================================================== Table 8 (RFF)


def deletion_auc(model, X, ranking, steps=10, Xfill=None):
    Xfill = X.mean(0) if Xfill is None else Xfill
    P0 = model.predict_proba(X); yhat = P0.argmax(1)
    d = X.shape[1]; vals = []
    for t in range(steps + 1):
        k = int(round(t * d / steps)); Xd = X.copy()
        Xd[:, ranking[:k]] = Xfill[ranking[:k]]
        vals.append(model.predict_proba(Xd)[np.arange(len(X)), yhat].mean())
    return float(np.trapezoid(vals, dx=1.0 / steps) if hasattr(np, "trapezoid") else np.trapz(vals, dx=1.0 / steps))


def synthetic_dataset(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)); X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=n)
    y = (np.sin(2 * X[:, 0]) + X[:, 2] ** 2 + 0.3 * rng.normal(size=n) > 1).astype(int)
    return X, y


def run_rff(args):
    from sklearn.model_selection import train_test_split
    sets = {}
    if "synthetic" in args.datasets:
        X, y = synthetic_dataset(seed=0); sets["Synthetic"] = (X, y)
    if "20ng" in args.datasets:
        Xtr_s, ytr, Xte_s, yte, _ = load_20ng(args.data_home)
        idx = screen(Xtr_s, ytr, "lr200", 0)
        sets["20NG"] = (np.asarray(Xtr_s[:, idx].todense()), ytr, np.asarray(Xte_s[:, idx].todense()), yte)
    if args.airquality:
        import pandas as pd
        df = pd.read_csv(args.airquality, sep=";", decimal=",").dropna(axis=1, how="all").dropna()
        num = df.select_dtypes("number").replace(-200, np.nan).dropna()
        y = (num["CO(GT)"] > num["CO(GT)"].median()).astype(int).values
        X = num.drop(columns=["CO(GT)"]).values
        sets["UCI-AirQuality"] = ((X - X.mean(0)) / X.std(0), y)
    print("dataset          D      runtime_s   max|s_RFF-s_exact|   DeletionAUC   (exact: runtime, AUC)")
    out = []
    for name, data in sets.items():
        if len(data) == 2:
            X, y = data
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        else:
            Xtr, ytr, Xte, yte = data
        model = train_model(Xtr, ytr, 0)
        rng = np.random.default_rng(0)
        Xe = Xte[rng.choice(len(Xte), size=min(args.n_eval, len(Xte)), replace=False)]
        P = model.predict_proba(Xe)
        t0 = time.time()
        L = output_gram(P); grams = [feature_gram(Xe[:, j]) for j in range(Xe.shape[1])]
        ex = kmcir_from_meta(*meta_gram_exact(grams, L), args.B, np.random.default_rng(1))
        t_ex = time.time() - t0
        auc_ex = deletion_auc(model, Xe, np.argsort(-ex["a"]))
        bws = [_median_bw(Xe[:, j]) for j in range(Xe.shape[1])]; bwy = _median_bw(P)
        for D in args.D:
            gaps, times, aucs = [], [], []
            for rs in range(args.rff_seeds):
                r = np.random.default_rng(100 + rs); t0 = time.time()
                Zs = [rff_factor(Xe[:, j], D, bws[j], r) for j in range(Xe.shape[1])]
                Zy = rff_factor(P, D, bwy, r)
                ap = kmcir_from_meta(*meta_gram_rff(Zs, Zy), args.B, np.random.default_rng(1))
                times.append(time.time() - t0)
                gaps.append(np.nanmax(np.abs(ap["s"] - ex["s"])))
                aucs.append(deletion_auc(model, Xe, np.argsort(-ap["a"])))
            row = dict(dataset=name, D=D, runtime=np.mean(times), gap=np.mean(gaps), gap_sd=np.std(gaps),
                       auc=np.mean(aucs), auc_sd=np.std(aucs), runtime_exact=t_ex, auc_exact=auc_ex)
            out.append(row)
            print(f"{name:15s} {D:5d}   {row['runtime']:8.2f}     {row['gap']:.3f} ± {row['gap_sd']:.3f}"
                  f"        {row['auc']:.3f} ± {row['auc_sd']:.3f}   ({t_ex:.2f}s, {auc_ex:.3f})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"raw rows -> {args.out}")


# ============================================================== self-test (offline)


def selftest():
    from scipy import sparse
    print("[selftest] 1) incremental-Cholesky K-MCIR == pinv formula, reviewer example")
    n = 60; rng = np.random.default_rng(0)
    a = rng.normal(size=n); a -= a.mean(); a /= np.linalg.norm(a)
    bb = rng.normal(size=n); bb -= bb.mean(); bb -= (bb @ a) * a; bb /= np.linalg.norm(bb)
    w = (a + bb) / np.sqrt(2)
    grams = [np.outer(a, a), np.outer(w, w)]; L = np.outer(w, w)
    M, b = meta_gram_exact(grams, L)
    # order (0,1) only: s_1 should be 0.75
    Lc = np.sqrt(M[0, 0]); c = M[0, 1] / Lc; r = c * (b[0] / Lc); u = b[1] - r
    print(f"   s_1 | Phi={{0}} = {abs(u)/(abs(u)+abs(r)):.3f} (expect 0.750)")
    print("[selftest] 2) exact-duplicate dilution 1/(k+1) with the fast estimator")
    X = rng.normal(size=(300, 4)); y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    L = output_gram(y); g = [feature_gram(X[:, j]) for j in range(4)]
    pre = kmcir_from_meta(*meta_gram_exact(g, L), 400, np.random.default_rng(1))
    for k in (1, 2, 3):
        post = kmcir_from_meta(*meta_gram_exact(g + [g[0]] * k, L), 400, np.random.default_rng(1))
        print(f"   k={k}: Ubar post/pre = {post['U'][0]/pre['U'][0]:.3f} (theory {1/(k+1):.3f}); "
              f"s_orig={post['s'][0]:.3f}, s_copy={post['s'][4]:.3f} (equal up to MC error)")
    print("[selftest] 3) RFF meta-Gram converges to exact")
    P = y.reshape(-1, 1); bws = [_median_bw(X[:, j]) for j in range(4)]
    ex = kmcir_from_meta(*meta_gram_exact(g, L), 100, np.random.default_rng(1))
    for D in (32, 128, 512):
        r = np.random.default_rng(5)
        ap = kmcir_from_meta(*meta_gram_rff([rff_factor(X[:, j], D, bws[j], r) for j in range(4)],
                                             rff_factor(P, D, _median_bw(P), r)), 100, np.random.default_rng(1))
        print(f"   D={D}: max|s_RFF - s_exact| = {np.nanmax(np.abs(ap['s'] - ex['s'])):.3f}")
    print("[selftest] 4) full dilution pipeline on a tiny fake sparse 'text' dataset")
    ntr, nte, V = 1500, 400, 300
    Xtr = sparse.random(ntr, V, density=0.05, random_state=1, format="csr")
    Xte = sparse.random(nte, V, density=0.05, random_state=2, format="csr")
    wv = rng.normal(size=(V, 4))
    ytr = np.asarray(Xtr @ wv).argmax(1); yte = np.asarray(Xte @ wv).argmax(1)
    a_ = argparse.Namespace(seed_start=0, seeds=2, B=30, n_eval=150, m=5, k=[1, 2], p=[0.0, 0.3], screen="lr40",
                            settings=["fixed", "retrain"], perm=True, sage=False, sage_n=200, sage_thresh=0.1, fresh=True, out="/tmp/selftest_rows.json",
                            data_home=None)
    run_dilution(a_, data=(Xtr, ytr, Xte, yte, None))


# ============================================================== CLI


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dilution")
    d.add_argument("--seeds", type=int, default=10)
    d.add_argument("--seed-start", dest="seed_start", type=int, default=0)
    d.add_argument("--B", type=int, default=200)
    d.add_argument("--n-eval", dest="n_eval", type=int, default=500)
    d.add_argument("--m", type=int, default=10, help="number of terms to duplicate")
    d.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    d.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.3], help="dropout of nonzeros in copies")
    d.add_argument("--screen", default="lr200", help="lr200 | df200 | lr500")
    d.add_argument("--settings", nargs="+", default=["fixed", "retrain"])
    d.add_argument("--no-perm", dest="perm", action="store_false")
    d.add_argument("--sage", action="store_true", help="SAGE only for k=1, p=0, retrained model (it is slow)")
    d.add_argument("--sage-n", dest="sage_n", type=int, default=200, help="eval points used by SAGE")
    d.add_argument("--sage-thresh", dest="sage_thresh", type=float, default=0.1, help="SAGE convergence threshold")
    d.add_argument("--data-home", default=None)
    d.add_argument("--out", default="dilution_rows.json")
    d.add_argument("--fresh", action="store_true", help="ignore an existing --out file and start over")
    r = sub.add_parser("rff")
    r.add_argument("--datasets", nargs="+", default=["synthetic", "20ng"])
    r.add_argument("--airquality", default=None, help="path to AirQualityUCI.csv")
    r.add_argument("--D", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    r.add_argument("--B", type=int, default=200)
    r.add_argument("--n-eval", dest="n_eval", type=int, default=500)
    r.add_argument("--rff-seeds", dest="rff_seeds", type=int, default=3)
    r.add_argument("--data-home", default=None)
    r.add_argument("--out", default="table8_rows.json")
    sub.add_parser("selftest")
    rp = sub.add_parser("report", help="merge saved dilution json files and print the table")
    rp.add_argument("files", nargs="+")
    rp.add_argument("--label", default="merged")
    a = ap.parse_args()
    if a.cmd == "report":
        rows = sum((json.load(open(f)) for f in a.files), [])
        report(rows, argparse.Namespace(screen=a.label, seeds=len({r["seed"] for r in rows}), B="?", n_eval="?",
                                        out=f"{a.label}_merged.json"))
        return
    {"dilution": run_dilution, "rff": run_rff, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
