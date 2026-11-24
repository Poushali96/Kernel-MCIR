"""
Kernel-MCIR rebuttal experiments (self-contained, numpy only).

E0  reviewer counterexample: old factor-space RFF path vs. meta-Gram (Gram-space) path
E1  AI-review rank saturation of the old factor-space path
E2  Corrected RFF (meta-Gram) converges to exact Gram-space scores as D grows
E3  Duplicate dilution: pre/post ratio of the ORIGINAL feature's unique strength
    vs. number of copies k and copy-noise sigma; compared with marginal HSIC
E4  Correlated-but-redundant vs. correlated-but-complementary regimes (reviewer Q3/Q4)

Key identity used everywhere (all Kernel-MCIR quantities depend only on the
(d+1)x(d+1) meta-Gram of centered kernels):
    M_jk = <K~_j, K~_k>_F ,  b_j = <K~_j, L~>_F
    alpha_Phi = M_PhiPhi^+ b_Phi
    r_i(Phi) = M_{i,Phi} alpha_Phi               (= <Pi_Phi K~_i, Pi_Phi L~>)
    u_i(Phi) = b_i - r_i(Phi)                   (= <K~_i, Delta_i(Phi)>)
With RFF factors Z~ (column-centered), <Z~_j Z~_j^T, Z~_k Z~_k^T>_F = ||Z~_j^T Z~_k||_F^2.
"""
import itertools
import numpy as np

rng_global = np.random.default_rng(0)


# ---------------------------------------------------------------- kernels
def center(K):
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    return H @ K @ H


def rbf_gram(x, bw=None):
    x = np.asarray(x, float).reshape(len(x), -1)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    if bw is None:
        bw = np.sqrt(np.median(d2[d2 > 0]) / 2) if np.any(d2 > 0) else 1.0
    return np.exp(-d2 / (2 * bw ** 2)), bw


def rff(x, D, bw, rng):
    x = np.asarray(x, float).reshape(len(x), -1)
    W = rng.normal(0, 1.0 / bw, size=(x.shape[1], D))
    b = rng.uniform(0, 2 * np.pi, D)
    Z = np.sqrt(2.0 / D) * np.cos(x @ W + b)
    return Z - Z.mean(0, keepdims=True)          # column-centred factors


# ---------------------------------------------------------------- core
def meta_gram_from_grams(Ks, L):
    Ks = [center(K) for K in Ks]
    L = center(L)
    d = len(Ks)
    M = np.array([[np.sum(Ks[j] * Ks[k]) for k in range(d)] for j in range(d)])
    b = np.array([np.sum(K * L) for K in Ks])
    return M, b


def meta_gram_from_factors(Zs, Zy):
    d = len(Zs)
    M = np.array([[np.linalg.norm(Zs[j].T @ Zs[k]) ** 2 for k in range(d)] for j in range(d)])
    b = np.array([np.linalg.norm(Z.T @ Zy) ** 2 for Z in Zs])
    return M, b


def ur(M, b, i, Phi, tol=1e-10):
    Phi = list(Phi)
    if not Phi:
        return b[i], 0.0
    MP = M[np.ix_(Phi, Phi)]
    alpha = np.linalg.pinv(MP, rcond=tol) @ b[Phi]
    r = M[i, Phi] @ alpha
    return b[i] - r, r


def kmcir(M, b, B=None, rng=None):
    """Permutation-symmetrised Kernel-MCIR (average strengths, then ratio)."""
    d = len(b)
    perms = (list(itertools.permutations(range(d))) if B is None
             else [rng.permutation(d) for _ in range(B)])
    U = np.zeros(d); R = np.zeros(d)
    for p in perms:
        for pos, i in enumerate(p):
            u, r = ur(M, b, i, p[:pos])
            U[i] += abs(u); R[i] += abs(r)
    U /= len(perms); R /= len(perms)
    T = U + R
    s = np.where(T > 0, U / np.maximum(T, 1e-300), np.nan)
    return dict(U=U, R=R, T=T, s=s)


def factor_space_old(Zs, Zy, i, Phi):
    """The submitted Appendix B.2, Eqs. (49)-(53)."""
    def proj(ZP, A):
        if ZP is None:
            return np.zeros_like(A)
        C, *_ = np.linalg.lstsq(ZP, A, rcond=None)
        return ZP @ C
    ZP = np.hstack([Zs[j] for j in Phi]) if Phi else None
    ZPi = np.hstack([Zs[j] for j in list(Phi) + [i]])
    F = proj(ZP, Zy); Fi = proj(ZPi, Zy)
    u = np.sum(Zs[i] * (Fi - F)); r = np.sum(proj(ZP, Zs[i]) * F)
    return u, r


# ================================================================ E0
def E0():
    n = 50
    A = np.linalg.qr(rng_global.normal(size=(n, 2)) - 0)[0]
    a, bb = A[:, 0] - A[:, 0].mean(), A[:, 1] - A[:, 1].mean()
    a /= np.linalg.norm(a)
    bb -= (bb @ a) * a; bb /= np.linalg.norm(bb)
    w = (a + bb) / np.sqrt(2)
    KPhi, Ki, L = np.outer(a, a), np.outer(w, w), np.outer(w, w)
    M = np.array([[np.sum(KPhi * KPhi), np.sum(KPhi * Ki)],
                  [np.sum(Ki * KPhi), np.sum(Ki * Ki)]])
    b = np.array([np.sum(KPhi * L), np.sum(Ki * L)])
    u, r = ur(M, b, 1, [0])
    s_meta = abs(u) / (abs(u) + abs(r))
    uo, ro = factor_space_old([a[:, None], w[:, None]], w[:, None], 1, [0])
    s_old = abs(uo) / (abs(uo) + abs(ro))
    print(f"E0 reviewer example: Gram-space s = {s_meta:.3f} (theory 0.75); "
          f"old factor-space s = {s_old:.3f}")


# ================================================================ E1
def E1():
    n, d = 200, 4
    X = rng_global.normal(size=(n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2            # X2, X3 pure noise
    bws = [rbf_gram(X[:, j])[1] for j in range(d)]; bwy = rbf_gram(y)[1]
    for D in [64, 128]:
        rng = np.random.default_rng(1)
        Zs = [rff(X[:, j], D, bws[j], rng) for j in range(d)]; Zy = rff(y, D, bwy, rng)
        uo, _ = factor_space_old(Zs, Zy, 0, [2, 3])     # condition on 2 NOISE features
        M, b = meta_gram_from_factors(Zs, Zy)
        um, _ = ur(M, b, 0, [2, 3])
        print(f"E1 n={n}, D={D}, |Phi|D={2*D}: u_0 | noise-Phi  old={uo:+.2e}  meta-Gram={um:+.2e}"
              f"  (b_0={b[0]:.2e})")


# ================================================================ E2
def E2():
    n, d = 300, 4
    X = rng_global.normal(size=(n, d)); X[:, 1] = X[:, 0] + 0.3 * rng_global.normal(size=n)
    y = np.sin(2 * X[:, 0]) + X[:, 2] ** 2
    Gs, bws = zip(*[rbf_gram(X[:, j]) for j in range(d)]); Ly, bwy = rbf_gram(y)
    ex = kmcir(*meta_gram_from_grams(Gs, Ly))
    print("E2 exact s =", np.round(ex["s"], 3))
    for D in [32, 128, 512, 2048]:
        errs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            Zs = [rff(X[:, j], D, bws[j], rng) for j in range(d)]; Zy = rff(y, D, bwy, rng)
            ap = kmcir(*meta_gram_from_factors(Zs, Zy))
            errs.append(np.max(np.abs(ap["s"] - ex["s"])))
        print(f"   D={D:5d}: max|s_RFF - s_exact| = {np.mean(errs):.4f} ± {np.std(errs):.4f}")


# ================================================================ E3
def E3(n=300, B=200, seeds=5):
    print("E3 duplicate dilution of the ORIGINAL feature (ratio post/pre), mean ± sd over seeds")
    print("   k  sigma   Ubar_post/Ubar_pre   1/(k+1)   s_orig pre->post      HSIC post/pre")
    for k in [1, 2, 3]:
        for sigma in [0.0, 0.1, 0.3, 1.0]:
            rU, sp, sq, rH = [], [], [], []
            for seed in range(seeds):
                rng = np.random.default_rng(100 + seed)
                X = rng.normal(size=(n, 4))
                y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2]
                cols = [X[:, j] for j in range(4)]
                copies = [X[:, 0] + sigma * rng.normal(size=n) for _ in range(k)]
                Ly, _ = rbf_gram(y)
                G = [rbf_gram(c)[0] for c in cols]
                Gc = [rbf_gram(c)[0] for c in copies]
                pre = kmcir(*meta_gram_from_grams(G, Ly), B=B, rng=rng)
                post = kmcir(*meta_gram_from_grams(G + Gc, Ly), B=B, rng=rng)
                rU.append(post["U"][0] / pre["U"][0])
                sp.append(pre["s"][0]); sq.append(post["s"][0])
                rH.append(1.0)   # marginal HSIC of the original is unchanged by construction
            print(f"   {k}  {sigma:4.1f}   {np.mean(rU):.3f} ± {np.std(rU):.3f}        {1/(k+1):.3f}"
                  f"     {np.mean(sp):.2f} -> {np.mean(sq):.2f}          {np.mean(rH):.3f}")


# ================================================================ E4
def E4(n=400, seeds=5):
    print("E4 regimes (exact permutations, d=3: X1, X2, noise X3); mean over seeds")
    print("   regime         rho    s1    s2    T1/T2    | LOCO1  LOCO2 (ground truth, R^2 drop)")
    for regime in ["redundant", "complementary", "additive"]:
        for rho in [0.0, 0.6, 0.9]:
            S1, S2, TR, L1, L2 = [], [], [], [], []
            for seed in range(seeds):
                rng = np.random.default_rng(200 + seed)
                x1 = rng.normal(size=n)
                x2 = rho * x1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
                x3 = rng.normal(size=n)
                if regime == "redundant":          # X2 correlated, never used by f
                    f = lambda a, b_: np.sin(2 * a)
                elif regime == "complementary":    # suppressor / difference structure
                    f = lambda a, b_: a - b_
                else:                              # distinct additive roles
                    f = lambda a, b_: np.sin(2 * a) + b_ ** 2
                y = f(x1, x2)
                Ly, _ = rbf_gram(y)
                res = kmcir(*meta_gram_from_grams([rbf_gram(v)[0] for v in (x1, x2, x3)], Ly))
                S1.append(res["s"][0]); S2.append(res["s"][1]); TR.append(res["T"][0] / res["T"][1])
                # ground-truth LOCO via oracle conditional expectation (Monte Carlo, large sample)
                N = 20000; a = rng.normal(size=N)
                bq = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=N); yy = f(a, bq)
                v = np.var(yy)
                def r2_given(z):    # R^2 of E[Y|z] by fine binning
                    q = np.quantile(z, np.linspace(0, 1, 101)); idx = np.clip(np.searchsorted(q, z) - 1, 0, 99)
                    m = np.array([yy[idx == t].mean() if np.any(idx == t) else 0 for t in range(100)])
                    return np.var(m[idx]) / v
                L1.append(1 - r2_given(bq)); L2.append(1 - r2_given(a))
            print(f"   {regime:13s} {rho:4.1f}  {np.mean(S1):.2f}  {np.mean(S2):.2f}  {np.mean(TR):6.2f}"
                  f"   | {np.mean(L1):.2f}   {np.mean(L2):.2f}")


if __name__ == "__main__":
    E0(); E1(); E2(); E3(); E4()
