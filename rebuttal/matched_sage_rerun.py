"""Matched re-run for the SAGE comparison (k=1, p=0, retrained MLP).

In run_dilution, Kernel-MCIR and permutation importance score the 500 evaluation documents of each seed, while SAGE scores
the first 200 of them (Xe[:sage_n]). This script replays the identical random stream of run_dilution (same screening,
evaluation sample, duplicated terms, copies and retrained model) and scores Kernel-MCIR and permutation importance on
exactly the 200 documents used by SAGE. SAGE values are taken from local_runs/dil_sage*.json (unchanged).
Sanity check: Kernel-MCIR on all 500 documents must reproduce the stored KMCIR_U values.

python matched_sage_rerun.py --data-home <dir with 20news-bydate_py3.pkz> --seeds 10
  -> local_runs/matched_sage.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kmcir_rebuttal_runs as K  # noqa: E402


def dil_auroc(pre, post, dup, d0):
    is_dup = np.isin(np.arange(d0), dup).astype(int)
    rel = (post[:d0] - pre[:d0]) / np.maximum(np.abs(pre[:d0]), 1e-12)
    return (float(np.nanmean(post[dup] / np.maximum(pre[dup], 1e-12))),
            float(roc_auc_score(is_dup, -np.nan_to_num(rel, nan=0.0))))


def kmcir_U(Xe, Xe_a, mdl0, mdl1, B, seed):
    d0 = Xe.shape[1]
    g0 = [K.feature_gram(Xe[:, j]) for j in range(d0)]
    M0, b0 = K.meta_gram_exact(g0, K.output_gram(mdl0.predict_proba(Xe)))
    g1 = g0 + [K.feature_gram(Xe_a[:, j]) for j in range(d0, Xe_a.shape[1])]
    M1, b1 = K.meta_gram_exact(g1, K.output_gram(mdl1.predict_proba(Xe_a)))
    return (K.kmcir_from_meta(M0, b0, B, np.random.default_rng(seed))["U"],
            K.kmcir_from_meta(M1, b1, B, np.random.default_rng(seed))["U"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--sage-n", type=int, default=200)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "local_runs", "matched_sage.json"))
    a = ap.parse_args()
    stored = []
    for f in sorted(glob.glob(os.path.join(HERE, "local_runs", "dil_sage*.json"))):
        stored += json.load(open(f))
    sel = lambda s, m: [r for r in stored if r["seed"] == s and r["method"] == m and r["setting"] == "retrain"
                        and r["k"] == 1 and r["p"] == 0]
    Xtr_s, ytr, Xte_s, yte, _ = K.load_20ng(a.data_home)
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)                       # identical stream to run_dilution
        idx = K.screen(Xtr_s, ytr, "lr200", seed)
        Xtr = np.asarray(Xtr_s[:, idx].todense()); Xte = np.asarray(Xte_s[:, idx].todense())
        model = K.train_model(Xtr, ytr, seed)
        ev = rng.choice(len(Xte), size=min(a.n_eval, len(Xte)), replace=False)
        Xe = Xte[ev]
        dup = K.pick_terms_to_duplicate(model, Xtr, a.m)
        d0 = Xe.shape[1]
        K.perm_importance(model, Xe, rng)                              # consumes rng exactly as in run_dilution
        copies_tr, copies_e = [], []
        for j in dup:                                                  # k = 1, p = 0 is the first (k, p) cell
            copies_tr.append(K.make_copy(Xtr[:, j], 0.0, np.random.default_rng(rng.integers(1 << 31))))
            copies_e.append(K.make_copy(Xe[:, j], 0.0, np.random.default_rng(rng.integers(1 << 31))))
        Xtr_a = np.column_stack([Xtr] + copies_tr); Xe_a = np.column_stack([Xe] + copies_e)
        mdl = K.train_model(Xtr_a, ytr, seed)
        # sanity: 500 documents reproduce the stored value
        pre5, post5 = kmcir_U(Xe, Xe_a, model, mdl, a.B, seed)
        d5, au5 = dil_auroc(pre5, post5, dup, d0)
        ns = a.sage_n
        pre2, post2 = kmcir_U(Xe[:ns], Xe_a[:ns], model, mdl, a.B, seed)
        d2, au2 = dil_auroc(pre2, post2, dup, d0)
        prng = np.random.default_rng(5000 + seed)
        pi_pre, pi_post = K.perm_importance(model, Xe[:ns], prng), K.perm_importance(mdl, Xe_a[:ns], prng)
        dp, aup = dil_auroc(pi_pre, pi_post, dup, d0)
        st = sel(seed, "KMCIR_U"); sg = sel(seed, "SAGE")
        row = dict(seed=seed, kmcir500_dil=d5, kmcir500_auroc=au5,
                   stored_kmcir500_dil=st[0]["dilution"] if st else None, stored_kmcir500_auroc=st[0]["auroc"] if st else None,
                   kmcir200_dil=d2, kmcir200_auroc=au2, perm200_dil=dp, perm200_auroc=aup,
                   sage200_dil=sg[0]["dilution"] if sg else None, sage200_auroc=sg[0]["auroc"] if sg else None)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
