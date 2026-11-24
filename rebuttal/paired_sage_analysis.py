"""Paired comparison of Kernel-MCIR U vs SAGE and permutation importance on IDENTICAL evaluation documents.
Uses local_runs/matched_sage.json (matched_sage_rerun.py): Kernel-MCIR and permutation importance re-scored on exactly
the 200 documents per seed that SAGE used (k=1, p=0, retrained MLP); SAGE values from dil_sage*.json.
(The first version paired Kernel-MCIR on 500 documents with SAGE on 200; that comparison is superseded.)
python paired_sage_analysis.py -> local_runs/paired_sage.json
"""
import json
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "local_runs", "matched_sage.json")))
rng = np.random.default_rng(0)
out = {"seeds": [r["seed"] for r in rows], "documents": "identical 200 per seed"}
for key in ("auroc", "dil"):
    a = np.array([r[f"kmcir200_{key}"] for r in rows])
    for other in ("sage200", "perm200"):
        b = np.array([r[f"{other}_{key}"] for r in rows]); d = a - b
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(20000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res = dict(kmcir_mean=a.mean(), kmcir_sd=a.std(), other_mean=b.mean(), other_sd=b.std(), diff=d.mean(),
                   ci95=[lo, hi], cohen_dz=d.mean() / d.std(ddof=1), t_p=ttest_rel(a, b).pvalue,
                   wilcoxon_p=wilcoxon(a, b).pvalue, wins=int((d > 0).sum()))
        out[f"{key}_vs_{other}"] = {k: (float(v) if np.isscalar(v) else [float(x) for x in v]) for k, v in res.items()}
        print(f"{key:6s} K-MCIR {a.mean():.3f}±{a.std():.3f} vs {other} {b.mean():.3f}±{b.std():.3f} | diff {d.mean():+.3f} "
              f"CI [{lo:+.3f},{hi:+.3f}] d_z={res['cohen_dz']:.2f} t p={res['t_p']:.2g} Wilcoxon p={res['wilcoxon_p']:.2g} "
              f"wins {res['wins']}/{len(d)}")
json.dump(out, open(os.path.join(HERE, "local_runs", "paired_sage.json"), "w"), indent=1)
