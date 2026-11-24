# Rebuttal-stage experiments

Additional experiments for the revised Kernel-MCIR formulation. The reference implementation in the repository root is unchanged.

## Contents

| Path | What |
|---|---|
| `kmcir_rebuttal_runs.py` | Main runner: 20 Newsgroups duplicate dilution (`dilution`), corrected random-feature (meta-Gram) estimator vs. exact (`rff`), result merging (`report`), offline `selftest`. Runs resume from the saved JSON. |
| `kmcir_rebuttal_experiments.py` | Small synthetic checks (numpy only): meta-Gram vs. factor-space RFF, RFF convergence, exact-copy dilution against 1/(k+1), redundant vs. complementary regimes. |
| `kmcir_fulln_runs.py` | Duplicate dilution on the full 20 Newsgroups test set via a streamed meta-Gram. |
| `kmcir_extra_analyses.py` | Related quantities on the same sampled orders: projection energy (kernel LMG), sequential HSIC gain, partial distance covariance. |
| `kmcir_dependence_sweep.py` | Controlled correlation sweep (5 regimes x 8 correlations) against population LOCO, DFI and DIP. |
| `crosscheck_official_dfi_dip.py`, `compare_official_crosscheck.py` | Cross-check of the sweep's oracle DFI/DIP values against the official estimators. |
| `replication_invariance.py`, `make_replication_table.py` | Representation-perturbation (feature replication) benchmark. |
| `matched_sage_rerun.py`, `paired_sage_analysis.py` | Kernel-MCIR vs. SAGE on matched seeds and documents; paired statistics. |
| `exact_rank_table.py` | Exact-rank table. |
| `make_appendix_tables.py`, `make_dependence_figures.py` | Build tables and figures from `local_runs/` (no hand-typed numbers). |
| `kmcir_rebuttal_colab.ipynb`, `COLAB_CELLS.md` | Colab notebook (and its cells as text) for the main runs. |
| `local_runs/` | Saved outputs (JSON, logs, generated LaTeX tables) of the runs above; see `local_runs/README.md`. |
| `figures/` | Figures generated from `local_runs/`. |

## Quick start

```bash
pip install numpy scipy scikit-learn pandas matplotlib   # optional: sage-importance
python kmcir_rebuttal_runs.py selftest
python kmcir_rebuttal_runs.py dilution --seeds 10 --B 200 --out local_runs/dil_main.json
python kmcir_rebuttal_runs.py rff --datasets synthetic 20ng --D 64 128 256 512 1024
```

Scripts are run from this folder; outputs are written to `local_runs/` and `figures/`.
