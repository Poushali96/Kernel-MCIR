# Kernel-MCIR — Anonymous Reproducibility Artifact

This repository is the **clean reference implementation for the revised Kernel-MCIR formulation** used in the anonymous manuscript. It is intentionally separated from exploratory development notebooks that contained superseded definitions.

## What this repository implements

The code follows the revised mathematical definition:

- exact Frobenius-space orthogonal projection onto spans of centered Gram matrices;
- signed alignments `u_i` and `r_i`;
- magnitude strengths `U_i = |u_i|` and `R_i = |r_i|`;
- the order-conditioned novelty ratio `s_i = U_i/(U_i+R_i)` when the denominator is positive;
- the extended aggregation convention `tilde{s}_i = 0` for an uninformative zero-strength context;
- uniform feature-order permutation averaging;
- `bar U`, `bar R`, `bar T`, and the optional activity-gated score;
- exact duplicate symmetry under uniform permutation averaging;
- exact-kernel and Random Fourier Feature (RFF) utilities.

The reference projector uses a **rank-revealing SVD**. It does not use `max(u,0)`, ReLU clipping, or an all-other-features conditioning set.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install pytest
pytest -q
```

Run all revised controlled diagnostics:

```bash
python experiments/run_all.py
```

Outputs are written to `results/` and are deliberately git-ignored so each user regenerates them from code.

## Individual diagnostics

```bash
python experiments/context_dependence.py
python experiments/duplicate_symmetry.py
python experiments/controlled_order.py --reference-orders 1000
python experiments/rff_sensitivity.py --n 240 --orders 100
```

`controlled_order.py` reuses one deterministic stream of uniformly sampled feature orders when comparing different values of `B`. `duplicate_symmetry.py` uses exhaustive permutation averaging on a five-feature controlled problem.

## Repository scope and archived manuscript results

The manuscript clearly distinguishes **archived core benchmark values** from experiments introduced for the revised symmetrized formulation. This cleaned artifact is the reference implementation for the revised formulation and reproduces the controlled mathematical/algorithmic diagnostics included with that revision.

Some archived UCI/Wind/Fashion-MNIST benchmark values predate this cleaned artifact. They are retained in the manuscript only as archived fixed-protocol results and are not used as evidence for the revised permutation-symmetrized estimator. This repository does not claim to recreate those unavailable historical pipelines.

## Anonymity

This artifact contains no author names, institutional affiliations, personal repository URLs, Google Drive identifiers, or acknowledgements. Please keep the review repository anonymous until the double-blind period ends.

## Numerical conventions

The theoretical score `s_i(Phi)` is undefined when `U_i + R_i = 0`. For permutation aggregation only, the manuscript defines an extended score `tilde{s}_i(Phi)=0` in that uninformative case. The implementation follows this convention using `strength_tol` to identify numerical zero.

For exact duplicate/near-collinear kernels, rank is determined by SVD with relative tolerance `rank_tol`.

## Reproducibility map

See `docs/MANUSCRIPT_MAPPING.md` for the correspondence between manuscript claims and scripts, and `docs/REPRODUCIBILITY.md` for seeds, dependencies, and expected outputs.
