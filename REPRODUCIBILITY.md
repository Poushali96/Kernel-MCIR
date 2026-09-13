# Reproducibility notes

## Environment

Recommended: Python 3.10–3.12 on Linux, macOS, or Windows.

Install with:

```bash
pip install -e .
pip install pytest
```

The controlled diagnostics are CPU-only. No GPU is required.

## Determinism

All scripts expose or fix explicit NumPy/scikit-learn seeds. For permutation experiments, sampled order matrices are created once and reused. This matters when comparing clean and perturbed inputs because Monte Carlo order noise should not be conflated with data perturbation.

## Core numerical definitions

For a conditioning set `Phi` and candidate `i`:

1. `L_phi` is the orthogonal projection of the centered output Gram matrix onto the span of feature Gram matrices in `Phi`.
2. `Delta_i = L_{Phi+i} - L_phi`.
3. `P_i` is the projection of the candidate Gram matrix onto the current span.
4. `u_i = <K_i, Delta_i>_F` and `r_i = <P_i, L_phi>_F` remain signed.
5. `U_i = abs(u_i)` and `R_i = abs(r_i)`.
6. `s_i = U_i/(U_i+R_i)` when positive; otherwise the theoretical score is undefined and the aggregation-only extended score is zero.

The implementation also records `signed_total_error`, numerically checking the identity `u_i + r_i = <K_i,L>_F`.

## Generated files

`experiments/run_all.py` produces:

- `results/context_dependence.json`
- `results/duplicate_symmetry.json`
- `results/controlled_order_convergence.csv`
- `results/controlled_order_summary.json`
- `results/rff_sensitivity.csv`

Generated outputs are ignored by git by default.

## Archived benchmarks

The cleaned artifact intentionally does not silently reconstruct missing historical benchmark pipelines. Archived manuscript values are labeled as such in the paper. The revised claims about permutation symmetrization, duplicate symmetry, magnitude strengths, and RFF approximation should be assessed using the clean implementation and controlled diagnostics in this repository.
