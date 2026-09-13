# Manuscript-to-code mapping

| Manuscript concept / result | Reference implementation |
|---|---|
| Centered Gram matrices | `src/kernel_mcir/kernels.py` |
| Orthogonal projector / rank-revealing implementation | `src/kernel_mcir/core.py::_ProjectionCache` |
| Signed alignments `u_i`, `r_i` | `src/kernel_mcir/core.py::context_score` |
| Magnitude strengths `U_i`, `R_i` | `src/kernel_mcir/core.py::context_score` |
| Zero-strength extended score | `src/kernel_mcir/core.py::context_score` |
| Fixed-order conditional diagnostic | `src/kernel_mcir/core.py::fixed_order_scores` |
| Permutation symmetrization | `src/kernel_mcir/core.py::symmetrized_scores` |
| Exact duplicate symmetry | `experiments/duplicate_symmetry.py`, `tests/test_core.py` |
| Context-dependence proposition | `experiments/context_dependence.py`, `tests/test_core.py` |
| Order-convergence diagnostic | `experiments/controlled_order.py` |
| RFF approximation diagnostic | `src/kernel_mcir/rff.py`, `experiments/rff_sensitivity.py` |
| Signed-total identity | checked by `ContextScore.signed_total_error` and unit tests |

## Important distinction

The revised reference implementation does **not** use the superseded development conventions found in older exploratory notebooks (positive clipping/ReLU, raw signed ratios, or conditioning on all other features). Those notebooks are not part of this anonymous review artifact to avoid ambiguity about which definition is canonical.
