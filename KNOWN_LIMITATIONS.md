# Known limitations and scope

- Kernel-MCIR is a statistical-geometric attribution diagnostic, not a causal method and not a conditional mutual-information estimator.
- The novelty ratio should be interpreted jointly with total target-aligned strength; the activity-gated score is optional when one scalar ranking is needed.
- Exact Gram matrices scale quadratically in sample count. RFF or other low-rank approximations are preferable at larger scale.
- Stability/consistency guarantees require local rank/non-degeneracy and a denominator bounded away from zero.
- Uniform permutation averaging addresses arbitrary feature order but increases computation.
- This artifact does not recreate unavailable historical pipelines for manuscript tables explicitly labeled as archived.
