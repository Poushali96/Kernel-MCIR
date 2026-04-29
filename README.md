# Kernel-MCIR: Anonymous Code Repository

## Overview

This repository contains the implementation of **Kernel-MCIR**, a redundancy-aware nonlinear feature attribution method based on incremental geometric projections in Reproducing Kernel Hilbert Spaces (RKHS).

Kernel-MCIR quantifies feature importance by separating:

* **Unique contribution** (new explanatory structure introduced by a feature)
* **Redundant contribution** (overlap with already explained structure)

This geometric formulation enables **stable, interpretable, and noise-robust feature attribution** across diverse data regimes.

---

## Key Features

* Redundancy-aware feature attribution
* Nonlinear modeling via kernel methods
* Stable rankings under noise and correlation
* No reliance on perturbation or density estimation
* Scalable using Random Fourier Features (RFF)
* GPU-enabled implementation for large-scale experiments

---

## Repository Structure

```
.
├── kernel_mcIR/                 # Core implementation
│   ├── kernel_utils.py         # Kernel computations (RBF, centering, RFF)
│   ├── projection.py           # RKHS projection operators
│   ├── attribution.py          # Kernel-MCIR computation (U, R, score)
│   └── metrics.py              # Evaluation metrics (Spearman, AUC)
│
├── experiments/
│   ├── synthetic.py            # Controlled redundancy experiments
│   ├── uci_air_quality.py      # Tabular dataset experiments
│   ├── wind_power.py           # Time-series experiments
│   ├── mnist_embeddings.py     # High-dimensional embedding experiments
│
├── models/
│   ├── mlp.py                  # Neural network model
│   ├── xgboost_model.py        # Tree-based model
│
├── notebooks/
│   └── kernelmcirextended.ipynb  # Full benchmarking notebook (main file)
│
├── configs/
│   └── default.yaml            # Experiment configurations
│
├── requirements.txt
├── run_experiment.py
└── README.md
```

---

## Installation

```bash
git clone <anonymous-repo-link>
cd kernel-mcir
pip install -r requirements.txt
```

---

## Quick Start (Recommended)

Run the full experimental pipeline directly:

```bash
jupyter notebook notebooks/kernelmcirextended.ipynb
```

This notebook includes:

## Datasets

The experiments cover a diverse set of datasets across synthetic, tabular, temporal, and deep representation settings:

### Real-world datasets
* UCI Air Quality dataset (tabular)
* Gas Sensor dataset (UCI / public repository)
* Wind power time-series (Norway, Open Power System Data / Renewables.ninja; programmatically downloaded)
* Fashion-MNIST (raw images and learned embeddings)

### Synthetic datasets
* Swiss Roll dataset (nonlinear manifold)
* Nonlinear interaction datasets (controlled feature dependencies)
* Large-scale synthetic datasets (up to 50k samples)

### Derived representations
* Deep embeddings extracted from CNN models
* Kernel representations (RBF and Random Fourier Features)

### Robustness settings
* Gaussian noise perturbations
* Adversarial noise perturbations

All datasets are publicly available or generated using standard libraries. Preprocessing, feature engineering (including lag, rolling, and calendar features for temporal data), and generation pipelines are fully included in the repository.
* Baseline comparisons:

  * HSIC
  * CKA
  * Mutual Information
  * Integrated Gradients
* Noise robustness experiments (Gaussian + adversarial)

---

## Usage

### Run a standard experiment

```bash
python run_experiment.py --dataset uci_air_quality
```

### Compute Kernel-MCIR attribution

```python
from kernel_mcIR.attribution import compute_kmcir

scores = compute_kmcir(X, y)
```

---

## Method Summary

Kernel-MCIR evaluates feature importance through **incremental RKHS projections**:

* Let Φ be the conditioning set
* Add feature ( f_i )
* Measure projection change ( \Delta_i )

Compute:

* Unique influence: alignment with new structure
* Redundant influence: alignment with existing structure

Final score:

[
\text{K-MCIR}(i \mid \Phi) =
\frac{U(i \mid \Phi)}{U(i \mid \Phi) + R(i \mid \Phi)}
]

---

## Datasets

All datasets used are publicly available:

* UCI Air Quality (tabular)
* Fashion-MNIST (deep embeddings)
* Swiss Roll synthetic dataset
* Wind power time-series (derived from public sources)
* Synthetic nonlinear interaction datasets

All preprocessing pipelines are included in the repository.

---

## Reproducibility

* Fixed random seeds across experiments
* Standard train/validation/test splits (70/15/15)
* Results averaged over multiple runs
* GPU and CPU implementations provided
* All experiment pipelines included in the notebook

---

## Notes for Reviewers

* This repository is anonymized for double-blind review
* No identifying information is included
* All code is self-contained and executable
* Notebook reproduces all key results in the paper
* Baseline comparisons are implemented consistently

---

## Limitations

* Kernel bandwidth selection may affect performance
* RFF approximation introduces variance at low dimensions
* Computational cost increases with feature dimensionality (without RFF)

---

## License

This repository is provided for academic review purposes.
A formal license will be added upon publication.

---

## Contact

Anonymous submission — contact information will be provided after review.
