from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernel_mcir.kernels import rbf_gram

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def save_json(name: str, payload: dict) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def controlled_dataset(seed: int = 42, n: int = 1600):
    """Five-feature stress test: main, exact duplicate, nonlinear, weak, noise."""
    rng = np.random.default_rng(seed)
    main = rng.normal(size=n)
    duplicate = main.copy()  # exact exchangeable duplicate for the symmetry diagnostic
    nonlinear = rng.normal(size=n)
    weak = rng.normal(size=n)
    noise = rng.normal(size=n)
    latent = 1.5 * main + 1.1 * np.sin(1.8 * nonlinear) + 0.20 * weak + 0.35 * rng.normal(size=n)
    y = (latent > np.median(latent)).astype(int)
    X = np.column_stack([main, duplicate, nonlinear, weak, noise])
    names = ["main", "duplicate", "nonlinear", "weak", "noise"]
    return X, y, names


def trained_controlled_problem(seed: int = 42, n: int = 1600, explain_n: int = 300):
    X, y, names = controlled_dataset(seed=seed, n=n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=3, n_jobs=-1)
    model.fit(Xtr, ytr)
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(len(Xte), size=min(explain_n, len(Xte)), replace=False)
    Xexp = Xte[idx]
    yhat = model.predict_proba(Xexp)[:, 1]
    K_feats = [rbf_gram(Xexp[:, j]) for j in range(Xexp.shape[1])]
    L = rbf_gram(yhat)
    return Xexp, yhat, names, K_feats, L, model, scaler
