"""Confidence intervals and aggregation helpers."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over the units in `values`."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def per_flow_success_ci(success: np.ndarray, **kw) -> Dict[str, float]:
    """success: (n_seeds, n_flows) bool.  Seeds are averaged inside each flow, the
    bootstrap resamples FLOWS, so the interval reflects test-set sampling error."""
    s = np.asarray(success, dtype=float)
    per_flow = s.mean(0)
    mean, lo, hi = bootstrap_ci(per_flow, **kw)
    out = dict(asr=mean, lo=lo, hi=hi, n_flows=int(s.shape[1]))
    if s.shape[0] > 1:
        out["seed_std"] = float(s.mean(1).std(ddof=1))
    return out


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap CI of mean(a-b) for paired per-flow outcomes."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot=n_boot, seed=seed)
