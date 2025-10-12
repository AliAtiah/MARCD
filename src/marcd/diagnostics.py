#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats  # type: ignore

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:  # pragma: no cover - optional dependency
    acorr_ljungbox = None


def ks_es_vs(scenarios: np.ndarray) -> Tuple[float, float, float]:
    flat = scenarios.flatten()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return 0.0, 0.0, 0.0
    ks = stats.kstest((flat - flat.mean()) / (flat.std() + 1e-12), 'norm').statistic
    es = np.mean(flat[flat <= np.quantile(flat, 0.05)])
    vs = np.var(flat)
    return float(ks), float(abs(es)), float(vs)


def ljung_box_p_abs(scenarios: np.ndarray, lags: int = 10) -> float:
    if acorr_ljungbox is None:
        return float('nan')
    flat = np.abs(scenarios.flatten())
    _, p = acorr_ljungbox(flat, lags=[lags], return_df=False)
    return float(p[0])


def var_coverage_uc(realized: np.ndarray, alpha: float = 0.95) -> float:
    q = np.quantile(realized, 1 - alpha)
    hits = realized < -q
    p = hits.mean()
    n = len(hits)
    if n == 0:
        return float('nan')
    expected = 1 - alpha
    with np.errstate(divide='ignore'):
        LR = -2 * ((n * (expected * np.log(expected) + (1 - expected) * np.log(1 - expected))) -
                   (hits.sum() * np.log(p + 1e-12) + (n - hits.sum()) * np.log(1 - p + 1e-12)))
    from scipy.stats import chi2  # type: ignore
    return float(1 - chi2.cdf(LR, df=1))


def cvar_error_bps(realized_losses: np.ndarray, alpha: float = 0.95, target: float | None = None) -> float:
    if len(realized_losses) == 0:
        return float('nan')
    q = np.quantile(realized_losses, alpha)
    tail = realized_losses[realized_losses >= q]
    cvar = tail.mean() if len(tail) else q
    if target is None:
        return float(cvar * 1e4)
    return float(abs(cvar - target) * 1e4)
