#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


def equal_weight_target(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def risk_parity_target(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov) + 1e-12)
    inv = 1.0 / np.maximum(vols, 1e-8)
    w = inv / inv.sum()
    return w


def black_litterman_target(mu: np.ndarray, tau: float = 0.05) -> np.ndarray:
    w = equal_weight_target(len(mu))
    tilt = tau * (mu - mu.mean())
    w = w + tilt
    w = np.maximum(0.0, w)
    s = w.sum()
    return w / (s if s > 0 else 1.0)
