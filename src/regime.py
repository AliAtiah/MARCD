#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None


@dataclass
class RegimeConfig:
    K: int = 3
    rolling_window_days: int = 756
    use_hmm: bool = True
    seed: int = 2020


class RegimeAgent:
    """
    Gaussian HMM regime inference with strict walk-forward updates.
    Produces posteriors and a context vector for conditioning.
    """

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def fit_update(self, returns_hist: pd.DataFrame) -> Dict[str, Any]:
        if not self.cfg.use_hmm or GaussianHMM is None:
            pi = np.array([1.0] + [0.0] * (self.cfg.K - 1))
            return {"pi": pi, "z": np.concatenate([pi, [returns_hist.std().mean()]])}

        X = returns_hist.values
        X = np.nan_to_num(X, nan=0.0)
        K = max(1, self.cfg.K)
        hmm = GaussianHMM(n_components=K, covariance_type="full", random_state=self.cfg.seed, n_iter=100)
        try:
            hmm.fit(X)
            post = hmm.predict_proba(X)
            pi = post[-1, :]
        except Exception:
            pi = np.zeros(K)
            pi[0] = 1.0
        z = np.concatenate([pi, [np.nan_to_num(np.std(X), nan=0.0)]])
        return {"pi": pi, "z": z}
