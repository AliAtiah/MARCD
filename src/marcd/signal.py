#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ledoit_wolf_shrinkage


@dataclass
class SignalConfig:
    lambda_blend: float = 0.5
    shrink_delta: float = 0.05


class SignalAgent:
    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def blend(self,
              hist_window: pd.DataFrame,
              synth_scenarios: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        mu_hist = hist_window.mean().values
        cov_hist = np.cov(hist_window.values, rowvar=False)
        mu_synth, cov_synth = None, None
        if synth_scenarios is not None:
            mu_synth = synth_scenarios.mean(axis=0)
            cov_synth = np.cov(synth_scenarios, rowvar=False)
        lam = self.cfg.lambda_blend if synth_scenarios is not None else 0.0
        mu_hat = lam * (mu_synth if mu_synth is not None else 0.0) + (1 - lam) * mu_hist
        cov_hat = lam * (cov_synth if cov_synth is not None else 0.0) + (1 - lam) * cov_hist
        cov_hat = ledoit_wolf_shrinkage(np.asarray(cov_hat), delta=self.cfg.shrink_delta)
        return np.asarray(mu_hat), np.asarray(cov_hat)
