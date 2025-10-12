#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DiffusionConfig:
    use_diffusion: bool = False
    tail_q: float = 0.05
    tail_eta: float = 2.0
    moe: bool = True
    n_scenarios: int = 1024
    seed: int = 2020


class ScenarioAgent:
    """
    Placeholder for a conditional diffusion model with Tail-weighted training
    and a Regime-MoE denoiser. Implement train() and sample() for full power.
    """

    def __init__(self, cfg: DiffusionConfig, n_assets: int):
        self.cfg = cfg
        self.n_assets = n_assets
        self._rng = np.random.default_rng(cfg.seed)

    def train(self, returns_train: pd.DataFrame, returns_val: Optional[pd.DataFrame] = None):
        # Intentionally left as a hook.
        return None

    def sample(self, z_t: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
        N = self.cfg.n_scenarios
        d = self.n_assets
        mean = np.zeros(d)
        cov = 0.0001 * np.eye(d)
        X = self._rng.multivariate_normal(mean, cov, size=N)
        return X
