#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cvxpy as cp
import numpy as np


@dataclass
class AllocatorConfig:
    alpha: float = 0.95
    gamma_mv: float = 1.0
    lambda_mu: float = 0.0
    tau: float = 0.20
    box_low: float = 0.0
    box_high: float = 0.3
    cost_bps: float = 10.0


class AllocationAgent:
    def __init__(self, cfg: AllocatorConfig, n_assets: int):
        self.cfg = cfg
        self.n_assets = n_assets

    def solve(self,
              w_prev: np.ndarray,
              mu_hat: np.ndarray,
              cov_hat: np.ndarray,
              scenarios: Optional[np.ndarray]) -> Dict[str, Any]:
        d = self.n_assets
        w = cp.Variable(d)
        zeta = cp.Variable()
        if scenarios is None or scenarios.size == 0:
            obj = -self.cfg.lambda_mu * mu_hat @ w + self.cfg.gamma_mv * cp.quad_form(w, cov_hat) + 0.0 * zeta
            constraints = [cp.sum(w) == 1,
                           w >= self.cfg.box_low, w <= self.cfg.box_high,
                           cp.norm1(w - w_prev) <= self.cfg.tau]
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(solver=cp.OSQP, verbose=False)
            return {"w": w.value, "zeta": float(zeta.value) if zeta.value is not None else 0.0,
                    "status": prob.status, "extras": {}}

        N = scenarios.shape[0]
        s = cp.Variable(N, nonneg=True)
        losses = -scenarios @ w
        obj = (-self.cfg.lambda_mu * mu_hat @ w
               + self.cfg.gamma_mv * cp.quad_form(w, cov_hat)
               + zeta
               + (1.0 / ((1 - self.cfg.alpha) * N)) * cp.sum(s))
        constraints = [
            cp.sum(w) == 1.0,
            w >= self.cfg.box_low, w <= self.cfg.box_high,
            cp.norm1(w - w_prev) <= self.cfg.tau,
            s >= losses - zeta,
        ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        extras = {"dual_box_low": None, "dual_box_high": None}
        return {"w": w.value, "zeta": float(zeta.value), "status": prob.status, "extras": extras}
