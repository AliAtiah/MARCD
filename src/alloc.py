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
        extras: Dict[str, Any] = {}
        if scenarios is None or scenarios.size == 0:
            obj = -self.cfg.lambda_mu * mu_hat @ w + self.cfg.gamma_mv * cp.quad_form(w, cov_hat) + 0.0 * zeta
            c_budget = cp.sum(w) == 1
            c_box_low = w >= self.cfg.box_low
            c_box_high = w <= self.cfg.box_high
            c_turnover = cp.norm1(w - w_prev) <= self.cfg.tau
            constraints = [c_budget, c_box_low, c_box_high, c_turnover]
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(solver=cp.OSQP, verbose=False)
            try:
                extras = {
                    "dual_budget": float(c_budget.dual_value) if c_budget.dual_value is not None else None,
                    "active_box_low": (np.array(c_box_low.dual_value) > 1e-9).astype(int).tolist() if c_box_low.dual_value is not None else None,
                    "active_box_high": (np.array(c_box_high.dual_value) > 1e-9).astype(int).tolist() if c_box_high.dual_value is not None else None,
                    "dual_turnover": float(c_turnover.dual_value) if c_turnover.dual_value is not None else None,
                }
            except Exception:
                extras = {}
            return {"w": w.value, "zeta": float(zeta.value) if zeta.value is not None else 0.0,
                    "status": prob.status, "extras": extras}

        N = scenarios.shape[0]
        s = cp.Variable(N, nonneg=True)
        losses = -scenarios @ w
        obj = (-self.cfg.lambda_mu * mu_hat @ w
               + self.cfg.gamma_mv * cp.quad_form(w, cov_hat)
               + zeta
               + (1.0 / ((1 - self.cfg.alpha) * N)) * cp.sum(s))
        c_budget = cp.sum(w) == 1.0
        c_box_low = w >= self.cfg.box_low
        c_box_high = w <= self.cfg.box_high
        c_turnover = cp.norm1(w - w_prev) <= self.cfg.tau
        c_cvar = s >= losses - zeta
        constraints = [c_budget, c_box_low, c_box_high, c_turnover, c_cvar]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        try:
            extras = {
                "dual_budget": float(c_budget.dual_value) if c_budget.dual_value is not None else None,
                "active_box_low": (np.array(c_box_low.dual_value) > 1e-9).astype(int).tolist() if c_box_low.dual_value is not None else None,
                "active_box_high": (np.array(c_box_high.dual_value) > 1e-9).astype(int).tolist() if c_box_high.dual_value is not None else None,
                "dual_turnover": float(c_turnover.dual_value) if c_turnover.dual_value is not None else None,
                "dual_cvar": (np.array(c_cvar.dual_value).reshape(-1).tolist()) if c_cvar.dual_value is not None else None,
            }
        except Exception:
            extras = {}
        return {"w": w.value, "zeta": float(zeta.value), "status": prob.status, "extras": extras}
