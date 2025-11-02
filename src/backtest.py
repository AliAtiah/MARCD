#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore

from .utils import compute_returns, rolling_last_trading_days, sharpe, sortino, max_drawdown, calmar, ensure_dir, setup_logging, log_json
from .regime import RegimeAgent, RegimeConfig
from .scenario import ScenarioAgent, DiffusionConfig
from .signal import SignalAgent, SignalConfig
from .alloc import AllocationAgent, AllocatorConfig
from .baselines import equal_weight_target


@dataclass
class BacktestConfig:
    rebalance: str = "monthly"
    alpha: float = 0.95
    tau: float = 0.20
    box_low: float = 0.0
    box_high: float = 0.3
    lambda_blend: float = 0.5
    gamma_mv: float = 1.0
    lambda_mu: float = 0.0
    cost_bps: float = 10.0
    use_hmm: bool = False
    use_diffusion: bool = False
    moe: bool = True
    tail_q: float = 0.05
    tail_eta: float = 2.0
    n_scenarios: int = 1024
    seed: int = 2020


def backtest_pipeline(prices: pd.DataFrame, cfg: BacktestConfig, outdir: Path) -> Dict[str, Any]:
    setup_logging()
    rets = compute_returns(prices).dropna()
    dates = rets.index
    n_assets = rets.shape[1]
    if cfg.rebalance.lower().startswith("month"):
        rb_dates = rolling_last_trading_days(dates, "M")
    else:
        rb_dates = dates

    regime = RegimeAgent(RegimeConfig(K=3, use_hmm=cfg.use_hmm, seed=cfg.seed))
    scen = ScenarioAgent(DiffusionConfig(use_diffusion=cfg.use_diffusion,
                                         tail_q=cfg.tail_q, tail_eta=cfg.tail_eta,
                                         moe=cfg.moe, n_scenarios=cfg.n_scenarios,
                                         seed=cfg.seed),
                         n_assets=n_assets)
    signal = SignalAgent(SignalConfig(lambda_blend=cfg.lambda_blend, shrink_delta=0.05))
    alloc = AllocationAgent(AllocatorConfig(alpha=cfg.alpha, gamma_mv=cfg.gamma_mv,
                                            lambda_mu=cfg.lambda_mu, tau=cfg.tau,
                                            box_low=cfg.box_low, box_high=cfg.box_high,
                                            cost_bps=cfg.cost_bps), n_assets=n_assets)

    w_prev = equal_weight_target(n_assets)
    nav = [1.0]
    w_path = [w_prev.copy()]
    pnl = []

    gov_rows = []

    for t in tqdm(range(1, len(rb_dates))):
        t_date = rb_dates[t]
        hist = rets.loc[:t_date].tail(252)
        z = np.zeros(1)
        if cfg.use_hmm:
            rz = regime.fit_update(rets.loc[:t_date].tail(756))
            z = rz["z"]

        scenarios = None
        if cfg.use_diffusion:
            scenarios = scen.sample(z_t=z)

        mu_hat, cov_hat = signal.blend(hist_window=hist, synth_scenarios=scenarios)
        sol = alloc.solve(w_prev=w_prev, mu_hat=mu_hat, cov_hat=cov_hat, scenarios=scenarios)

        w_t = np.array(sol["w"]).reshape(-1)
        w_t = np.clip(w_t, cfg.box_low, cfg.box_high)
        s = w_t.sum()
        if s > 0:
            w_t = w_t / s
        next_idx = rets.index.get_indexer([t_date], method="backfill")[0] + 1
        if next_idx >= len(rets):
            break
        r_next = rets.iloc[next_idx].values
        trade_cost = cfg.cost_bps * 1e-4 * np.abs(w_t - w_prev).sum()
        port_ret = float(w_t @ r_next) - trade_cost
        pnl.append(port_ret)
        nav.append(nav[-1] * (1 + port_ret))

        gov_rows.append({
            "date": str(t_date.date()),
            "dual_budget": sol["extras"].get("dual_budget"),
            "dual_turnover": sol["extras"].get("dual_turnover"),
            "box_low": sol["extras"].get("active_box_low"),
            "box_high": sol["extras"].get("active_box_high"),
        })
        log_json(logger=__import__(__name__), event="rebalance", date=str(t_date.date()), ret=port_ret)

        w_prev = w_t
        w_path.append(w_prev.copy())

    pnl = pd.Series(pnl, index=rb_dates[1:len(pnl)+1])
    nav = pd.Series(nav, index=[rb_dates[0]] + list(rb_dates[1:len(pnl)+1]))
    metrics = {
        "Return(ann%)": (1 + pnl).prod() ** (252 / len(pnl)) - 1 if len(pnl) else np.nan,
        "Vol(ann%)": pnl.std(ddof=0) * math.sqrt(252) if len(pnl) else np.nan,
        "Sharpe": sharpe(pnl) if len(pnl) else np.nan,
        "Sortino": sortino(pnl) if len(pnl) else np.nan,
        "MaxDD(%)": max_drawdown(nav) if len(pnl) else np.nan,
        "Calmar": calmar(pnl) if len(pnl) else np.nan,
    }
    ensure_dir(outdir)
    pd.DataFrame({"pnl": pnl, "nav": nav}).to_csv(outdir / "timeseries.csv")
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)
    np.save(outdir / "weights.npy", np.array(w_path))
    if gov_rows:
        pd.DataFrame(gov_rows).to_csv(outdir / "governance.csv", index=False)
    return {"pnl": pnl, "nav": nav, "metrics": metrics}


def print_metrics(metrics: Dict[str, Any]) -> None:
    print(json.dumps(metrics, indent=2, default=lambda x: float(x)))
