#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, backtest_pipeline


def run_ablations(prices: pd.DataFrame, base: BacktestConfig, outdir: Path) -> pd.DataFrame:
    tests = {
        "MARCD(base)": base,
        "UncondDiff": dataclasses.replace(base, use_diffusion=False),
        "NoCVaR": dataclasses.replace(base, lambda_mu=base.lambda_mu, gamma_mv=0.0),
        "Lambda0": dataclasses.replace(base, lambda_blend=0.0),
        "Lambda1": dataclasses.replace(base, lambda_blend=1.0),
    }
    rows = []
    for name, cfg in tests.items():
        res = backtest_pipeline(prices, cfg, outdir / f"abl_{name}")
        row = {"variant": name} | res["metrics"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "ablations.csv", index=False)
    return df


def run_sensitivity(prices: pd.DataFrame, base: BacktestConfig, outdir: Path) -> pd.DataFrame:
    grid = []
    for K in [2, 3, 4]:
        cfg = dataclasses.replace(base)
        cfg.use_hmm = True
        res = backtest_pipeline(prices, cfg, outdir / f"sens_K{K}")
        grid.append({"param": "K", "value": K} | res["metrics"])
    for a in [0.90, 0.95, 0.99]:
        cfg = dataclasses.replace(base, alpha=a)
        res = backtest_pipeline(prices, cfg, outdir / f"sens_alpha{a}")
        grid.append({"param": "alpha", "value": a} | res["metrics"])
    for lam in [0.3, 0.5, 0.7]:
        cfg = dataclasses.replace(base, lambda_blend=lam)
        res = backtest_pipeline(prices, cfg, outdir / f"sens_lambda{lam}")
        grid.append({"param": "lambda", "value": lam} | res["metrics"])
    for tau in [0.10, 0.20, 0.30]:
        cfg = dataclasses.replace(base, tau=tau)
        res = backtest_pipeline(prices, cfg, outdir / f"sens_tau{tau}")
        grid.append({"param": "tau", "value": tau} | res["metrics"])
    df = pd.DataFrame(grid)
    df.to_csv(outdir / "sensitivity.csv", index=False)
    return df
