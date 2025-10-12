#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, backtest_pipeline, print_metrics
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MARCD scaffold")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--data", type=str, required=True)
        sp.add_argument("--start-train", type=str, default=None)
        sp.add_argument("--end-train", type=str, default=None)
        sp.add_argument("--start-val", type=str, default=None)
        sp.add_argument("--end-val", type=str, default=None)
        sp.add_argument("--start-test", type=str, default=None)
        sp.add_argument("--end-test", type=str, default=None)
        sp.add_argument("--rebalance", type=str, default="monthly")
        sp.add_argument("--alpha", type=float, default=0.95)
        sp.add_argument("--tau", type=float, default=0.20)
        sp.add_argument("--box-low", type=float, default=0.0)
        sp.add_argument("--box-high", type=float, default=0.3)
        sp.add_argument("--lambda-blend", type=float, default=0.5)
        sp.add_argument("--gamma-mv", type=float, default=1.0)
        sp.add_argument("--lambda-mu", type=float, default=0.0)
        sp.add_argument("--cost-bps", type=float, default=10.0)
        sp.add_argument("--use-hmm", action="store_true")
        sp.add_argument("--use-diffusion", action="store_true")
        sp.add_argument("--moe", action="store_true")
        sp.add_argument("--tail-q", type=float, default=0.05)
        sp.add_argument("--tail-eta", type=float, default=2.0)
        sp.add_argument("--n-scenarios", type=int, default=1024)
        sp.add_argument("--seed", type=int, default=2020)
        sp.add_argument("--outdir", type=str, required=True)

    sp_bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    add_common(sp_bt)

    sp_ab = sub.add_parser("ablations", help="Run component ablations")
    add_common(sp_ab)

    sp_sens = sub.add_parser("sensitivity", help="Run parameter sensitivity sweeps")
    add_common(sp_sens)

    return p.parse_args()


def load_prices(path: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    if start:
        df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[:pd.Timestamp(end)]
    df = df.dropna(axis=1, how="all").dropna(how="all")
    return df


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    prices = load_prices(args.data, start=args.start_train or args.start_test, end=args.end_test)

    base = BacktestConfig(
        rebalance=args.rebalance,
        alpha=args.alpha,
        tau=args.tau,
        box_low=args.box_low,
        box_high=args.box_high,
        lambda_blend=args.lambda_blend,
        gamma_mv=args.gamma_mv,
        lambda_mu=args.lambda_mu,
        cost_bps=args.cost_bps,
        use_hmm=args.use_hmm,
        use_diffusion=args.use_diffusion,
        moe=args.moe,
        tail_q=args.tail_q,
        tail_eta=args.tail_eta,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
    )

    if args.cmd == "backtest":
        res = backtest_pipeline(prices, base, outdir)
        print_metrics(res["metrics"])  # pretty print JSON metrics
    elif args.cmd == "ablations":
        from .experiments import run_ablations
        df = run_ablations(prices, base, outdir)
        print(df.to_string(index=False))
    elif args.cmd == "sensitivity":
        from .experiments import run_sensitivity
        df = run_sensitivity(prices, base, outdir)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
