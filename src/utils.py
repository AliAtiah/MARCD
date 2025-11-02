#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility helpers for MARCD: filesystem, returns, metrics, and basic math helpers.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_json(logger: logging.Logger, event: str, **kwargs) -> None:
    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload))


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    rets = prices.pct_change().dropna(how="all")
    return rets


def rolling_last_trading_days(dates: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    months = pd.PeriodIndex(dates, freq=freq)
    last = dates.groupby(months).max()
    return pd.DatetimeIndex(last)


def annualize_ret_vol(returns: pd.Series) -> Tuple[float, float]:
    mu = (1 + returns).prod() ** (252 / max(1, len(returns))) - 1
    vol = returns.std(ddof=0) * math.sqrt(252)
    return mu, vol


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    mean_excess = returns.mean() * 252 - rf
    vol = returns.std(ddof=0) * math.sqrt(252) + 1e-12
    return mean_excess / vol


def sortino(returns: pd.Series, mar: float = 0.0) -> float:
    downside = returns[returns < mar]
    dd = downside.std(ddof=0) * math.sqrt(252) + 1e-12
    return (returns.mean() * 252 - mar) / dd


def max_drawdown(nav: pd.Series) -> float:
    roll_max = nav.cummax()
    dd = (nav / roll_max - 1.0).min()
    return abs(float(dd))


def calmar(returns: pd.Series) -> float:
    nav = (1 + returns).cumprod()
    mdd = max_drawdown(nav) + 1e-12
    return ((1 + returns).prod() ** (252 / len(returns)) - 1) / mdd


def ledoit_wolf_shrinkage(cov: np.ndarray, delta: float = 0.05) -> np.ndarray:
    d = cov.shape[0]
    return (1 - delta) * cov + delta * np.eye(d) * np.trace(cov) / d
