import pandas as pd
import numpy as np
from pathlib import Path

from src.backtest import BacktestConfig, backtest_pipeline


def test_backtest_smoke(tmp_path: Path):
    # Create synthetic price data for 6 assets over 260 business days
    dates = pd.bdate_range(start="2020-01-01", periods=260)
    prices = pd.DataFrame(
        100.0 + np.cumsum(np.random.randn(len(dates), 6) * 0.1, axis=0),
        index=dates,
        columns=[f"A{i}" for i in range(6)],
    )
    cfg = BacktestConfig(rebalance="monthly", n_scenarios=128, use_hmm=False, use_diffusion=False)
    res = backtest_pipeline(prices, cfg, outdir=tmp_path)
    assert "metrics" in res and isinstance(res["metrics"], dict)
