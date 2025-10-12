import numpy as np

import marcd as M


def test_imports_and_versions():
    assert isinstance(M.__version__, str)


def test_basic_backtest_config():
    cfg = M.BacktestConfig()
    assert cfg.rebalance == "monthly"
