#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "BacktestConfig",
    "backtest_pipeline",
    "print_metrics",
    "RegimeConfig",
    "RegimeAgent",
    "DiffusionConfig",
    "ScenarioAgent",
    "SignalConfig",
    "SignalAgent",
    "AllocatorConfig",
    "AllocationAgent",
]

__version__ = "0.1.0"

from .backtest import BacktestConfig, backtest_pipeline, print_metrics  # noqa: E402
from .regime import RegimeConfig, RegimeAgent  # noqa: E402
from .scenario import DiffusionConfig, ScenarioAgent  # noqa: E402
from .signal import SignalConfig, SignalAgent  # noqa: E402
from .alloc import AllocatorConfig, AllocationAgent  # noqa: E402
