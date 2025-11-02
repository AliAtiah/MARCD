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
    backend: str = "numpy"  # "numpy" or "torch"
    steps: int = 50  # diffusion steps when torch backend is used


class ScenarioAgent:
    """
    Scenario generator with optional diffusion backend.
    - Numpy fallback: simple Gaussian draws (fast, dependency-free)
    - Torch backend: minimal DDPM-style sampling via a tiny UNet (optional)
    """

    def __init__(self, cfg: DiffusionConfig, n_assets: int):
        self.cfg = cfg
        self.n_assets = n_assets
        self._rng = np.random.default_rng(cfg.seed)
        self._torch_initialized = False
        self._torch = None
        self._unet = None
        if self.cfg.use_diffusion and self.cfg.backend == "torch":
            self._init_torch_backend()

    def _init_torch_backend(self) -> None:
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            class TinyUNet(nn.Module):
                def __init__(self, d: int):
                    super().__init__()
                    h = max(16, d * 2)
                    self.net = nn.Sequential(
                        nn.Linear(d, h), nn.GELU(),
                        nn.Linear(h, h), nn.GELU(),
                        nn.Linear(h, d),
                    )

                def forward(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
                    # Time embedding (simple scale)
                    emb = (t.float().view(-1, 1) + 1.0) / (self.training and 100.0 or 100.0)
                    return self.net(x) * (1.0 + emb)

            self._torch = torch
            self._unet = TinyUNet(self.n_assets).eval()
            self._torch_initialized = True
        except Exception:
            # Fallback if torch isn't available
            self._torch_initialized = False
            self._torch = None
            self._unet = None

    def train(self, returns_train: pd.DataFrame, returns_val: Optional[pd.DataFrame] = None):
        # Hook for future training; keep no-op for CI and lightweight usage
        return None

    def _sample_numpy(self, N: int) -> np.ndarray:
        d = self.n_assets
        mean = np.zeros(d)
        cov = 0.0001 * np.eye(d)
        X = self._rng.multivariate_normal(mean, cov, size=N)
        return X

    def _sample_torch(self, N: int) -> np.ndarray:
        if not self._torch_initialized or self._torch is None or self._unet is None:
            return self._sample_numpy(N)
        torch = self._torch
        device = torch.device("cpu")
        with torch.no_grad():
            x = torch.randn(N, self.n_assets, device=device)
            T = self.cfg.steps
            for t in reversed(range(T)):
                tt = torch.full((N,), t, device=device)
                eps_pred = self._unet(x, tt)
                # simple variance schedule
                beta = 0.01
                x = (x - beta * eps_pred)
            out = x.cpu().numpy()
        return out

    def sample(self, z_t: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
        N = self.cfg.n_scenarios
        if not self.cfg.use_diffusion:
            return self._sample_numpy(N)
        if self.cfg.backend == "torch":
            return self._sample_torch(N)
        return self._sample_numpy(N)
