<p align="center">
  <img src="./MARCD.png" alt="MARCD" width="720" />
</p>

# 🧠 MARCD: Multi‑Agent Regime‑Conditioned Diffusion for CVaR Allocation

[![CI](https://github.com/AliAtiah/MARCD/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAtiah/MARCD/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

This repository implements the end‑to‑end framework described in:

> **Ali Atiah Alzahrani (Public Investment Fund)**  
> **Crisis‑Aware Regime‑Conditioned Diffusion with CVaR Allocation (MARCD)**  

MARCD is a **generative‑to‑decision** pipeline that couples:  
1) **Regime inference** with a Gaussian HMM,  
2) a **regime‑conditioned diffusion** generator trained with a **tail‑weighted loss** and a **Regime‑MoE denoiser**, and  
3) a **convex CVaR epigraph QP allocator** with turnover/box constraints and governance (KKT) logging.  

The system runs **strict walk‑forward**, translating realistic, tail‑faithful scenarios into auditable portfolio decisions with improved drawdown control.

---

## 🔗 Quick Links

- Paper PDF: `./Paper___Long___Oct_1_EXTENDED.pdf`
- CLI entrypoint: `marcd` or `python -m marcd`
- Suggested dataset location: `./data/etf_prices.csv` (Adj Close, wide format)

---

## 🧱 Repository Layout

```
.
├── src/
│   ├── __init__.py
│   ├── __main__.py               # enables: python -m marcd
│   ├── cli.py                    # argparse CLI
│   ├── backtest.py               # walk-forward engine
│   ├── regime.py                 # Gaussian HMM agent
│   ├── scenario.py               # diffusion scaffolding
│   ├── signal.py                 # blending + shrinkage
│   ├── alloc.py                  # CVaR epigraph QP
│   ├── baselines.py              # EW / RP / BL(stub)
│   ├── diagnostics.py            # KS/ES/VS, LB p(|r|), VaR UC, CVaR err
│   └── experiments.py            # ablations & sensitivity
├── tests/
│   ├── test_imports.py
│   └── test_utils.py
├── MARCD.png
├── marcd_main.py                 # legacy wrapper → delegates to CLI
├── pyproject.toml                # packaging + tooling
├── .pre-commit-config.yaml
├── .editorconfig
├── .gitignore
├── .github/workflows/ci.yml
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Python ≥3.10 is recommended.

```bash
pip install -U pip
pip install -e .[dev,full]
```

> For a lightweight start, you can install without extras and run historical-only backtests.

---

## ▶️ Quick Start

1) Prepare data: `./data/etf_prices.csv` (Adjusted Close wide by ticker).  
2) Run a walk‑forward backtest (historical moments only, smoke test):

```bash
marcd backtest \
  --data ./data/etf_prices.csv \
  --start-train 2005-01-01 --end-train 2018-12-31 \
  --start-val 2019-01-01   --end-val 2019-12-31 \
  --start-test 2020-01-01  --end-test 2025-01-01 \
  --rebalance monthly \
  --alpha 0.95 --tau 0.20 \
  --lambda-blend 0.5 \
  --box-low 0.0 --box-high 0.3 \
  --n-scenarios 1024 \
  --cost-bps 10 \
  --outdir ./results
```

Or using the module runner:

```bash
python -m marcd backtest --data ./data/etf_prices.csv --outdir ./results
```

3) Enable regime and diffusion (placeholders provided):

```bash
marcd backtest --data ./data/etf_prices.csv --use-hmm --use-diffusion \
  --tail-q 0.05 --tail-eta 2.0 --moe --lambda-blend 0.5 --outdir ./results
```

4) Run ablations:

```bash
marcd ablations --data ./data/etf_prices.csv --outdir ./results
```

5) Run sensitivity sweeps:

```bash
marcd sensitivity --data ./data/etf_prices.csv --outdir ./results
```

---

## 🧠 Core Concepts & Components

### 1) Regime Agent (Gaussian HMM)
- Rolling walk‑forward estimation on returns to produce regime posteriors `π_t` and context `z_t`.
- `K=3` states is a strong default; `--hmm-k` sets it.
- Used to condition diffusion and gate MoE; optionally throttle risk in allocation.

### 2) Scenario Agent (Conditional Diffusion + Tail Loss + Regime‑MoE)
- Conditional DDPM UNet denoiser (compact ~1–2M params suggested).
- Tail‑weighted loss emphasizes lower‑q region of single‑asset worst loss.
- Regime‑MoE denoiser blends Base/Crisis experts via gate `g_t = σ(MLP(z_t))`.

### 3) Signal Agent (Moment Blending + Shrinkage)
- Blends moments from generated scenarios and rolling historical windows.

### 4) Allocation Agent (CVaR Epigraph QP + Governance)
- Solves convex QP minimizing CVaR under constraints, with audit-friendly logs.

---

## 📊 Default Experimental Protocol

- Universe: Liquid ETFs (daily Adjusted Close); any universe size supported.  
- Splits: Train 2005–2018, Val 2019, Test 2020–2025.  
- Rebalance: Monthly (last trading day), cost = 10 bps per rebalance.  
- Parity: All strategies share identical settings.  
- Scenarios: `N=1024` per month (configurable).  
- Metrics: CAGR, Vol, Sharpe, Sortino, MaxDD, Calmar; scenario diagnostics.

---

## 🧾 Governance & Auditability

- Each rebalance logs: active constraints, CVaR dual weights, HMM posteriors, MoE gates.  
- Logs saved under `./results/`.

---

## 🧩 Extending the Code

- Plug in your diffusion backend by implementing `ScenarioAgent.train/sample`.
- Consider decision‑aware training via implicit differentiation.
- Add multi‑step drawdown CVaR via convex surrogate (allocator stubs are ready).

---

## 🔒 Reproducibility

Set seeds via `--seed`; CLI args are recorded via outputs. For unit tests, run the smoke test first; then enable regime/diffusion progressively.

---

## 📜 Citation

```
@article{alzahrani2025marcd,
  title={Crisis-Aware Regime-Conditioned Diffusion with CVaR Allocation},
  author={Ali Atiah Alzahrani},
  year={2025},
  note={Working paper}
}
```

---

## ⚠️ Disclaimer

This codebase is intended for research/educational purposes. Financial results are illustrative and depend on data quality, modeling choices, and market conditions.
