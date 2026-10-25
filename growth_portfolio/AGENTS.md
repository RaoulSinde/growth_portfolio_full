# AGENTS.md

## Project

Growth USA portfolio: single composite PIT strategy (Core + Satellites) benchmarked
against the iShares ESG Aware MSCI USA Growth ETF (**EGUS**).

`Z_Composite = 0.25*Z_Acceleration + 0.45*Z_Momentum + 0.30*Z_Quality_GARP`

Backtest window: `PERF_START=2024-01-25` -> `SIM_END=2026-09-30`, 11 quarterly
`filing_date` rebalances. Point-in-Time strict on both prices and fundamentals.

## Environment

- Python 3.13, dependencies installed globally (`requirements.txt`).
- Windows / PowerShell 5.1. `rg` is not available in the shell.
- Use the dedicated Read/Edit/Grep/Glob tools for file operations.

## Commands

```powershell
python -m pytest tests/ -q            # full audit suite (expect: 135 passed)
python -m src.build_fundamentals_sec  # regenerate data/fundamentals_sec.parquet (SEC XBRL, PIT)
streamlit run app.py                  # launch the dashboard (7 tabs)
```

Always run `python -m pytest tests/ -q` after touching `src/` or `app.py`.

## Architecture

- `src/config.py` — rebalance dates, factor weights, portfolio constraints.
- `src/data_loader.py` — PIT prices/fundamentals; `annual_flow_points` /
  `annual_stock_points` build the annual `N, N-1, N-2, N-3` aggregates.
- `src/build_universe.py` / `src/build_fundamentals_sec.py` — data ingestion.
- `src/factors.py` — 3-pillar static PIT scores, `composite_weights_apply`,
  `build_factor_panel` / `compute_strategy_scores`.
- `src/portfolio.py` — Core + Satellites construction, `build_all_portfolios`.
- `src/backtest.py` / `src/metrics.py` — NAV, turnover, risk metrics.
- `src/mohanram.py` — **Mohanram Growth Tilted** research engine (G-Score 0-8,
  incl. G8 = ADV/Assets): annual PIT fundamentals cache
  (`data/cache/mohanram_annual.parquet`), deterministic tilt of EGUS weights,
  strict sector neutrality, iterative capping, 6-month Shumway (1997) horizon.
- `app.py` — Streamlit dashboard (7 tabs): single composite strategy (3 weight
  sliders 25/45/30) plus a **Pure Accélération** research tab (pillars 100/0/0,
  `w_max` default 15 %) and a **Stratégie Growth (Mohanram)** tab, with stats,
  Brinson attribution, conditional beta, alpha significance and sector/Top-10
  vs EGUS.

## Conventions

- No comments unless strictly necessary.
- **Look-ahead bias is the cardinal sin**: any factor must only use data with
  `pub_date <= t`; scores must be invariant to future price/fundamental truncation.
- Z-scores/winsorization are cross-sectional, per rebalance date only.
- Do NOT read fundamental metrics from `ticker.info` (identification only).
- Tests in `tests/` are the audit contract; update them alongside `src/`.
