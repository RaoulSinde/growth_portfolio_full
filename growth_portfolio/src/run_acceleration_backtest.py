"""Backtest de la strategie d'acceleration MSCI USA Growth et exports."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from . import acceleration
from . import backtest
from . import metrics
from . import msci_config as cfg
from . import msci_universe as universe

logger = logging.getLogger(__name__)


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Series):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return {str(k): v for k, v in value.to_dict(orient="index").items()}
    raise TypeError(f"Non serialisable: {type(value)}")


def _weights_by_date(weights: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    out: dict[pd.Timestamp, pd.Series] = {}
    for col in weights.columns:
        ser = weights[col].astype(float)
        ser = ser[ser > 0].rename("weight")
        out[pd.Timestamp(col)] = ser
    return out


def _index_weights_multi(uni: dict) -> pd.DataFrame:
    rows = []
    for date in uni["dates"]:
        day = uni["weights"]
        day = day[(day["date"] == date) & (day["weight"] > 0)][["ticker", "weight"]]
        if day.empty:
            continue
        part = day.copy()
        part["date"] = date
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True)
    return frame.set_index(["date", "ticker"])[["weight"]].sort_index()


def _period_returns(nav: pd.Series, freq: str) -> pd.Series:
    r = nav.pct_change().dropna()
    if freq == "Y":
        grouped = r.groupby(r.index.year)
        labels = [str(y) for y, _ in grouped]
    else:
        grouped = r.groupby([r.index.year, r.index.quarter])
        labels = [f"{y}Q{q}" for (y, q), _ in grouped]
    vals = [(1.0 + g).prod() - 1.0 for _, g in grouped]
    return pd.Series(vals, index=labels, name="return")


def _annual_table(nav_map: dict[str, pd.Series]) -> pd.DataFrame:
    series = {name: _period_returns(nav, "Y") for name, nav in nav_map.items()}
    return pd.DataFrame(series)


def _quarterly_table(nav_map: dict[str, pd.Series]) -> pd.DataFrame:
    series = {name: _period_returns(nav, "Q") for name, nav in nav_map.items()}
    return pd.DataFrame(series)


def _sector_exposures(uni: dict, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sectors = uni["sectors"]
    weights = uni["weights"]
    for date in uni["dates"]:
        bench = weights[(weights["date"] == date) & (weights["weight"] > 0)][["ticker", "weight"]]
        sec = sectors[sectors["date"] == date][["ticker", "sector"]]
        bench = bench.merge(sec, on="ticker", how="left")
        bench["weight"] = bench["weight"] / bench["weight"].sum()
        b = bench.groupby("sector")["weight"].sum()

        sel = scores[scores["date"] == date][["sector", "target_weight"]]
        s = sel.groupby("sector")["target_weight"].sum()
        for sector in sorted(set(b.index) | set(s.index)):
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "strategy": float(s.get(sector, 0.0)),
                    "benchmark": float(b.get(sector, 0.0)),
                    "active": float(s.get(sector, 0.0) - b.get(sector, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _coverage_table(uni: dict) -> pd.DataFrame:
    rows = []
    for date in uni["dates"]:
        all_day = uni["weights"]
        all_day = all_day[(all_day["date"] == date) & (all_day["weight"] > 0)]
        elig = universe.eligible_universe(uni, date)
        total = float(all_day["weight"].sum())
        rows.append(
            {
                "date": date,
                "n_constituents": int(len(all_day)),
                "n_eligible": int(len(elig)),
                "covered_weight": float(elig["weight"].sum() / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _run_variant(
    uni: dict,
    fund: pd.DataFrame,
    lam: float,
    tx_bps: float = cfg.TX_COST_BPS,
) -> dict:
    portfolios = acceleration.build_acceleration_portfolios(uni, fund, lam=lam)
    strat = _weights_by_date(portfolios["weights"])
    idx_w = _index_weights_multi(uni)
    result = backtest.run_backtest(
        {"Acceleration": strat},
        prices=uni["prices"],
        bench=uni["benchmark"],
        tx_cost_bps=tx_bps,
        rebalance_dates=uni["dates"],
        start=cfg.START,
        end=cfg.END,
    )
    nav = result["Acceleration"]["nav"]
    bench_nav = result["Benchmark"]["nav"]
    stats = metrics.performance_stats(
        nav,
        bench_nav,
        weights_daily=result["Acceleration"]["weights_daily"],
        index_weights=idx_w,
        turnover=result["Acceleration"]["turnover"],
    )
    bench_stats = metrics.performance_stats(bench_nav, bench_nav)
    return {
        "portfolios": portfolios,
        "result": result,
        "stats": stats,
        "bench_stats": bench_stats,
    }


def run_acceleration_backtest(save: bool = True) -> dict:
    uni = universe.build_universe()
    fund = pd.read_parquet(cfg.FUNDAMENTALS_MSCI)
    logger.info("Univers: %d tickers, %d dates", len(uni["tickers"]), len(uni["dates"]))

    main = _run_variant(uni, fund, lam=cfg.LAMBDA, tx_bps=cfg.TX_COST_BPS)
    portfolios = main["portfolios"]
    scores = portfolios["scores"]
    weights = portfolios["weights"]
    summary = portfolios["summary"]
    result = main["result"]
    stats = main["stats"]
    bench_stats = main["bench_stats"]

    nav_strat = result["Acceleration"]["nav"]
    nav_bench = result["Benchmark"]["nav"]
    nav_map = {"Strategie": nav_strat, "Benchmark": nav_bench}
    annual = _annual_table(nav_map)
    quarterly = _quarterly_table(nav_map)
    sectors = _sector_exposures(uni, scores)
    coverage = _coverage_table(uni)

    daily = pd.DataFrame(
        {
            "nav_strategy": nav_strat,
            "nav_benchmark": nav_bench,
            "ret_strategy": nav_strat.pct_change(),
            "ret_benchmark": nav_bench.pct_change(),
        }
    )
    daily.index.name = "date"

    # Sensibilite au multiplicateur exponentiel.
    sens_rows = []
    for lam in (0.35, 0.50, 0.70):
        variant = main if np.isclose(lam, cfg.LAMBDA) else _run_variant(uni, fund, lam=lam, tx_bps=cfg.TX_COST_BPS)
        s = variant["stats"]
        sens_rows.append(
            {
                "lambda": lam,
                "cum_return": s["cum_return"],
                "ann_return": s["ann_return"],
                "vol_ann": s["vol_ann"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_dd"],
                "te_ann": s["te_ann"],
                "info_ratio": s["info_ratio"],
                "turnover_avg": s["turnover_avg"],
            }
        )
    sensitivity = pd.DataFrame(sens_rows)

    last_date = uni["dates"][-1]
    last_scores = scores[scores["date"] == last_date].sort_values(
        "target_weight", ascending=False
    )
    top_holdings = last_scores[
        ["date", "ticker", "sector", "z", "benchmark_weight", "target_weight"]
    ].copy()

    # Attribution approximative par titre : poids de fin de journee * rendement
    # du lendemain, agregee par annee.
    wd = result["Acceleration"]["weights_daily"]
    rets = uni["prices"].pct_change(fill_method=None).reindex(wd.index).fillna(0.0)
    contrib = wd.shift(1).fillna(0.0) * rets
    annual_contrib = contrib.groupby(contrib.index.year).sum()
    overall_contrib = contrib.sum().sort_values(ascending=False)
    contribution_rows = []
    for year, row in annual_contrib.iterrows():
        for ticker, value in row.sort_values(ascending=False).head(10).items():
            contribution_rows.append(
                {"year": int(year), "ticker": ticker, "contribution": float(value)}
            )
    contributions = pd.DataFrame(contribution_rows)

    payload = {
        "config": {
            "start": cfg.START,
            "end": cfg.END,
            "n_holdings": cfg.N_HOLDINGS,
            "lambda": cfg.LAMBDA,
            "z_clip": cfg.Z_CLIP,
            "tx_cost_bps": cfg.TX_COST_BPS,
            "min_end": str(cfg.MIN_END),
            "n_tickers": len(uni["tickers"]),
            "n_dates": len(uni["dates"]),
            "first_date": uni["dates"][0],
            "last_date": uni["dates"][-1],
        },
        "universe_summary": universe.universe_summary(uni),
        "stats": stats,
        "bench_stats": bench_stats,
        "annual_returns": annual,
        "quarterly_returns": quarterly,
        "sector_exposure": sectors,
        "coverage": coverage,
        "portfolio_summary": summary,
        "top_holdings": top_holdings,
        "contributions": contributions,
        "overall_contributions": overall_contrib.head(20).rename("contribution"),
        "sensitivity": sensitivity,
        "turnover": result["Acceleration"]["turnover"],
    }

    if save:
        cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        acceleration.save_acceleration_outputs(portfolios)
        daily.to_csv(cfg.REPORT_CSV)
        wd.to_parquet(cfg.REPORTS_DIR / "acceleration_weights_daily.parquet")
        annual.to_csv(cfg.REPORTS_DIR / "acceleration_annual.csv")
        quarterly.to_csv(cfg.REPORTS_DIR / "acceleration_quarterly.csv")
        sectors.to_csv(cfg.REPORTS_DIR / "acceleration_sector_exposure.csv", index=False)
        coverage.to_csv(cfg.REPORTS_DIR / "acceleration_coverage.csv", index=False)
        summary.to_csv(cfg.REPORTS_DIR / "acceleration_portfolio_summary.csv", index=False)
        top_holdings.to_csv(cfg.REPORTS_DIR / "acceleration_top_holdings.csv", index=False)
        contributions.to_csv(cfg.REPORTS_DIR / "acceleration_contributions.csv", index=False)
        sensitivity.to_csv(cfg.REPORTS_DIR / "acceleration_sensitivity.csv", index=False)
        with open(cfg.REPORTS_DIR / "acceleration_summary.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        logger.info("Exports ecrits dans %s", cfg.REPORTS_DIR)

    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    payload = run_acceleration_backtest(save=True)
    print(json.dumps(payload["stats"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
