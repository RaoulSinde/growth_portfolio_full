"""Backtest, metriques et exports du portefeuille two-pocket."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from growth_portfolio.src import backtest as gp_backtest
from growth_portfolio.src import metrics as gp_metrics

from . import two_pocket_construction as tpc
from . import two_pocket_data as tpd

__all__ = [
    "VERSIONS",
    "VERSION_LABELS",
    "COMPOSITE_LABEL",
    "COMPOSITE_CAP_LABEL",
    "DEFAULT_MOMENTUM_WEIGHT",
    "OUTPUT_DIR",
    "build_portfolios",
    "compute_results",
    "save_results",
    "run_two_pocket_backtest",
]

COMPOSITE_LABEL = "Quality-Momentum"
COMPOSITE_CAP_LABEL = "Quality-Momentum 3x"
DEFAULT_MOMENTUM_WEIGHT = 0.30
VERSION_CONFIGS = {
    COMPOSITE_LABEL: {
        "score_column": "score_composite",
        "max_weight_multiple": None,
    },
    COMPOSITE_CAP_LABEL: {
        "score_column": "score_composite",
        "max_weight_multiple": 3.0,
    },
}
VERSIONS = tuple(VERSION_CONFIGS)
VERSION_LABELS = {label: label for label in VERSIONS}
OUTPUT_DIR = tpd.OUTPUT_DIR
START = tpd.START
END = tpd.END

_STAT_COLUMNS = [
    "cum_return",
    "ann_return",
    "vol_ann",
    "sharpe",
    "sortino",
    "calmar",
    "max_dd",
    "beta",
    "alpha_ann",
    "active_return_ann",
    "te_ann",
    "info_ratio",
    "up_capture",
    "down_capture",
    "active_share",
    "active_share_latest",
    "turnover_avg",
    "n_positions_latest",
    "n_positions_avg",
    "mega_weight_latest",
    "quant_weight_latest",
    "top10_weight",
    "n_effective",
    "sector_gap_max",
    "best_year",
    "worst_year",
    "positive_quarters",
    "skew",
    "kurtosis",
]


def _as_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _period_returns(nav: pd.Series, freq: str) -> pd.Series:
    returns = nav.pct_change().dropna()
    if freq == "Y":
        grouped = returns.groupby(returns.index.year)
        labels = [str(year) for year, _ in grouped]
    else:
        grouped = returns.groupby([returns.index.year, returns.index.quarter])
        labels = [f"{year}Q{quarter}" for (year, quarter), _ in grouped]
    values = [(1.0 + group).prod() - 1.0 for _, group in grouped]
    return pd.Series(values, index=labels, name="return")


def _period_table(nav_map: dict[str, pd.Series], freq: str) -> pd.DataFrame:
    table = pd.DataFrame({name: _period_returns(nav, freq) for name, nav in nav_map.items()})
    if freq == "Y":
        table = table[table.index.astype(int) >= 2021]
    else:
        table = table[[idx[:4].isdigit() and int(idx[:4]) >= 2021 for idx in table.index]]
    return table


def _composite_score(day: pd.DataFrame, momentum_weight: float) -> pd.Series:
    """Score composite Quality/Momentum, robuste aux composantes manquantes."""
    quality = day["score_quality_pure"].astype(float)
    momentum = day["score_momentum"].astype(float)
    weight = float(momentum_weight)
    if weight <= 0.0:
        return quality
    if weight >= 1.0:
        return momentum
    score = (1.0 - weight) * quality + weight * momentum
    return score.where(quality.notna() & momentum.notna())


def _heuristic_portfolios(
    panel: pd.DataFrame,
    label: str,
    max_weight_multiple: float | None = None,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
) -> dict:
    holding_parts = []
    budget_parts = []
    diagnostics_rows = []

    for date, day in panel.groupby("date", sort=True):
        date = _as_ts(date)
        frame = day[
            [
                "ticker",
                "sector",
                "benchmark_weight",
                "score_quality_pure",
                "score_momentum",
                "is_megacap",
            ]
        ].copy()
        frame["score_composite"] = _composite_score(frame, momentum_weight)
        eligible = set(day.loc[day["eligible"].astype(bool), "ticker"])
        result = tpc.build_two_pocket_portfolio(
            frame,
            n_quant=40,
            eligible_tickers=eligible,
            score_column="score_composite",
            max_weight_multiple=max_weight_multiple,
        )
        holdings = result.holdings.copy()
        holdings.insert(0, "date", date)
        holdings.insert(0, "version", label)
        holding_parts.append(holdings)

        budget = result.sector_budget.copy()
        budget.insert(0, "date", date)
        budget.insert(0, "version", label)
        budget_parts.append(budget)

        checks = result.diagnostics["checks"]
        diagnostics_rows.append(
            {
                "date": date,
                "version": label,
                "score_type": "score_composite",
                "sum_weight": result.diagnostics["total_weight"],
                "max_sector_gap": result.diagnostics["max_sector_gap"],
                "budget_residual": result.diagnostics["budget_residual"],
                "n_positions": result.diagnostics["n_positions"],
                "n_mega": result.diagnostics["n_mega"],
                "n_mega_companies": result.diagnostics["n_mega_companies"],
                "n_quant": result.diagnostics["n_quant_selected"],
                "n_quant_target": result.diagnostics["n_quant_target"],
                "mega_weight": result.diagnostics["mega_weight"],
                "quant_weight": result.diagnostics["quant_weight"],
                "active_share": result.diagnostics["active_share"],
                "max_weight_ratio_quant": result.diagnostics[
                    "max_weight_ratio_quant"
                ],
                "cap_relaxed": result.diagnostics["cap_relaxed"],
                "ok": bool(checks["ok"].all()),
            }
        )

    holdings = pd.concat(holding_parts, ignore_index=True)
    budget = pd.concat(budget_parts, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostics_rows)
    weights_by_date = {
        _as_ts(date): part.set_index("ticker")["weight"].astype(float)
        for date, part in holdings.groupby("date", sort=True)
    }
    return {
        "n_quant": 40,
        "score_column": "score_composite",
        "momentum_weight": float(momentum_weight),
        "max_weight_multiple": max_weight_multiple,
        "holdings": holdings,
        "sector_budget": budget,
        "diagnostics": diagnostics,
        "weights": weights_by_date,
    }


def build_portfolios(
    panel: pd.DataFrame,
    market: dict | None = None,
    versions: Iterable[str] = VERSIONS,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
) -> dict[str, dict]:
    """Construit les portefeuilles Quant 40 pour chaque variante."""
    if not 0.0 <= float(momentum_weight) <= 1.0:
        raise ValueError("momentum_weight doit etre compris entre 0 et 1")
    output: dict[str, dict] = {}
    for label in versions:
        if label not in VERSION_CONFIGS:
            raise ValueError(f"Version inconnue: {label}")
        config = VERSION_CONFIGS[label]
        output[label] = _heuristic_portfolios(
            panel,
            label,
            config["max_weight_multiple"],
            momentum_weight,
        )
    return output


def _benchmark_structure(panel: pd.DataFrame) -> dict:
    top10 = []
    n_eff = []
    n_names = []
    mega_weight = []
    for _, day in panel.groupby("date", sort=True):
        weights = day.set_index("ticker")["benchmark_weight"].astype(float)
        top10.append(float(weights.nlargest(10).sum()))
        n_eff.append(float(1.0 / float((weights.to_numpy() ** 2).sum())))
        n_names.append(int(len(weights)))
        mega_weight.append(
            float(weights.reindex(day.loc[day["is_megacap"], "ticker"]).fillna(0).sum())
        )
    return {
        "top10_weight": float(np.mean(top10)),
        "n_effective": float(np.mean(n_eff)),
        "n_positions_avg": float(np.mean(n_names)),
        "n_positions_latest": int(n_names[-1]),
        "mega_weight_latest": float(mega_weight[-1]),
        "quant_weight_latest": float(1.0 - mega_weight[-1]),
    }


def _strategy_stats(
    label: str,
    nav: pd.Series,
    bench_nav: pd.Series,
    weights_daily: pd.DataFrame,
    index_weights: pd.DataFrame,
    turnover: pd.Series,
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict:
    stats = gp_metrics.performance_stats(
        nav,
        bench_nav,
        weights_daily=weights_daily,
        index_weights=index_weights,
        turnover=turnover,
    )
    returns = nav.pct_change().dropna()
    max_dd = float(stats.get("max_dd", np.nan))
    stats["calmar"] = (
        float(stats["ann_return"]) / abs(max_dd)
        if np.isfinite(max_dd) and max_dd < 0 and np.isfinite(stats.get("ann_return", np.nan))
        else np.nan
    )
    stats["skew"] = float(returns.skew()) if len(returns) > 2 else np.nan
    stats["kurtosis"] = float(returns.kurtosis()) if len(returns) > 2 else np.nan

    annual_label = annual[label].dropna() if label in annual.columns else pd.Series(dtype=float)
    stats["best_year"] = float(annual_label.max()) if len(annual_label) else np.nan
    stats["worst_year"] = float(annual_label.min()) if len(annual_label) else np.nan
    quarterly_label = (
        quarterly[label].dropna() if label in quarterly.columns else pd.Series(dtype=float)
    )
    stats["positive_quarters"] = (
        float((quarterly_label > 0).mean()) if len(quarterly_label) else np.nan
    )

    diag = diagnostics[diagnostics["version"] == label].sort_values("date")
    stats["active_share_latest"] = float(diag["active_share"].iloc[-1])
    stats["n_positions_latest"] = int(diag["n_positions"].iloc[-1])
    stats["n_positions_avg"] = float(diag["n_positions"].mean())
    stats["mega_weight_latest"] = float(diag["mega_weight"].iloc[-1])
    stats["quant_weight_latest"] = float(diag["quant_weight"].iloc[-1])
    stats["sector_gap_max"] = float(diag["max_sector_gap"].max())
    return {key: stats.get(key, np.nan) for key in _STAT_COLUMNS}


def _benchmark_stats(
    nav: pd.Series,
    panel: pd.DataFrame,
    annual: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> dict:
    stats = gp_metrics.performance_stats(nav, nav)
    returns = nav.pct_change().dropna()
    max_dd = float(stats.get("max_dd", np.nan))
    structure = _benchmark_structure(panel)
    annual_label = annual["Benchmark"].dropna() if "Benchmark" in annual.columns else pd.Series(dtype=float)
    quarterly_label = (
        quarterly["Benchmark"].dropna() if "Benchmark" in quarterly.columns else pd.Series(dtype=float)
    )
    stats.update(
        {
            "calmar": (
                float(stats["ann_return"]) / abs(max_dd)
                if np.isfinite(max_dd) and max_dd < 0 and np.isfinite(stats.get("ann_return", np.nan))
                else np.nan
            ),
            "skew": float(returns.skew()) if len(returns) > 2 else np.nan,
            "kurtosis": float(returns.kurtosis()) if len(returns) > 2 else np.nan,
            "active_share": 0.0,
            "active_share_latest": 0.0,
            "turnover_avg": 0.0,
            "sector_gap_max": 0.0,
            "best_year": float(annual_label.max()) if len(annual_label) else np.nan,
            "worst_year": float(annual_label.min()) if len(annual_label) else np.nan,
            "positive_quarters": float((quarterly_label > 0).mean()) if len(quarterly_label) else np.nan,
            **structure,
        }
    )
    return {key: stats.get(key, np.nan) for key in _STAT_COLUMNS}


def _active_positions(panel: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (date, version), held in holdings.groupby(["date", "version"], sort=True):
        day = panel[panel["date"] == date].copy()
        benchmark = day.set_index("ticker")["benchmark_weight"]
        universe = benchmark.index.union(held["ticker"])
        part = pd.DataFrame(index=universe)
        part["company"] = day.set_index("ticker")["company"].reindex(universe)
        part["sector"] = day.set_index("ticker")["sector"].reindex(universe)
        part["benchmark_weight"] = benchmark.reindex(universe).fillna(0.0)
        part["portfolio_weight"] = (
            held.set_index("ticker")["weight"].reindex(universe).fillna(0.0)
        )
        part["active_weight"] = part["portfolio_weight"] - part["benchmark_weight"]
        part["pocket"] = held.set_index("ticker")["pocket"].reindex(universe).fillna("Non detenu")
        part["score"] = held.set_index("ticker")["score"].reindex(universe)
        part["z"] = held.set_index("ticker")["z"].reindex(universe)
        part = part.reset_index().rename(columns={"index": "ticker"})
        part.insert(0, "date", date)
        part.insert(0, "version", version)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _contributions(
    label: str,
    weights_daily: pd.DataFrame,
    prices: pd.DataFrame,
    sector_map: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None).reindex(weights_daily.index).fillna(0.0)
    contribution = weights_daily.shift(1).fillna(0.0) * returns
    overall = contribution.sum()
    overall = overall[overall.abs() > 1e-12]
    annual = contribution.groupby(contribution.index.year).sum()
    sectors = sector_map.reindex(overall.index)

    overall_frame = pd.DataFrame(
        {
            "version": label,
            "ticker": overall.index,
            "sector": sectors.to_numpy(),
            "contribution": overall.to_numpy(dtype=float),
        }
    )
    annual_rows = []
    for year, row in annual.iterrows():
        row = row[row.abs() > 1e-12]
        annual_rows.append(
            pd.DataFrame(
                {
                    "version": label,
                    "year": int(year),
                    "ticker": row.index,
                    "sector": sectors.reindex(row.index).to_numpy(),
                    "contribution": row.to_numpy(dtype=float),
                }
            )
        )
    annual_frame = pd.concat(annual_rows, ignore_index=True)
    sector_overall = (
        overall_frame.groupby("sector", as_index=False)["contribution"].sum()
    )
    sector_overall.insert(0, "version", label)
    sector_overall.insert(1, "year", np.nan)
    sector_overall = sector_overall[["version", "year", "sector", "contribution"]]
    sector_annual = (
        annual_frame.groupby(["year", "sector"], as_index=False)["contribution"].sum()
    )
    sector_annual.insert(0, "version", label)
    sector_annual = sector_annual[["version", "year", "sector", "contribution"]]
    return overall_frame, annual_frame, sector_overall, sector_annual


def compute_results(
    panel: pd.DataFrame,
    market: dict,
    tx_cost_bps: float = 10.0,
    portfolios: dict | None = None,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
) -> dict:
    """Construit, backteste et agrege toutes les sorties analytiques."""
    if not 0.0 <= float(momentum_weight) <= 1.0:
        raise ValueError("momentum_weight doit etre compris entre 0 et 1")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    dates = sorted(panel["date"].unique())
    if portfolios is None:
        portfolios = build_portfolios(
            panel,
            market,
            VERSIONS,
            momentum_weight=momentum_weight,
        )
    else:
        first = next(iter(portfolios.values()))
        momentum_weight = float(first.get("momentum_weight", momentum_weight))

    strategies = {label: data["weights"] for label, data in portfolios.items()}
    nav_result = gp_backtest.run_backtest(
        strategies,
        prices=market["prices"],
        bench=market["benchmark"],
        tx_cost_bps=float(tx_cost_bps),
        rebalance_dates=dates,
        start=START,
        end=END,
    )

    benchmark_nav = nav_result["Benchmark"]["nav"]
    nav_daily = pd.DataFrame({"Benchmark": benchmark_nav})
    for label in strategies:
        nav_daily[label] = nav_result[label]["nav"]
    nav_daily.index.name = "date"

    index_weights = (
        panel[["date", "ticker", "benchmark_weight"]]
        .rename(columns={"benchmark_weight": "weight"})
        .set_index(["date", "ticker"])
        .sort_index()
    )

    annual = _period_table({c: nav_daily[c] for c in nav_daily.columns}, "Y")
    quarterly = _period_table({c: nav_daily[c] for c in nav_daily.columns}, "Q")

    diagnostics = pd.concat(
        [data["diagnostics"] for data in portfolios.values()], ignore_index=True
    )
    stats_rows = {
        "Benchmark": _benchmark_stats(benchmark_nav, panel, annual, quarterly)
    }
    for label in strategies:
        stats_rows[label] = _strategy_stats(
            label,
            nav_result[label]["nav"],
            benchmark_nav,
            nav_result[label]["weights_daily"],
            index_weights,
            nav_result[label]["turnover"],
            annual,
            quarterly,
            diagnostics,
        )
    stats = pd.DataFrame(stats_rows).T.reindex(columns=_STAT_COLUMNS)
    stats.index.name = "strategy"

    holdings = pd.concat(
        [data["holdings"] for data in portfolios.values()], ignore_index=True
    )
    sector_budget = pd.concat(
        [data["sector_budget"] for data in portfolios.values()], ignore_index=True
    )
    sector_exposure = sector_budget[
        [
            "date",
            "version",
            "sector",
            "benchmark_weight",
            "portfolio_weight",
            "active_weight",
        ]
    ].copy()
    active_positions = _active_positions(panel, holdings)

    turnover_rows = []
    for label in strategies:
        part = nav_result[label]["turnover"].rename("turnover").reset_index()
        part["version"] = label
        turnover_rows.append(part[["date", "version", "turnover"]])
    turnover = pd.concat(turnover_rows, ignore_index=True)

    sector_map = (
        panel.sort_values("date")
        .drop_duplicates("ticker", keep="last")
        .set_index("ticker")["sector"]
    )
    contribution_parts = [
        _contributions(
            label,
            nav_result[label]["weights_daily"],
            market["prices"],
            sector_map,
        )
        for label in strategies
    ]
    contributions_overall = pd.concat(
        [part[0] for part in contribution_parts], ignore_index=True
    )
    contributions_annual = pd.concat(
        [part[1] for part in contribution_parts], ignore_index=True
    )
    sector_contributions = pd.concat(
        [part[2] for part in contribution_parts]
        + [part[3] for part in contribution_parts],
        ignore_index=True,
    )

    validation_columns = [
        "date",
        "version",
        "sum_weight",
        "max_sector_gap",
        "budget_residual",
        "pocket_budget_residual",
        "n_positions",
        "n_mega",
        "n_mega_companies",
        "n_quant",
        "n_quant_target",
        "mega_weight",
        "quant_weight",
        "active_share",
        "max_weight_ratio_quant",
        "cap_relaxed",
        "ok",
    ]
    validation = diagnostics.reindex(columns=validation_columns).copy()

    mega_base = panel.loc[
        panel["is_megacap"].astype(bool),
        ["date", "ticker", "company", "sector", "benchmark_weight"],
    ].copy()
    mega_parts = []
    for label, data in portfolios.items():
        held = data["holdings"][
            ["date", "ticker", "weight", "pocket", "active_weight"]
        ].copy()
        part = mega_base.merge(held, on=["date", "ticker"], how="left")
        part["version"] = label
        part["weight"] = part["weight"].fillna(0.0)
        part["active_weight"] = part["active_weight"].fillna(
            -part["benchmark_weight"]
        )
        part["pocket"] = part["pocket"].fillna("Mega-cap")
        part["held"] = part["weight"] > 0.0
        mega_parts.append(
            part[
                [
                    "date",
                    "version",
                    "ticker",
                    "company",
                    "sector",
                    "benchmark_weight",
                    "weight",
                    "active_weight",
                    "pocket",
                    "held",
                ]
            ]
        )
    mega_caps = pd.concat(mega_parts, ignore_index=True)

    latest_date = pd.Timestamp(max(dates))
    latest_mega = (
        panel.loc[
            (panel["date"] == latest_date) & panel["is_megacap"].astype(bool),
            "company",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    metadata = {
        "start": str(START.date()),
        "end": str(END.date()),
        "rebalance_dates": [str(pd.Timestamp(d).date()) for d in dates],
        "n_rebalances": len(dates),
        "versions": {
            label: {
                "n_quant": int(data["n_quant"]),
                "score": data["score_column"],
                "quality_weight": 1.0 - float(momentum_weight),
                "momentum_weight": float(momentum_weight),
                "max_weight_multiple": data["max_weight_multiple"],
            }
            for label, data in portfolios.items()
        },
        "quality_weight": 1.0 - float(momentum_weight),
        "momentum_weight": float(momentum_weight),
        "momentum_definition": (
            "6M-1M risk-adjusted price momentum + 200d trend, "
            "sector-neutral z-score on adjusted prices"
        ),
        "mega_weight_threshold": float(tpc.MEGA_WEIGHT_THRESHOLD),
        "alphabet_tickers": list(tpc.ALPHABET_TICKERS),
        "mega_caps_latest": latest_mega,
        "tx_cost_bps": float(tx_cost_bps),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "MSCI USA Growth",
    }

    return {
        "panel": panel,
        "metadata": metadata,
        "nav_daily": nav_daily.reset_index(),
        "stats": stats,
        "annual_returns": annual,
        "quarterly_returns": quarterly,
        "holdings": holdings,
        "sector_budget": sector_budget,
        "sector_exposure": sector_exposure,
        "active_positions": active_positions,
        "turnover": turnover,
        "contributions_overall": contributions_overall,
        "contributions_annual": contributions_annual,
        "sector_contributions": sector_contributions,
        "validation": validation,
        "mega_caps": mega_caps,
        "portfolios": portfolios,
        "nav_result": nav_result,
    }


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def save_results(results: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """Ecrit tous les exports dans ``reports/two_pocket``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results["nav_daily"].to_parquet(output_dir / "nav_daily.parquet", index=False)
    results["stats"].to_csv(output_dir / "stats.csv")
    results["annual_returns"].to_csv(output_dir / "annual_returns.csv")
    results["quarterly_returns"].to_csv(output_dir / "quarterly_returns.csv")
    results["holdings"].to_parquet(output_dir / "holdings.parquet", index=False)
    results["sector_budget"].to_parquet(
        output_dir / "sector_budget.parquet", index=False
    )
    results["sector_exposure"].to_parquet(
        output_dir / "sector_exposure.parquet", index=False
    )
    results["active_positions"].to_parquet(
        output_dir / "active_positions.parquet", index=False
    )
    results["turnover"].to_parquet(output_dir / "turnover.parquet", index=False)
    results["contributions_overall"].to_parquet(
        output_dir / "contributions_overall.parquet", index=False
    )
    results["contributions_annual"].to_parquet(
        output_dir / "contributions_annual.parquet", index=False
    )
    results["sector_contributions"].to_parquet(
        output_dir / "sector_contributions.parquet", index=False
    )
    results["validation"].to_parquet(output_dir / "validation.parquet", index=False)
    stale_optimizer = output_dir / "optimizer_diagnostics.parquet"
    if stale_optimizer.exists():
        stale_optimizer.unlink()
    results["mega_caps"].to_parquet(
        output_dir / "mega_caps.parquet", index=False
    )
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(results["metadata"], handle, indent=2, default=_json_default)
    return output_dir


def run_two_pocket_backtest(
    tx_cost_bps: float = 10.0,
    force_panel: bool = False,
    save: bool = True,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
) -> dict:
    """Pipeline complet : panel PIT -> portefeuilles -> backtest -> exports."""
    panel = tpd.build_universe_panel(force=force_panel, save=save)
    market = tpd.load_market_prices()
    results = compute_results(
        panel,
        market,
        tx_cost_bps=tx_cost_bps,
        momentum_weight=momentum_weight,
    )
    if save:
        save_results(results)
    return results
