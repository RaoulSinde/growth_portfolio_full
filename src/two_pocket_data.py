"""Adaptateur de donnees PIT pour le portefeuille two-pocket MSCI USA Growth.

Le panel produit contient, pour chaque date de rebalancement et chaque titre du
benchmark, les seules informations necessaires a la construction :

- ``ticker``, ``sector``, ``benchmark_weight`` ;
- ``score_quality_pure`` et ``score_momentum`` calcules point-in-time ;
- ``is_megacap``, ``price_available`` et ``eligible``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from growth_portfolio.src import factors as gp_factors
from growth_portfolio.src import msci_universe as gp_universe

from . import config
from .two_pocket_construction import (
    MEGA_WEIGHT_THRESHOLD,
    company_name,
)

__all__ = [
    "START",
    "END",
    "PANEL_FILE",
    "OUTPUT_DIR",
    "load_market_data",
    "load_market_prices",
    "load_adjusted_prices",
    "build_universe_panel",
]

START = pd.Timestamp("2020-12-31")
END = pd.Timestamp("2025-12-31")

OUTPUT_DIR = Path(config.REPORTS_DIR) / "two_pocket"
PANEL_FILE = OUTPUT_DIR / "universe_panel.parquet"

_FUNDAMENTALS_FILE = Path(config.PROCESSED_DIR) / "fundamentals_sec_msci.parquet"


def _quarter_dates(weights: pd.DataFrame) -> list[pd.Timestamp]:
    dates = gp_universe.quarter_end_dates(weights)
    dates = [pd.Timestamp(d).normalize() for d in dates if START <= pd.Timestamp(d) <= END]
    if not dates:
        raise ValueError("Aucune date de rebalancement dans la fenetre demandee")
    return dates


def load_market_data() -> dict:
    """Charge poids, prix, secteurs PIT, benchmark et dates trimestrielles."""
    weights = gp_universe.load_weights()
    prices = gp_universe.load_prices()
    sectors = gp_universe.build_pit_sector_map()
    benchmark = gp_universe.load_benchmark()
    dates = _quarter_dates(weights)
    return {
        "weights": weights,
        "prices": prices,
        "sectors": sectors,
        "benchmark": benchmark,
        "dates": dates,
    }


def load_adjusted_prices() -> pd.DataFrame:
    """Prix ajustes, utilises uniquement pour estimer la covariance."""
    return gp_universe.load_prices(adj=True)


def load_market_prices() -> dict:
    """Version allegee pour l'app : prix, benchmark et dates uniquement."""
    prices = gp_universe.load_prices()
    benchmark = gp_universe.load_benchmark()
    weights = gp_universe.load_weights()
    return {
        "prices": prices,
        "adjusted_prices": load_adjusted_prices(),
        "benchmark": benchmark,
        "dates": _quarter_dates(weights),
    }


def _blend_scores(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    num = None
    den = None
    for series, weight in parts:
        series = series.astype(float)
        present = series.notna().astype(float)
        num = series.fillna(0.0) * weight if num is None else num + series.fillna(0.0) * weight
        den = present * weight if den is None else den + present * weight
    if num is None or den is None:
        return pd.Series(dtype=float)
    return num / den.replace(0.0, np.nan)


def _price_available(prices: pd.DataFrame, date: pd.Timestamp, tickers: pd.Series) -> pd.Series:
    index = prices.index.union(pd.DatetimeIndex([date])).sort_values()
    px = prices.reindex(index).ffill(limit=5).loc[date]
    values = px.reindex(tickers.to_numpy())
    return pd.Series(np.isfinite(values.to_numpy()) & (values.to_numpy() > 0), index=tickers.index)


def _momentum_scores(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    tickers: pd.Series,
    sector: pd.Series,
) -> pd.DataFrame:
    """Momentum prix 6M-1M ajuste du risque + tendance 200j, PIT.

    Le calcul est strictement tronque a ``date``. Les prix ajustes evitent les
    distorsions de splits/dividendes. Le score final est un z-score sectoriel.
    """
    index = pd.Index(tickers.to_numpy())
    empty = pd.DataFrame(
        np.nan,
        index=index,
        columns=["score_momentum", "momentum_ram_z", "momentum_trend_z"],
        dtype=float,
    )
    hist = prices.loc[prices.index <= date]
    if hist.empty or len(hist) < gp_factors.MOM_6M + 1:
        return empty

    px = hist.ffill(limit=5)
    tickers_idx = pd.Index(tickers.to_numpy())
    p1m = px.iloc[-gp_factors.MOM_1M].reindex(tickers_idx)
    p6m = px.iloc[-gp_factors.MOM_6M].reindex(tickers_idx)
    ret_6_1 = p1m / p6m - 1.0

    returns = px.pct_change(fill_method=None)
    vol_window = returns.tail(gp_factors.VOL_WINDOW_6M)
    n_obs = vol_window.notna().sum()
    vol_6m = vol_window.std() * np.sqrt(252.0)
    vol_6m = vol_6m.where(n_obs >= gp_factors.MIN_VOL_OBS)
    ram = ret_6_1 / vol_6m.reindex(tickers_idx)

    last_price = px.iloc[-1].reindex(tickers_idx)
    trend_window = px.tail(gp_factors.SMA_TREND)
    sma200 = trend_window.mean().reindex(tickers_idx)
    sma200 = sma200.where(
        trend_window.notna().sum().reindex(tickers_idx) >= gp_factors.SMA_TREND
    )
    trend = last_price / sma200 - 1.0

    sector_series = pd.Series(sector).reindex(index)
    s_ram = gp_factors.zscore(ram.reindex(index), by_sector=True, sector=sector_series)
    s_trend = gp_factors.zscore(trend.reindex(index), by_sector=True, sector=sector_series)
    raw = _blend_scores([(s_ram, 0.65), (s_trend, 0.35)])
    score = gp_factors.zscore(raw, by_sector=True, sector=sector_series)

    return pd.DataFrame(
        {
            "score_momentum": score,
            "momentum_ram_z": s_ram,
            "momentum_trend_z": s_trend,
        },
        index=index,
    )


def build_universe_panel(
    force: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """Construit le panel PIT multi-dates du benchmark."""
    if not force and PANEL_FILE.exists():
        panel = pd.read_parquet(PANEL_FILE)
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        panel = panel[(panel["date"] >= START) & (panel["date"] <= END)].copy()
        if (
            not panel.empty
            and "score_quality_pure" in panel.columns
            and "score_momentum" in panel.columns
            and "score_quality_garp" not in panel.columns
        ):
            return panel.reset_index(drop=True)

    market = load_market_data()
    weights = market["weights"]
    sectors = market["sectors"]
    prices = market["prices"]
    adjusted_prices = load_adjusted_prices()
    benchmark = market["benchmark"]
    benchmark_returns = benchmark.pct_change(fill_method=None)
    fund = pd.read_parquet(_FUNDAMENTALS_FILE)

    frames = []
    for date in market["dates"]:
        day = weights[(weights["date"] == date) & (weights["weight"] > 0)][
            ["ticker", "weight"]
        ].copy()
        sec_date = sectors[sectors["date"] == date][["ticker", "sector"]].drop_duplicates(
            "ticker"
        )
        day = day.merge(sec_date, on="ticker", how="left")
        if day["sector"].isna().any():
            missing = day.loc[day["sector"].isna(), "ticker"].tolist()
            raise ValueError(f"Secteur manquant a {date.date()}: {missing}")
        day = (
            day.groupby(["ticker", "sector"], as_index=False)["weight"]
            .sum()
            .sort_values("ticker")
            .reset_index(drop=True)
        )
        total = float(day["weight"].sum())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Poids benchmark non normalises a {date.date()}: {total}")

        universe = day.set_index("ticker").rename(
            columns={"sector": "gics_sector", "weight": "weight"}
        )[["gics_sector", "weight"]]
        scores = gp_factors.compute_strategy_scores(
            date,
            prices,
            fund,
            benchmark_returns,
            universe,
        )
        quality_pure_raw = _blend_scores(
            [(scores["S_ROE"], 0.55), (scores["S_FCFYield"], 0.45)]
        )
        quality_pure = gp_factors.zscore(
            quality_pure_raw.reindex(day["ticker"].to_numpy()),
            by_sector=True,
            sector=day.set_index("ticker")["sector"],
        )
        momentum = _momentum_scores(
            adjusted_prices,
            date,
            day["ticker"],
            day.set_index("ticker")["sector"],
        )
        frame = pd.DataFrame(
            {
                "date": date,
                "ticker": day["ticker"].to_numpy(),
                "sector": day["sector"].to_numpy(),
                "benchmark_weight": day["weight"].to_numpy(dtype=float),
                "score_quality_pure": day["ticker"].map(quality_pure).to_numpy(dtype=float),
                "score_momentum": day["ticker"].map(momentum["score_momentum"]).to_numpy(dtype=float),
                "quality_roe": day["ticker"].map(scores["S_ROE"]).to_numpy(dtype=float),
                "quality_fcf_yield": day["ticker"].map(scores["S_FCFYield"]).to_numpy(dtype=float),
                "momentum_ram_z": day["ticker"].map(momentum["momentum_ram_z"]).to_numpy(dtype=float),
                "momentum_trend_z": day["ticker"].map(momentum["momentum_trend_z"]).to_numpy(dtype=float),
            }
        )
        frame["company"] = frame["ticker"].map(company_name)
        company_weight = frame.groupby("company")["benchmark_weight"].transform("sum")
        frame["is_megacap"] = company_weight > MEGA_WEIGHT_THRESHOLD
        frame["price_available"] = _price_available(
            prices, date, frame["ticker"]
        ).to_numpy()
        frame["eligible"] = (
            (~frame["is_megacap"])
            & frame["price_available"]
            & (
                frame["score_quality_pure"].notna()
                | frame["score_momentum"].notna()
            )
            & (frame["benchmark_weight"] > 0.0)
        )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    first = ["date", "ticker", "company", "sector", "benchmark_weight"]
    rest = [c for c in panel.columns if c not in first]
    panel = panel[first + rest].sort_values(["date", "ticker"]).reset_index(drop=True)

    totals = panel.groupby("date")["benchmark_weight"].sum()
    if not totals.between(1.0 - 1e-6, 1.0 + 1e-6).all():
        raise ValueError(f"Poids benchmark non normalises:\n{totals}")

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL_FILE, index=False)
    return panel
