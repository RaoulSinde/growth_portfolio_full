"""Univers PIT, prix et benchmark MSCI USA Growth."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import msci_config as cfg

_EXPECTED_CUSIPS = 494
_TOP_SECTOR_RE = re.compile(r"^(\d{2})\s+(.+)$")
_INDEX_SYMBOL = "105825"


def load_weights() -> pd.DataFrame:
    df = pd.read_parquet(cfg.MSCI_WEIGHTS)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["cusip"] = df["cusip"].astype(str)
    df = df[["date", "ticker", "cusip", "name", "weight"]].copy()
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("index_weights: plusieurs lignes pour un meme (date, ticker)")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_prices(adj: bool = False) -> pd.DataFrame:
    path = cfg.COMPONENT_PRICES_ADJ if adj else cfg.COMPONENT_PRICES
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [str(c) for c in df.columns]
    return df


def load_benchmark() -> pd.Series:
    df = pd.read_parquet(cfg.MSCI_PRICES)
    df["date"] = pd.to_datetime(df["date"])
    ser = df.set_index("date")["index_level"].sort_index()
    ser.name = "index_level"
    return ser


def load_reconstructed() -> pd.DataFrame:
    df = pd.read_parquet(cfg.RECONSTRUCTED)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_ticker_map() -> pd.DataFrame:
    df = pd.read_parquet(cfg.TICKER_MAP)
    df["cusip"] = df["cusip"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    return df[["cusip", "ticker"]].drop_duplicates("cusip").reset_index(drop=True)


def load_current_cik_tickers() -> set[str]:
    with open(cfg.SEC_TICKERS_CACHE, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.values() if isinstance(payload, dict) else payload
    tickers = {
        str(row["ticker"]).strip().upper()
        for row in rows
        if row.get("ticker") and row.get("cik_str") is not None
    }
    return {t for t in tickers if t}


def build_pit_sector_map() -> pd.DataFrame:
    raw = pd.read_excel(cfg.CONSTITUENTS_XLSX, header=5)
    weight_cols = [c for c in raw.columns if str(c).endswith(" Weight")]
    if not weight_cols:
        raise ValueError("Aucune colonne de poids dans le fichier des constituants")

    sector: str | None = None
    sectors: list[str | None] = []
    for _, row in raw.iterrows():
        symbol = row["Symbol"]
        name = str(row["Name"]).strip()
        if pd.isna(symbol):
            match = _TOP_SECTOR_RE.match(name)
            if match:
                sector = match.group(2).strip()
        sectors.append(sector)
    raw = raw.assign(sector=sectors)

    symbol = raw["Symbol"].astype("string").str.strip()
    mask = symbol.notna() & (symbol != _INDEX_SYMBOL)
    titles = raw.loc[mask, ["Name", "Symbol", "sector", *weight_cols]].copy()
    titles["cusip"] = titles["Symbol"].astype(str)
    if titles["sector"].isna().any():
        raise ValueError("Constituants sans secteur de niveau 2")

    long = titles.melt(
        id_vars=["Name", "cusip", "sector"],
        value_vars=weight_cols,
        var_name="snapshot",
        value_name="weight",
    )
    long["date"] = pd.to_datetime(
        long["snapshot"].str.replace(" Weight", "", regex=False),
        format="%d-%b-%Y",
    )
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce").fillna(0.0)
    long = long[long["weight"] > 0].copy()

    n_sectors = long.groupby(["date", "cusip"])["sector"].nunique()
    if (n_sectors > 1).any():
        bad = n_sectors[n_sectors > 1]
        raise ValueError(f"Plusieurs secteurs positifs pour (date, cusip): {bad.head().to_dict()}")

    ticker_map = load_ticker_map()
    out = long.merge(ticker_map, on="cusip", how="left", validate="many_to_one")
    out = out[out["ticker"].notna()].copy()

    n_cusips = out["cusip"].nunique()
    if n_cusips != _EXPECTED_CUSIPS:
        raise ValueError(f"Couverture secteur: {n_cusips} CUSIP au lieu de {_EXPECTED_CUSIPS}")

    out = out.rename(columns={"Name": "name"})
    return (
        out[["date", "cusip", "ticker", "sector", "weight"]]
        .sort_values(["date", "cusip"])
        .reset_index(drop=True)
    )


def quarter_end_dates(weights: pd.DataFrame | None = None) -> list[pd.Timestamp]:
    if weights is None:
        weights = load_weights()
    snapshots = pd.DatetimeIndex(sorted(pd.to_datetime(weights["date"]).unique()))
    out: list[pd.Timestamp] = []
    for year in range(cfg.START.year, cfg.END.year + 1):
        for month in (3, 6, 9, 12):
            candidates = snapshots[(snapshots.year == year) & (snapshots.month == month)]
            if len(candidates):
                out.append(pd.Timestamp(candidates.max()))
    out = sorted(out)
    if pd.Timestamp("2020-12-31") not in out or pd.Timestamp("2025-12-31") not in out:
        raise ValueError("Les dates trimestrielles doivent inclure 2020-12-31 et 2025-12-31")
    if len(out) != 21:
        raise ValueError(f"21 dates trimestrielles attendues, {len(out)} obtenues")
    return out


def build_universe() -> dict:
    weights = load_weights()
    prices = load_prices()
    sectors = build_pit_sector_map()
    cik_tickers = load_current_cik_tickers()

    positive = set(weights.loc[weights["weight"] > 0, "ticker"])
    tickers = sorted(cik_tickers & set(prices.columns) & positive)

    weights_u = weights[weights["ticker"].isin(tickers)].copy()
    prices_u = prices.loc[:, tickers].copy()
    sectors_u = sectors[sectors["ticker"].isin(tickers)].copy()

    return {
        "tickers": tickers,
        "weights": weights_u,
        "prices": prices_u,
        "sectors": sectors_u,
        "dates": quarter_end_dates(weights),
        "benchmark": load_benchmark(),
        "reconstructed": load_reconstructed(),
    }


def eligible_universe(universe: dict, date: pd.Timestamp) -> pd.DataFrame:
    date = pd.Timestamp(date)
    weights = universe["weights"]
    selected = weights[(weights["date"] == date) & (weights["weight"] > 0)][["ticker", "weight"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=["ticker", "weight", "sector", "price"])

    prices = universe["prices"]
    index = prices.index.union(pd.DatetimeIndex([date])).sort_values()
    px = prices.reindex(index).ffill(limit=5).loc[date]
    selected["price"] = selected["ticker"].map(px)

    sectors = universe["sectors"]
    sectors_date = sectors[sectors["date"] == date][["ticker", "sector"]].drop_duplicates("ticker")
    selected = selected.merge(sectors_date, on="ticker", how="left")
    selected = selected[selected["price"].notna() & (selected["price"] > 0)].copy()
    if selected["sector"].isna().any():
        missing = selected.loc[selected["sector"].isna(), "ticker"].tolist()
        raise ValueError(f"Secteur manquant pour {date.date()}: {missing}")

    return (
        selected[["ticker", "weight", "sector", "price"]]
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def build_benchmark_excluding(
    universe: dict,
    excluded_tickers: list[str] | tuple[str, ...],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    tx_cost_bps: float = 0.0,
) -> pd.Series:
    """Reconstruit un benchmark pondere en excluant certains titres.

    Les poids d'indice sont renormalises a chaque date de rebalancement, puis
    les titres derivent en buy&hold jusqu'au rebalancement suivant. Les couts
    sont nuls par defaut afin de reproduire un indice de reference.
    """
    from . import backtest

    excluded = {str(ticker) for ticker in excluded_tickers}
    weights = universe["weights"]
    targets: dict[pd.Timestamp, pd.Series] = {}
    for date in universe["dates"]:
        day = weights[
            (weights["date"] == date)
            & (weights["weight"] > 0)
            & (~weights["ticker"].isin(excluded))
        ][["ticker", "weight"]].copy()
        if day.empty:
            raise ValueError(f"Aucun titre restant apres exclusion a {date.date()}")
        day["weight"] = day["weight"] / day["weight"].sum()
        targets[pd.Timestamp(date)] = day.set_index("ticker")["weight"]

    result = backtest.run_backtest(
        {"BenchmarkExcluded": targets},
        prices=universe["prices"],
        bench=universe["benchmark"],
        tx_cost_bps=tx_cost_bps,
        rebalance_dates=universe["dates"],
        start=start if start is not None else cfg.START,
        end=end if end is not None else cfg.END,
    )
    return result["BenchmarkExcluded"]["nav"]


def universe_summary(universe: dict) -> dict:
    constituents: list[int] = []
    eligible: list[int] = []
    coverage: list[float] = []
    weights = universe["weights"]

    for date in universe["dates"]:
        day_weights = weights[(weights["date"] == date) & (weights["weight"] > 0)]
        day_eligible = eligible_universe(universe, date)
        constituents.append(len(day_weights))
        eligible.append(len(day_eligible))
        total = day_weights["weight"].sum()
        coverage.append(float(day_eligible["weight"].sum() / total) if total > 0 else 0.0)

    sector_tickers = set(universe["sectors"]["ticker"])
    missing_sector = len(set(universe["tickers"]) - sector_tickers)

    return {
        "n_tickers": len(universe["tickers"]),
        "n_dates": len(universe["dates"]),
        "n_constituents_min": int(np.min(constituents)),
        "n_constituents_max": int(np.max(constituents)),
        "n_eligible_min": int(np.min(eligible)),
        "n_eligible_max": int(np.max(eligible)),
        "coverage_weight_min": float(np.min(coverage)),
        "coverage_weight_max": float(np.max(coverage)),
        "missing_sector_count": int(missing_sector),
    }
