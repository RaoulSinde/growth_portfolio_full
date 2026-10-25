"""Chargement des donnees de marche et fondamentales pour le moteur quant.

Pipeline Point-in-Time (PIT) trimestriel base sur l'inventaire N-PORT SEC
(`data/inventaire_consolide_S000079196.csv`) consolide dans
`data/universe_quarterly.csv`.

Principes :
- Le panier/poids d'un `report_period` n'est connu qu'a sa `filing_date`
  (= `rebalance_date`). Aucune donnee future n'est utilisee.
- Le cache local (`data/cache/*.parquet`) evite de solliciter l'API Yahoo
  Finance a chaque appel (notamment a chaque clic dans Streamlit).
- SECURITE : on n'utilise JAMAIS l'attribut `info` d'un Ticker pour des
  fondamentaux (fuite temporelle). Seuls `.info['symbol']` / `.info['sector']`
  ont pu servir a l'identification (hors de ce module).

Note : `get_macro` a ete retire (aucune dependance macro dans le pipeline).
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)

__all__ = [
    "load_universe",
    "get_rebalance_dates",
    "get_universe",
    "get_index_weights",
    "get_index_sector_weights",
    "get_sector_map",
    "get_prices",
    "get_benchmark",
    "get_benchmarks",
    "get_fundamentals",
    "get_earnings_dates",
    "fundamentals_asof",
    "annual_flow_points",
    "annual_stock_points",
    "clear_cache",
]

# --- Chemins de cache -------------------------------------------------------
PRICES_CACHE = os.path.join(config.CACHE_DIR, "prices.parquet")
BENCH_CACHE = os.path.join(config.CACHE_DIR, "benchmark.parquet")
FUND_CACHE = os.path.join(config.CACHE_DIR, "fundamentals.parquet")
FUND_SEC_PARQUET = os.path.join(config.DATA_DIR, "fundamentals_sec.parquet")
EARN_CACHE = os.path.join(config.CACHE_DIR, "earnings_dates.parquet")
ISIN_CACHE = os.path.join(config.CACHE_DIR, "isin_ticker.parquet")

# Tickers sans historique Yahoo Finance / delistes (exclus de l'univers eligible
# de l'ancien `universe.csv`, conserve pour retro-compatibilite).
_INELIGIBLE_TICKERS = frozenset({"EA", "PSTG", "SATS"})

# Batches de telechargement (robustesse reseau).
_BATCH_SIZE = 40

# Tolérance (jours) pour considérer un `period_end` Yahoo comme le doublon d'un
# trimestre SEC : Yahoo arrondit la fin de trimestre à la fin de mois, SEC non.
_SEC_YF_PERIOD_TOL = pd.Timedelta(days=45)


# ===========================================================================
# Helpers generiques
# ===========================================================================
def _normalize_ticker(ticker: str) -> str:
    """Normalise un ticker au format yfinance (BRK.B -> BRK-B, etc.)."""
    return str(ticker).strip().replace(".", "-").replace("/", "-")


def _to_ts(date) -> pd.Timestamp:
    """Convertit en Timestamp timezone-naive normalise a minuit."""
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _end_exclusive(end: pd.Timestamp) -> pd.Timestamp:
    """`yfinance` exclut la borne de fin : on ajoute 1 jour pour inclure `end`."""
    return _to_ts(end) + pd.Timedelta(days=1)


def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    """Lit un parquet en cache, retourne None si absent/illisible."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - cache corrompu
        logger.warning("Cache illisible %s : %s", path, exc)
        return None


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    """Ecrit un DataFrame/Série au format parquet (cree le dossier au besoin)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.to_parquet(path)
    except Exception as exc:  # pragma: no cover - disque indisponible
        logger.warning("Ecriture cache impossible %s : %s", path, exc)


def clear_cache() -> None:
    """Supprime tous les fichiers de cache locaux (dont les benchmarks par ticker)."""
    import glob
    paths = [PRICES_CACHE, BENCH_CACHE, FUND_CACHE, EARN_CACHE, ISIN_CACHE]
    paths += glob.glob(os.path.join(config.CACHE_DIR, "benchmark_*.parquet"))
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Suppression cache impossible %s : %s", path, exc)


def _extract_close(data: pd.DataFrame) -> pd.DataFrame:
    """Extrait les colonnes de cloture ajustee d'un retour `yf.download`."""
    if data is None or len(data) == 0:
        return pd.DataFrame()
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        lvl0 = list(cols.get_level_values(0))
        if "Close" in lvl0:
            close = data["Close"].copy()
        elif "Adj Close" in lvl0:
            close = data["Adj Close"].copy()
        else:
            close = data.xs(lvl0[0], axis=1, level=0).copy()
    else:
        if "Close" in cols:
            close = data["Close"]
        elif "Adj Close" in cols:
            close = data["Adj Close"]
        else:
            close = data.iloc[:, 0]
        if isinstance(close, pd.Series):
            close = close.to_frame()
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()
    close = close.loc[:, ~close.columns.duplicated()]
    return close


def _download_close(tickers: Sequence[str], start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    """Telecharge la cloture ajustee par lots avec gestion d'erreur reseau."""
    tickers = [t for t in tickers]
    if not tickers:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), _BATCH_SIZE):
        chunk = tickers[i:i + _BATCH_SIZE]
        try:
            data = yf.download(
                chunk,
                start=start,
                end=_end_exclusive(end),
                auto_adjust=True,   # Close == Adj Close
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as exc:
            logger.warning("Echec telechargement %s: %s", chunk[:3], exc)
            continue
        close = _extract_close(data)
        if not close.empty:
            frames.append(close)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.sort_index()
    return out


# ===========================================================================
# Univers PIT trimestriel
# ===========================================================================
def get_rebalance_dates() -> list[pd.Timestamp]:
    """Dates de rebalancement (= `filing_date`) triees, depuis `config`."""
    return sorted({_to_ts(d) for d in config.REBALANCE_DATES})


def _load_universe_quarterly() -> pd.DataFrame:
    """Charge `data/universe_quarterly.csv` (poids PIT reels par trimestre)."""
    if not os.path.exists(config.UNIVERSE_QUARTERLY_CSV):
        raise FileNotFoundError(
            f"{config.UNIVERSE_QUARTERLY_CSV} absent. "
            "Lancer `python -m src.build_universe`."
        )
    df = pd.read_csv(config.UNIVERSE_QUARTERLY_CSV)
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"]).dt.normalize()
    df["report_period"] = pd.to_datetime(df["report_period"]).dt.normalize()
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    return df


def get_universe(asof: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Constituants PIT du (des) trimestre(s).

    `asof=None` -> tout l'historique trimestriel. Sinon, on renvoie le panier du
    dernier trimestre dont la `rebalance_date` <= `asof` (i.e. la derniere
    `filing_date` connue a cette date). Colonnes : rebalance_date, report_period,
    ticker, name, isin, cusip, weight.
    """
    uni = _load_universe_quarterly()
    if asof is None:
        return uni.reset_index(drop=True)
    asof = _to_ts(asof)
    avail = sorted(d for d in uni["rebalance_date"].unique() if d <= asof)
    if not avail:
        return uni.iloc[0:0].reset_index(drop=True)
    last = max(avail)
    return uni[uni["rebalance_date"] == last].reset_index(drop=True)


def get_sector_map() -> pd.Series:
    """Mapping ticker -> secteur GICS (`data/sector_map.csv`)."""
    df = pd.read_csv(config.SECTOR_MAP_CSV)
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    return df.set_index("ticker")["gics_sector"]


def _index_weights_long(dates: Optional[Sequence]) -> pd.DataFrame:
    """Poids PIT (long) : colonnes date, ticker, weight ; somme=1 par date."""
    uni = _load_universe_quarterly()
    if dates is None:
        dates = get_rebalance_dates()
    dates = sorted({_to_ts(d) for d in dates})

    sub = uni[uni["rebalance_date"].isin(dates)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "ticker", "weight"])

    # Renormalisation de securite a 1.0 par date.
    grp = sub.groupby("rebalance_date")["weight"].transform("sum")
    sub["weight"] = sub["weight"] / grp
    return sub.rename(columns={"rebalance_date": "date"})[["date", "ticker", "weight"]]


def get_index_weights(dates: Optional[Sequence] = None):
    """Poids PIT reels de l'indice (pas de derive), source `universe_quarterly`.

    MultiIndex `(date, ticker)` avec colonne `weight`, chaque date somme a 1
    (poids exacts du trimestre tel que publie dans le N-PORT a la `filing_date`).
    Si `dates` ne contient qu'une seule date, renvoie une `pd.Series`
    (SingleIndex) indexee par ticker — retro-compatibilite d'usage.
    """
    long = _index_weights_long(dates)
    if long.empty:
        return long
    if long["date"].nunique() == 1:
        d = long["date"].iloc[0]
        ser = long.set_index("ticker")["weight"]
        ser.name = "weight"
        ser.index.name = "ticker"
        logger.debug("get_index_weights(%s) -> Series (%d tickers)", d, len(ser))
        return ser
    out = long.set_index(["date", "ticker"])["weight"].to_frame("weight")
    out.index.names = ["date", "ticker"]
    return out


def get_index_sector_weights(dates: Optional[Sequence] = None) -> pd.DataFrame:
    """Poids sectoriels de l'indice, index=dates, colonnes=secteurs GICS (Σ=1)."""
    long = _index_weights_long(dates)
    if long.empty:
        return pd.DataFrame(columns=config.GICS_SECTORS)
    smap = get_sector_map()
    long = long.copy()
    long["gics_sector"] = long["ticker"].map(smap).fillna("Unknown")
    sec = (long.groupby(["date", "gics_sector"])["weight"].sum()
               .unstack(fill_value=0.0))
    sec = sec.reindex(columns=config.GICS_SECTORS, fill_value=0.0)
    sec = sec.sort_index()
    totals = sec.sum(axis=1).replace(0.0, np.nan)
    sec = sec.div(totals, axis=0).fillna(0.0)
    sec.index.name = "date"
    return sec


def load_universe(eligible: bool = True) -> pd.DataFrame:
    """DEPRECATED — ancien univers statique (`data/universe.csv`).

    Conserve pour retro-compatibilite (benchmark synthetique, scripts legacy).
    Le pipeline multi-dates utilise desormais `get_universe()` /
    `get_index_weights()` bases sur `data/universe_quarterly.csv`.
    """
    df = pd.read_csv(config.UNIVERSE_CSV)
    df.columns = [str(c).strip() for c in df.columns]
    df["ticker"] = df["ticker"].map(_normalize_ticker)

    if eligible:
        df = df[~df["ticker"].isin(_INELIGIBLE_TICKERS)].copy()
        total = df["weight_pct"].sum()
        if total > 0:
            df["weight_pct"] = df["weight_pct"] / total * 100.0

    df = df.reset_index(drop=True)
    return df[["ticker", "name", "prop_sector", "gics_sector", "weight_pct"]]


# ===========================================================================
# Prix actions
# ===========================================================================
def get_prices(tickers: Optional[Iterable[str]] = None,
               start: str = config.PRICE_START,
               end: str = config.PRICE_END) -> pd.DataFrame:
    """Cours de cloture ajustes (Adj Close), index date, colonnes tickers.

    Par defaut : union de TOUS les tickers de `universe_quarterly` (2023-01 ->
    2026-09-30). Lit `data/cache/prices.parquet` s'il existe, sinon telecharge
    puis met en cache. Les tickers sans donnees sur la fenetre sont exclus.
    """
    if tickers is None:
        uni = _load_universe_quarterly()
        tickers = sorted(uni["ticker"].unique())
    tickers = [_normalize_ticker(t) for t in tickers]
    start_ts, end_ts = _to_ts(start), _to_ts(end)

    cache = _read_parquet(PRICES_CACHE)
    if cache is None or cache.empty:
        cache = pd.DataFrame()

    if not cache.empty:
        cache.index = pd.to_datetime(cache.index)
        if cache.index.tz is not None:
            cache.index = cache.index.tz_localize(None)
        cache.index = cache.index.normalize()

    missing = [t for t in tickers if t not in cache.columns]
    if missing:
        new = _download_close(missing, start_ts, end_ts)
        if not new.empty:
            cache = new if cache.empty else cache.join(new, how="outer")

    if cache.empty:
        return pd.DataFrame(columns=tickers)

    cache = cache.sort_index()
    cache = cache.loc[:, ~cache.columns.duplicated()]
    _write_parquet(cache, PRICES_CACHE)

    present = [t for t in tickers if t in cache.columns]
    out = cache[present]
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    # Exclut les colonnes entierement vides sur la fenetre demandee.
    out = out.loc[:, out.notna().any(axis=0)]
    return out


# ===========================================================================
# Benchmark
# ===========================================================================
def _synthetic_benchmark(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Indice synthetique = derive quotidienne des poids initiaux (legacy)."""
    uni = load_universe(eligible=False)
    tickers = list(uni["ticker"])
    w0 = uni.set_index("ticker")["weight_pct"]
    w0 = (w0 / w0.sum()).reindex(tickers).fillna(0.0)

    prices = get_prices(tickers, start, end)
    if prices.empty or len(prices) < 2:
        return pd.Series(dtype=float, name="Benchmark")

    rets = prices.pct_change(fill_method=None).fillna(0.0)
    growth = (1.0 + rets).cumprod()
    level = growth.mul(w0, axis=1).sum(axis=1)
    first = level.iloc[0]
    if first and np.isfinite(first):
        level = level / first * 100.0
    level.name = "Benchmark"
    return level


def _bench_cache_path(ticker: str) -> str:
    """Chemin de cache par ticker de benchmark (ex. `EGUS` -> benchmark_EGUS.parquet)."""
    key = _normalize_ticker(ticker).lstrip("^").split("-")[0].strip()
    return os.path.join(config.CACHE_DIR, f"benchmark_{key or 'bench'}.parquet")


def _normalize_bench_index(ser: pd.Series) -> pd.Series:
    """Normalise l'index d'une serie de prix (timezone-naive, minuit)."""
    ser.index = pd.to_datetime(ser.index)
    if ser.index.tz is not None:
        ser.index = ser.index.tz_localize(None)
    ser.index = ser.index.normalize()
    return ser


def _load_benchmark_series(ticker: str, start_ts: pd.Timestamp,
                           end_ts: pd.Timestamp) -> Optional[pd.Series]:
    """Charge (cache par ticker) ou telecharge la serie de prix d'un benchmark."""
    path = _bench_cache_path(ticker)
    cached = _read_parquet(path)
    if cached is not None and not cached.empty:
        return _normalize_bench_index(cached.iloc[:, 0])

    try:
        data = yf.download(
            ticker,
            start=start_ts,
            end=_end_exclusive(end_ts),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = _extract_close(data)
        if not close.empty:
            ser = close.iloc[:, 0]
            _write_parquet(ser.rename(ticker).to_frame(), path)
            return ser
    except Exception as exc:
        logger.warning("Echec benchmark %s : %s", ticker, exc)
    return None


def _clean_benchmark(ser: Optional[pd.Series], start_ts: pd.Timestamp,
                     end_ts: pd.Timestamp) -> Optional[pd.Series]:
    """Nettoie (dropna) et borne une serie de benchmark sur la fenetre demandee."""
    if ser is None or ser.empty:
        return None
    ser = ser.dropna()
    ser = ser.loc[(ser.index >= start_ts) & (ser.index <= end_ts)]
    return ser if not ser.empty else None


def get_benchmark(start: str = config.PRICE_START,
                  end: str = config.PRICE_END,
                  ticker: str = config.BENCHMARK_TICKER) -> pd.Series:
    """Serie de prix du benchmark de reference (defaut : EGUS). Rétro-compatible."""
    start_ts, end_ts = _to_ts(start), _to_ts(end)
    ser = _clean_benchmark(_load_benchmark_series(ticker, start_ts, end_ts),
                           start_ts, end_ts)

    n_bus = len(pd.bdate_range(start_ts, end_ts))
    coverage = (len(ser) / n_bus) if (ser is not None and n_bus) else 0.0
    if ser is None or coverage < 0.90:
        logger.info("Benchmark %s absent/incomplet (%.1f%%) -> synthetique.",
                    ticker, coverage * 100)
        ser = _synthetic_benchmark(start_ts, end_ts)

    ser = ser.copy()
    ser.name = "Benchmark"
    return ser


def get_benchmarks(start: str = config.PRICE_START,
                   end: str = config.PRICE_END) -> dict[str, pd.Series]:
    """Benchmark unique : {"EGUS": serie de prix}. Cache par ticker ; fallback
    synthetique si le ticker est absent/incomplet."""
    start_ts, end_ts = _to_ts(start), _to_ts(end)
    n_bus = len(pd.bdate_range(start_ts, end_ts))
    out: dict[str, pd.Series] = {}

    for name, ticker in config.BENCHMARKS.items():
        ser = _clean_benchmark(_load_benchmark_series(ticker, start_ts, end_ts),
                               start_ts, end_ts)
        coverage = (len(ser) / n_bus) if (ser is not None and n_bus) else 0.0
        if ser is None or coverage < 0.90:
            logger.info("Benchmark %s (%s) absent/incomplet (%.1f%%) -> synthetique.",
                        name, ticker, coverage * 100)
            ser = _synthetic_benchmark(start_ts, end_ts)
        ser = ser.copy()
        ser.name = name
        out[name] = ser

    return out


# ===========================================================================
# Fondamentaux Point-in-Time
# ===========================================================================
def _default_tickers() -> list[str]:
    """Union des tickers de l'univers trimestriel (fallback univers statique)."""
    try:
        uni = _load_universe_quarterly()
        return sorted(uni["ticker"].unique())
    except FileNotFoundError:
        return list(load_universe(eligible=False)["ticker"])


def get_earnings_dates(tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Dates de publication des resultats et surprise (%) par ticker.

    Colonnes : ticker, earnings_date, surprise_pct.
    """
    if tickers is None:
        tickers = _default_tickers()
    tickers = [_normalize_ticker(t) for t in tickers]

    cache = _read_parquet(EARN_CACHE)
    if cache is None or cache.empty:
        cache = pd.DataFrame(columns=["ticker", "earnings_date", "surprise_pct"])
        have: set[str] = set()
    else:
        cache["earnings_date"] = pd.to_datetime(cache["earnings_date"])
        have = set(cache["ticker"].unique())

    missing = [t for t in tickers if t not in have]
    buffer: list[dict] = []

    def _flush(rows: list[dict]) -> None:
        nonlocal cache
        if not rows:
            return
        new = pd.DataFrame(rows)
        cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
        cache = cache.drop_duplicates(subset=["ticker", "earnings_date"])
        _write_parquet(cache, EARN_CACHE)

    for n, t in enumerate(missing, 1):
        try:
            ed = yf.Ticker(t).earnings_dates
        except Exception as exc:
            logger.warning("Echec earnings_dates %s : %s", t, exc)
            continue
        if ed is None or len(ed) == 0:
            continue
        ed = ed.reset_index()
        date_col = ed.columns[0]
        for _, r in ed.iterrows():
            buffer.append({
                "ticker": t,
                "earnings_date": _to_ts(r[date_col]),
                "surprise_pct": pd.to_numeric(r.get("Surprise(%)", np.nan),
                                              errors="coerce"),
            })
        if n % 20 == 0:
            _flush(buffer)
            buffer = []
    _flush(buffer)

    out = cache[cache["ticker"].isin(tickers)].reset_index(drop=True)
    return out


def _pick(df: Optional[pd.DataFrame], candidates: Sequence[str]) -> Optional[pd.Series]:
    """Cherche une ligne de statement parmi plusieurs libelles possibles."""
    if df is None or getattr(df, "empty", True):
        return None
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    lower = {str(i).strip().lower(): i for i in df.index}
    for c in candidates:
        key = c.strip().lower()
        if key in lower:
            return df.loc[lower[key]]
    return None


def _value_at(row: Optional[pd.Series], period_end: pd.Timestamp) -> float:
    """Valeur d'une ligne de statement pour une periode donnee."""
    if row is None:
        return np.nan
    for c in row.index:
        if pd.Timestamp(c) == pd.Timestamp(period_end):
            return pd.to_numeric(row[c], errors="coerce")
    return np.nan


def _publication_date(period_end: pd.Timestamp,
                      ed_dates: Sequence[pd.Timestamp],
                      ed_surprise: dict) -> tuple[pd.Timestamp, float]:
    """Date de publication PIT + surprise associee.

    `pub_date` = premiere date d'earnings >= period_end - 5j (bornee a period_end),
    sinon period_end + 45 jours (lag conservateur).
    """
    tol = pd.Timedelta(days=5)
    cands = [d for d in ed_dates if d >= period_end - tol]
    if cands:
        pub = min(cands)
        surprise = ed_surprise.get(pub, np.nan)
        return max(pub, period_end), surprise
    return period_end + pd.Timedelta(days=45), np.nan


def _fetch_fundamentals_one(ticker: str) -> pd.DataFrame:
    """Construit le panel PIT pour un ticker a partir de ses statements."""
    try:
        tk = yf.Ticker(ticker)
        inc = tk.quarterly_income_stmt
        bs = tk.quarterly_balance_sheet
        cf = tk.quarterly_cashflow
        fin = tk.quarterly_financials
    except Exception as exc:
        logger.warning("Echec fondamentaux %s : %s", ticker, exc)
        return pd.DataFrame()

    period_ends: set[pd.Timestamp] = set()
    for df in (inc, bs, cf, fin):
        if df is not None and getattr(df, "empty", True) is False:
            period_ends.update(_to_ts(c) for c in df.columns)
    if not period_ends:
        return pd.DataFrame()
    period_ends = sorted(period_ends, reverse=True)

    ed = get_earnings_dates([ticker])
    ed_dates = [_to_ts(d) for d in ed["earnings_date"].tolist()] if not ed.empty else []
    ed_surprise = {_to_ts(r["earnings_date"]): r["surprise_pct"]
                   for _, r in ed.iterrows()} if not ed.empty else {}

    rows: list[dict] = []
    for pe in period_ends:
        pub, surprise = _publication_date(pe, ed_dates, ed_surprise)
        rows.append({
            "ticker": ticker,
            "period_end": pe,
            "pub_date": pub,
            "total_assets": _value_at(_pick(bs, ["Total Assets"]), pe),
            "stockholders_equity": _value_at(
                _pick(bs, ["Stockholders Equity", "Total Stockholder Equity",
                           "Common Stock Equity", "Total Equity Gross Minority Interest"]), pe),
            "total_debt": _value_at(_pick(bs, ["Total Debt"]), pe),
            "shares": _value_at(
                _pick(bs, ["Ordinary Shares Number", "Share Issued"]), pe),
            "net_income_q": _value_at(
                _pick(inc, ["Net Income", "Net Income Common Stockholders",
                            "Net Income Continuous Operations"]), pe),
            "gross_profit_q": _value_at(_pick(inc, ["Gross Profit"]), pe),
            "ocf_q": _value_at(
                _pick(cf, ["Operating Cash Flow",
                           "Total Cash From Operating Activities",
                           "Cash Flow From Continuing Operating Activities"]), pe),
            "revenue_q": _value_at(
                _pick(inc, ["Total Revenue", "Operating Revenue"]), pe),
            "research_dev_q": _value_at(
                _pick(inc, ["Research And Development", "Research Development"]), pe),
            "capex_q": _value_at(
                _pick(cf, ["Capital Expenditure", "Capital Expenditures",
                           "Purchase Of PPE"]), pe),
            "buyback_q": _value_at(
                _pick(cf, ["Repurchase Of Capital Stock", "Repurchase Of Equity",
                           "Common Stock Payments"]), pe),
            "dividends_q": _value_at(
                _pick(cf, ["Common Stock Dividend Paid", "Cash Dividends Paid",
                           "Common Dividends Paid"]), pe),
            "surprise_pct": surprise,
        })
    return pd.DataFrame(rows)


_FUND_COLUMNS = [
    "ticker", "period_end", "pub_date", "total_assets", "stockholders_equity",
    "total_debt", "shares", "net_income_q", "gross_profit_q",
    "operating_income_q", "ocf_q",
    "revenue_q", "research_dev_q", "capex_q", "buyback_q", "dividends_q",
    "surprise_pct",
]


def get_fundamentals(tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """PANEL POINT-IN-TIME : une ligne par (ticker, period_end).

    `pub_date` est la date de publication (earnings >= period_end, sinon +45j).
    Aucune donnee issue de l'attribut `info` d'un Ticker.
    """
    if tickers is None:
        tickers = _default_tickers()
    tickers = [_normalize_ticker(t) for t in tickers]

    cache = _read_parquet(FUND_CACHE)
    if cache is None or cache.empty:
        cache = pd.DataFrame(columns=_FUND_COLUMNS)
        have: set[str] = set()
    else:
        for col in ("period_end", "pub_date"):
            if col in cache.columns:
                cache[col] = pd.to_datetime(cache[col])
        have = set(cache["ticker"].unique())

    missing = [t for t in tickers if t not in have]
    buffer: list[pd.DataFrame] = []

    def _flush(frames: list[pd.DataFrame]) -> None:
        nonlocal cache
        frames = [f for f in frames if not f.empty]
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)
        cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
        cache = cache.drop_duplicates(subset=["ticker", "period_end"], keep="last")
        _write_parquet(cache, FUND_CACHE)

    for n, t in enumerate(missing, 1):
        buffer.append(_fetch_fundamentals_one(t))
        if n % 20 == 0:
            _flush(buffer)
            buffer = []
    _flush(buffer)

    out = cache[cache["ticker"].isin(tickers)].copy()
    out = out.reindex(columns=_FUND_COLUMNS)

    # --- Union avec les fondamentaux SEC XBRL (historique 2023-2024) --------
    # La SEC fournit `pub_date` = date de depot (`filed`) : PIT strict. On la
    # prefere la ou elle existe et on complete avec Yahoo pour les periodes
    # absentes (typiquement les trimestres les plus recents).
    sec = _read_parquet(FUND_SEC_PARQUET)
    if sec is not None and not sec.empty:
        sec = sec.copy()
        for col in ("period_end", "pub_date"):
            if col in sec.columns:
                sec[col] = pd.to_datetime(sec[col])
        sec = sec[sec["ticker"].isin(tickers)]
        # Report de la surprise Yahoo sur les periodes SEC quand disponible.
        sur = (out[["ticker", "period_end", "surprise_pct"]]
               .dropna(subset=["surprise_pct"]))
        if not sur.empty:
            sec = sec.drop(columns=["surprise_pct"]).merge(
                sur, on=["ticker", "period_end"], how="left")
        sec = sec.reindex(columns=_FUND_COLUMNS)
        sec = sec.drop_duplicates(subset=["ticker", "period_end"], keep="last")
        # Ne conserver de Yahoo que les trimestres réellement absents de SEC :
        # Yahoo arrondit `period_end` à la fin de mois, donc une dédup sur la
        # date exacte laisse survivre un quasi-doublon qui décalerait la fenêtre
        # TTM. On retire les lignes Yahoo à moins de `_SEC_YF_PERIOD_TOL` d'un
        # `period_end` SEC du même ticker.
        out = out.reset_index(drop=True)
        if not out.empty:
            pairs = out[["ticker", "period_end"]].reset_index().merge(
                sec[["ticker", "period_end"]], on="ticker",
                how="inner", suffixes=("", "_sec"))
            if not pairs.empty:
                close = ((pairs["period_end"] - pairs["period_end_sec"]).abs()
                         <= _SEC_YF_PERIOD_TOL)
                dup_idx = pairs.loc[close, "index"].unique()
                if len(dup_idx):
                    out = out.drop(index=dup_idx)
        out = pd.concat([sec, out], ignore_index=True)
        out = out.drop_duplicates(subset=["ticker", "period_end"], keep="first")

    return out.reset_index(drop=True)


def fundamentals_asof(fund: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Derniere ligne disponible (pub_date <= asof) pour chaque ticker."""
    if fund is None or fund.empty:
        return pd.DataFrame(columns=_FUND_COLUMNS)
    asof = _to_ts(asof)
    df = fund.copy()
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df = df[df["pub_date"] <= asof]
    if df.empty:
        return df.set_index("ticker")
    df = df.sort_values("pub_date").groupby("ticker").tail(1)
    return df.set_index("ticker")


# ===========================================================================
# Agregats annuels Point-in-Time (points N / N-1 / N-2 / N-3)
# ===========================================================================
def _annual_points(fund: pd.DataFrame, col: str, tickers,
                   kind: str = "flow") -> pd.DataFrame:
    """Points annuels ``N, N-1, N-2, N-3`` par titre, depuis un panel PIT.

    ``fund`` doit etre **deja filtre** (``pub_date <= t``) par l'appelant : ce
    helper ne fait aucune coupe temporelle, il agrege l'historique disponible.

    - ``kind='flow'`` : agrégats de flux (CA, résultat op., résultat net, OCF,
      CapEx) -> **somme des 4 derniers trimestres** (N), des 4 précédents (N-1),
      etc. NaN si l'historique est trop court (4/8/12/16 trimestres).
    - ``kind='stock'`` : agrégats de stock (Total Assets, Equity, shares) ->
      valeur au dernier ``period_end`` (N), puis 4 trimestres plus tôt (N-1),
      etc.

    Retourne un ``DataFrame`` indexé par ticker, colonnes ``["N","N1","N2","N3"]``.
    """
    idx = pd.Index(tickers).unique()
    out = pd.DataFrame(np.nan, index=idx,
                       columns=["N", "N1", "N2", "N3"], dtype=float)
    if fund is None or col not in getattr(fund, "columns", []):
        return out
    tmp = fund[["ticker", "period_end", col]].copy()
    tmp["period_end"] = pd.to_datetime(tmp["period_end"])
    tmp = tmp.dropna(subset=[col])
    tmp = tmp.drop_duplicates(subset=["ticker", "period_end"], keep="last")
    tmp = tmp.sort_values(["ticker", "period_end"])
    if tmp.empty:
        return out

    for ticker, g in tmp.groupby("ticker", sort=False):
        if ticker not in out.index:
            continue
        v = g[col].to_numpy(dtype=float)
        n = v.size
        if kind == "flow":
            if n >= 4:
                out.at[ticker, "N"] = float(v[-4:].sum())
            if n >= 8:
                out.at[ticker, "N1"] = float(v[-8:-4].sum())
            if n >= 12:
                out.at[ticker, "N2"] = float(v[-12:-8].sum())
            if n >= 16:
                out.at[ticker, "N3"] = float(v[-16:-12].sum())
        else:  # stock
            if n >= 1:
                out.at[ticker, "N"] = float(v[-1])
            if n >= 5:
                out.at[ticker, "N1"] = float(v[-5])
            if n >= 9:
                out.at[ticker, "N2"] = float(v[-9])
            if n >= 13:
                out.at[ticker, "N3"] = float(v[-13])
    return out


def annual_flow_points(fund: pd.DataFrame, col: str, tickers) -> pd.DataFrame:
    """Points annuels d'un agrégat de FLUX (somme de 4 trimestres). Cf. `_annual_points`."""
    return _annual_points(fund, col, tickers, kind="flow")


def annual_stock_points(fund: pd.DataFrame, col: str, tickers) -> pd.DataFrame:
    """Points annuels d'un agrégat de STOCK (valeur ponctuelle). Cf. `_annual_points`."""
    return _annual_points(fund, col, tickers, kind="stock")
