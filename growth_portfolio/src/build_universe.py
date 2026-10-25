"""Construction de l'univers/ponderation PIT trimestriel de l'ETF EGUS.

Source : inventaire N-PORT SEC (`data/inventaire_consolide_S000079196.csv`).
On ne retient QUE les actions US (`asset_cat == 'EC'` ET `inv_country == 'US'`),
on exclut donc les fonds monetaires (STIV) et les futures/produits derives (DE).

Regles :
- `rebalance_date` = `filing_date` (date de connaissance publique des poids).
- Les poids (`pct_val`) sont renormalises a 1.0 PAR `report_period` sur les titres
  retenus (apres retrait des ISIN non resolus).
- Resolution ISIN -> ticker via yfinance, avec cache disque
  `data/cache/isin_ticker.parquet` (retries).

Production de :
- `data/universe_quarterly.csv`  (source de verite des poids PIT)
- `data/sector_map.csv`          (ticker -> secteur GICS)

Usage :  python -m src.build_universe
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)

# Mapping des libelles sectoriels Yahoo Finance -> 11 secteurs GICS de config.
YAHOO_TO_GICS: Dict[str, str] = {
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "communication services": "Communication Services",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "financial services": "Financials",
    "financials": "Financials",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "industrials": "Industrials",
    "energy": "Energy",
    "basic materials": "Materials",
    "materials": "Materials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
}

# Corrections manuelles lorsque Yahoo ne renvoie pas de secteur (gap de donnees).
SECTOR_OVERRIDES: Dict[str, str] = {
    "FISV": "Financials",  # Fiserv (ancien ticker, pas de `info['sector']`)
}


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().replace(".", "-").replace("/", "-")


# Certains ISIN US sont (mal) resolus par Yahoo vers une cote etrangere de la
# meme societe. On les ramene vers la cote primaire US (verifie manuellement).
_FOREIGN_ALIASES: Dict[str, str] = {
    "1WBD-MI": "WBD",   # Warner Bros Discovery (ISIN US9344231041)
    "LRCX-MX": "LRCX",  # Lam Research (ISIN US5128071082)
    "COO-MX": "COO",    # The Cooper Companies (ISIN US2166484020)
}


def _us_primary(symbol: Optional[str]) -> Optional[str]:
    """Ramene une cote etrangere connue vers son ticker primaire US."""
    if not symbol:
        return None
    sym = _normalize_ticker(symbol)
    return _FOREIGN_ALIASES.get(sym, sym)


def _norm_name(name: str) -> str:
    return " ".join(str(name).upper().replace(".", " ").replace(",", " ").split())


# ===========================================================================
# Resolution ISIN -> ticker (cache disque + retries)
# ===========================================================================
def resolve_isins(isins, cache_path: Optional[str] = None,
                  attempts: int = 3, sleep_s: float = 1.0) -> pd.Series:
    """Resout une collection d'ISIN vers leur ticker yfinance.

    Le cache `data/cache/isin_ticker.parquet` (colonnes `isin`, `ticker`) evite
    de re-interroger Yahoo. Les ISIN non resolus sont absents du cache (mais on
    memorise une valeur vide pour ne pas boucler a chaque execution).
    """
    if cache_path is None:
        cache_path = os.path.join(config.CACHE_DIR, "isin_ticker.parquet")

    cache: dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            cdf = pd.read_parquet(cache_path)
            cache = {str(r["isin"]): (None if pd.isna(r["ticker"])
                                      else _us_primary(str(r["ticker"])))
                     for _, r in cdf.iterrows()}
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache ISIN illisible : %s", exc)

    def _persist() -> None:
        out = pd.DataFrame(
            {"isin": list(cache.keys()),
             "ticker": [cache[k] for k in cache.keys()]}
        )
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            out.to_parquet(cache_path, index=False)
        except Exception as exc:  # pragma: no cover
            logger.warning("Ecriture cache ISIN impossible : %s", exc)

    to_fetch = [i for i in dict.fromkeys(isins) if i not in cache]
    for n, isin in enumerate(to_fetch, 1):
        symbol = None
        for _ in range(attempts):
            try:
                info = yf.Ticker(isin).info or {}
                symbol = info.get("symbol")
                if symbol:
                    break
            except Exception as exc:
                logger.debug("Echec resolution %s : %s", isin, exc)
            time.sleep(sleep_s)
        cache[isin] = _us_primary(symbol)
        if n % 25 == 0:
            _persist()

    _persist()
    return pd.Series({k: v for k, v in cache.items()}, name="ticker")


# ===========================================================================
# Univers trimestriel
# ===========================================================================
def build_universe_quarterly(save: bool = True) -> pd.DataFrame:
    """Construit `data/universe_quarterly.csv` (une ligne par (date, ticker))."""
    inv = pd.read_csv(config.INVENTORY_CSV)
    inv.columns = [c.strip() for c in inv.columns]
    us = inv[(inv["asset_cat"] == "EC") & (inv["inv_country"] == "US")].copy()

    resolved = resolve_isins(us["isin"].tolist())
    us["ticker"] = us["isin"].map(resolved)
    unresolved = sorted(us.loc[us["ticker"].isna(), "isin"].unique().tolist())
    if unresolved:
        logger.warning("%d ISIN non resolus retires : %s", len(unresolved), unresolved)
    us = us.dropna(subset=["ticker"]).copy()
    us["ticker"] = us["ticker"].map(_normalize_ticker)

    # Doublons (isin, report_period) : on consolide en sommant la valeur.
    us = (us.groupby(["report_period", "filing_date", "ticker"], as_index=False)
            .agg(name=("name", "first"), isin=("isin", "first"),
                 cusip=("cusip", "first"), value_usd=("value_usd", "sum"),
                 pct_val=("pct_val", "sum")))

    # Renormalisation a 1.0 par (report_period).
    grp = us.groupby("report_period")["pct_val"]
    us["weight"] = us["pct_val"] / grp.transform("sum")

    us["rebalance_date"] = pd.to_datetime(us["filing_date"])
    us["report_period"] = pd.to_datetime(us["report_period"])

    # Secteur GICS (issu de sector_map.csv s'il existe deja).
    sector_map: dict[str, str] = {}
    if os.path.exists(config.SECTOR_MAP_CSV):
        try:
            sm = pd.read_csv(config.SECTOR_MAP_CSV)
            sector_map = dict(zip(sm["ticker"].map(_normalize_ticker),
                                  sm["gics_sector"]))
        except Exception:  # pragma: no cover
            sector_map = {}
    us["gics_sector"] = us["ticker"].map(sector_map).fillna("Unknown")

    out = us[["rebalance_date", "report_period", "ticker", "name", "isin",
              "cusip", "gics_sector", "weight"]].sort_values(["rebalance_date", "ticker"])
    out = out.reset_index(drop=True)

    if save:
        out.to_csv(config.UNIVERSE_QUARTERLY_CSV, index=False)
        logger.info("Ecrit %s (%d lignes)", config.UNIVERSE_QUARTERLY_CSV, len(out))
    return out


# ===========================================================================
# Mapping sectoriel
# ===========================================================================
def _load_universe_csv_map() -> tuple[dict[str, str], dict[str, str]]:
    """Retourne (ticker->secteur, nom_normalise->secteur) depuis universe.csv."""
    if not os.path.exists(config.UNIVERSE_CSV):
        return {}, {}
    uni = pd.read_csv(config.UNIVERSE_CSV)
    uni["ticker"] = uni["ticker"].map(_normalize_ticker)
    by_ticker = dict(zip(uni["ticker"], uni["gics_sector"]))
    by_name = {_norm_name(n): s for n, s in zip(uni["name"], uni["gics_sector"])}
    return by_ticker, by_name


def build_sector_map(save: bool = True) -> pd.Series:
    """Construit `data/sector_map.csv` (ticker -> secteur GICS).

    Priorite : ticker connu dans `universe.csv` ; sinon nom connu ; sinon
    `yfinance info['sector']` (libelle Yahoo mappe vers GICS) ; sinon "Unknown".
    """
    uni_path = config.UNIVERSE_QUARTERLY_CSV
    if not os.path.exists(uni_path):
        build_universe_quarterly(save=True)
    uni = pd.read_csv(uni_path)
    tickers = sorted(uni["ticker"].unique())

    # noms de reference par ticker (depuis l'inventaire).
    names = (uni.sort_values("rebalance_date")
                .groupby("ticker")["name"].last().to_dict())

    by_ticker, by_name = _load_universe_csv_map()

    # Reprise d'un eventuel mapping deja construit (incremental).
    existing: dict[str, str] = {}
    if os.path.exists(config.SECTOR_MAP_CSV):
        try:
            sm = pd.read_csv(config.SECTOR_MAP_CSV)
            existing = dict(zip(sm["ticker"], sm["gics_sector"]))
        except Exception:  # pragma: no cover
            pass

    mapping: dict[str, str] = {}

    def _persist_map() -> None:
        if not save or not mapping:
            return
        out = pd.Series(mapping, name="gics_sector").sort_index()
        out.rename_axis("ticker").reset_index().to_csv(config.SECTOR_MAP_CSV, index=False)

    for n, t in enumerate(tickers, 1):
        if t in SECTOR_OVERRIDES:
            mapping[t] = SECTOR_OVERRIDES[t]
            continue
        if t in by_ticker and by_ticker[t] in config.GICS_SECTORS:
            mapping[t] = by_ticker[t]
            continue
        nm = _norm_name(names.get(t, ""))
        if nm and nm in by_name:
            mapping[t] = by_name[nm]
            continue
        prev = existing.get(t)
        if prev and prev != "Unknown":
            mapping[t] = prev
            continue

        sector = None
        try:
            raw = (yf.Ticker(t).info or {}).get("sector")
            if raw:
                sector = YAHOO_TO_GICS.get(str(raw).strip().lower())
        except Exception as exc:
            logger.debug("Echec secteur %s : %s", t, exc)
        mapping[t] = sector or "Unknown"
        if n % 25 == 0:
            _persist_map()

    _persist_map()
    out = pd.Series(mapping, name="gics_sector").sort_index()
    if save:
        logger.info("Ecrit %s (%d tickers, %d Unknown)",
                    config.SECTOR_MAP_CSV, len(out), int((out == "Unknown").sum()))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_universe_quarterly(save=True)
    build_sector_map(save=True)
    # Backfill de la colonne `gics_sector` dans universe_quarterly.csv.
    build_universe_quarterly(save=True)


if __name__ == "__main__":
    main()
