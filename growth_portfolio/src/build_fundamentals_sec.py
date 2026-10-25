"""Reconstruction Point-in-Time des fondamentaux via SEC EDGAR XBRL.

Contexte / probleme resolu
--------------------------
`yfinance` ne conserve que ~5-6 trimestres glissants de fondamentaux : le cache
`data/cache/fundamentals.parquet` ne couvre donc que 2025Q1 -> 2026Q3. Les 8
premiers rebalancements du backtest (2023-2024) n'ont AUCUNE donnee
fondamentale, ce qui rend `Z_Quality` entierement NaN.

Ce module reconstruit l'historique PIT depuis l'API XBRL publique de la SEC :

- `https://www.sec.gov/files/company_tickers.json`  -> mapping ticker -> CIK.
- `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
  -> `facts.us-gaap.<Concept>.units.USD[]`, chaque fait etant
  `{start?, end, val, accn, fy, fp, form, filed}`. `filed` = date de depot =
  **date de publication PIT**.

Regles PIT
----------
- `period_end` = `end` du fait.
- `pub_date` = date de depot (`filed`). Pour rester conservateur, pour un
  `period_end` donne on retient `max(filed)` parmi les concepts effectivement
  utilises : une ligne n'est exploitable que lorsque TOUS ses agrégats sont
  publics. Un bilan du T1 clos le 31/03 n'est donc jamais utilise avant son
  depot (souvent en mai).
- Instantanes (bilan) : on garde le PREMIER `filed` publiant une `end` donnee
  (PIT strict, pas de restatement futur).
- Flux trimestriels : on ne retient que les durees ~80-100 jours ; si seul du
  cumulatif YTD existe, on derive le trimestre par difference avec la periode
  cumulee precedente (meme chaine start/end), avec `filed` = depot le plus tard
  des deux (donc PIT correct).

Contraintes / securite
----------------------
- En-tetes OBLIGATOIRES : `User-Agent` identifiant + `Accept-Encoding`.
- Debit <= 10 req/s (`sleep ~0.15s`).
- Cache brut disque `data/cache/sec_companyfacts/CIK##########.json` pour
  eviter tout re-telechargement ; cache du mapping dans
  `data/cache/sec_company_tickers.json`.
- JAMAIS d'usage de `ticker.info`.

Sortie : `data/fundamentals_sec.parquet` (meme schema que
`data_loader.get_fundamentals`).

Usage :  python -m src.build_fundamentals_sec
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)

__all__ = [
    "SEC_FUNDAMENTALS_PATH",
    "SEC_TICKERS_CACHE",
    "SEC_COMPANYFACTS_DIR",
    "MIN_END",
    "get_ticker_cik_map",
    "download_companyfacts",
    "extract_company",
    "build_fundamentals_sec",
    "get_universe_tickers",
    "main",
]

# --- Chemins ----------------------------------------------------------------
SEC_FUNDAMENTALS_PATH = os.path.join(config.DATA_DIR, "fundamentals_sec.parquet")
SEC_TICKERS_CACHE = os.path.join(config.CACHE_DIR, "sec_company_tickers.json")
SEC_COMPANYFACTS_DIR = os.path.join(config.CACHE_DIR, "sec_companyfacts")

# Bornes temporelles / reseau.
MIN_END = date(2018, 1, 1)          # on garde les faits `end >= 2018-01-01`
# (indispensable aux deltas annuels TTM(N) vs TTM(N-1)/TTM(N-2), a la croissance
# BPA 3 ans ET a la fenetre de 20 trimestres (~5 ans) requise par les criteres
# de variance du G-Score de Mohanram sur la fenetre de test 2024-2026 ;
# l'historique complet 2008+ est deja present dans le cache brut
# `sec_companyfacts/`, le rebuild est donc local et sans reseau).
SEC_SLEEP_S = 0.15                  # <= ~6 req/s (limite SEC : 10 req/s)
SEC_TIMEOUT = 30
SEC_ATTEMPTS = 3
USER_AGENT = "growth_portfolio research raoulsinde@gmail.com"
_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Colonnes du schema final (= `data_loader._FUND_COLUMNS`).
FUND_COLUMNS = [
    "ticker", "period_end", "pub_date", "total_assets", "stockholders_equity",
    "total_debt", "shares", "net_income_q", "gross_profit_q",
    "operating_income_q", "ocf_q",
    "revenue_q", "research_dev_q", "capex_q", "buyback_q", "dividends_q",
    "surprise_pct",
]

# --- Concepts XBRL ----------------------------------------------------------
# Instantanes (bilan).
_INSTANT_CONCEPTS = {
    "assets": ["Assets"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "liabilities": ["Liabilities"],
    "ltd_noncurrent": ["LongTermDebtNoncurrent"],
    "ltd_current": ["LongTermDebtCurrent"],
    "ltd_total": ["LongTermDebt"],
}
# Flux trimestriels (duree ~80-100 j ; derivation YTD sinon).
_DURATION_CONCEPTS = {
    "net_income_q": ["NetIncomeLoss"],
    "gross_profit_q": ["GrossProfit"],
    "operating_income_q": ["OperatingIncomeLoss"],
    "ocf_q": ["NetCashProvidedByUsedInOperatingActivities"],
    "revenue_q": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ],
    "research_dev_q": ["ResearchAndDevelopmentExpense"],
    "capex_q": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buyback_q": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_q": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
}


# ===========================================================================
# Utilitaires
# ===========================================================================
def _normalize_ticker(ticker: str) -> str:
    """Normalise un ticker au format yfinance (BRK.B -> BRK-B)."""
    return str(ticker).strip().replace(".", "-").replace("/", "-")


def _parse_date(value) -> Optional[date]:
    if value is None:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _get(url: str) -> Optional[requests.Response]:
    """GET robuste avec retries/timeout/backoff exponentiel."""
    for attempt in range(SEC_ATTEMPTS):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=SEC_TIMEOUT)
            if resp.status_code == 200:
                return resp
            logger.warning("SEC %s -> HTTP %s (essai %d)", url, resp.status_code,
                           attempt + 1)
        except Exception as exc:  # pragma: no cover - reseau
            logger.warning("SEC %s -> %s (essai %d)", url, exc, attempt + 1)
        time.sleep(SEC_SLEEP_S * (2 ** attempt))
    return None


# ===========================================================================
# Mapping ticker -> CIK
# ===========================================================================
def get_ticker_cik_map(refresh: bool = False) -> Dict[str, str]:
    """Mapping `TICKER` -> CIK 10 chiffres, cache disque.

    Source : `https://www.sec.gov/files/company_tickers.json`.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    raw: Optional[dict] = None
    if not refresh and os.path.exists(SEC_TICKERS_CACHE):
        try:
            with open(SEC_TICKERS_CACHE, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache SEC tickers illisible : %s", exc)
    if raw is None:
        resp = _get("https://www.sec.gov/files/company_tickers.json")
        if resp is None:
            raise RuntimeError("Impossible de recuperer company_tickers.json")
        raw = resp.json()
        try:
            with open(SEC_TICKERS_CACHE, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)
        except Exception as exc:  # pragma: no cover
            logger.warning("Ecriture cache SEC tickers impossible : %s", exc)

    mapping: Dict[str, str] = {}
    for entry in raw.values():
        tk = entry.get("ticker")
        cik = entry.get("cik_str")
        if tk is None or cik is None:
            continue
        mapping.setdefault(str(tk).upper(), f"{int(cik):010d}")
    return mapping


# ===========================================================================
# Telechargement companyfacts (cache brut)
# ===========================================================================
def _companyfacts_path(cik: str) -> str:
    return os.path.join(SEC_COMPANYFACTS_DIR, f"CIK{cik}.json")


def download_companyfacts(cik: str, refresh: bool = False) -> Optional[dict]:
    """Telecharge (ou lit le cache) le JSON `companyfacts` d'un CIK."""
    os.makedirs(SEC_COMPANYFACTS_DIR, exist_ok=True)
    path = _companyfacts_path(cik)
    if not refresh and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache companyfacts %s illisible : %s", cik, exc)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = _get(url)
    time.sleep(SEC_SLEEP_S)
    if resp is None:
        logger.warning("companyfacts indisponible pour CIK %s", cik)
        return None
    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover
        logger.warning("companyfacts JSON invalide pour CIK %s : %s", cik, exc)
        return None
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception as exc:  # pragma: no cover
        logger.warning("Ecriture cache companyfacts %s impossible : %s", cik, exc)
    return data


# ===========================================================================
# Extraction des faits
# ===========================================================================
def _facts_for(facts: dict, taxonomy: str, concepts: Sequence[str],
               unit: str = "USD") -> List[dict]:
    """Concatene les listes de faits de PLUSIEURS concepts candidats.

    On fusionne (et non on s'arrete au premier concept non vide) car un
    emetteur peut basculer d'un tag a un autre selon les periodes (ex. NVDA :
    revenue en ``Revenues`` avant, ``RevenueFromContractWithCustomerExcluding
    AssessedTax`` ensuite). Les doublons de periode sont dedupliques en aval.
    """
    tax = facts.get(taxonomy, {})
    out: List[dict] = []
    for concept in concepts:
        node = tax.get(concept)
        if not node:
            continue
        arr = node.get("units", {}).get(unit)
        if arr:
            out.extend(arr)
    return out


def _extract_instant(records: Iterable[dict],
                      min_end: date = MIN_END) -> Dict[date, tuple]:
    """Instantanes keyes par `end` : garde le PREMIER `filed` (PIT strict)."""
    out: Dict[date, tuple] = {}
    for f in records:
        start = f.get("start")
        if start is not None and start != f.get("end"):
            continue  # ce n'est pas un instantane
        end = _parse_date(f.get("end"))
        filed = _parse_date(f.get("filed"))
        val = f.get("val")
        if end is None or filed is None or end < min_end or val is None:
            continue
        prev = out.get(end)
        if prev is None or filed < prev[1]:
            out[end] = (float(val), filed)
    return out


def _extract_duration(records: Iterable[dict],
                      min_end: date = MIN_END) -> Dict[date, tuple]:
    """Flux trimestriels keyes par `end`.

    On prefere une duree 80-100 jours (trimestre direct). Sinon, on derive le
    trimestre depuis un cumulatif YTD en le differenciant du cumulatif
    immediatement precedent (meme chaine `start`/`end`).
    """
    recs: List[dict] = []
    for f in records:
        start = _parse_date(f.get("start"))
        end = _parse_date(f.get("end"))
        filed = _parse_date(f.get("filed"))
        val = f.get("val")
        if start is None or end is None or filed is None or val is None:
            continue
        if end < min_end:
            continue
        recs.append({"start": start, "end": end, "filed": filed, "val": float(val)})

    quarters: Dict[date, tuple] = {}
    for r in recs:
        dur = (r["end"] - r["start"]).days
        if 80 <= dur <= 100:
            prev = quarters.get(r["end"])
            if prev is None or r["filed"] < prev[1]:
                quarters[r["end"]] = (r["val"], r["filed"])

    # Derivation des trimestres manquants (ex. Q4 d'un 10-K) depuis les chaines
    # YTD : base = fait de MEME debut de periode (`start` proche) dont la fin est
    # la plus proche mais anterieure, et telle que l'intervalle derive soit bien
    # un trimestre (~80-105 j). Cela evite de differencier avec l'exercice
    # precedent (meme `end`/`start` annuel) ou avec le seul T1 direct.
    for r in recs:
        dur = (r["end"] - r["start"]).days
        if dur <= 100 or r["end"] in quarters:
            continue
        best = None
        for p in recs:
            if p is r or p["end"] >= r["end"]:
                continue
            if abs((p["start"] - r["start"]).days) > 10:
                continue
            qlen = (r["end"] - p["end"]).days
            if not (80 <= qlen <= 105):
                continue
            if best is None or p["end"] > best["end"]:
                best = p
        if best is None:
            continue
        val = r["val"] - best["val"]
        filed = max(r["filed"], best["filed"])  # PIT : depot le plus tardif requis
        prev = quarters.get(r["end"])
        if prev is None or filed < prev[1]:
            quarters[r["end"]] = (val, filed)
    return quarters


def _extract_shares(facts: dict, min_end: date = MIN_END) -> Dict[date, tuple]:
    """Actions en circulation (dei prioritaire, fallback us-gaap), par `end`."""
    def _one(records: Iterable[dict]) -> Dict[date, tuple]:
        d: Dict[date, tuple] = {}
        for f in records:
            end = _parse_date(f.get("end"))
            filed = _parse_date(f.get("filed"))
            val = f.get("val")
            if end is None or filed is None or end < min_end or val is None:
                continue
            prev = d.get(end)
            if prev is None or filed < prev[1]:
                d[end] = (float(val), filed)
        return d

    us_gaap = _one(_facts_for(facts, "us-gaap",
                              ["CommonStockSharesOutstanding"], unit="shares"))
    dei = _one(_facts_for(facts, "dei",
                          ["EntityCommonStockSharesOutstanding"], unit="shares"))
    out = dict(us_gaap)
    out.update(dei)  # dei prioritaire pour une meme `end`
    return out


def _nearest_shares(shares: Dict[date, tuple],
                    period_end: date) -> Optional[tuple]:
    """Mappe des actions sur un `period_end` (date de couverture proche).

    `EntityCommonStockSharesOutstanding` porte generalement la date de
    couverture (quelques jours apres la cloture) : on accepte un ecart de
    [0, +45] jours et on prend la date la plus proche.
    """
    best = None
    for end, rec in shares.items():
        delta = (end - period_end).days
        if 0 <= delta <= 45:
            if best is None or delta < best[0]:
                best = (delta, rec)
    return best[1] if best is not None else None


# ===========================================================================
# Extraction par societe
# ===========================================================================
def extract_company(ticker: str, facts: dict,
                    min_end: date = MIN_END) -> pd.DataFrame:
    """Construit le panel PIT SEC d'une societe (une ligne par `period_end`)."""
    if not facts or "facts" not in facts:
        return pd.DataFrame(columns=FUND_COLUMNS)
    froot = facts["facts"]

    instants = {
        name: _extract_instant(_facts_for(froot, "us-gaap", concepts), min_end)
        for name, concepts in _INSTANT_CONCEPTS.items()
    }
    durations = {
        name: _extract_duration(_facts_for(froot, "us-gaap", concepts), min_end)
        for name, concepts in _DURATION_CONCEPTS.items()
    }
    shares = _extract_shares(froot, min_end)

    period_ends: set[date] = set()
    for series in list(instants.values()) + list(durations.values()):
        period_ends.update(series.keys())
    if not period_ends:
        return pd.DataFrame(columns=FUND_COLUMNS)

    rows: List[dict] = []
    for pe in sorted(period_ends):
        used_filed: List[date] = []

        def _take(series: Dict[date, tuple]) -> float:
            rec = series.get(pe)
            if rec is None:
                return np.nan
            used_filed.append(rec[1])
            return rec[0]

        total_assets = _take(instants["assets"])
        equity = _take(instants["stockholders_equity"])
        # (Liabilities extrait pour completude mais absent du schema cible.)

        # total_debt = LongTermDebtNoncurrent + LongTermDebtCurrent si les DEUX
        # sont disponibles, sinon LongTermDebt, sinon NaN.
        nc = instants["ltd_noncurrent"].get(pe)
        cur = instants["ltd_current"].get(pe)
        ltd = instants["ltd_total"].get(pe)
        if nc is not None and cur is not None:
            total_debt = nc[0] + cur[0]
            used_filed.extend([nc[1], cur[1]])
        elif ltd is not None:
            total_debt = ltd[0]
            used_filed.append(ltd[1])
        else:
            total_debt = np.nan

        sh = _nearest_shares(shares, pe)
        if sh is not None:
            shares_val = sh[0]
            used_filed.append(sh[1])
        else:
            shares_val = np.nan

        net_income_q = _take(durations["net_income_q"])
        gross_profit_q = _take(durations["gross_profit_q"])
        operating_income_q = _take(durations["operating_income_q"])
        ocf_q = _take(durations["ocf_q"])
        revenue_q = _take(durations["revenue_q"])
        research_dev_q = _take(durations["research_dev_q"])
        capex_q = _take(durations["capex_q"])
        buyback_q = _take(durations["buyback_q"])
        dividends_q = _take(durations["dividends_q"])

        # Aucun agregat exploitable -> on ignore la periode.
        if all(pd.isna(v) for v in (
                total_assets, equity, total_debt, net_income_q, gross_profit_q,
                operating_income_q, ocf_q, revenue_q)):
            continue

        pub_date = max(used_filed) if used_filed else None
        rows.append({
            "ticker": ticker,
            "period_end": pd.Timestamp(pe),
            "pub_date": pd.Timestamp(pub_date) if pub_date else pd.NaT,
            "total_assets": total_assets,
            "stockholders_equity": equity,
            "total_debt": total_debt,
            "shares": shares_val,
            "net_income_q": net_income_q,
            "gross_profit_q": gross_profit_q,
            "operating_income_q": operating_income_q,
            "ocf_q": ocf_q,
            "revenue_q": revenue_q,
            "research_dev_q": research_dev_q,
            "capex_q": capex_q,
            "buyback_q": buyback_q,
            "dividends_q": dividends_q,
            "surprise_pct": np.nan,
        })
    return pd.DataFrame(rows, columns=FUND_COLUMNS)


# ===========================================================================
# Univers & pipeline global
# ===========================================================================
def get_universe_tickers() -> List[str]:
    """Union des tickers de `universe_quarterly.csv` et du cache yfinance."""
    tickers: set[str] = set()
    if os.path.exists(config.UNIVERSE_QUARTERLY_CSV):
        uni = pd.read_csv(config.UNIVERSE_QUARTERLY_CSV)
        tickers.update(_normalize_ticker(t) for t in uni["ticker"].dropna())
    yf_cache = os.path.join(config.CACHE_DIR, "fundamentals.parquet")
    if os.path.exists(yf_cache):
        try:
            yfc = pd.read_parquet(yf_cache)
            tickers.update(_normalize_ticker(t) for t in yfc["ticker"].dropna())
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache yfinance illisible : %s", exc)
    return sorted(t for t in tickers if t)


def build_fundamentals_sec(tickers: Optional[Iterable[str]] = None,
                           save: bool = True,
                           min_end: date = MIN_END,
                           output_path: Optional[str] = None) -> pd.DataFrame:
    """Construit `data/fundamentals_sec.parquet` pour l'univers demande."""
    if tickers is None:
        tickers = get_universe_tickers()
    tickers = sorted({_normalize_ticker(t) for t in tickers})
    logger.info("Univers SEC : %d tickers", len(tickers))

    cik_map = get_ticker_cik_map()
    mapped: Dict[str, str] = {}
    unmapped: List[str] = []
    for t in tickers:
        cik = cik_map.get(t.upper())
        if cik:
            mapped[t] = cik
        else:
            unmapped.append(t)
    if unmapped:
        logger.warning("%d tickers NON mappes CIK (ignores) : %s",
                       len(unmapped), unmapped)

    frames: List[pd.DataFrame] = []
    failed: List[str] = []
    for n, (t, cik) in enumerate(sorted(mapped.items()), 1):
        facts = download_companyfacts(cik)
        if facts is None:
            failed.append(t)
            continue
        try:
            df = extract_company(t, facts, min_end=min_end)
        except Exception as exc:  # pragma: no cover - robustesse
            logger.warning("Extraction %s echouee : %s", t, exc)
            failed.append(t)
            continue
        if not df.empty:
            frames.append(df)
        if n % 25 == 0:
            logger.info("... %d/%d societes traitees", n, len(mapped))

    if frames:
        out = pd.concat(frames, ignore_index=True)[FUND_COLUMNS]
    else:
        out = pd.DataFrame(columns=FUND_COLUMNS)

    if failed:
        logger.warning("%d societes en echec (companyfacts) : %s",
                       len(failed), failed)

    if not out.empty:
        out["period_end"] = pd.to_datetime(out["period_end"])
        out["pub_date"] = pd.to_datetime(out["pub_date"])
        out = out.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    dest = output_path or SEC_FUNDAMENTALS_PATH
    if save:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        out.to_parquet(dest, index=False)
        logger.info("Ecrit %s (%d lignes, %d tickers)",
                    dest, len(out), out["ticker"].nunique()
                    if not out.empty else 0)

    # Resume pour l'operateur.
    logger.info("Resume SEC : %d mappes / %d non mappes / %d en echec / %d lignes",
                len(mapped), len(unmapped), len(failed), len(out))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_fundamentals_sec(save=True)


if __name__ == "__main__":
    main()
