"""Strategie << Mohanram Growth Tilted Portfolio >> (PIT strict, 100 % deterministe).

Nouvelle strategie independante du moteur << Core + Satellites >> du projet :

- univers = poids Point-in-Time de l'indice de reference (poids N-PORT EGUS,
  proxy MSCI USA Growth) ;
- 8 signaux binaires G1..G8 de Partha Mohanram (2005), calcules en coupe
  transversale sur les fondamentaux ANNUELS publies (``filed <= t0``) ;
- tilting deterministe des poids de l'indice, neutralite sectorielle puis
  plafonnement iteratif (capping) ;
- simulation 6 mois en buy-and-hold (derive des poids), gestion des radiations
  (methode de Shumway, 1997) et attribution de Brinson-Fachler.

Politique de donnees : AUCUN fallback ni mock data. Toute donnee critique
manquante leve ``MohanramDataError`` (Level-1). Les cas partiels prevus par la
methodologie (historique court, R&D/CAPEX/ADV non publies, secteur etroit) sont
journalises (Level-2) sans bloquer.

SECURITE : aucune donnee issue de l'attribut ``info`` d'un Ticker. Toute la
matiere fondamentale provient de l'API XBRL publique de la SEC (cache disque).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl
from . import metrics as M

logger = logging.getLogger(__name__)

__all__ = [
    "MohanramDataError",
    "MULTIPLIERS",
    "FINANCIALS_SECTOR",
    "DEFAULT_CAP_ABS",
    "DEFAULT_CAP_REL",
    "DEFAULT_TX_BPS",
    "HORIZON_MONTHS",
    "SHUMWAY_DELIST_RETURN",
    "MIN_VALID_HISTORY",
    "candidate_rebalance_dates",
    "load_universe_at",
    "build_annual_fundamentals",
    "compute_gscores",
    "apply_tilt",
    "sector_neutralize",
    "iterative_cap",
    "simulate_horizon",
    "brinson_fachler",
    "run_mohanram",
]

# ===========================================================================
# Constantes
# ===========================================================================
FINANCIALS_SECTOR = "Financials"

# G_SCORE -> multiplicateur deterministe de tilting.
MULTIPLIERS = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.50, 5: 0.50,
               6: 1.50, 7: 2.00, 8: 2.50}

DEFAULT_CAP_ABS = 0.08        # plafond absolu par titre
DEFAULT_CAP_REL = 0.03        # marge au-dessus du poids indiciel
DEFAULT_TX_BPS = 10.0         # couts de transaction (bps)
HORIZON_MONTHS = 6
SHUMWAY_DELIST_RETURN = -0.30  # Shumway (1997)
MIN_VALID_HISTORY = 3          # annees minimales pour VARROA / VARSGR

# Fenetre annuelle requise (T, T-1, ... T-4 ; +1 pour TA_{T-1}).
N_YEARS = 6
# Marge anti look-ahead : un exercice doit etre clos >= 4 mois avant t0.
FILING_LAG_MONTHS = 4

# Cache annuel reconstruit depuis l'API XBRL SEC.
ANNUAL_CACHE = os.path.join(config.CACHE_DIR, "mohanram_annual.parquet")
SEC_TICKERS_CACHE = os.path.join(config.CACHE_DIR, "sec_company_tickers.json")
SEC_COMPANYFACTS_DIR = os.path.join(config.CACHE_DIR, "sec_companyfacts")

# Budget de duree d'un exercice annuel (jour), tolere pour les calendriers 52/53.
_ANNUAL_MIN_DAYS = 330
_ANNUAL_MAX_DAYS = 400

# Concepts XBRL (us-gaap), du plus specifique (priorite 0) au plus general.
_FLOW_CONCEPTS = {
    "net_income": [
        ("NetIncomeLoss", 0),
        ("NetIncomeLossAvailableToCommonStockholdersBasic", 1),
        ("NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest", 1),
        ("ProfitLoss", 2),
    ],
    "cfo": [
        ("NetCashProvidedByUsedInOperatingActivities", 0),
        ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", 1),
    ],
    "sales": [
        ("RevenueFromContractWithCustomerExcludingAssessedTax", 0),
        ("RevenueFromContractWithCustomerIncludingAssessedTax", 1),
        ("Revenues", 2),
        ("SalesRevenueNet", 3),
        ("SalesRevenueServicesNet", 3),
        ("SalesRevenueGoodsNet", 3),
        ("OperatingRevenue", 3),
    ],
    "rd": [("ResearchAndDevelopmentExpense", 0)],
    "capex": [
        ("PaymentsToAcquirePropertyPlantAndEquipment", 0),
        ("PaymentsToAcquireProductiveAssets", 1),
    ],
    "adv": [("AdvertisingExpense", 0)],
}
_INSTANT_CONCEPTS = {"total_assets": [("Assets", 0)]}

_ANNUAL_COLUMNS = [
    "ticker", "period_end", "pub_date", "pub_rd", "pub_capex", "pub_adv",
    "net_income", "cfo", "sales", "rd", "capex", "adv", "total_assets",
]


class MohanramDataError(Exception):
    """Erreur critique (Level-1) : donnee obligatoire absente/en incoherente.

    ``issues`` est une liste de dictionnaires serialisable en DataFrame pour
    l'affichage du tableau de diagnostic dans Streamlit.
    """

    def __init__(self, message: str, issues: Optional[list] = None):
        super().__init__(message)
        self.issues = list(issues or [])


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().replace(".", "-").replace("/", "-")


def _to_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _parse_date(value) -> Optional[date]:
    if value is None:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - cache corrompu
        logger.warning("Cache illisible %s : %s", path, exc)
        return None


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Ecriture cache impossible %s : %s", path, exc)


# ===========================================================================
# Univers PIT a la date t0
# ===========================================================================
def candidate_rebalance_dates() -> list[pd.Timestamp]:
    """Dates de rebalancement disposant d'une fenetre 6 mois complete."""
    uni = dl._load_universe_quarterly()
    end = _to_ts(config.PRICE_END)
    out = []
    for d in sorted(uni["rebalance_date"].unique()):
        d = _to_ts(d)
        if d + pd.DateOffset(months=HORIZON_MONTHS) <= end:
            out.append(d)
    return out


def load_universe_at(t0: pd.Timestamp) -> pd.DataFrame:
    """Panier PIT de l'indice (dernier trimestre connu a ``t0``).

    Colonnes : ``ticker, name, gics_sector, w_bench`` (poids renormalises a 1).
    """
    t0 = _to_ts(t0)
    uni = dl._load_universe_quarterly()
    avail = uni[uni["rebalance_date"] <= t0]
    if avail.empty:
        raise MohanramDataError(
            f"Aucune composition d'indice connue a {t0:%d/%m/%Y} "
            f"(plus ancienne date : {uni['rebalance_date'].min():%d/%m/%Y}).")
    last = avail["rebalance_date"].max()
    comp = avail[avail["rebalance_date"] == last].copy()
    comp = comp.sort_values("weight", ascending=False)
    comp = comp.drop_duplicates(subset=["ticker"], keep="first")
    comp = comp.set_index("ticker")
    comp["ticker"] = comp.index
    total = float(comp["weight"].sum())
    issues = []
    if not np.isfinite(total) or total <= 0:
        raise MohanramDataError(
            f"Somme des poids de l'indice nulle/negative a {last:%d/%m/%Y}.")
    dev = abs(total - 1.0)
    if dev > 0.005:
        issues.append({"niveau": "warning", "categorie": "poids_indice",
                       "detail": f"Somme brute des poids = {total:.4f} "
                                 f"(deviation {dev:.2%}), renormalisee a 1."})
    comp["w_bench"] = comp["weight"] / total
    comp.attrs["rebalance_date"] = last
    comp.attrs["issues"] = issues
    if comp["gics_sector"].isna().any():
        missing = comp.index[comp["gics_sector"].isna()].tolist()
        raise MohanramDataError(
            "Code sectoriel manquant pour des titres de l'indice.",
            [{"niveau": "error", "categorie": "secteur", "ticker": t,
              "detail": "gics_sector absent"} for t in missing])
    return comp


# ===========================================================================
# Fondamentaux annuels PIT (SEC EDGAR XBRL, cache parquet)
# ===========================================================================
def _load_cik_map() -> dict[str, str]:
    if not os.path.exists(SEC_TICKERS_CACHE):
        raise MohanramDataError(
            f"Mapping ticker->CIK absent ({SEC_TICKERS_CACHE}). Fournir le cache "
            "SEC ou lancer `python -m src.build_fundamentals_sec`.")
    with open(SEC_TICKERS_CACHE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, str] = {}
    for entry in raw.values():
        tk = entry.get("ticker")
        cik = entry.get("cik_str")
        if tk is None or cik is None:
            continue
        out.setdefault(str(tk).upper(), f"{int(cik):010d}")
    return out


def _extract_annual(facts: dict) -> pd.DataFrame:
    """Panel annuel (une ligne par exercice fiscal) depuis ``companyfacts``."""
    froot = facts.get("facts", {})
    gaap = froot.get("us-gaap", {})
    if not gaap:
        return pd.DataFrame(columns=_ANNUAL_COLUMNS)

    def _records(concept: str, unit: str) -> list:
        node = gaap.get(concept)
        if not node:
            return []
        return node.get("units", {}).get(unit, []) or []

    flows: dict[str, dict] = {name: {} for name in _FLOW_CONCEPTS}
    for name, concepts in _FLOW_CONCEPTS.items():
        for concept, prio in concepts:
            for f in _records(concept, "USD"):
                start = _parse_date(f.get("start"))
                end = _parse_date(f.get("end"))
                filed = _parse_date(f.get("filed"))
                val = f.get("val")
                if start is None or end is None or filed is None or val is None:
                    continue
                if not (_ANNUAL_MIN_DAYS <= (end - start).days <= _ANNUAL_MAX_DAYS):
                    continue
                prev = flows[name].get(end)
                # PIT : on retient la PREMIERE divulgation (filed le plus tot),
                # puis le concept le plus specifique (prio la plus basse).
                cand = (filed, prio, float(val), concept)
                if prev is None or cand[:2] < prev[:2]:
                    flows[name][end] = cand

    instants: dict = {}
    for concepts in _INSTANT_CONCEPTS.values():
        for concept, prio in concepts:
            for f in _records(concept, "USD"):
                start = _parse_date(f.get("start"))
                end = _parse_date(f.get("end"))
                filed = _parse_date(f.get("filed"))
                val = f.get("val")
                if end is None or filed is None or val is None:
                    continue
                if start is not None and start != end:
                    continue
                prev = instants.get(end)
                cand = (prio, filed, float(val))
                if prev is None or cand[:2] < prev[:2]:
                    instants[end] = cand

    year_ends: set = set()
    for name in ("net_income", "sales", "cfo"):
        year_ends.update(flows[name].keys())
    if not year_ends:
        for name in _FLOW_CONCEPTS:
            year_ends.update(flows[name].keys())
    if not year_ends:
        return pd.DataFrame(columns=_ANNUAL_COLUMNS)

    instant_ends = sorted(instants)
    rows: list[dict] = []
    for end in sorted(year_ends):
        recs = {name: flows[name].get(end) for name in _FLOW_CONCEPTS}
        ta = instants.get(end)
        if ta is None and instant_ends:
            near = min(instant_ends, key=lambda d: abs((d - end).days))
            if abs((near - end).days) <= 20:
                ta = instants[near]
        core_filed = [recs[n][0] for n in ("net_income", "cfo", "sales")
                      if recs[n] is not None]
        if ta is not None:
            core_filed.append(ta[1])
        if not core_filed:
            continue

        def _pub(name: str) -> Optional[date]:
            return recs[name][0] if recs[name] is not None else None

        def _val(name: str) -> float:
            return recs[name][2] if recs[name] is not None else np.nan

        rows.append({
            "ticker": None,
            "period_end": pd.Timestamp(end),
            "pub_date": pd.Timestamp(max(core_filed)),
            "pub_rd": pd.Timestamp(_pub("rd")) if _pub("rd") else pd.NaT,
            "pub_capex": pd.Timestamp(_pub("capex")) if _pub("capex") else pd.NaT,
            "pub_adv": pd.Timestamp(_pub("adv")) if _pub("adv") else pd.NaT,
            "net_income": _val("net_income"),
            "cfo": _val("cfo"),
            "sales": _val("sales"),
            "rd": _val("rd"),
            "capex": _val("capex"),
            "adv": _val("adv"),
            "total_assets": ta[2] if ta else np.nan,
        })
    return pd.DataFrame(rows, columns=_ANNUAL_COLUMNS)


def build_annual_fundamentals(tickers: Optional[Iterable[str]] = None,
                              refresh: bool = False) -> pd.DataFrame:
    """Panel fondamental ANNUEL PIT reconstruit depuis le cache SEC companyfacts.

    ATTENTION : aucun fallback. Si un ticker est absent du cache CIK /
    companyfacts, il est simplement omis (il sera signale comme donnee manquante
    en aval par ``compute_gscores``).
    """
    if not refresh:
        cached = _read_parquet(ANNUAL_CACHE)
        if cached is not None and not cached.empty and \
                set(_ANNUAL_COLUMNS).issubset(cached.columns):
            cached["period_end"] = pd.to_datetime(cached["period_end"])
            cached["pub_date"] = pd.to_datetime(cached["pub_date"])
            for c in ("pub_rd", "pub_capex", "pub_adv"):
                cached[c] = pd.to_datetime(cached[c])
            if tickers is None:
                return cached
            wanted = {_normalize_ticker(t) for t in tickers}
            have = set(cached["ticker"].unique())
            missing = wanted - have
            if not missing:
                return cached[cached["ticker"].isin(wanted)].reset_index(drop=True)

    cik_map = _load_cik_map()
    uni = dl._load_universe_quarterly()
    universe = {_normalize_ticker(t) for t in uni["ticker"].dropna()}
    if tickers is not None:
        universe = {_normalize_ticker(t) for t in tickers}

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for t in sorted(universe):
        cik = cik_map.get(t.upper())
        if cik is None:
            failed.append(t)
            continue
        path = os.path.join(SEC_COMPANYFACTS_DIR, f"CIK{cik}.json")
        if not os.path.exists(path):
            failed.append(t)
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                facts = json.load(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning("companyfacts illisible %s : %s", t, exc)
            failed.append(t)
            continue
        df = _extract_annual(facts)
        if df.empty:
            failed.append(t)
            continue
        df = df.copy()
        df["ticker"] = t
        frames.append(df)

    if not frames:
        raise MohanramDataError(
            "Aucun fondamental annuel SEC exploitable (cache companyfacts vide "
            f"ou illisible : {SEC_COMPANYFACTS_DIR}).")
    out = pd.concat(frames, ignore_index=True)[_ANNUAL_COLUMNS]
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["pub_date"] = pd.to_datetime(out["pub_date"])
    out = out.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    _write_parquet(out, ANNUAL_CACHE)
    if failed:
        logger.warning("%d tickers sans fondamentaux annuels SEC : %s",
                       len(failed), failed)
    return out


# ===========================================================================
# Signaux G1..G8
# ===========================================================================
def _annual_panel(ann: pd.DataFrame, col: str, tickers: pd.Index,
                  n_years: int = N_YEARS) -> pd.DataFrame:
    """Valeurs annuelles decroissantes ``A0..A_{n-1}`` par titre (PIT)."""
    out = pd.DataFrame(np.nan, index=tickers,
                       columns=[f"A{k}" for k in range(n_years)], dtype=float)
    if ann is None or ann.empty or col not in ann.columns:
        return out
    sub = ann[["ticker", col]].copy()
    sub["period_end"] = ann["period_end"].to_numpy()
    sub = sub.dropna(subset=[col]).sort_values(["ticker", "period_end"])
    for tkr, g in sub.groupby("ticker", sort=False):
        if tkr not in out.index:
            continue
        v = g[col].to_numpy(dtype=float)[::-1]
        for k in range(min(n_years, v.size)):
            out.at[tkr, f"A{k}"] = v[k]
    return out


def _sector_median(metric: pd.Series, sector: pd.Series) -> pd.Series:
    valid = metric.notna() & sector.notna()
    med = metric[valid].groupby(sector[valid]).median()
    return sector.map(med)


def compute_gscores(ann: pd.DataFrame, universe: pd.DataFrame,
                    t0: pd.Timestamp) -> tuple[pd.DataFrame, list]:
    """Calcule les 7 ratios, les 8 signaux et le G_SCORE (0-8) a ``t0``.

    ``ann`` : panel annuel PIT. ``universe`` : index=ticker, colonnes
    ``gics_sector`` / ``w_bench``. Retourne ``(scores, issues)`` ; leve
    ``MohanramDataError`` si une variable de base obligatoire manque pour un
    titre NON financier.
    """
    t0 = _to_ts(t0)
    tickers = pd.Index(universe.index)
    sector = universe["gics_sector"].reindex(tickers)
    adv_issues: list = []

    eligible = ann[(ann["pub_date"] <= t0)].copy()
    cutoff = t0 - pd.DateOffset(months=FILING_LAG_MONTHS)
    eligible = eligible[eligible["period_end"] <= cutoff]
    eligible = eligible[eligible["ticker"].isin(tickers)]
    eligible = eligible.drop_duplicates(subset=["ticker", "period_end"], keep="last")

    # PIT par champ : R&D/CAPEX/ADV ne sont exploitables que publies avant t0.
    for src, pubcol in (("rd", "pub_rd"), ("capex", "pub_capex"),
                        ("adv", "pub_adv")):
        if pubcol in eligible.columns:
            late = eligible[pubcol].isna() | (eligible[pubcol] > t0)
            eligible.loc[late, src] = np.nan

    ni = _annual_panel(eligible, "net_income", tickers)
    cfo = _annual_panel(eligible, "cfo", tickers)
    sales = _annual_panel(eligible, "sales", tickers)
    rd = _annual_panel(eligible, "rd", tickers)
    capex = _annual_panel(eligible, "capex", tickers)
    adv = _annual_panel(eligible, "adv", tickers)
    ta = _annual_panel(eligible, "total_assets", tickers)

    is_fin = (sector == FINANCIALS_SECTOR).reindex(tickers).fillna(False)

    # --- Level-1 : variables comptables de base obligatoires (non financiers) --
    base = {"net_income": ni["A0"], "cfo": cfo["A0"], "sales": sales["A0"],
            "total_assets_T1": ta["A1"]}
    missing_any = pd.Series(False, index=tickers)
    missing_all = pd.Series(True, index=tickers)
    for series in base.values():
        missing_any |= series.isna()
        missing_all &= series.isna()
    missing_any &= ~is_fin
    missing_all &= ~is_fin

    # Introduction recente (IPO/spin-off) : aucune variable de base -> Level-2.
    for tkr in tickers[missing_all]:
        adv_issues.append({"niveau": "warning", "categorie": "historique_court",
                           "ticker": tkr, "champ": "base",
                           "detail": f"Aucun compte annuel publie a {t0:%d/%m/%Y} "
                                     f"(introduction recente) : titre exclu du "
                                     f"scoring (poids mis a 0)."})

    partial = missing_any & ~missing_all
    critical: list = []
    for name, series in base.items():
        miss = series.isna() & (~is_fin) & (missing_any & ~missing_all)
        for tkr in tickers[miss]:
            critical.append({"niveau": "error", "categorie": "donnee_base",
                             "ticker": tkr, "champ": name,
                             "detail": f"{name} absent (exercice T / T-1) a "
                                       f"{t0:%d/%m/%Y}"})
    if critical:
        raise MohanramDataError(
            "Variables comptables obligatoires manquantes pour des titres non "
            "financiers. Fournir les champs XBRL correspondants "
            "(net_income, cfo, sales, total_assets).", critical)
    data_ok = ~missing_all

    # --- Ratios ---------------------------------------------------------------
    ta_pos = ta["A1"].where(ta["A1"] > 0)
    roa = ni["A0"] / ta_pos
    cfroa = cfo["A0"] / ta_pos

    rd0 = rd["A0"].fillna(0.0)
    capex0 = capex["A0"].abs().fillna(0.0)
    adv0 = adv["A0"].fillna(0.0)
    for name, raw in (("rd", rd), ("capex", capex), ("adv", adv)):
        miss = raw["A0"].isna() & (~is_fin)
        for tkr in tickers[miss]:
            adv_issues.append({"niveau": "warning", "categorie": "impute_zero",
                               "ticker": tkr, "champ": name,
                               "detail": f"{name}_T non publie -> traite a 0 "
                                         f"(G associe = 0 si mediane >= 0)."})
    rd_int = rd0 / ta_pos
    capex_int = capex0 / ta_pos
    adv_int = adv0 / ta_pos

    # VARROA : Var(ROA_{T-4..T}) ; VARSGR : Var(SGR_{T-4..T})
    n_hist = N_YEARS - 1
    roa_hist = pd.DataFrame(index=tickers, dtype=float)
    for k in range(n_hist):
        roa_hist[k] = ni[f"A{k}"] / ta[f"A{k + 1}"].where(ta[f"A{k + 1}"] > 0)
    varroa = roa_hist.apply(
        lambda r: float(r.dropna().var(ddof=1)) if r.notna().sum() >= MIN_VALID_HISTORY
        else np.nan, axis=1)

    sgr_hist = pd.DataFrame(index=tickers, dtype=float)
    for k in range(n_hist):
        prev = sales[f"A{k + 1}"].where(sales[f"A{k + 1}"] > 0)
        sgr_hist[k] = sales[f"A{k}"] / prev - 1.0
    varsgr = sgr_hist.apply(
        lambda r: float(r.dropna().var(ddof=1)) if r.notna().sum() >= MIN_VALID_HISTORY
        else np.nan, axis=1)

    for name, series in (("VARROA", varroa), ("VARSGR", varsgr)):
        miss = series.isna() & (~is_fin)
        for tkr in tickers[miss]:
            adv_issues.append({"niveau": "warning", "categorie": "historique_court",
                               "ticker": tkr, "champ": name,
                               "detail": f"Historique < {MIN_VALID_HISTORY} ans -> "
                                         f"signal force a 0."})

    # --- Signaux binaires ------------------------------------------------------
    def _gt(metric: pd.Series) -> pd.Series:
        med = _sector_median(metric, sector)
        out = (metric > med).astype(float)
        return out.where(metric.notna())

    def _lt_or_zero(metric: pd.Series) -> pd.Series:
        med = _sector_median(metric, sector)
        out = (metric < med).astype(float)
        return out.where(metric.notna(), 0.0)

    g1 = _gt(roa)
    g2 = _gt(cfroa)
    g3 = (cfo["A0"] > ni["A0"]).astype(float).where(
        cfo["A0"].notna() & ni["A0"].notna())
    g4 = _lt_or_zero(varroa)
    g5 = _lt_or_zero(varsgr)
    g6 = _gt(rd_int).fillna(0.0)
    g7 = _gt(capex_int).fillna(0.0)
    g8 = _gt(adv_int).fillna(0.0)

    scores = pd.DataFrame({
        "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6,
        "G7": g7, "G8": g8,
        "ROA": roa, "CFROA": cfroa, "VARROA": varroa, "VARSGR": varsgr,
        "RDINT": rd_int, "CAPINT": capex_int, "ADINT": adv_int,
    }, index=tickers)

    gcols = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    nonzero = scores[gcols][~is_fin].fillna(0.0)
    scores["G_SCORE"] = np.nan
    scores.loc[~is_fin & data_ok, "G_SCORE"] = nonzero.sum(axis=1)
    scores["is_financial"] = is_fin
    scores["data_ok"] = data_ok
    scores["statut"] = np.where(
        is_fin, "Neutre (financier)",
        np.where(data_ok, "Complet", "Exclu (historique insuffisant)"))
    return scores, adv_issues


# ===========================================================================
# Tilting, neutralite sectorielle, plafonnement
# ===========================================================================
def apply_tilt(w_bench: pd.Series, gscore: pd.Series,
               multipliers: Optional[dict] = None,
               data_ok: Optional[pd.Series] = None
               ) -> tuple[pd.Series, pd.Series]:
    """Multiplicateurs deterministes -> poids bruts tildes.

    Les titres sans historique exploitable (``data_ok`` faux) recoivent un
    multiplicateur nul (exclusion stricte).
    """
    mult_cfg = dict(MULTIPLIERS if multipliers is None else multipliers)
    g = gscore.reindex(w_bench.index)
    mult = pd.Series(np.nan, index=w_bench.index, dtype=float)
    for score, m in mult_cfg.items():
        mult[g == score] = float(m)
    if data_ok is not None:
        ok = data_ok.reindex(w_bench.index).fillna(True)
        mult[(~ok) & g.isna()] = 0.0
    # Titres financiers : multiplicateur neutre (poids d'indice conserve).
    fin = g.isna() & (mult.isna())
    mult[fin] = 1.0
    mult = mult.fillna(1.0)
    return w_bench * mult, mult


def sector_neutralize(w_tilt: pd.Series, w_bench: pd.Series,
                      sector: pd.Series) -> tuple[pd.Series, list]:
    """Renormalisation intra-sectorielle (neutralite sectorielle parfaite).

    Les financiers restent strictement a leur poids d'indice. Un secteur dont
    tous les titres sont exclus (``sum w_tilt == 0``) est ramene a la neutralite
    pure et signale.
    """
    out = pd.Series(0.0, index=w_bench.index, dtype=float)
    warnings: list = []
    for s, idx in sector.groupby(sector).groups.items():
        idx = pd.Index(idx)
        wb_s = float(w_bench.loc[idx].sum())
        if s == FINANCIALS_SECTOR:
            out.loc[idx] = w_bench.loc[idx]
            continue
        tilt_s = float(w_tilt.loc[idx].sum())
        if tilt_s <= 0:
            out.loc[idx] = w_bench.loc[idx]
            warnings.append({"niveau": "warning", "categorie": "secteur_degenere",
                             "secteur": s,
                             "detail": "Aucun titre avec G_SCORE >= 4 : secteur "
                                       "ramene a la neutralite pure."})
            continue
        out.loc[idx] = wb_s * w_tilt.loc[idx] / tilt_s
    total = float(out.sum())
    if total > 0:
        out = out / total
    return out, warnings


def iterative_cap(w: pd.Series, w_bench: pd.Series, sector: pd.Series,
                  cap_abs: float = DEFAULT_CAP_ABS,
                  cap_rel: float = DEFAULT_CAP_REL,
                  max_iter: int = 100) -> tuple[pd.Series, list]:
    """Plafonnement iteratif : ``min(cap_abs, w_bench + cap_rel)`` par titre.

    L'excedent est redistribue au prorata sur les titres non plafonnes du meme
    secteur ; si le secteur sature, la redistribution devient globale (avertie).
    """
    w = w.astype(float).copy()
    cap = np.minimum(cap_abs, w_bench.reindex(w.index).fillna(0.0) + cap_rel)
    warnings: list = []
    eps = 1e-12
    global_excess = 0.0
    for s, idx in sector.groupby(sector).groups.items():
        idx = list(idx)
        for _ in range(max_iter):
            over = [t for t in idx if w[t] > cap[t] + eps]
            if not over:
                break
            excess = float(sum(w[t] - cap[t] for t in over))
            for t in over:
                w[t] = cap[t]
            elig = [t for t in idx if w[t] < cap[t] - eps]
            if not elig:
                global_excess += excess
                warnings.append({"niveau": "warning", "categorie": "secteur_sature",
                                 "secteur": s,
                                 "detail": "Tous les titres du secteur plafonnes : "
                                           "redistribution globale."})
                break
            tot = float(sum(w[t] for t in elig))
            if tot <= 0:
                for t in elig:
                    w[t] += excess / len(elig)
            else:
                for t in elig:
                    w[t] += excess * w[t] / tot
    if global_excess > 0:
        elig = [t for t in w.index if w[t] < cap[t] - eps]
        tot = float(sum(w[t] for t in elig))
        if tot > 0:
            for t in elig:
                w[t] += global_excess * w[t] / tot
    total = float(w.sum())
    if total > 0:
        w = w / total
    return w, warnings


# ===========================================================================
# Simulation 6 mois (buy & hold, delisting Shumway)
# ===========================================================================
def _price_diagnostics(px: pd.DataFrame, tickers: Sequence[str],
                       dates: pd.DatetimeIndex) -> tuple[dict, list]:
    """Detecte les radiations (arret terminal) et les trous de prix."""
    delist: dict = {}
    critical: list = []
    for t in tickers:
        s = px[t]
        valid = s.dropna()
        if valid.empty:
            critical.append({"niveau": "error", "categorie": "prix", "ticker": t,
                             "detail": "Aucune cotation sur la fenetre."})
            continue
        first, last = valid.index[0], valid.index[-1]
        inner = s.loc[(s.index >= first) & (s.index <= last)]
        if inner.isna().any():
            gap = inner.index[inner.isna()][0]
            critical.append({"niveau": "error", "categorie": "prix", "ticker": t,
                             "detail": f"Cotation manquante (NaN) au {gap:%d/%m/%Y}."})
            continue
        if first > dates[0]:
            critical.append({"niveau": "error", "categorie": "prix", "ticker": t,
                             "detail": f"Pas de cotation au debut de fenetre "
                                       f"({dates[0]:%d/%m/%Y})."})
            continue
        if last < dates[-1]:
            delist[t] = last
    return delist, critical


def simulate_horizon(w_final: pd.Series, universe: pd.DataFrame,
                     prices: pd.DataFrame, bench: pd.Series,
                     t0: pd.Timestamp, tx_bps: float = DEFAULT_TX_BPS,
                     horizon_months: int = HORIZON_MONTHS) -> dict:
    """Simule le portefeuille 6 mois (buy & hold + radiations Shumway)."""
    t0 = _to_ts(t0)
    end = t0 + pd.DateOffset(months=horizon_months)
    dates = pd.DatetimeIndex(prices.index).intersection(pd.DatetimeIndex(bench.index))
    dates = dates[(dates >= t0) & (dates <= end)]
    if len(dates) < 2:
        raise MohanramDataError(
            f"Fenetre [{t0:%d/%m/%Y} ; {end:%d/%m/%Y}] insuffisante en cotations.")

    held = w_final[w_final > 0].index
    missing_px = [t for t in held if t not in prices.columns]
    if missing_px:
        raise MohanramDataError(
            "Series de prix absentes pour des titres detenus.",
            [{"niveau": "error", "categorie": "prix", "ticker": t,
              "detail": "colonne absente de prices.parquet"} for t in missing_px])

    px = prices.reindex(dates, columns=list(held))
    delist, critical = _price_diagnostics(px, list(held), dates)
    if critical:
        raise MohanramDataError(
            "Cotaitons manquantes sur des titres detenus (hors radiation "
            "terminale).", critical)

    sector = universe["gics_sector"].reindex(held)
    rets = px.pct_change(fill_method=None)

    w = w_final.reindex(held).astype(float).copy()
    w = w / w.sum()
    turnover = float(0.5 * np.abs(w).sum())
    cost = float(tx_bps) / 1e4 * turnover
    nav = 100.0 * (1.0 - cost)

    navs = [nav]
    weights = [w.copy()]
    delist_events: list = []
    for i in range(1, len(dates)):
        day = dates[i]
        r = rets.loc[day].reindex(held).fillna(0.0)
        for t, dd in delist.items():
            if day == dd:
                r[t] = SHUMWAY_DELIST_RETURN
            elif day > dd:
                r[t] = 0.0
        r_port = float((w * r).sum())
        denom = 1.0 + r_port
        nav *= denom
        if abs(denom) > 1e-12:
            w = (w * (1.0 + r)) / denom
        else:  # pragma: no cover
            w = w * (1.0 + r)
        for t, dd in delist.items():
            if day == dd and w.get(t, 0.0) > 0:
                proceeds = float(w[t])
                w[t] = 0.0
                sec = sector.get(t)
                pool = [u for u in held if w[u] > 0
                        and sector.get(u) == sec
                        and (u not in delist or dd < delist[u])]
                if not pool:
                    pool = [u for u in held if w[u] > 0]
                tot = float(sum(w[u] for u in pool))
                if tot > 0:
                    for u in pool:
                        w[u] += proceeds * w[u] / tot
                delist_events.append({"niveau": "warning", "categorie": "radiation",
                                      "ticker": t, "date": day,
                                      "detail": f"Radiation au {day:%d/%m/%Y} "
                                                f"(Shumway {SHUMWAY_DELIST_RETURN:+.0%}), "
                                                f"reinvesti."})
        navs.append(nav)
        weights.append(w.copy())

    result = pd.DataFrame(weights, index=dates)
    nav_ser = pd.Series(navs, index=dates, name="Mohanram")
    bench_win = bench.reindex(dates).astype(float)
    bench_nav = (bench_win / bench_win.dropna().iloc[0] * 100.0).ffill()
    return {"nav": nav_ser, "bench_nav": bench_nav, "weights_daily": result,
            "turnover": turnover, "cost": cost, "dates": dates,
            "delist_events": delist_events, "delist": delist}


# ===========================================================================
# Attribution Brinson-Fachler
# ===========================================================================
def brinson_fachler(w_final: pd.Series, w_bench: pd.Series,
                    sector: pd.Series, total_returns: pd.Series) -> pd.DataFrame:
    """Decomposition Brinson-Fachler (allocation / selection / interaction)."""
    cols = pd.Index(w_final.index).union(w_bench.index)
    wp = w_final.reindex(cols).fillna(0.0)
    wb = w_bench.reindex(cols).fillna(0.0)
    r = total_returns.reindex(cols).fillna(0.0)
    sec = sector.reindex(cols).fillna("__NA__")
    rows = []
    for s in sorted(sec.unique()):
        m = sec == s
        Wp, Wb = float(wp[m].sum()), float(wb[m].sum())
        if Wp <= 0 and Wb <= 0:
            continue
        Rp = float((wp[m] * r[m]).sum() / Wp) if Wp > 0 else 0.0
        Rb = float((wb[m] * r[m]).sum() / Wb) if Wb > 0 else 0.0
        rows.append({"secteur": s, "Wp": Wp, "Wb": Wb, "R_p": Rp, "R_b": Rb})
    if not rows:
        return pd.DataFrame(columns=["Wp", "Wb", "R_p", "R_b", "Allocation",
                                     "Sélection", "Interaction", "Total"])
    df = pd.DataFrame(rows).set_index("secteur")
    Rb_total = float((df["Wb"] * df["R_b"]).sum())
    df["Allocation"] = (df["Wp"] - df["Wb"]) * (df["R_b"] - Rb_total)
    df["Sélection"] = df["Wb"] * (df["R_p"] - df["R_b"])
    df["Interaction"] = (df["Wp"] - df["Wb"]) * (df["R_p"] - df["R_b"])
    df["Total"] = df["Allocation"] + df["Sélection"] + df["Interaction"]
    return df


# ===========================================================================
# Orchestration
# ===========================================================================
def run_mohanram(t0: pd.Timestamp,
                 params: Optional[dict] = None,
                 refresh_data: bool = False) -> dict:
    """Pipeline complet de la strategie Mohanram pour la date ``t0``."""
    params = dict(params or {})
    cap_abs = float(params.get("cap_abs", DEFAULT_CAP_ABS))
    cap_rel = float(params.get("cap_rel", DEFAULT_CAP_REL))
    tx_bps = float(params.get("tx_bps", DEFAULT_TX_BPS))
    multipliers = params.get("multipliers")
    horizon_months = int(params.get("horizon_months", HORIZON_MONTHS))

    t0 = _to_ts(t0)
    universe = load_universe_at(t0)
    issues = list(universe.attrs.get("issues", []))

    prices = dl.get_prices()
    bench = dl.get_benchmarks()[config.BENCHMARK_TICKER]
    ann = build_annual_fundamentals(refresh=refresh_data)

    scores, score_issues = compute_gscores(ann, universe, t0)
    issues.extend(score_issues)

    w_bench = universe["w_bench"]
    sector = universe["gics_sector"]
    gscore = scores["G_SCORE"]

    w_tilt, mult = apply_tilt(w_bench, gscore, multipliers,
                              data_ok=scores["data_ok"])
    w_neutral, neutral_issues = sector_neutralize(w_tilt, w_bench, sector)
    issues.extend(neutral_issues)
    w_final, cap_issues = iterative_cap(w_neutral, w_bench, sector,
                                        cap_abs=cap_abs, cap_rel=cap_rel)
    issues.extend(cap_issues)

    sim = simulate_horizon(w_final, universe, prices, bench, t0, tx_bps=tx_bps,
                           horizon_months=horizon_months)
    issues.extend(sim["delist_events"])

    nav, bench_nav = sim["nav"], sim["bench_nav"]
    stats = M.performance_stats(nav, bench_nav, rf=float(params.get("rf", 0.0)))

    held = w_final[w_final > 0]
    excluded = w_bench.index[(gscore.fillna(0) <= 3)
                             & (sector != FINANCIALS_SECTOR)]
    excluded_w = float(w_bench.reindex(excluded).sum())
    total_w = float(w_bench.sum())
    active_share = float(0.5 * np.abs(
        w_final.reindex(pd.Index(w_final.index).union(w_bench.index)).fillna(0.0)
        - w_bench.reindex(pd.Index(w_final.index).union(w_bench.index)).fillna(0.0)
    ).sum())

    common = nav.index.intersection(bench_nav.index)
    total_ret = (prices.loc[common[-1], pd.Index(held.index)] /
                 prices.loc[common[0], pd.Index(held.index)] - 1.0)
    brinson = brinson_fachler(w_final, w_bench, sector, total_ret)

    end_row = sim["weights_daily"].iloc[-1]
    sector_t0 = w_final.groupby(sector).sum().reindex(config.GICS_SECTORS).fillna(0.0)
    sector_t1 = end_row.groupby(sector).sum().reindex(config.GICS_SECTORS).fillna(0.0)
    sector_bench = w_bench.groupby(sector).sum().reindex(config.GICS_SECTORS).fillna(0.0)

    portfolio = pd.DataFrame({
        "ticker": universe.index,
        "name": universe["name"].to_numpy(),
        "sector": sector.to_numpy(),
        "w_bench": w_bench.to_numpy(),
        "multiplier": mult.to_numpy(),
        "w_final": w_final.to_numpy(),
    })
    portfolio = portfolio.join(scores[[f"G{k}" for k in range(1, 9)]
                                      + ["G_SCORE", "statut"]], on="ticker")
    portfolio["w_actif"] = portfolio["w_final"] - portfolio["w_bench"]
    portfolio["rendement_6m"] = total_ret.reindex(portfolio["ticker"]).to_numpy()

    level1 = [i for i in issues if i.get("niveau") == "error"]
    level2 = [i for i in issues if i.get("niveau") == "warning"]
    status = "ok" if not level2 else "warning"

    return {
        "t0": t0,
        "rebalance_date": universe.attrs.get("rebalance_date"),
        "horizon_end": nav.index[-1],
        "universe": universe,
        "scores": scores,
        "portfolio": portfolio,
        "w_bench": w_bench,
        "w_final": w_final,
        "nav": nav,
        "bench_nav": bench_nav,
        "weights_daily": sim["weights_daily"],
        "stats": stats,
        "active_share": active_share,
        "excluded_weight": excluded_w / total_w if total_w else np.nan,
        "n_held": int(len(held)),
        "n_bench": int(len(w_bench)),
        "n_excluded": int(len(excluded)),
        "brinson": brinson,
        "sector_t0": sector_t0,
        "sector_t1": sector_t1,
        "sector_bench": sector_bench,
        "issues": issues,
        "level1": level1,
        "level2": level2,
        "status": status,
        "tx_bps": tx_bps,
        "cap_abs": cap_abs,
        "cap_rel": cap_rel,
    }
