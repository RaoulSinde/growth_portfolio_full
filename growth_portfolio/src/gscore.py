"""G-Score de Mohanram (pilier de RECHERCHE) — calcul PIT strict.

Onglet de recherche : ce module n'altere PAS la strategie composite a trois
piliers. Il fournit un score entier 0-7 (Mohanram 2005, critere de publicite G8
volontairement exclu faute de donnee), agrege en un Z-score intra-sectoriel
utilisable comme signal de selection.

Criteres (1 point chacun), tous calcules en coupe transversale a la date ``t``
sur les seuls fondamentaux publies (``pub_date <= t``) :

    Rentabilite
      G1 : ROA    = NI_TTM / Actifs       > mediane sectorielle
      G2 : CFO/TA = OCF_TTM / Actifs      > mediane sectorielle
      G3 : Accruals : OCF_TTM > NI_TTM
    Stabilite
      G4 : Var(ROA) sur ~20 trimestres    < mediane sectorielle
      G5 : Var(croissance ventes)         < mediane sectorielle
    Investissement
      G6 : R&D / Actifs                   > mediane sectorielle
      G7 : CAPEX / Actifs                 > mediane sectorielle

Conventions :
- R&D / CAPEX non rapportes sont imputes a 0 (convention Mohanram) : l'absence
  est interpretee comme une intensite nulle.
- G4 / G5 exigent un historique minimal (``MIN_QUARTERS``) ; sinon le critere est
  NaN et n'entre pas dans le compte (score renormalise via ``G_N``).
- La mediane sectorielle suit une regle a 3 niveaux (secteur GICS -> super-secteur
  -> univers) pour eviter toute mediane sur un secteur minuscule (cf.
  ``sector_median_at_t``).

SECURITE : aucune donnee issue de l'attribut ``info`` d'un Ticker. Aucune donnee
de prix n'entre dans le score (le G-Score est independant du prix) ; les prix ne
servent qu'a ecarter les titres non tradables de l'univers eligibile.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl
from . import factors

__all__ = [
    "GSCORE_COLUMNS",
    "MIN_MEDIAN_N",
    "MIN_QUARTERS",
    "VARIANCE_WINDOW",
    "SUPER_SECTORS",
    "sector_median_at_t",
    "quarterly_ratio_std",
    "quarterly_growth_std",
    "compute_gscores",
    "build_gscore_panel",
    "gscore_quality_correlation",
    "gscore_dispersion",
    "clear_cache",
]

# Nombre minimal de titres pour qu'une mediane (secteur ou super-secteur) soit
# jugee robuste. Superieur a `factors.SECTOR_MIN_N` (5) : une mediane est plus
# sensible a la taille que la moyenne d'un Z-score.
MIN_MEDIAN_N = 8
# Historique trimestriel minimal pour les criteres de variance (G4/G5).
MIN_QUARTERS = 16
# Fenetre glissante (trimestres) pour les variances (cible ~5 ans).
VARIANCE_WINDOW = 20

# Regroupement des micro-secteurs GICS en poches economiquement homogenes
# (niveau 2 de la mediane, avant le repli sur l'univers).
SUPER_SECTORS = {
    "Energy": "Resources & Utilities",
    "Materials": "Resources & Utilities",
    "Utilities": "Resources & Utilities",
    "Financials": "Rate-sensitive",
    "Real Estate": "Rate-sensitive",
    "Consumer Staples": "Defensives",
    "Health Care": "Defensives",
    "Information Technology": "Information Technology",
    "Communication Services": "Communication Services",
    "Consumer Discretionary": "Consumer Discretionary",
    "Industrials": "Industrials",
}

# Colonnes du panel G-Score.
GSCORE_COLUMNS = [
    "G_Roa", "G_Cfo", "G_Accruals", "G_RoaVar", "G_SalesVar", "G_Rd", "G_Capex",
    "G_Count", "G_N", "G_Score", "G_Core", "G_Ext",
    "Z_GScore", "Z_GScore_Core", "Z_GScore_Ext", "eligible",
]

GSCORE_CACHE = os.path.join(config.CACHE_DIR, "gscore_panel.parquet")

# Fragments de code verifies par l'audit (anti-derive de la formule).
_CRITERIA_KEYS = ["G_Roa", "G_Cfo", "G_Accruals", "G_RoaVar", "G_SalesVar",
                  "G_Rd", "G_Capex"]


# ===========================================================================
# Cache
# ===========================================================================
def clear_cache() -> None:
    """Supprime le cache du panel G-Score."""
    try:
        if os.path.exists(GSCORE_CACHE):
            os.remove(GSCORE_CACHE)
    except Exception:  # pragma: no cover - disque indisponible
        pass


def _read_cache() -> Optional[pd.DataFrame]:
    if not os.path.exists(GSCORE_CACHE):
        return None
    try:
        df = pd.read_parquet(GSCORE_CACHE)
    except Exception:  # pragma: no cover
        return None
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_levels(
            pd.to_datetime(df.index.levels[0]), level=0)
    return df


def _write_cache(panel: pd.DataFrame) -> None:
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        panel.to_parquet(GSCORE_CACHE)
    except Exception:  # pragma: no cover
        pass


# ===========================================================================
# Mediane sectorielle PIT (3 niveaux)
# ===========================================================================
def sector_median_at_t(metric: pd.Series, sector: pd.Series,
                       tickers: pd.Index,
                       min_n: int = MIN_MEDIAN_N
                       ) -> tuple[pd.Series, pd.Series]:
    """Mediane de reference par titre, en coupe transversale a une date.

    Regle a 3 niveaux (du plus fin au plus large), chaque niveau n'etant utilise
    que si l'echantillon VALIDE (metrique non NaN a la date) atteint ``min_n`` :

    1. **secteur GICS** ;
    2. **super-secteur** (regroupement explicite, cf. ``SUPER_SECTORS``) ;
    3. **univers** (dernier recours).

    Retourne ``(median_ref, level)`` : la mediane a laquelle chaque titre est
    compare (broadcast) et le niveau d'agregation effectivement employe
    (``"sector"`` / ``"super"`` / ``"universe"`` / ``"none"``).
    """
    tickers = pd.Index(tickers)
    metric = pd.Series(metric).reindex(tickers).astype(float)
    sec = pd.Series(sector).reindex(tickers)
    valid = metric.notna()

    med = pd.Series(np.nan, index=tickers, dtype=float)
    level = pd.Series("none", index=tickers, dtype=object)

    # Niveau 1 : secteur GICS.
    sec_valid = sec.where(valid)
    for g, idx in sec_valid.dropna().groupby(sec_valid.dropna()).groups.items():
        idx = pd.Index(idx)
        if len(idx) >= min_n:
            med.loc[idx] = float(metric.loc[idx].median())
            level.loc[idx] = "sector"

    # Niveau 2 : super-secteur pour les titres non encore couverts.
    rem = valid & med.isna() & sec.notna()
    if rem.any():
        sup = sec[rem].map(SUPER_SECTORS).fillna(sec[rem])
        for g, idx in sup.groupby(sup).groups.items():
            idx = pd.Index(idx)
            if len(idx) >= min_n:
                med.loc[idx] = float(metric.loc[idx].median())
                level.loc[idx] = "super"

    # Niveau 3 : univers (dernier recours).
    rem = valid & med.isna()
    if rem.any():
        med.loc[rem] = float(metric[valid].median())
        level.loc[rem] = "universe"

    return med, level


# ===========================================================================
# Variances trimestrielles PIT (G4 / G5)
# ===========================================================================
def quarterly_ratio_std(fund: pd.DataFrame, num_col: str, den_col: str,
                        tickers: pd.Index,
                        window: int = VARIANCE_WINDOW,
                        min_obs: int = MIN_QUARTERS) -> pd.Series:
    """Ecart-type trimestriel de ``num_col / den_col`` sur ~``window`` trimestres.

    ``fund`` doit etre DEJA filtre (``pub_date <= t``). Le ratio n'est conserve
    que sur les ``(ticker, period_end)`` ou les deux colonnes sont non nulles et
    le denominateur strictement positif ; l'ecart-type (``ddof=1``) exige au moins
    ``min_obs`` observations, sinon NaN.
    """
    tickers = pd.Index(tickers)
    out = pd.Series(np.nan, index=tickers, dtype=float)
    if fund is None or fund.empty:
        return out
    cols = {"ticker", "period_end", num_col, den_col}
    if not cols.issubset(fund.columns):
        return out
    sub = fund[["ticker", "period_end", num_col, den_col]].dropna()
    sub = sub[sub[den_col] > 0]
    if sub.empty:
        return out
    sub = sub.sort_values(["ticker", "period_end"])
    for t, g in sub.groupby("ticker", sort=False):
        if t not in out.index:
            continue
        v = (g[num_col] / g[den_col]).to_numpy(dtype=float)[-window:]
        if v.size >= min_obs and np.isfinite(v).all():
            out.at[t] = float(np.std(v, ddof=1))
    return out


def quarterly_growth_std(fund: pd.DataFrame, col: str, tickers: pd.Index,
                         window: int = VARIANCE_WINDOW + 1,
                         min_obs: int = MIN_QUARTERS - 1) -> pd.Series:
    """Ecart-type de la croissance trimestrielle (``pct_change``) d'un flux."""
    tickers = pd.Index(tickers)
    out = pd.Series(np.nan, index=tickers, dtype=float)
    if fund is None or fund.empty or col not in fund.columns:
        return out
    sub = fund[["ticker", "period_end", col]].dropna()
    if sub.empty:
        return out
    sub = sub.sort_values(["ticker", "period_end"])
    for t, g in sub.groupby("ticker", sort=False):
        if t not in out.index:
            continue
        v = g[col].to_numpy(dtype=float)[-window:]
        if v.size < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = v[1:] / v[:-1] - 1.0
        growth = growth[np.isfinite(growth)]
        if growth.size >= min_obs:
            out.at[t] = float(np.std(growth, ddof=1))
    return out


# ===========================================================================
# Coupe transversale : G-Score d'une date
# ===========================================================================
def compute_gscores(date: pd.Timestamp, prices: pd.DataFrame,
                    fund: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Calcule les 7 criteres G-Score et le score agrege a la date ``t``.

    ``universe`` : coupe par titre (``gics_sector``, optionnellement ``weight``).
    ``fund`` : panel fondamental PIT (la coupe ``pub_date <= t`` est appliquee
    ici). Les prix ne servent qu'a marquer ``eligible`` (au moins 2 observations).
    """
    date = pd.Timestamp(date)
    if getattr(date, "tzinfo", None) is not None:
        date = date.tz_localize(None)
    date = date.normalize()

    if "ticker" in getattr(universe, "columns", []):
        uni = universe.set_index("ticker")
    else:
        uni = universe.copy()
    tickers = pd.Index(uni.index)
    if "gics_sector" in uni.columns:
        sector = uni["gics_sector"].reindex(tickers)
    else:
        sector = uni.get("prop_sector",
                         pd.Series(index=tickers, dtype=object)).reindex(tickers)

    # --- Panel PIT -----------------------------------------------------------
    f = fund.copy()
    if "ticker" not in f.columns:
        f = f.reset_index()
        if "ticker" not in f.columns:
            f = f.rename(columns={f.columns[0]: "ticker"})
    if "pub_date" in f.columns:
        f = f[pd.to_datetime(f["pub_date"]) <= date]
    f = f[f["ticker"].isin(tickers)]
    if not f.empty:
        f = f.drop_duplicates(subset=["ticker", "period_end"])

    # --- Agregats TTM --------------------------------------------------------
    ni_ttm = factors._ttm_panel(f, "net_income_q", tickers)
    ocf_ttm = factors._ttm_panel(f, "ocf_q", tickers)
    rd_ttm = factors._ttm_panel(f, "research_dev_q", tickers).fillna(0.0)
    capex_ttm = factors._ttm_panel(f, "capex_q", tickers).abs().fillna(0.0)
    ta = dl.annual_stock_points(f, "total_assets", tickers)["N"]
    ta_pos = ta.where(ta > 0)

    roa = ni_ttm / ta_pos
    cfo = ocf_ttm / ta_pos
    rd_int = rd_ttm / ta_pos
    capex_int = capex_ttm / ta_pos

    roa_var = quarterly_ratio_std(f, "net_income_q", "total_assets", tickers)
    sales_var = quarterly_growth_std(f, "revenue_q", tickers)

    # --- Criteres (1 point / 0) ---------------------------------------------
    def _gt(metric: pd.Series) -> pd.Series:
        med, _ = sector_median_at_t(metric, sector, tickers)
        out = (metric > med).astype(float)
        return out.where(metric.notna())

    def _lt(metric: pd.Series) -> pd.Series:
        med, _ = sector_median_at_t(metric, sector, tickers)
        out = (metric < med).astype(float)
        return out.where(metric.notna())

    G_roa = _gt(roa)
    G_cfo = _gt(cfo)
    both_ni = ni_ttm.notna() & ocf_ttm.notna()
    G_accruals = (ocf_ttm > ni_ttm).astype(float).where(both_ni)
    G_roavar = _lt(roa_var)
    G_salesvar = _lt(sales_var)
    G_rd = _gt(rd_int)
    G_capex = _gt(capex_int)

    criteria = pd.DataFrame({
        "G_Roa": G_roa, "G_Cfo": G_cfo, "G_Accruals": G_accruals,
        "G_RoaVar": G_roavar, "G_SalesVar": G_salesvar,
        "G_Rd": G_rd, "G_Capex": G_capex,
    }, index=tickers)

    avail = criteria.notna()
    points = criteria.fillna(0.0)
    G_count = points.sum(axis=1)
    G_n = avail.sum(axis=1)
    # Score renormalise sur 7 (comparable entre titres a couverture differente).
    G_score = (G_count / G_n.replace(0, np.nan)) * len(_CRITERIA_KEYS)
    core_keys = ["G_Roa", "G_Cfo", "G_Accruals", "G_RoaVar", "G_SalesVar"]
    avail_core = avail[core_keys].any(axis=1)
    avail_ext = avail[_CRITERIA_KEYS].any(axis=1)
    G_core = points[core_keys].sum(axis=1).where(avail_core)
    G_ext = points[_CRITERIA_KEYS].sum(axis=1).where(avail_ext)

    # --- Z-scores intra-sectoriels ------------------------------------------
    def _z(metric: pd.Series) -> pd.Series:
        return factors.zscore(metric.reindex(tickers), by_sector=True,
                              sector=sector)

    Z_full = _z(G_score)
    Z_core = _z(G_core)
    Z_ext = _z(G_ext)

    # --- Eligibilite (tradable : >= 2 prix observes <= t) --------------------
    px = prices.loc[prices.index <= date] if prices is not None else pd.DataFrame()
    has_px = pd.Series(False, index=tickers)
    px_cols = [t for t in tickers if t in px.columns]
    if px_cols:
        has_px.loc[px_cols] = (px[px_cols].notna().sum() >= 2).to_numpy()
    eligible = (G_score.notna()) & (G_n >= 3) & has_px

    out = pd.DataFrame({
        "G_Roa": G_roa, "G_Cfo": G_cfo, "G_Accruals": G_accruals,
        "G_RoaVar": G_roavar, "G_SalesVar": G_salesvar,
        "G_Rd": G_rd, "G_Capex": G_capex,
        "G_Count": G_count, "G_N": G_n, "G_Score": G_score,
        "G_Core": G_core, "G_Ext": G_ext,
        "Z_GScore": Z_full, "Z_GScore_Core": Z_core, "Z_GScore_Ext": Z_ext,
        "eligible": eligible,
    }, index=tickers)

    # Recentrage transversal final (coherence avec `factors`).
    for col in ("Z_GScore", "Z_GScore_Core", "Z_GScore_Ext"):
        out[col] = out[col] - out[col].mean()
    return out[GSCORE_COLUMNS]


# ===========================================================================
# Panel complet (toutes les dates)
# ===========================================================================
def build_gscore_panel(rebalance_dates: Optional[Sequence] = None,
                       use_cache: bool = True) -> pd.DataFrame:
    """Panel G-Score MultiIndex (date, ticker) sur les dates de rebalancement."""
    if rebalance_dates is None:
        dates = [pd.Timestamp(d) for d in config.REBALANCE_DATES]
    else:
        dates = sorted(pd.Timestamp(d) for d in rebalance_dates)

    cached = _read_cache() if use_cache else None
    have: set = set()
    if cached is not None and not cached.empty:
        have = set(pd.to_datetime(
            cached.index.get_level_values(0)).normalize().unique())

    missing = [d for d in dates if d.normalize() not in have]
    if missing:
        prices = dl.get_prices()
        fund = dl.get_fundamentals()
        smap = dl.get_sector_map()
        w_all = dl.get_index_weights(dates)

        frames = []
        for d in missing:
            try:
                w_d = w_all.xs(pd.Timestamp(d), level="date")["weight"]
            except (KeyError, TypeError):
                continue
            if w_d.empty:
                continue
            tickers = pd.Index(w_d.index)
            uni = pd.DataFrame({"gics_sector": smap.reindex(tickers).to_numpy(),
                                "weight": w_d.to_numpy()}, index=tickers)
            sc = compute_gscores(d, prices, fund, uni)
            sc = sc.copy()
            sc.index = pd.MultiIndex.from_product([[pd.Timestamp(d)], sc.index],
                                                  names=["date", "ticker"])
            frames.append(sc)

        if not frames:
            panel = cached if cached is not None else pd.DataFrame(
                columns=GSCORE_COLUMNS)
        else:
            new = pd.concat(frames)
            if cached is not None and not cached.empty:
                new = pd.concat([cached, new])
                new = new[~new.index.duplicated(keep="last")]
            panel = new.sort_index()
            if use_cache:
                _write_cache(panel)
    else:
        panel = cached

    mask = pd.to_datetime(panel.index.get_level_values(0)).normalize().isin(
        [d.normalize() for d in dates])
    result = panel[mask].copy()
    result.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(result.index.get_level_values(0)),
         result.index.get_level_values(1)], names=["date", "ticker"])
    return result[GSCORE_COLUMNS]


# ===========================================================================
# Diagnostics (risque residuel)
# ===========================================================================
def gscore_quality_correlation(gs_panel: pd.DataFrame,
                               factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Correlation transversale G-Score vs pilier Qualite/GARP, par date.

    Mesure le recouvrement du G-Score avec le pilier Qualite existant (risque de
    quasi-colinearite) : Pearson et Spearman entre ``Z_GScore`` et
    ``Z_Quality_GARP`` sur la coupe de chaque date.
    """
    rows = []
    dates = sorted(pd.Index(gs_panel.index.get_level_values(0)).unique())
    for d in dates:
        sub = gs_panel.xs(d, level=0)
        try:
            fac = factor_panel.xs(d, level=0)
        except KeyError:
            continue
        cols = sub.index.intersection(fac.index)
        a = sub.loc[cols, "Z_GScore"].astype(float)
        b = fac.loc[cols, "Z_Quality_GARP"].astype(float)
        both = pd.concat([a.rename("g"), b.rename("q")], axis=1).dropna()
        if len(both) < 3:
            continue
        pear = float(both["g"].corr(both["q"]))
        spear = float(both["g"].corr(both["q"], method="spearman"))
        rows.append({"date": pd.Timestamp(d), "n": int(len(both)),
                     "Pearson": pear, "Spearman": spear})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["n", "Pearson", "Spearman"])


def gscore_dispersion(gs_panel: pd.DataFrame) -> pd.DataFrame:
    """Distribution du score entier par date (dispersion de l'univers).

    Retourne, par date, le nombre de titres, la moyenne / ecart-type de
    ``G_Score``, et la part (%) des titres dans chaque classe de points
    (``G_Count`` de 0 a 7).
    """
    rows = []
    dates = sorted(pd.Index(gs_panel.index.get_level_values(0)).unique())
    for d in dates:
        sub = gs_panel.xs(d, level=0)
        s = sub["G_Score"].dropna()
        cnt = sub["G_Count"].dropna()
        row = {"date": pd.Timestamp(d), "n": int(s.size),
               "mean": float(s.mean()) if s.size else np.nan,
               "std": float(s.std(ddof=1)) if s.size > 1 else np.nan,
               "share_extreme": float((cnt >= 6).mean()) if cnt.size else np.nan}
        for k in range(8):
            row[f"p_{k}"] = float((cnt == k).mean()) if cnt.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
