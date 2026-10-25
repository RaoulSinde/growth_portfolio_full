"""Calcul des signaux factoriels (3 piliers statiques PIT) et des Z-scores.

Strategie UNIQUE « Growth USA » : a chaque date de rebalancement `t` on calcule
une coupe transversale PURE de l'univers (aucune statistique agregee sur la
matrice temporelle complete) :

    Z_Composite = w_accel*Z_Acceleration + w_mom*Z_Momentum + w_qual*Z_Quality_GARP

Piliers (agregats annuels Point-in-Time, ``pub_date <= t``) :

1. **Acceleration fondamentale & expansion operationnelle** (25 %) :
   - Sales_Acceleration   = [CA(N)/CA(N-1) - 1] - [CA(N-1)/CA(N-2) - 1]
   - Margin_Expansion     = OI(N)/CA(N) - OI(N-1)/CA(N-1)
   - Asset_Efficiency_Delta = CA(N)/TA(N) - CA(N-1)/TA(N-1)
     (secteur **Financials** : remplacé par ROA(N) - ROA(N-1))
2. **Momentum de prix ajuste du risque** (45 %) :
   - Risk_Adjusted_Momentum = (P[t-21]/P[t-126] - 1) / (std(ret ; 6M)*sqrt(252))
   - Trend_Filter           = Price / SMA200 - 1
3. **Qualite intrinseque & valorisation GARP** (30 %) :
   - ROE_TTM       = Net_Income_TTM / Total_Stockholders_Equity
   - FCF_Yield     = (OCF_TTM - |CapEx_TTM|) / Market_Cap
   - GARP_Inverted = EPS_Growth_3Y / (Price / EPS_TTM)
     (secteur **Financials** : FCF omis -> 0.60*Z_ROE + 0.40*Z_GARP)

Chaque metrique brute est (1) winsorisee aux 5e/95e centiles, puis (2)
transformee en Z-score intra-sectoriel GICS (fallback univers si le secteur
compte < 5 titres), pour rendre les scores comparables entre secteurs.

Étape 1 — le filtre bilan ``purge_junk_growth`` s'applique AVANT le scoring.

SECURITE : aucune donnee issue de l'attribut ``info`` d'un Ticker.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl

__all__ = [
    "FACTOR_COLUMNS",
    "ttm",
    "winsorize",
    "zscore",
    "purge_junk_growth",
    "compute_strategy_scores",
    "composite_weights_apply",
    "build_factor_panel",
    "clear_cache",
]

# Nombre minimal de titres dans un secteur pour un Z-score intra-sectoriel.
SECTOR_MIN_N = 5
# Nombre minimal de titres d'un secteur pour appliquer la purge par décile.
PURGE_MIN_N = 5
# Seuil de detection du noyau Mega-Cap (poids indiciel, fraction).
MEGA_CAP_THRESHOLD = 0.025
# Fenetres (jours de bourse).
MOM_1M = 21
MOM_6M = 126
MOM_12M = 252          # conserve (compatibilite) ; le moteur utilise 6M-1M
VOL_WINDOW_6M = 126    # volatilite realisee 6M (annualisee)
SMA_TREND = 200        # moyenne mobile de tendance
# Historique minimal (rendements quotidiens) requis pour estimer la vol 6M.
MIN_VOL_OBS = 60
# Secteur GICS avec traitement specifique (Financials).
FINANCIALS_SECTOR = "Financials"
# Nombre d'annees pour la croissance BPA historique.
EPS_GROWTH_YEARS = 3

FACTOR_COLUMNS = [
    "Z_Acceleration", "Z_Momentum", "Z_Quality_GARP", "Z_Composite",
    "S_SalesAccel", "S_MarginExp", "S_AssetEff",
    "S_RAM", "S_Trend",
    "S_ROE", "S_FCFYield", "S_GARP",
    "eligible",
]

FACTOR_CACHE = os.path.join(config.CACHE_DIR, "factor_panel.parquet")


# ===========================================================================
# Cache
# ===========================================================================
def clear_cache() -> None:
    """Supprime le cache du panel factoriel."""
    try:
        if os.path.exists(FACTOR_CACHE):
            os.remove(FACTOR_CACHE)
    except Exception:  # pragma: no cover - disque indisponible
        pass


def _read_cache() -> Optional[pd.DataFrame]:
    if not os.path.exists(FACTOR_CACHE):
        return None
    try:
        df = pd.read_parquet(FACTOR_CACHE)
    except Exception:  # pragma: no cover - cache corrompu
        return None
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_levels(
            pd.to_datetime(df.index.levels[0]), level=0
        )
    return df


def _write_cache(panel: pd.DataFrame) -> None:
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        panel.to_parquet(FACTOR_CACHE)
    except Exception:  # pragma: no cover - disque indisponible
        pass


# ===========================================================================
# Pretraitement statistique (winsorisation + z-score)
# ===========================================================================
def winsorize(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    """Winsorise une série aux quantiles ``lo`` / ``hi`` (NaN préservés)."""
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    if s.notna().sum() == 0:
        return s
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)


def _zscore_raw(x: pd.Series, w: Optional[pd.Series] = None) -> pd.Series:
    """Z-score d'un sous-échantillon (option : pondération par capitalisation)."""
    x = x.astype(float)
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=x.index)
    if w is not None:
        w = w.reindex(x.index).fillna(0.0).astype(float)
        if w.sum() > 0:
            mean = float(np.average(x, weights=w))
            var = float(np.average((x - mean) ** 2, weights=w))
            sd = np.sqrt(var)
        else:
            mean, sd = float(x.mean()), float(x.std(ddof=0))
    else:
        mean, sd = float(x.mean()), float(x.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - mean) / sd


def zscore(s: pd.Series, by_sector: bool = True, cap: Optional[pd.Series] = None,
           sector: Optional[pd.Series] = None) -> pd.Series:
    """Winsorise puis standardise (Z-score) en coupe transversale.

    - ``by_sector`` et ``sector`` fournis : Z intra-sectoriel GICS si le secteur
      compte >= 5 titres, sinon Z universel (fallback).
    - ``cap`` : pondération optionnelle par capitalisation (désactivée par défaut).
    """
    s = winsorize(s)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.notna()
    z_uni = _zscore_raw(s[valid], cap)

    if not by_sector or sector is None:
        out.loc[s[valid].index] = z_uni
        return out

    sec = pd.Series(sector).reindex(s.index)
    fallback_idx: list = []
    for g, sub in s[valid].groupby(sec[valid]):
        w = cap.reindex(sub.index) if cap is not None else None
        if len(sub) >= SECTOR_MIN_N:
            out.loc[sub.index] = _zscore_raw(sub, w)
        else:
            fallback_idx.extend(list(sub.index))
    if fallback_idx:
        out.loc[fallback_idx] = z_uni.reindex(fallback_idx)
    return out


def _blend(pairs: Sequence[tuple]) -> pd.Series:
    """Moyenne pondérée robuste aux NaN (renormalise les poids disponibles).

    ``pairs`` = séquence de ``(série, poids)``. Un composant manquant (NaN) est
    ignoré et son poids redistribué ; le résultat est NaN si aucun composant
    n'est disponible.
    """
    num: Optional[pd.Series] = None
    den: Optional[pd.Series] = None
    for s, w in pairs:
        s = pd.Series(s).astype(float)
        present = s.notna().astype(float)
        contrib = s.fillna(0.0) * float(w)
        wmask = present * float(w)
        num = contrib if num is None else num + contrib
        den = wmask if den is None else den + wmask
    if num is None:
        return pd.Series(dtype=float)
    return num / den.replace(0.0, np.nan)


# ===========================================================================
# TTM robuste (utilise par la purge bilan et la compatibilite)
# ===========================================================================
def ttm(x: Iterable[float]) -> float:
    """Annualisation robuste : ``mean(min(4, n) derniers trimestres non-NaN) * 4``."""
    x = pd.Series(x).dropna()
    if x.empty:
        return float("nan")
    return float(x.tail(4).mean() * 4.0)


def _ttm_panel(fund: pd.DataFrame, col: str, tickers: pd.Index) -> pd.Series:
    """TTM par titre sur un panel PIT déjà filtré (pub_date <= t)."""
    if col not in fund.columns:
        return pd.Series(np.nan, index=tickers, dtype=float)
    tmp = fund[["ticker", "period_end", col]].dropna(subset=[col])
    if tmp.empty:
        return pd.Series(np.nan, index=tickers, dtype=float)
    tmp = tmp.sort_values(["ticker", "period_end"])
    agg = tmp.groupby("ticker")[col].apply(lambda x: ttm(x))
    return agg.reindex(tickers)


# ===========================================================================
# Purge Junk Growth (Étape 1 : filtre bilan)
# ===========================================================================
def _series_or(df: pd.DataFrame, primary: str, fallback: Optional[pd.Series] = None,
               default: float = np.nan) -> pd.Series:
    if primary in df.columns:
        return df[primary].astype(float)
    if fallback is not None:
        return fallback.astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def purge_junk_growth(fund_asof: pd.DataFrame, sector: pd.Series,
                      mega_cap: Optional[pd.Series] = None,
                      decile: float = 0.10) -> pd.Series:
    """Indicateur booléen ``eligible`` (purge de la queue "Junk Growth").

    False si :
      - GP / Assets dans le décile inférieur intra-sectoriel, OU
      - (OCF < NetIncome ET Debt/Equity dans le décile supérieur intra-sectoriel).
    Les titres du noyau Mega-Cap (``mega_cap=True``) restent toujours éligibles.
    """
    df = fund_asof
    gpa = _series_or(
        df, "gpa",
        fallback=(df["gross_profit_q"] / df["total_assets"]) if
        {"gross_profit_q", "total_assets"}.issubset(df.columns) else None,
    )
    ocf = _series_or(df, "ocf_ttm", fallback=df.get("ocf_q") if "ocf_q" in df else None)
    ni = _series_or(df, "net_income_ttm",
                    fallback=df.get("net_income_q") if "net_income_q" in df else None)
    d2e = _series_or(
        df, "d2e",
        fallback=(df["total_debt"] / df["stockholders_equity"]) if
        {"total_debt", "stockholders_equity"}.issubset(df.columns) else None,
    )

    out = pd.Series(True, index=df.index, dtype=bool)
    sec = pd.Series(sector).reindex(df.index)

    for _, idx in sec.groupby(sec).groups.items():
        idx = pd.Index(idx)
        if len(idx) < PURGE_MIN_N:
            continue
        g = gpa.reindex(idx).dropna()
        if len(g) > 0:
            thr = g.quantile(decile)
            out.loc[g.index[g <= thr]] = False
        d = d2e.reindex(idx).dropna()
        if len(d) > 0:
            thr = d.quantile(1.0 - decile)
            cand = d.index[d >= thr]
            bad = (ocf.reindex(cand) < ni.reindex(cand)).fillna(False)
            out.loc[cand[bad.values]] = False

    if mega_cap is not None:
        out.loc[mega_cap.reindex(out.index).fillna(False)] = True
    return out


# ===========================================================================
# Coupe transversale a une date
# ===========================================================================
def compute_strategy_scores(date: pd.Timestamp, prices: pd.DataFrame,
                            fund_asof: pd.DataFrame, bench_returns: pd.Series,
                            universe: pd.DataFrame) -> pd.DataFrame:
    """Calcule les scores factoriels (3 piliers) pour une date ``t``.

    ``fund_asof`` = panel PIT (les lignes ``pub_date <= t`` sont retenues) ;
    ``universe`` = coupe par titre (``gics_sector`` et ``weight`` en fraction, ou
    ``weight_pct``). ``bench_returns`` est conservé pour la signature mais n'est
    plus utilisé (pas de momentum de résultats dans le modèle statique).
    """
    date = pd.Timestamp(date)
    if getattr(date, "tzinfo", None) is not None:
        date = date.tz_localize(None)
    date = date.normalize()

    # --- Univers / secteurs -------------------------------------------------
    if "ticker" in getattr(universe, "columns", []):
        uni = universe.set_index("ticker")
    else:
        uni = universe.copy()

    tickers = pd.Index(uni.index)
    if "gics_sector" in uni.columns:
        sector = uni["gics_sector"].reindex(tickers)
    else:
        sector = uni.get("prop_sector", pd.Series(index=tickers, dtype=object)).reindex(tickers)
    if "weight" in uni.columns:
        weight = uni["weight"].astype(float).reindex(tickers).fillna(0.0)
    elif "weight_pct" in uni.columns:
        weight = (uni["weight_pct"].astype(float) / 100.0).reindex(tickers).fillna(0.0)
    else:
        weight = pd.Series(0.0, index=tickers)
    mega_cap = weight >= MEGA_CAP_THRESHOLD
    is_fin = (sector == FINANCIALS_SECTOR).reindex(tickers).fillna(False)

    # --- Prix jusqu'à t -----------------------------------------------------
    px = prices.loc[prices.index <= date]
    px_cols = [t for t in tickers if t in px.columns]
    has_px = pd.Series(False, index=tickers)
    if px_cols:
        has_px.loc[px_cols] = (px[px_cols].notna().sum() >= 2).to_numpy()

    # --- Panel fondamental PIT ---------------------------------------------
    fund = fund_asof.copy()
    if "ticker" not in fund.columns:
        fund = fund.reset_index()
        if "ticker" not in fund.columns:
            fund = fund.rename(columns={fund.columns[0]: "ticker"})
    if "pub_date" in fund.columns:
        fund = fund[pd.to_datetime(fund["pub_date"]) <= date]
    fund = fund[fund["ticker"].isin(tickers)]
    if not fund.empty:
        fund = fund.drop_duplicates(subset=["ticker", "period_end"])

    def Z(metric: pd.Series) -> pd.Series:
        """Z-score intra-sectoriel d'une métrique brute."""
        return zscore(metric.reindex(tickers), by_sector=True, sector=sector)

    def Zagg(metric: pd.Series) -> pd.Series:
        """Re-standardisation intra-sectorielle d'un score agrégé (pilier)."""
        return zscore(metric.reindex(tickers), by_sector=True, sector=sector)

    # --- Agregats annuels PIT (N, N-1, N-2, N-3) ---------------------------
    rev = dl.annual_flow_points(fund, "revenue_q", tickers)
    oi = dl.annual_flow_points(fund, "operating_income_q", tickers)
    ni = dl.annual_flow_points(fund, "net_income_q", tickers)
    ocf = dl.annual_flow_points(fund, "ocf_q", tickers)
    capex = dl.annual_flow_points(fund, "capex_q", tickers)
    ta = dl.annual_stock_points(fund, "total_assets", tickers)
    eq = dl.annual_stock_points(fund, "stockholders_equity", tickers)
    shares = dl.annual_stock_points(fund, "shares", tickers)

    # =======================================================================
    # Pilier 1 : Acceleration fondamentale & expansion operationnelle (25 %)
    # =======================================================================
    sales_accel = (rev["N"] / rev["N1"] - 1.0) - (rev["N1"] / rev["N2"] - 1.0)
    margin_exp = oi["N"] / rev["N"] - oi["N1"] / rev["N1"]
    asset_eff = rev["N"] / ta["N"] - rev["N1"] / ta["N1"]
    roa_delta = ni["N"] / ta["N"] - ni["N1"] / ta["N1"]
    asset_eff = asset_eff.where(~is_fin, roa_delta)

    S_SalesAccel = Z(sales_accel)
    S_MarginExp = Z(margin_exp)
    S_AssetEff = Z(asset_eff)
    Z_Acceleration = Zagg(_blend([
        (S_SalesAccel, 1.0), (S_MarginExp, 1.0), (S_AssetEff, 1.0),
    ]))

    # =======================================================================
    # Pilier 2 : Momentum de prix ajuste du risque (45 %)
    # =======================================================================
    pxf = px.ffill()

    def _price_at(offset: int) -> pd.Series:
        if len(pxf) >= offset:
            return pxf.iloc[-offset].reindex(tickers)
        return pd.Series(np.nan, index=tickers)

    r6_1 = _price_at(MOM_1M) / _price_at(MOM_6M) - 1.0
    rets = px.pct_change(fill_method=None)
    vol_window = rets.tail(VOL_WINDOW_6M)
    n_obs = vol_window.notna().sum()
    sigma_6m = vol_window.std() * np.sqrt(252.0)
    sigma_6m = sigma_6m.where(n_obs >= MIN_VOL_OBS)
    ram = r6_1 / sigma_6m.reindex(tickers)

    last_price = (pxf.iloc[-1].reindex(tickers)
                  if len(pxf) else pd.Series(np.nan, index=tickers))
    sma200 = (pxf.tail(SMA_TREND).mean().reindex(tickers)
              if len(pxf) else pd.Series(np.nan, index=tickers))
    trend = last_price / sma200 - 1.0

    S_RAM = Z(ram)
    S_Trend = Z(trend)
    Z_Momentum = Zagg(_blend([(S_RAM, 0.65), (S_Trend, 0.35)]))

    # =======================================================================
    # Pilier 3 : Qualite intrinseque & valorisation GARP (30 %)
    # =======================================================================
    equity = eq["N"]
    roe = (ni["N"] / equity).where(equity > 0)

    ocf_n = ocf["N"]
    capex_n = capex["N"].abs()
    mktcap = shares["N"] * last_price
    fcf_yield = (ocf_n - capex_n) / mktcap.where(mktcap > 0)

    eps_ttm = ni["N"] / shares["N"]
    eps_3y = ni["N3"] / shares["N"]
    pe = last_price / eps_ttm
    # Croissance BPA 3 ans : CAGR (neutralise si non positive des deux cotes).
    valid_eps = (eps_ttm > 0) & (eps_3y > 0)
    eps_growth_3y = pd.Series(np.nan, index=tickers, dtype=float)
    eps_growth_3y.loc[valid_eps] = (
        (eps_ttm.loc[valid_eps] / eps_3y.loc[valid_eps])
        ** (1.0 / EPS_GROWTH_YEARS) - 1.0
    )
    garp = (eps_growth_3y / pe).where(pe > 0)

    S_ROE = Z(roe)
    S_FCFYield = Z(fcf_yield)
    S_GARP = Z(garp)
    Z_qual_std = _blend([(S_ROE, 0.40), (S_FCFYield, 0.35), (S_GARP, 0.25)])
    Z_qual_fin = _blend([(S_ROE, 0.60), (S_GARP, 0.40)])
    Z_Quality_GARP = Zagg(Z_qual_std.where(~is_fin, Z_qual_fin))

    # =======================================================================
    # Étape 1 (rappel) : purge Junk Growth + eligibilite
    # =======================================================================
    gp_ttm = _ttm_panel(fund, "gross_profit_q", tickers)
    ni_ttm = _ttm_panel(fund, "net_income_q", tickers)
    ocf_ttm = _ttm_panel(fund, "ocf_q", tickers)
    latest = (fund.sort_values("period_end").groupby("ticker").tail(1)
              .set_index("ticker").reindex(tickers)) if not fund.empty else pd.DataFrame(index=tickers)
    ta_latest = (latest["total_assets"].astype(float) if "total_assets" in latest.columns
                 else pd.Series(np.nan, index=tickers))
    eq_latest = (latest["stockholders_equity"].astype(float) if "stockholders_equity" in latest.columns
                 else pd.Series(np.nan, index=tickers))
    debt_latest = (latest["total_debt"].astype(float) if "total_debt" in latest.columns
                   else pd.Series(np.nan, index=tickers))
    gpa = gp_ttm / ta_latest
    d2e = (debt_latest / eq_latest).where(eq_latest > 0)

    metrics = pd.DataFrame({"gpa": gpa, "ocf_ttm": ocf_ttm,
                            "net_income_ttm": ni_ttm, "d2e": d2e}, index=tickers)
    eligible = purge_junk_growth(metrics, sector, mega_cap=mega_cap)
    has_score = (Z_Acceleration.notna() | Z_Momentum.notna() | Z_Quality_GARP.notna())
    eligible = eligible & has_score & has_px
    eligible = eligible.fillna(False).astype(bool)

    out = pd.DataFrame({
        "Z_Acceleration": Z_Acceleration,
        "Z_Momentum": Z_Momentum,
        "Z_Quality_GARP": Z_Quality_GARP,
        "S_SalesAccel": S_SalesAccel,
        "S_MarginExp": S_MarginExp,
        "S_AssetEff": S_AssetEff,
        "S_RAM": S_RAM,
        "S_Trend": S_Trend,
        "S_ROE": S_ROE,
        "S_FCFYield": S_FCFYield,
        "S_GARP": S_GARP,
        "eligible": eligible,
    }, index=tickers)
    # Recentrage transversal final (le fallback "univers" des secteurs < 5 titres
    # peut décaler légèrement la moyenne globale ; on la force à 0 en coupe).
    for col in FACTOR_COLUMNS:
        if col in ("eligible", "Z_Composite"):
            continue
        out[col] = out[col] - out[col].mean()
    out["Z_Composite"] = composite_weights_apply(
        out, config.DEFAULT_W_ACCEL, config.DEFAULT_W_MOMENTUM,
        config.DEFAULT_W_QUALITY,
    )
    return out[FACTOR_COLUMNS]


# ===========================================================================
# Combinaison & panel complet
# ===========================================================================
def composite_weights_apply(panel: pd.DataFrame, w_accel: float, w_mom: float,
                            w_qual: float) -> pd.Series:
    """Z_Composite = w_accel*Z_Acceleration + w_mom*Z_Momentum + w_qual*Z_Quality_GARP.

    Poids normalisés à 1. Un pilier manquant (NaN) est traité comme 0 : la somme
    pondérée reste centrée car chaque pilier l'est intra-sectoriellement.
    """
    total = float(w_accel) + float(w_mom) + float(w_qual)
    if total <= 0:
        w_accel, w_mom, w_qual = (config.DEFAULT_W_ACCEL, config.DEFAULT_W_MOMENTUM,
                                  config.DEFAULT_W_QUALITY)
        total = w_accel + w_mom + w_qual
    wa, wm, wq = w_accel / total, w_mom / total, w_qual / total
    z = (wa * panel["Z_Acceleration"].fillna(0.0)
         + wm * panel["Z_Momentum"].fillna(0.0)
         + wq * panel["Z_Quality_GARP"].fillna(0.0))
    return z.rename("Z_Composite")


def _weights_for_date(w_all, date: pd.Timestamp) -> pd.Series:
    """Extrait les poids PIT d'une date depuis la sortie de `get_index_weights`."""
    if isinstance(w_all, pd.Series):
        return w_all.astype(float)
    try:
        sub = w_all.xs(date, level="date")
    except KeyError:
        return pd.Series(dtype=float)
    if isinstance(sub, pd.DataFrame):
        sub = sub["weight"]
    sub = sub.astype(float)
    sub.name = "weight"
    return sub


def build_factor_panel(rebalance_dates: Optional[Sequence] = None) -> pd.DataFrame:
    """Panel factoriel MultiIndex (date, ticker) sur les dates de rebalancement.

    L'univers est **reconstruit a chaque date** : titres et poids PIT proviennent
    de :func:`data_loader.get_index_weights` (panier N-PORT du trimestre) et les
    secteurs de :func:`data_loader.get_sector_map`. Mise en cache dans
    ``data/cache/factor_panel.parquet``.
    """
    if rebalance_dates is None:
        dates = [pd.Timestamp(d) for d in config.REBALANCE_DATES]
    else:
        dates = sorted(pd.Timestamp(d) for d in rebalance_dates)

    cached = _read_cache()
    have: set = set()
    if cached is not None and not cached.empty:
        have = set(pd.to_datetime(cached.index.get_level_values(0)).normalize().unique())

    missing = [d for d in dates if d.normalize() not in have]
    if missing:
        prices = dl.get_prices()
        bench = dl.get_benchmark()
        bench_ret = bench.pct_change(fill_method=None)
        fund = dl.get_fundamentals()
        smap = dl.get_sector_map()
        w_all = dl.get_index_weights(dates)

        frames = []
        for d in missing:
            w_d = _weights_for_date(w_all, d)
            if w_d.empty:
                continue
            tickers = pd.Index(w_d.index)
            uni = pd.DataFrame(
                {
                    "gics_sector": smap.reindex(tickers).to_numpy(),
                    "weight": w_d.to_numpy(),
                },
                index=tickers,
            )
            sc = compute_strategy_scores(d, prices, fund, bench_ret, uni)
            sc = sc.copy()
            sc.index = pd.MultiIndex.from_product([[d], sc.index],
                                                  names=["date", "ticker"])
            frames.append(sc)
        if not frames:
            panel = cached if cached is not None else pd.DataFrame(columns=FACTOR_COLUMNS)
        else:
            new = pd.concat(frames)
            if cached is not None and not cached.empty:
                new = pd.concat([cached, new])
                new = new[~new.index.duplicated(keep="last")]
            panel = new.sort_index()
            _write_cache(panel)
    else:
        panel = cached

    mask = pd.to_datetime(panel.index.get_level_values(0)).normalize().isin(
        [d.normalize() for d in dates])
    result = panel[mask].copy()
    result.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(result.index.get_level_values(0)),
         result.index.get_level_values(1)],
        names=["date", "ticker"],
    )
    return result[FACTOR_COLUMNS]
