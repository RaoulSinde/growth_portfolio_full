"""Statistiques de performance, de risque et de structure du portefeuille.

Toutes les annualisations utilisent 252 jours de bourse. Les metriques relatives
(alpha, tracking error, ratio d'information, beta, capture) sont calculees sur
les rendements quotidiens du portefeuille et du benchmark. Les metriques de
structure (active share, top 10, nombre effectif de titres) sont moyennees sur
les snapshots de rebalancement.

SECURITE : aucune donnee issue de l'attribut ``info`` d'un Ticker.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["drawdown_series", "performance_stats", "TRADING_DAYS"]

TRADING_DAYS = 252


# ===========================================================================
# Helpers
# ===========================================================================
def _clean_returns(series: pd.Series) -> pd.Series:
    """Rendements quotidiens propres (index trie, NaN supprimes)."""
    s = pd.Series(series).astype(float).sort_index()
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def _ann_return(nav: pd.Series) -> float:
    """Rendement annualise a partir d'une NAV."""
    n = len(nav.dropna())
    if n < 2:
        return np.nan
    total = float(nav.iloc[-1] / nav.iloc[0])
    if total <= 0:
        return np.nan
    return total ** (TRADING_DAYS / (n - 1)) - 1.0


def _downside_deviation(returns: pd.Series) -> float:
    """Ecart-type de baisse annualise (semideviation sur les rendements < 0)."""
    neg = returns[returns < 0.0]
    if len(neg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(neg.to_numpy()))) * np.sqrt(TRADING_DAYS))


def _capture(returns: pd.Series, bench_returns: pd.Series) -> tuple[float, float]:
    """Up / down capture ratios composes sur les mois haussiers/baissiers.

    Les rendements quotidiens sont d'abord composes au pas mensuel. Sans ce
    changement d'echelle, les ratios journaliers sur plusieurs annees sont
    illisibles : le compartiment baissier compose proche de -100 % transforme
    un ecart de rendement minime en ecart de richesse majeur.
    """
    frame = pd.concat(
        [returns.rename("p"), bench_returns.rename("b")], axis=1
    ).dropna()
    if frame.empty:
        return np.nan, np.nan
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = (1.0 + frame).resample("ME").prod(min_count=1) - 1.0
        frame = frame.dropna()
    if frame.empty:
        return np.nan, np.nan

    up = frame["b"] > 0
    down = frame["b"] < 0

    def _ratio(mask: pd.Series) -> float:
        sub = frame.loc[mask]
        if sub.empty:
            return np.nan
        comp_p = float((1.0 + sub["p"]).prod() - 1.0)
        comp_b = float((1.0 + sub["b"]).prod() - 1.0)
        if abs(comp_b) < 1e-12:
            return np.nan
        return comp_p / comp_b

    return _ratio(up), _ratio(down)


def _asof_row(df: pd.DataFrame, date: pd.Timestamp,
              fallback_first: bool = False) -> Optional[pd.Series]:
    """Derniere ligne d'un DataFrame (index de dates) a une date donnee.

    ``fallback_first=True`` : si la date demandee precede l'index (cas ou le
    rebalancement initial est anterieur a la fenetre de performance), on retourne
    la premiere ligne disponible plutot que ``None``. Cela permet de calculer les
    metriques de structure (top10, active share, n effectif) a partir de la
    composition effectivement detenue des le premier jour.
    """
    if df is None or len(df) == 0:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    pos = int(idx.searchsorted(pd.Timestamp(date), side="right")) - 1
    if pos < 0:
        if not fallback_first:
            return None
        pos = 0
    row = df.iloc[pos]
    if isinstance(row, pd.DataFrame):  # pragma: no cover - index duplique
        row = row.iloc[-1]
    return row.astype(float)


def _index_weights_dates(index_weights) -> list:
    """Dates de rebalancement disponibles dans ``index_weights`` (MultiIndex ok)."""
    if index_weights is None or len(index_weights) == 0:
        return []
    if isinstance(index_weights.index, pd.MultiIndex):
        raw = index_weights.index.get_level_values(0)
    else:
        raw = index_weights.index
    dates = pd.Index(pd.to_datetime(raw)).normalize().unique()
    return sorted(pd.DatetimeIndex(dates))


def _index_weight_row(index_weights, date: pd.Timestamp) -> Optional[pd.Series]:
    """Poids indiciels d'une date (``Series`` index=ticker), quel que soit le format.

    Gere les formats produits par :func:`data_loader.get_index_weights` :
    MultiIndex ``(date, ticker)`` (multi-dates), ``Series`` indexee par ticker
    (date unique) ou ``DataFrame`` index=dates / colonnes=tickers (ancien format).
    """
    if index_weights is None or len(index_weights) == 0:
        return None
    d = pd.Timestamp(date).normalize()

    if isinstance(index_weights, pd.Series):
        return index_weights.astype(float)

    if isinstance(index_weights.index, pd.MultiIndex):
        for level in ("date", 0):
            try:
                sub = index_weights.xs(d, level=level)
            except (KeyError, TypeError):
                continue
            if isinstance(sub, pd.DataFrame):
                sub = sub["weight"] if "weight" in sub.columns else sub.iloc[:, 0]
            return sub.astype(float)
        return None

    idx = pd.DatetimeIndex(pd.to_datetime(index_weights.index)).normalize()
    if d not in idx:
        return None
    row = index_weights.loc[d]
    if isinstance(row, pd.DataFrame):
        row = row["weight"] if "weight" in row.columns else row.iloc[:, 0]
    return row.astype(float)


# ===========================================================================
# Drawdowns
# ===========================================================================
def drawdown_series(nav: pd.Series) -> pd.Series:
    """Serie temporelle des drawdowns (``nav / cummax - 1``)."""
    s = pd.Series(nav).astype(float).sort_index()
    if s.empty:
        return s.rename("drawdown")
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return dd.rename("drawdown")


# ===========================================================================
# Statistiques de performance
# ===========================================================================
def performance_stats(nav: pd.Series,
                      bench_nav: pd.Series,
                      weights_daily: Optional[pd.DataFrame] = None,
                      index_weights: Optional[pd.DataFrame] = None,
                      bench_returns: Optional[pd.Series] = None,
                      rf: float = 0.0,
                      turnover: Optional[pd.Series] = None) -> dict:
    """Calcule le bloc complet de statistiques de performance et de risque.

    ``nav`` / ``bench_nav`` : series de NAV. ``weights_daily`` : poids quotidiens
    (derive buy&hold). ``index_weights`` : poids indiciels aux dates de
    rebalancement (MultiIndex ``(date, ticker)`` ou ``Series``), utilises pour
    l'active share et le Top10. ``bench_returns`` : rendements du benchmark
    (sinon derives de ``bench_nav``). ``turnover`` : turnover par rebalancement
    (trimestriel) ; ``turnover_avg`` en est la moyenne.

    Les metriques de structure (``active_share``, ``top10_weight``,
    ``n_effective``) sont **moyennees** sur les snapshots de composition aux dates
    de rebalancement (composition effective ``_asof_row`` a chaque date, avec
    repli sur la premiere composition detenue si une date precede la fenetre).
    """
    nav = pd.Series(nav).astype(float).sort_index()
    bench_nav = pd.Series(bench_nav).astype(float).sort_index()
    common = nav.index.intersection(bench_nav.index)
    nav = nav.reindex(common)
    bench_nav = bench_nav.reindex(common)

    ret_p = _clean_returns(nav.pct_change())
    if bench_returns is not None:
        ret_b = _clean_returns(pd.Series(bench_returns))
    else:
        ret_b = _clean_returns(bench_nav.pct_change())
    aligned = pd.concat([ret_p.rename("p"), ret_b.rename("b")], axis=1).dropna()
    rp = aligned["p"]
    rb = aligned["b"]

    # --- Rendements de base -------------------------------------------------
    cum_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) > 1 else np.nan
    ann_return = _ann_return(nav)

    vol_ann = float(rp.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rp) > 1 else np.nan
    mean_ann = float(rp.mean() * TRADING_DAYS) if len(rp) else np.nan
    excess = mean_ann - float(rf)
    sharpe = excess / vol_ann if vol_ann and np.isfinite(vol_ann) and vol_ann > 0 else np.nan

    dstd = _downside_deviation(rp)
    sortino = excess / dstd if dstd and np.isfinite(dstd) and dstd > 0 else np.nan

    # --- Metriques relatives ------------------------------------------------
    if len(aligned) > 1:
        active = (rp - rb)
        active_return_ann = float(active.mean() * TRADING_DAYS)
        te_ann = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        info_ratio = active_return_ann / te_ann if te_ann and te_ann > 0 else np.nan
        var_b = float(rb.var(ddof=1))
        beta = float(rp.cov(rb) / var_b) if var_b > 0 else np.nan
        rf_daily = float(rf) / TRADING_DAYS
        alpha_ann = float(
            ((rp.mean() - rf_daily) - beta * (rb.mean() - rf_daily))
            * TRADING_DAYS
        )
    else:
        alpha_ann = active_return_ann = te_ann = info_ratio = beta = np.nan

    up_capture, down_capture = _capture(rp, rb)

    # --- Risque de queue ----------------------------------------------------
    dd = drawdown_series(nav)
    max_dd = float(dd.min()) if len(dd) else np.nan

    # --- Structure ----------------------------------------------------------
    snapshots = _weight_snapshots(weights_daily, index_weights)
    if snapshots:
        top10 = float(np.mean([float(np.sort(r.to_numpy())[::-1][:10].sum())
                               for r in snapshots]))
        n_eff = float(np.mean([1.0 / float((r.to_numpy() ** 2).sum())
                               for r in snapshots
                               if float((r.to_numpy() ** 2).sum()) > 0]))
    else:
        top10 = np.nan
        n_eff = np.nan

    active_share = _active_share(weights_daily, index_weights)

    turnover_avg = np.nan
    if turnover is not None and len(pd.Series(turnover)) > 0:
        turnover_avg = float(pd.Series(turnover).astype(float).mean())

    return {
        "cum_return": cum_return,
        "ann_return": ann_return,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "alpha_ann": alpha_ann,
        "active_return_ann": active_return_ann,
        "te_ann": te_ann,
        "info_ratio": info_ratio,
        "beta": beta,
        "max_dd": max_dd,
        "active_share": active_share,
        "top10_weight": top10,
        "turnover_avg": turnover_avg,
        "n_effective": n_eff,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "tracking_error_ex_post": te_ann,
    }


# ===========================================================================
# Helpers de structure
# ===========================================================================
def _weight_snapshots(weights_daily: Optional[pd.DataFrame],
                      index_weights: Optional[pd.DataFrame]) -> list:
    """Snapshots de poids aux dates de rebalancement (ou toutes les lignes).

    On echantillonne la composition **effectivement detenue** a chaque date de
    rebalancement de l'indice (``index_weights``, MultiIndex ``(date, ticker)``),
    avec repli sur le premier jour detenu si la date precede la fenetre de NAV.
    """
    if weights_daily is None or len(weights_daily) == 0:
        return []
    reb_dates = _index_weights_dates(index_weights)
    if reb_dates:
        rows = []
        for d in reb_dates:
            row = _asof_row(weights_daily, d, fallback_first=True)
            if row is not None:
                rows.append(row)
        if rows:
            return rows
        # Repli : composition du premier jour detenu.
        return [weights_daily.iloc[0].astype(float)]
    return [weights_daily.iloc[i].astype(float) for i in range(len(weights_daily))]


def _active_share(weights_daily: Optional[pd.DataFrame],
                  index_weights: Optional[pd.DataFrame]) -> float:
    """Active share moyen (``0.5 * sum |w_p - w_b|``) sur les rebalancements."""
    if weights_daily is None or index_weights is None or len(index_weights) == 0:
        return np.nan
    vals = []
    for d in _index_weights_dates(index_weights):
        wp = _asof_row(weights_daily, d, fallback_first=True)
        wb = _index_weight_row(index_weights, d)
        if wp is None or wb is None:
            continue
        wb = pd.Series(wb).astype(float)
        cols = pd.Index(wp.index).union(pd.Index(wb.index))
        diff = (wp.reindex(cols).fillna(0.0) - wb.reindex(cols).fillna(0.0))
        vals.append(float(0.5 * np.abs(diff).sum()))
    return float(np.mean(vals)) if vals else np.nan
