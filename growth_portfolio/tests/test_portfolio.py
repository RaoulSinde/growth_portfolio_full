"""Tests du constructeur de portefeuille, du backtest et des metriques.

Contexte : backtest **trimestriel Point-in-Time** de la **strategie unique**
« Growth Composite » (11 dates de rebalancement, 25/01/2024 -> 27/07/2026).
Le portefeuille est **sector-neutre** (cibles = poids GICS de l'indice EGUS du
trimestre) et ne depend d'aucune macro (`src.macro` supprime).

Execution :  python -m pytest tests/test_portfolio.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest as bt
from src import config
from src import metrics as mt
from src import portfolio as pf

from conftest import STRATEGY_NAME

TOL = 1e-6
SUM_TOL = 1e-8
N_DATES = 11

METRIC_KEYS = [
    "cum_return", "ann_return", "vol_ann", "sharpe", "sortino", "alpha_ann",
    "active_return_ann", "te_ann", "info_ratio", "beta", "max_dd",
    "active_share", "top10_weight",
    "turnover_avg", "n_effective", "up_capture", "down_capture",
    "tracking_error_ex_post",
]


def _iw_row(index_weights, date) -> pd.Series:
    """Vecteur de poids indiciels d'une date depuis le MultiIndex (date,ticker)."""
    sub = index_weights.xs(pd.Timestamp(date).normalize(), level="date")
    return sub["weight"] if isinstance(sub, pd.DataFrame) else sub


# ===========================================================================
# Construction du portefeuille (multi-dates)
# ===========================================================================
def test_all_rebalance_dates(composite_portfolios):
    """Un portefeuille par date de rebalancement (11 dates)."""
    assert len(composite_portfolios) == len(config.REBALANCE_DATES) == N_DATES
    assert set(composite_portfolios) == set(config.REBALANCE_DATES)


def test_weights_sum_one_and_nonnegative(composite_portfolios):
    """Somme des poids = 1 et poids >= 0, pour chaque date."""
    for d, w in composite_portfolios.items():
        assert abs(float(w.sum()) - 1.0) <= SUM_TOL, (d, float(w.sum()))
        assert (w >= -1e-12).all(), (d, float(w.min()))
        assert np.isfinite(w.to_numpy()).all(), d


def test_number_of_titles(composite_portfolios):
    """Nombre de titres dans [40, 50] (N par defaut = 40)."""
    for d, w in composite_portfolios.items():
        n = int((w > 0).sum())
        assert config.N_MIN <= n <= config.N_MAX, (d, n)
        assert n == config.DEFAULT_N, (d, n)


def test_n_titles_out_of_range_raises(panel, index_weights, index_sector_w):
    """``n_titles`` hors plage [40, 50] leve une erreur explicite."""
    for bad in (20, 80):
        with pytest.raises(ValueError):
            pf.build_all_portfolios(params={"n_titles": bad}, panel=panel,
                                    index_weights=index_weights,
                                    index_sector_w=index_sector_w)


def test_active_bounds_and_cap(composite_portfolios, index_weights):
    """Bornes actives [dw_min, dw_max] et plafond absolu w_max respectes."""
    for d, w in composite_portfolios.items():
        wi = _iw_row(index_weights, d).reindex(w.index).fillna(0.0)
        dw = (w - wi).to_numpy()
        assert (dw >= config.DEFAULT_DW_MIN - TOL).all(), (d, float(dw.min()))
        assert (dw <= config.DEFAULT_DW_MAX + TOL).all(), (d, float(dw.max()))
        assert (w <= config.DEFAULT_W_MAX + TOL).all(), (d, float(w.max()))


def test_core_present(composite_portfolios, panel, index_weights):
    """Toutes les Mega-Caps eligibles (w_idx >= 2.5%) sont presentes."""
    for d, w in composite_portfolios.items():
        panel_date = panel.loc[pd.Timestamp(d)]
        eligible = panel_date.index[panel_date["eligible"].astype(bool)]
        wi = _iw_row(index_weights, d)
        core = [t for t in eligible if float(wi.get(t, 0.0)) >= pf.CORE_THRESHOLD]
        assert core, d
        missing = [t for t in core if t not in w.index]
        assert not missing, (d, missing)


def test_sector_band(composite_portfolios, sector_map, index_sector_w):
    """Expositions sectorielles dans la bande (+/- 4%) autour de l'indice EGUS."""
    band = config.DEFAULT_SECTOR_BAND
    for d, w in composite_portfolios.items():
        sw = (w.groupby(sector_map.reindex(w.index)).sum()
              .reindex(config.GICS_SECTORS).fillna(0.0))
        iw = index_sector_w.loc[d].reindex(config.GICS_SECTORS).fillna(0.0)
        dev = float((sw - iw).abs().max())
        assert dev <= band + 1e-6, (d, dev)


def test_no_short_positions_and_finite(composite_portfolios):
    """Aucune vente a decouvert, poids finis."""
    for d, w in composite_portfolios.items():
        assert (w >= 0).all(), d
        assert np.isfinite(w.to_numpy()).all(), d


def test_no_macro_reference_in_portfolio_source():
    """``src/portfolio.py`` ne referencie plus le module macro (supprime)."""
    import pathlib
    src = pathlib.Path(pf.__file__).read_text(encoding="utf-8")
    assert "macro_mod" not in src
    assert "get_macro" not in src


# ===========================================================================
# Backtest
# ===========================================================================
def test_backtest_keys(backtest_result):
    """Benchmark EGUS + la strategie composite, chacun avec les 4 blocs."""
    assert "EGUS" in backtest_result
    assert STRATEGY_NAME in backtest_result, list(backtest_result)
    for name, blk in backtest_result.items():
        for k in ("nav", "returns", "turnover", "weights_daily"):
            assert k in blk, (name, k)


def test_backtest_benchmark_series_retrocompat(composite_portfolios, prices, bench):
    """Une Series seule produit la cle retro-compatible "Benchmark"."""
    res = bt.run_backtest({STRATEGY_NAME: composite_portfolios}, prices,
                          bench, tx_cost_bps=10.0)
    assert "Benchmark" in res


def test_backtest_nav(backtest_result, prices):
    """NAV finie, Base 100 au 1er jour ouvre, longueur coherente."""
    perf_start = pd.Timestamp(config.PERF_START)
    expected_days = prices.loc[perf_start:config.SIM_END]
    n_days = len(expected_days)
    for name in (STRATEGY_NAME, "EGUS"):
        nav = backtest_result[name]["nav"]
        assert len(nav) == n_days, (name, len(nav), n_days)
        assert np.isfinite(nav.to_numpy()).all(), name
        assert nav.index[0] >= perf_start, name
        assert nav.index[0] == expected_days.index[0].normalize(), name
        assert abs(float(nav.iloc[0]) - 100.0) < 1e-8, (name, float(nav.iloc[0]))
        assert (nav > 0).all(), name


def test_backtest_turnover_length(backtest_result):
    """Le turnover est enregistre a chacune des 11 dates de rebalancement."""
    turnover = backtest_result[STRATEGY_NAME]["turnover"]
    assert len(turnover) == len(config.REBALANCE_DATES) == N_DATES
    assert set(turnover.index) == set(config.REBALANCE_DATES)
    assert (turnover.to_numpy() >= 0).all()


def test_backtest_no_lookahead_weights(backtest_result):
    """Les poids quotidiens de la strategie somment a 1 apres le 1er jour."""
    for name, blk in backtest_result.items():
        wd = blk["weights_daily"]
        if wd is None:  # benchmark
            continue
        sums = wd.sum(axis=1)
        assert (sums > 0).all(), name
        assert np.allclose(sums.iloc[1:].to_numpy(), 1.0, atol=1e-8), name


# ===========================================================================
# Metriques
# ===========================================================================
def test_drawdown_series(backtest_result):
    nav = backtest_result[STRATEGY_NAME]["nav"]
    dd = mt.drawdown_series(nav)
    assert len(dd) == len(nav)
    assert (dd <= 1e-12).all()
    assert dd.min() >= -1.0


def test_performance_stats_keys_and_finite(backtest_result, index_weights):
    """Toutes les cles attendues sont presentes et finies (vs EGUS)."""
    bench_blk = backtest_result["EGUS"]
    strat = backtest_result[STRATEGY_NAME]
    stats = mt.performance_stats(
        strat["nav"], bench_blk["nav"],
        weights_daily=strat["weights_daily"],
        index_weights=index_weights,
        bench_returns=bench_blk["returns"],
        rf=0.0, turnover=strat["turnover"],
    )
    for k in METRIC_KEYS:
        assert k in stats, k
        v = stats[k]
        assert v is not None and np.isfinite(float(v)), (k, v)

    assert abs(stats["tracking_error_ex_post"] - stats["te_ann"]) < 1e-12
    assert stats["cum_return"] > -1.0
    assert stats["vol_ann"] > 0.0
    assert 0.0 < stats["top10_weight"] <= 0.90, stats["top10_weight"]
    assert 0.0 <= stats["active_share"] <= 1.0, stats["active_share"]
    assert stats["turnover_avg"] >= 0.0


def test_performance_stats_benchmark_vs_self(backtest_result):
    """Un portefeuille identique au benchmark a un alpha/beta ~ 0/1."""
    nav = backtest_result["EGUS"]["nav"]
    stats = mt.performance_stats(nav, nav, rf=0.0)
    assert abs(stats["alpha_ann"]) < 1e-9
    assert abs(stats["active_return_ann"]) < 1e-9
    assert abs(stats["te_ann"]) < 1e-9
    assert abs(stats["beta"] - 1.0) < 1e-6
    assert stats["up_capture"] == pytest.approx(1.0)
    assert stats["down_capture"] == pytest.approx(1.0)


def test_capture_compounds_daily_to_monthly():
    """Les captures sont composees mensuellement, pas sur les jours bruts."""
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01"])
    bench = pd.Series([0.10, -0.01, -0.05], index=index)
    port = pd.Series([0.10, -0.50, -0.025], index=index)

    up_capture, down_capture = mt._capture(port, bench)

    jan_bench = 1.10 * 0.99 - 1.0
    jan_port = 1.10 * 0.50 - 1.0
    assert up_capture == pytest.approx(jan_port / jan_bench)
    assert down_capture == pytest.approx(0.5)
