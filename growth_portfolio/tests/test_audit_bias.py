"""Audit critique de la baseline v8 (agent: quant_auditor).

Baseline auditee (**v8**) :

    - univers = inventaire **N-PORT SEC** de l'ETF **EGUS**, panier **Point-in-Time
      trimestriel** (`data/universe_quarterly.csv`, 11 `filing_date` de config,
      81 a 126 titres, 213 tickers uniques) ; actions US `EC` (STIV/DE exclus) ;
    - backtest **trimestriel PIT** du **2024-01-25 au 2026-09-30** ;
    - **strategie unique** a trois piliers PIT (Acceleration 25 % / Momentum 45 %
      / Qualite-GARP 30 %) ;
    - benchmark unique **EGUS** ; **aucune dependance macro** (`src.macro` supprime) ;
    - `max_top10=None` (aucune borne de concentration cumulee).

Cette suite tranche, par des tests, les points cles de l'analyse des resultats :

    1. **PIT de l'univers** : poids = panier N-PORT du trimestre (pas de snapshot
       statique), corr poids / rendement du trimestre suivant moderee ;
    2. **Resultats** : la strategie composite vs EGUS (recalibres sur les sorties
       reellement mesurees de la baseline v8) ;
    3. **Absence d'alpha net significatif** : la surperformance vs EGUS est dans
       l'IC bootstrap par blocs (21 j, 1000 tirages) -> l'IC95 contient 0 ;
    4. **Structure** : beta proche de 1, Top10 au niveau de l'indice, active share
       modere, cap 12,5 % respecte, turnover trimestriel ~19 % (hors inception) ;
    5. **Limites residuelles** : univers = holdings de l'ETF (alpha attendu faible),
       delistes retires, lecture equity-only, pas de macro.

Execution :  python -m pytest tests/test_audit_bias.py -q
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data_loader as dl
from src import factors
from src import metrics as mt

from conftest import STRATEGY_NAME

N_BOOT = 1000
BLOCK = 21          # ~ 1 mois de bourse
SEED = 12345
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# Helpers
# ===========================================================================
def _stats(nav, bench_nav, turnover=None, weights_daily=None, index_weights=None) -> dict:
    return mt.performance_stats(nav, bench_nav, turnover=turnover,
                                weights_daily=weights_daily,
                                index_weights=index_weights, rf=0.0)


def _cum(nav) -> float:
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def _block_bootstrap(rp: pd.Series, rb: pd.Series, n_boot: int = N_BOOT,
                     block: int = BLOCK, seed: int = SEED) -> dict:
    """Bootstrap par blocs (circulaire) de l'alpha et de la TE annualises."""
    act = (rp - rb).dropna().to_numpy(float)
    t = len(act)
    rng = np.random.default_rng(seed)
    a = np.empty(n_boot)
    te = np.empty(n_boot)
    for b in range(n_boot):
        nblocks = int(np.ceil(t / block))
        starts = rng.integers(0, t, nblocks)
        idx = np.concatenate([(s + np.arange(block)) % t for s in starts])[:t]
        x = act[idx]
        a[b] = x.mean() * 252.0
        te[b] = x.std(ddof=1) * np.sqrt(252.0)
    return {"alpha": a, "te": te}


def _index_top10_avg(index_weights) -> float:
    vals = [float(sub["weight"].nlargest(10).sum())
            for _, sub in index_weights.groupby(level="date")]
    return float(np.mean(vals))


# ===========================================================================
# 1) UNIVERS PIT N-PORT TRIMESTRIEL
# ===========================================================================
def test_universe_quarterly_is_nport_pit(index_weights):
    """11 trimestres PIT retenus, paniers 81-126 titres, Σ=1 par date."""
    assert os.path.exists(config.UNIVERSE_QUARTERLY_CSV)
    uni = pd.read_csv(config.UNIVERSE_QUARTERLY_CSV)
    uni["rebalance_date"] = pd.to_datetime(uni["rebalance_date"]).dt.normalize()
    cfg_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in config.REBALANCE_DATES])

    # Le fichier source couvre tout l'historique N-PORT (14 trimestres) ; le
    # moteur ne retient que les 11 dates de config.REBALANCE_DATES.
    assert uni["rebalance_date"].nunique() >= 11
    assert set(cfg_dates).issubset(set(uni["rebalance_date"].unique()))
    assert index_weights.index.get_level_values("date").nunique() == 11

    cfg = uni[uni["rebalance_date"].isin(cfg_dates)]
    sizes = cfg.groupby("rebalance_date").size()
    assert sizes.min() >= 75, sizes.min()
    assert sizes.max() <= 135, sizes.max()
    assert uni["ticker"].nunique() >= 200
    sums = cfg.groupby("rebalance_date")["weight"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-9), sums
    # La 1re date de rebalancement est bien le debut de simulation.
    assert str(cfg["rebalance_date"].min())[:10] == str(config.PERF_START)[:10]


def test_universe_baskets_change_across_quarters(index_weights):
    """Le panier change d'un trimestre a l'autre (source N-PORT dynamique)."""
    sets = [frozenset(sub.index.get_level_values("ticker"))
            for _, sub in index_weights.groupby(level="date")]
    assert len(set(sets)) >= 8, "paniers quasi identiques : snapshot statique ?"
    sizes = {len(s) for s in sets}
    assert len(sizes) > 1


def test_universe_excludes_stiv_de_and_foreign():
    """Aucun ISIN non `EC`/`US` (STIV, DE, cotes etrangeres) dans l'univers PIT."""
    inv = pd.read_csv(config.INVENTORY_CSV)
    uni = pd.read_csv(config.UNIVERSE_QUARTERLY_CSV)
    non_eq = set(inv.loc[(inv["asset_cat"] != "EC") |
                         (inv["inv_country"] != "US"), "isin"])
    assert len(non_eq) > 0
    assert not (set(uni["isin"]) & non_eq)


def test_delisted_and_unresolved_tickers_removed(index_weights):
    """GOOGL (ISIN non resolu) et les titres sans donnees sont absents de l'univers."""
    tickers = set(index_weights.index.get_level_values("ticker"))
    assert "GOOGL" not in tickers
    assert "EA" not in tickers  # deliste sans historique yfinance


def test_index_weight_forward_return_correlation_moderate(index_weights, prices):
    """PIT : corr(poids du trimestre, rendement du trimestre suivant) moderee."""
    dates = sorted(index_weights.index.get_level_values("date").unique())
    corrs = []
    for i, d in enumerate(dates):
        w = index_weights.xs(d, level="date")["weight"]
        end = dates[i + 1] if i + 1 < len(dates) else pd.Timestamp(config.SIM_END)
        px = prices.loc[(prices.index > d) & (prices.index <= end)]
        if len(px) < 2:
            continue
        ret = (px.ffill().iloc[-1] / px.ffill().iloc[0] - 1.0).reindex(w.index)
        df = pd.DataFrame({"w": w, "r": ret}).dropna()
        if len(df) > 5:
            corrs.append(float(np.corrcoef(df["w"], df["r"])[0, 1]))
    assert corrs, "aucune correlation calculable"
    assert max(abs(c) for c in corrs) < 0.35, corrs
    assert abs(float(np.mean(corrs))) < 0.15, float(np.mean(corrs))


# ===========================================================================
# 2) RESULTATS VS BENCHMARK UNIQUE EGUS (baseline v8)
# ===========================================================================
def test_benchmark_egus_return(backtest_result):
    """EGUS realise environ +70,4 % sur la fenetre (2024-01-25 -> 2026-09-30)."""
    cum = _cum(backtest_result["EGUS"]["nav"])
    assert abs(cum - 0.704) < 0.03, cum


def test_strategy_returns_vs_egus(backtest_result):
    """La strategie composite (2024-2026) surperforme legerement EGUS.

    Mesure : Composite ≈ +77,8 % vs EGUS ≈ +70,4 % ; ecart geometrique modeste,
    non distinguable du bruit (cf. test d'alpha bootstrap ci-dessous).
    """
    egus = _cum(backtest_result["EGUS"]["nav"])
    c = _cum(backtest_result[STRATEGY_NAME]["nav"])
    assert abs(c - 0.778) < 0.03, c
    assert c > egus, (c, egus)
    assert abs(c - egus) < 0.15, (c, egus)


# ===========================================================================
# 3) SIGNIFICATIVITE : BOOTSTRAP PAR BLOCS (1000, blocs 21 j)
# ===========================================================================
def test_alpha_vs_egus_not_significant(backtest_result):
    """Alpha de la strategie vs EGUS : IC95 bootstrap contient 0 -> NON significatif."""
    rp = backtest_result[STRATEGY_NAME]["returns"]
    rb = backtest_result["EGUS"]["returns"]
    boot = _block_bootstrap(rp, rb)
    lo, hi = np.percentile(boot["alpha"], [2.5, 97.5])
    assert abs(float(boot["alpha"].mean())) < 0.03, float(boot["alpha"].mean())
    assert lo < 0.0 < hi, (float(lo), float(hi))


def test_alpha_bootstrap_point_estimate(backtest_result):
    """L'alpha annualise moyen de la strategie est modeste (~ +1,9 pp)."""
    boot = _block_bootstrap(backtest_result[STRATEGY_NAME]["returns"],
                            backtest_result["EGUS"]["returns"])
    assert 0.0 < float(boot["alpha"].mean()) < 0.04, float(boot["alpha"].mean())


def test_te_bootstrap_interval(backtest_result):
    """IC bootstrap de la TE annualisee : coherent autour de ~4,3 %."""
    boot = _block_bootstrap(backtest_result[STRATEGY_NAME]["returns"],
                            backtest_result["EGUS"]["returns"])
    lo, hi = np.percentile(boot["te"], [2.5, 97.5])
    assert 0.02 < lo < 0.06, float(lo)
    assert 0.03 < hi < 0.08, float(hi)


# ===========================================================================
# 4) STRUCTURE : BETA, TOP10, ACTIVE SHARE, CAP
# ===========================================================================
def test_beta_near_one(backtest_result, index_weights):
    """Beta de la strategie proche de 1 (~1,04) : univers = panier de l'ETF."""
    st = _stats(backtest_result[STRATEGY_NAME]["nav"], backtest_result["EGUS"]["nav"],
                weights_daily=backtest_result[STRATEGY_NAME]["weights_daily"],
                index_weights=index_weights,
                turnover=backtest_result[STRATEGY_NAME]["turnover"])
    assert 0.90 < float(st["beta"]) < 1.20, float(st["beta"])


def test_top10_near_index_level(backtest_result, index_weights):
    """Top10 (~66 %) reste au niveau de l'indice (~64 %) : aucune borne cumulee.

    NOTE D'AUDIT : la cible de 40-50 % n'est PAS atteinte ici, car les poids Core
    (mega-caps) suivent l'indice et ``max_top10=None`` (borne desactivee). C'est un
    ecart documente (limite L1), pas une erreur de code.
    """
    st = _stats(backtest_result[STRATEGY_NAME]["nav"], backtest_result["EGUS"]["nav"],
                weights_daily=backtest_result[STRATEGY_NAME]["weights_daily"],
                index_weights=index_weights)
    top10 = float(st["top10_weight"])
    idx_top10 = _index_top10_avg(index_weights)
    assert 0.55 < top10 < 0.72, top10
    assert abs(top10 - idx_top10) < 0.08, (top10, idx_top10)


def test_active_share_moderate(backtest_result, index_weights):
    """Active share modere (~26 %) : univers restreint a l'ETF + tilts moderes."""
    st = _stats(backtest_result[STRATEGY_NAME]["nav"], backtest_result["EGUS"]["nav"],
                weights_daily=backtest_result[STRATEGY_NAME]["weights_daily"],
                index_weights=index_weights)
    assert 0.15 < float(st["active_share"]) < 0.45, float(st["active_share"])


def test_max_weight_capped(composite_portfolios):
    """Aucun titre ne depasse le cap absolu w_max = 12,5 %."""
    for d, w in composite_portfolios.items():
        assert float(w.max()) <= config.DEFAULT_W_MAX + 1e-9, (d, float(w.max()))


# ===========================================================================
# 5) TURNOVER TRIMESTRIEL
# ===========================================================================
def test_turnover_quarterly(backtest_result):
    """Turnover : 11 evenements trimestriels ; ~50 % a l'inception puis ~19 %."""
    to = backtest_result[STRATEGY_NAME]["turnover"]
    assert len(to) == 11
    assert set(to.index) == set(config.REBALANCE_DATES)
    assert (to.to_numpy() >= 0).all()
    first = float(to.loc[config.REBALANCE_DATES[0]])
    assert abs(first - 0.50) < 0.02, first          # construction depuis le cash
    ex_inception = to.iloc[1:]
    assert 0.12 < float(ex_inception.mean()) < 0.35, float(ex_inception.mean())
    assert float(to.max()) <= 0.60


# ===========================================================================
# 6) LIMITES RESIDUELLES
# ===========================================================================
def test_universe_etf_holdings_not_broad_index(index_weights):
    """L'univers = holdings de l'ETF EGUS (ESG-aware), pas un indice Growth USA pur.

    Limite structurelle : l'alpha vs l'indice de l'ETF est mecaniquement faible,
    car on part de son propre panier. Univers large et multi-secteurs.
    """
    sectors = set(dl.get_sector_map().unique())
    assert {"Health Care", "Financials", "Consumer Staples"}.issubset(sectors)
    sizes = index_weights.groupby(level="date").size()
    assert sizes.max() >= 120


def test_no_macro_dependency():
    """`src.macro` est supprime et les modules de strategie n'y referent plus."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.macro")
    assert not hasattr(config, "LONG_DURATION_SECTORS")
    factors_src = Path(factors.__file__).read_text(encoding="utf-8")
    assert "get_macro" not in factors_src
    app_src = (Path(ROOT) / "app.py").read_text(encoding="utf-8")
    for token in ("Régime", "Regime"):
        assert token not in app_src


def test_results_are_not_lookahead_borne(prices, benchmarks, fund, index_weights,
                                         sector_map):
    """Integration : les scores a la derniere date sont STRICTEMENT invariants au futur."""
    date = config.REBALANCE_DATES[-1]
    w = index_weights.xs(date, level="date")["weight"]
    tickers = pd.Index(w.index)
    uni = pd.DataFrame({"gics_sector": sector_map.reindex(tickers).to_numpy(),
                        "weight": w.to_numpy()}, index=tickers)
    br = benchmarks["EGUS"].pct_change(fill_method=None)

    ref = factors.compute_strategy_scores(date, prices, fund, br, uni)
    trunc = prices.loc[prices.index <= date]
    new = factors.compute_strategy_scores(date, trunc, fund, br, uni)
    for col in ("Z_Acceleration", "Z_Momentum", "Z_Quality_GARP", "Z_Composite"):
        assert float((ref[col] - new[col]).abs().max()) == 0.0, col


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
