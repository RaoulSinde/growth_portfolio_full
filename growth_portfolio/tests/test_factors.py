"""Tests du module factoriel (src/factors.py) multi-dates PIT trimestriel.

Strategie **unique a trois piliers statiques PIT** :

    Z_Composite = w_accel*Z_Acceleration + w_mom*Z_Momentum + w_qual*Z_Quality_GARP

Univers = panier N-PORT de chaque trimestre (`get_index_weights`) ; **11 dates**
de rebalancement (= `filing_date`, 2024-2026). Le module macro a ete supprime
(plus aucune dependance `src.macro`).

Execution :  python -m pytest tests/test_factors.py -q
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from src import config, factors, data_loader as dl

N_DATES = 11

Z_COLUMNS = ["Z_Acceleration", "Z_Momentum", "Z_Quality_GARP"]
S_COLUMNS = ["S_SalesAccel", "S_MarginExp", "S_AssetEff",
             "S_RAM", "S_Trend", "S_ROE", "S_FCFYield", "S_GARP"]


def _date_means(panel: pd.DataFrame, col: str) -> pd.Series:
    """Moyenne (NaN ignore) par date pour une colonne."""
    return panel.groupby(level="date")[col].apply(
        lambda s: s.dropna().mean() if s.notna().any() else np.nan)


# ===========================================================================
# Structure du panel
# ===========================================================================
def test_panel_shape_and_columns(panel):
    """MultiIndex (date, ticker), 11 dates = config.REBALANCE_DATES, colonnes."""
    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["date", "ticker"]

    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique())
    assert len(dates) == N_DATES == len(config.REBALANCE_DATES), list(dates)
    assert sorted(dates) == sorted(config.REBALANCE_DATES)

    for col in factors.FACTOR_COLUMNS:
        assert col in panel.columns, col
    assert list(panel.columns) == factors.FACTOR_COLUMNS


def test_panel_universe_matches_pit_basket(panel):
    """Chaque date contient exactement le panier/poids PIT de ce trimestre."""
    w = dl.get_index_weights()
    for d, sub in panel.groupby(level="date"):
        expected = set(w.xs(d, level="date")["weight"].index)
        got = set(sub.index.get_level_values("ticker"))
        assert got == expected, (d, len(expected), len(got))
        assert len(got) > 0


def test_panel_missing_price_tickers_ineligible(panel):
    """Un titre du panier sans historique de prix doit finir ineligible."""
    prices = dl.get_prices()
    no_price = [t for t in panel.index.get_level_values("ticker").unique()
                if t not in prices.columns]
    assert no_price, "aucun titre sans prix : test non discriminant"
    for t in no_price:
        sub = panel.xs(t, level="ticker")
        assert not bool(sub["eligible"].any()), t
        assert sub["Z_Composite"].isna().all() or sub["eligible"].sum() == 0


# ===========================================================================
# Pretraitement : Z-scores finis et centres par date
# ===========================================================================
def test_zscore_columns_zero_mean_finite(panel):
    """Chaque Z_* est fini la ou les donnees existent, moyenne ~0 par date."""
    for col in Z_COLUMNS:
        for d, sub in panel.groupby(level="date"):
            v = sub[col].dropna()
            if len(v) == 0:
                continue  # pas de donnees ce trimestre
            assert np.isfinite(v.to_numpy()).all(), (col, d)
            assert abs(v.mean()) < 0.05, (col, d, float(v.mean()))
    # Z_Composite doit etre disponible a chaque date (combinaison robuste).
    for d, sub in panel.groupby(level="date"):
        assert sub["Z_Composite"].notna().any(), d


def test_subpillars_zero_mean_finite(panel):
    """Les sous-piliers S_* sont centres (~0) et finis la ou ils existent."""
    for col in S_COLUMNS:
        means = _date_means(panel, col).dropna()
        assert len(means) > 0, f"{col}: aucune valeur"
        assert (means.abs() < 0.05).all(), (col, means.abs().max())
        v = panel[col].dropna()
        assert np.isfinite(v.to_numpy()).all(), col


def test_momentum_short_history_no_crash(panel):
    """Le momentum 6M-1M / vol 6M reste calculable malgre les IPO recentes."""
    v = panel["Z_Momentum"].dropna()
    assert len(v) >= 500, len(v)
    assert np.isfinite(v.to_numpy()).all()


# ===========================================================================
# Eligibility (purge Junk Growth par date)
# ===========================================================================
def test_eligible_count(panel):
    """Le nombre d'eligibles par date reste plausible (~75 a ~115)."""
    counts = panel.groupby(level="date")["eligible"].sum()
    n_rows = panel.groupby(level="date").size()
    assert panel["eligible"].dtype == bool
    assert (counts >= 50).all(), counts.to_dict()
    assert (counts <= n_rows).all(), counts.to_dict()
    # Borne souple : le panier maximal compte 126 titres (2024-01-25).
    assert (counts <= 145).all(), counts.to_dict()


# ===========================================================================
# Combinaison lineaire des piliers
# ===========================================================================
def test_composite_weights_apply():
    """La combinaison normalise les poids et respecte la formule lineaire."""
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-23")], ["A", "B", "C"]], names=["date", "ticker"])
    p = pd.DataFrame({
        "Z_Acceleration": [1.0, 0.0, -1.0],
        "Z_Momentum": [0.0, 1.0, 0.0],
        "Z_Quality_GARP": [-1.0, 0.0, 1.0],
    }, index=idx)

    # Poids egaux (deja normalises) -> moyenne simple des trois piliers.
    z = factors.composite_weights_apply(p, 1.0, 1.0, 1.0)
    expected = (p["Z_Acceleration"] + p["Z_Momentum"] + p["Z_Quality_GARP"]) / 3.0
    assert np.allclose(z.to_numpy(), expected.to_numpy())
    assert z.name == "Z_Composite"

    # Poids quelconques -> normalises a 1 avant application.
    wa, wm, wq = 2.0, 4.0, 2.0
    z2 = factors.composite_weights_apply(p, wa, wm, wq)
    tot = wa + wm + wq
    exp2 = (wa * p["Z_Acceleration"] + wm * p["Z_Momentum"]
            + wq * p["Z_Quality_GARP"]) / tot
    assert np.allclose(z2.to_numpy(), exp2.to_numpy())
    assert z2.name == "Z_Composite"


# ===========================================================================
# Suppression de la macro
# ===========================================================================
def test_macro_module_removed():
    """Le module src.macro a ete supprime et factors ne l'importe plus."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.macro")
    assert not hasattr(config, "LONG_DURATION_SECTORS")
    assert not hasattr(config, "DEFENSIVE_GROWTH_SECTORS")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
