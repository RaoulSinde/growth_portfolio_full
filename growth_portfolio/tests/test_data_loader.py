"""Tests d'integration du pipeline d'ingestion PIT trimestriel (src/data_loader).

Univers = inventaire N-PORT SEC consolide (`data/universe_quarterly.csv`) :
**11 dates** de rebalancement retenues (= `filing_date`, 2024-2026), poids reels
du trimestre (pas de derive). Ces tests sollicitent Yahoo Finance au premier
lancement puis utilisent le cache local `data/cache/`.

Execution :  python -m pytest tests/test_data_loader.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data_loader as dl

N_REBALANCE = 11
# Historique minimal de jours ouvres (SMA200 / momentum 6M et tendance).
MIN_HISTORY_DAYS = 600


def test_rebalance_dates():
    """1) 11 dates de rebalancement (= filing_date), triees et uniques."""
    dates = dl.get_rebalance_dates()
    assert len(dates) == N_REBALANCE, f"attendu {N_REBALANCE}, obtenu {len(dates)}"
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)
    assert dates == sorted(config.REBALANCE_DATES)
    assert dates[0] == pd.Timestamp("2024-01-25")
    assert dates[-1] == pd.Timestamp("2026-07-27")


def test_get_index_weights():
    """2) MultiIndex (date, ticker), Σ=1 par date, aucun poids hors panier."""
    w = dl.get_index_weights()
    assert isinstance(w, pd.DataFrame)
    assert list(w.columns) == ["weight"]
    assert isinstance(w.index, pd.MultiIndex)
    assert w.index.names == ["date", "ticker"]
    assert w.index.get_level_values("date").nunique() == N_REBALANCE

    sums = w.groupby(level="date")["weight"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-6), sums
    assert (w["weight"].values >= -1e-9).all()

    # Chaque (date, ticker) appartient bien au panier de ce trimestre.
    uni = dl.get_universe()
    pairs = set(map(tuple, w.index.to_frame(index=False).values))
    allowed = set(map(tuple, uni[["rebalance_date", "ticker"]].values))
    assert pairs.issubset(allowed)

    # Une date unique -> Series (SingleIndex) indexee par ticker.
    one = dl.get_index_weights([pd.Timestamp("2025-01-23")])
    assert isinstance(one, pd.Series)
    assert abs(one.sum() - 1.0) < 1e-6
    assert one.index.name == "ticker"
    assert len(one) > 0


def test_no_weight_before_filing():
    """3) Aucun poids avant la `filing_date` (regle Point-in-Time)."""
    w = dl.get_index_weights()
    dates = set(w.index.get_level_values("date").unique())
    assert dates.issubset(set(config.REBALANCE_DATES))

    # get_universe(asof) ne renvoie que le dernier trimestre connu a `asof`.
    asof = pd.Timestamp("2024-06-30")
    u = dl.get_universe(asof=asof)
    assert (u["rebalance_date"] <= asof).all()
    assert u["rebalance_date"].nunique() == 1
    last_known = max(d for d in config.REBALANCE_DATES if d <= asof)
    assert u["rebalance_date"].iloc[0] == last_known


def test_get_index_sector_weights():
    """4) Poids sectoriels : index=dates, colonnes=11 GICS, Σ=1 par date."""
    sec = dl.get_index_sector_weights()
    assert sec.index.nunique() == N_REBALANCE
    assert set(sec.columns) == set(config.GICS_SECTORS)
    assert np.allclose(sec.sum(axis=1).values, 1.0, atol=1e-6)


def test_get_sector_map():
    """5) Mapping secteur non vide, couverture maximale, libelles valides."""
    sm = dl.get_sector_map()
    assert len(sm) > 0
    assert sm.index.is_unique
    valid = set(config.GICS_SECTORS) | {"Unknown"}
    assert set(sm.unique()).issubset(valid), set(sm.unique()) - valid
    # Couverture : au moins 99% des tickers avec un secteur GICS connu.
    known = (sm != "Unknown").mean()
    assert known >= 0.99, f"couverture secteur insuffisante : {known:.1%}"


def test_get_prices():
    """6) Prix : union des tickers, index trie, couverture 2023-01 -> 2026-09."""
    px = dl.get_prices()
    assert isinstance(px.index, pd.DatetimeIndex)
    assert px.index.is_monotonic_increasing
    assert len(px) >= MIN_HISTORY_DAYS, f"seulement {len(px)} jours"
    assert px.index.min() >= pd.Timestamp(config.PRICE_START)
    assert px.index.max() <= pd.Timestamp(config.PRICE_END)
    assert px.index.min() <= pd.Timestamp("2023-02-01")
    assert px.index.max() >= pd.Timestamp("2026-08-01")

    # Colonnes = sous-ensemble des tickers de l'univers (tickers sans donnees exclus).
    uni = set(dl.get_universe()["ticker"].unique())
    assert set(px.columns).issubset(uni)
    assert len(px.columns) >= int(0.95 * len(uni))

    # Au moins 90% des titres avec un historique suffisant (3 ans).
    counts = px.notna().sum()
    frac_ok = (counts >= MIN_HISTORY_DAYS).mean()
    assert frac_ok >= 0.90, f"seulement {frac_ok:.1%} des titres avec >= {MIN_HISTORY_DAYS} jours"


def test_get_benchmarks():
    """7) Benchmark unique EGUS, serie non vide."""
    benchs = dl.get_benchmarks()
    assert set(benchs.keys()) == set(config.BENCHMARKS.keys()) == {"EGUS"}
    egus = benchs["EGUS"]
    assert len(egus) >= 900
    assert egus.notna().sum() >= 900
    assert egus.index.min() <= pd.Timestamp("2023-02-10")


def test_get_fundamentals_pit():
    """8) Panel PIT : pub_date >= period_end ET regle asof pub_date <= asof."""
    fund = dl.get_fundamentals()
    assert len(fund) > 0
    assert fund["ticker"].nunique() >= 200

    pe = pd.to_datetime(fund["period_end"])
    pub = pd.to_datetime(fund["pub_date"])
    assert (pub >= pe).all(), (pub - pe).min()

    # Point-in-Time : fundamentals_asof ne retient que pub_date <= asof.
    asof = pd.Timestamp("2026-03-31")
    snap = dl.fundamentals_asof(fund, asof)
    assert (pd.to_datetime(snap["pub_date"]) <= asof).all()

    # Aucune donnee issue de ticker.info : colonnes controlees uniquement.
    assert "ticker" in fund.columns and "period_end" in fund.columns


def test_get_earnings_dates():
    """9) Dates de publication presentes par ticker."""
    ed = dl.get_earnings_dates()
    assert set(["ticker", "earnings_date", "surprise_pct"]).issubset(ed.columns)
    assert ed["ticker"].nunique() >= 200
    assert pd.to_datetime(ed["earnings_date"]).notna().all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
