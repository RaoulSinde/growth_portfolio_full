"""Tests d'audit du G-Score de Mohanram (src/gscore.py) — pilier de recherche.

Le G-Score est un score entier 0-7 (critere de publicite G8 exclu) calcule en
coupe transversale PIT. La suite verifie :

    1. l'absence de look-ahead : invariance a la troncature / perturbation des
       prix et des fondamentaux posterieurs a `t`, et a l'injection d'une ligne
       publiee apres `t` ;
    2. le calcul de la mediane sectorielle (3 niveaux, coupe transversale, titres
       non publies exclus, jamais de mediane sur un secteur minuscule) ;
    3. la robustesse du score (bornes, NaN R&D -> 0, historique minimal G4/G5) ;
    4. l'absence de critere de publicite et l'absence d'acces `.info` ;
    5. les diagnostics de risque (correlation Qualite, dispersion).

Execution :  python -m pytest tests/test_gscore.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src import gscore

AUDIT_DATES = list(config.REBALANCE_DATES)
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
STRICT = 1e-10
Z_COLS = ["Z_GScore", "Z_GScore_Core", "Z_GScore_Ext"]
CRITERIA = ["G_Roa", "G_Cfo", "G_Accruals", "G_RoaVar", "G_SalesVar",
            "G_Rd", "G_Capex"]


def _as_ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _universe_for_date(date, index_weights, sector_map) -> pd.DataFrame:
    d = _as_ts(date)
    w = index_weights.xs(d, level="date")["weight"]
    tickers = pd.Index(w.index)
    return pd.DataFrame({"gics_sector": sector_map.reindex(tickers).to_numpy(),
                         "weight": w.to_numpy()}, index=tickers)


def _toy_universe(sectors: dict, weights=None) -> pd.DataFrame:
    tickers = list(sectors)
    w = weights if weights is not None else {t: 1.0 / len(tickers)
                                             for t in tickers}
    return pd.DataFrame({"gics_sector": [sectors[t] for t in tickers],
                         "weight": [w[t] for t in tickers]}, index=tickers)


def _toy_fund(rows) -> pd.DataFrame:
    """Construit un panel fondamental minimal (une ligne par dict)."""
    return pd.DataFrame(rows)


# ===========================================================================
# 1) Structure & bornes du score
# ===========================================================================
def test_gscore_panel_shape_and_columns(gscore_panel):
    assert isinstance(gscore_panel.index, pd.MultiIndex)
    assert gscore_panel.index.names == ["date", "ticker"]
    dates = pd.DatetimeIndex(gscore_panel.index.get_level_values("date").unique())
    assert len(dates) == len(AUDIT_DATES)
    assert list(gscore_panel.columns) == gscore.GSCORE_COLUMNS


def test_gscore_count_range_and_integers(gscore_panel):
    """G_Count in {0..7}, G_Core in {0..5}, G_Ext in {0..7} ; jamais de NaN->point."""
    for col, hi in (("G_Count", 7), ("G_Core", 5), ("G_Ext", 7)):
        v = gscore_panel[col].dropna()
        assert (v >= 0).all() and (v <= hi).all(), col
        assert np.allclose(v.to_numpy(), np.round(v.to_numpy()))  # entier


def test_gscore_criteria_are_binary(gscore_panel):
    for col in CRITERIA:
        v = gscore_panel[col].dropna()
        assert set(np.unique(v.to_numpy())).issubset({0.0, 1.0}), col


def test_gscore_score_normalised_consistency(gscore_panel):
    """G_Score = G_Count / G_N * 7 (avec G_N = nombre de criteres evaluables)."""
    sub = gscore_panel.dropna(subset=["G_Score", "G_N"])
    sub = sub[sub["G_N"] > 0]
    expect = sub["G_Count"] / sub["G_N"] * 7.0
    assert np.allclose(sub["G_Score"].to_numpy(), expect.to_numpy(), atol=1e-9)


def test_gscore_no_advertising_criterion(gscore_panel):
    """G8 (publicite) est explicitement absent (donnee indisponible)."""
    assert not any("Pub" in c or "Adv" in c for c in gscore_panel.columns)


def test_gscore_zscore_centered(gscore_panel):
    for col in Z_COLS:
        for d, sub in gscore_panel.groupby(level="date"):
            m = sub[col].dropna().mean()
            if np.isfinite(m):
                assert abs(m) < 1e-6, (col, d, m)


# ===========================================================================
# 2) ANTI LOOK-AHEAD
# ===========================================================================
def test_gscore_invariance_future_fundamentals(prices, fund, index_weights,
                                               sector_map):
    """Une ligne publiee apres `t` ne modifie aucun critere ni le score a `t`."""
    d = AUDIT_DATES[-1]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = gscore.compute_gscores(d, prices, fund, u)

    injected = fund.copy()
    row = injected.iloc[[0]].copy()
    row["period_end"] = d + pd.Timedelta(days=60)
    row["pub_date"] = d + pd.Timedelta(days=60)
    for c in injected.columns:
        if c not in ("ticker", "period_end", "pub_date") and \
                pd.api.types.is_numeric_dtype(injected[c]):
            row[c] = float(row[c].iloc[0]) * 100.0 + 1e6
    injected = pd.concat([injected, row], ignore_index=True)

    new = gscore.compute_gscores(d, prices, injected, u)
    for col in CRITERIA + Z_COLS + ["G_Count", "G_Score"]:
        assert np.allclose(ref[col].to_numpy(float), new[col].to_numpy(float),
                           atol=STRICT, rtol=0.0, equal_nan=True), col


def test_gscore_invariance_fund_truncation(prices, fund, index_weights,
                                           sector_map):
    d = AUDIT_DATES[5]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = gscore.compute_gscores(d, prices, fund, u)
    trunc = fund[pd.to_datetime(fund["pub_date"]) <= d]
    new = gscore.compute_gscores(d, prices, trunc, u)
    for col in Z_COLS:
        assert np.allclose(ref[col].to_numpy(float), new[col].to_numpy(float),
                           atol=STRICT, rtol=0.0, equal_nan=True), col


def test_gscore_independent_of_prices(prices, fund, index_weights, sector_map):
    """Perturber les prix (meme futurs) ne change AUCUN critere (score prix-agnostique)."""
    d = AUDIT_DATES[-1]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = gscore.compute_gscores(d, prices, fund, u)

    pert = prices.copy()
    pert.loc[pert.index > d] = pert.loc[pert.index > d] * 1e6 + 12345.0
    win = pert.index <= d
    pert.loc[win] = pert.loc[win] * 1.7 + 3.0
    new = gscore.compute_gscores(d, pert, fund, u)
    for col in CRITERIA + Z_COLS + ["G_Count"]:
        assert np.allclose(ref[col].to_numpy(float), new[col].to_numpy(float),
                           atol=STRICT, rtol=0.0, equal_nan=True), col


def test_gscore_pit_invariance_all_dates(prices, fund, index_weights, sector_map):
    """A chaque date : troncature prix <= t donne exactement le meme score."""
    for d in AUDIT_DATES:
        u = _universe_for_date(d, index_weights, sector_map)
        ref = gscore.compute_gscores(d, prices, fund, u)
        trunc = prices.loc[prices.index <= d]
        got = gscore.compute_gscores(d, trunc, fund, u)
        for col in Z_COLS:
            assert np.allclose(ref[col].to_numpy(float), got[col].to_numpy(float),
                               atol=STRICT, rtol=0.0, equal_nan=True), (d, col)


# ===========================================================================
# 3) Mediane sectorielle (3 niveaux)
# ===========================================================================
def test_sector_median_excludes_nan_and_unpublished():
    """La mediane ne porte que sur les titres VALIDES de la coupe a `t`."""
    tickers = pd.Index(list("abcdefgh"))
    metric = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan], index=tickers)
    sector = pd.Series(["IT"] * 8, index=tickers)
    med, level = gscore.sector_median_at_t(metric, sector, tickers, min_n=5)
    expected = float(metric.dropna().median())
    assert np.allclose(med.dropna().to_numpy(), expected)
    assert (level.reindex(metric.dropna().index) == "sector").all()
    assert level.loc["h"] == "none"  # metrique manquante -> aucune mediane


def test_sector_median_three_levels_fallback():
    """Secteurs < min_n -> super-secteur ; super-secteur < min_n -> univers."""
    # IT = 6 titres (>=5) -> niveau secteur.
    # Energy 2 + Materials 2 + Utilities 2 = 6 (>=5) -> super-secteur.
    # Health 2 (<5, Defensives) -> univers.
    sectors = {"i1": "Information Technology", "i2": "Information Technology",
               "i3": "Information Technology", "i4": "Information Technology",
               "i5": "Information Technology", "i6": "Information Technology",
               "e1": "Energy", "e2": "Energy", "m1": "Materials", "m2": "Materials",
               "u1": "Utilities", "u2": "Utilities",
               "h1": "Health Care", "h2": "Health Care"}
    tickers = pd.Index(list(sectors))
    metric = pd.Series(np.arange(1, len(tickers) + 1, dtype=float), index=tickers)
    sector = pd.Series(sectors)
    med, level = gscore.sector_median_at_t(metric, sector, tickers, min_n=5)

    assert (level.loc[["i1", "i6"]] == "sector").all()
    assert (level.loc[["e1", "m1", "u1"]] == "super").all()
    assert (level.loc[["h1", "h2"]] == "universe").all()
    # Super-secteur : mediane sur les 6 titres Energy/Materials/Utilities.
    sub = metric.loc[["e1", "e2", "m1", "m2", "u1", "u2"]]
    assert abs(med["e1"] - float(sub.median())) < 1e-12


def test_sector_median_never_on_singleton():
    """Un secteur a 1-2 titres n'a JAMAIS sa propre mediane."""
    sectors = {"a": "Utilities", "b": "Information Technology",
               "c": "Information Technology", "d": "Information Technology",
               "e": "Information Technology", "f": "Information Technology"}
    tickers = pd.Index(list(sectors))
    metric = pd.Series([100.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=tickers)
    sector = pd.Series(sectors)
    med, level = gscore.sector_median_at_t(metric, sector, tickers, min_n=5)
    assert level["a"] != "sector"


# ===========================================================================
# 4) Robustesse R&D / CAPEX / historique minimal
# ===========================================================================
def test_missing_rd_imputed_zero():
    """R&D non rapportee -> intensite 0 (convention Mohanram), pas NaN."""
    tickers = pd.Index([f"t{i}" for i in range(10)])
    sectors = {t: "Information Technology" for t in tickers}
    # t0 : R&D manquante (-> 0) ; t1 : R&D forte (200) ; t2..t9 : R&D mediane (50).
    rd_vals = [np.nan, 200.0] + [50.0] * 8
    rows = []
    for i, t in enumerate(tickers):
        rows.append({"ticker": t, "period_end": pd.Timestamp("2024-03-31"),
                     "pub_date": pd.Timestamp("2024-05-01"),
                     "total_assets": 1000.0, "net_income_q": 10.0,
                     "ocf_q": 20.0, "revenue_q": 100.0,
                     "research_dev_q": rd_vals[i], "capex_q": 5.0})
    fund = pd.DataFrame(rows)
    u = _toy_universe(sectors)
    d = pd.Timestamp("2025-01-01")
    prices = pd.DataFrame(index=[d], columns=tickers, data=100.0)
    res = gscore.compute_gscores(d, prices, fund, u)
    # R&D = 0 (manquante) -> critere evalue a 0, jamais NaN ni point gratuit.
    assert res.loc[tickers[0], "G_Rd"] == 0.0
    # R&D forte -> strictement au-dessus de la mediane sectorielle (0,05).
    assert res.loc[tickers[1], "G_Rd"] == 1.0


def test_quarterly_ratio_std_min_obs():
    """G4 (variance ROA) exige MIN_QUARTERS observations, sinon NaN."""
    tickers = pd.Index(["A"])
    rows = []
    for k in range(4):  # 4 trimestres seulement
        rows.append({"ticker": "A", "period_end": pd.Timestamp("2024-03-31")
                     + pd.DateOffset(months=3 * k),
                     "net_income_q": 10.0 + k, "total_assets": 1000.0})
    fund = pd.DataFrame(rows)
    s = gscore.quarterly_ratio_std(fund, "net_income_q", "total_assets", tickers)
    assert np.isnan(s["A"])
    # Avec assez d'observations -> valeur finie.
    rows = []
    for k in range(20):
        rows.append({"ticker": "A", "period_end": pd.Timestamp("2019-03-31")
                     + pd.DateOffset(months=3 * k),
                     "net_income_q": 10.0 + (k % 3), "total_assets": 1000.0})
    fund = pd.DataFrame(rows)
    s2 = gscore.quarterly_ratio_std(fund, "net_income_q", "total_assets", tickers)
    assert np.isfinite(s2["A"])


# ===========================================================================
# 5) Diagnostics
# ===========================================================================
def test_gscore_quality_correlation_diagnostic(gscore_panel, panel):
    corr = gscore.gscore_quality_correlation(gscore_panel, panel)
    assert not corr.empty
    assert {"n", "Pearson", "Spearman"}.issubset(corr.columns)
    assert corr["Pearson"].between(-1, 1).all()


def test_gscore_dispersion_diagnostic(gscore_panel):
    disp = gscore.gscore_dispersion(gscore_panel)
    assert not disp.empty
    for k in range(8):
        assert f"p_{k}" in disp.columns
    # Les parts somment a ~1 par date (hors NaN).
    share_cols = [f"p_{k}" for k in range(8)]
    sums = disp[share_cols].sum(axis=1)
    assert np.allclose(sums.dropna().to_numpy(), 1.0, atol=1e-6)


# ===========================================================================
# 6) Absence d'acces `.info` dans le nouveau module
# ===========================================================================
def test_no_info_access_in_gscore():
    tree = ast.parse((SRC_DIR / "gscore.py").read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "info":
            val = node.value
            if isinstance(val, ast.Name) and val.id in {"logger", "logging"}:
                continue
            hits.append((node.lineno, ast.unparse(node)))
    assert not hits, f"acces .info detecte dans gscore.py: {hits}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
