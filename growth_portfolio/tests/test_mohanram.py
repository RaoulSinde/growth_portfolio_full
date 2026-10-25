"""Audit du moteur Mohanram Growth Tilted (``src/mohanram.py``).

Contrat : scores PIT stricts (invariance a la troncature future), neutralite
sectorielle parfaite apres plafonnement, multiplicateurs deterministes,
respiration Shumway sur les radiations, et absence de look-ahead.

Execution :  python -m pytest tests/test_mohanram.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import mohanram as mo

T0 = pd.Timestamp("2025-01-23")


@pytest.fixture(scope="module")
def ann() -> pd.DataFrame:
    return mo.build_annual_fundamentals()


@pytest.fixture(scope="module")
def res() -> dict:
    return mo.run_mohanram(T0)


def test_candidate_dates_have_full_window():
    """Chaque date candidate dispose d'une fenetre de 6 mois complete."""
    cands = mo.candidate_rebalance_dates()
    assert cands, "aucune date candidate"
    assert cands == sorted(cands)
    end_limit = pd.Timestamp(mo.config.PRICE_END)
    for d in cands:
        assert d + pd.DateOffset(months=mo.HORIZON_MONTHS) <= end_limit


def test_annual_extraction_has_per_field_pubdates(ann):
    """Le panel annuel expose les dates de publication par champ optionnel."""
    for col in ("period_end", "pub_date", "pub_rd", "pub_capex", "pub_adv"):
        assert col in ann.columns, f"colonne '{col}' absente"
    assert ann["ticker"].nunique() >= 200
    assert (ann["pub_date"] >= ann["period_end"]).all()


def test_gscores_are_integer_0_8(res):
    """G_SCORE est entier, dans [0, 8], pour les non financiers exploitables."""
    scores = res["scores"]
    ok = (~scores["is_financial"]) & scores["data_ok"]
    g = scores.loc[ok, "G_SCORE"].dropna()
    assert len(g) > 0
    assert np.allclose(g, g.round())
    assert g.min() >= 0 and g.max() <= 8


def test_multiplier_mapping_deterministic(res):
    """Le multiplicateur suit exactement la table MULTIPLIERS / torpedo."""
    p = res["portfolio"]
    for _, row in p.iterrows():
        g = row["G_SCORE"]
        m = row["multiplier"]
        if pd.isna(g):
            continue
        assert g in mo.MULTIPLIERS
        assert m == pytest.approx(mo.MULTIPLIERS[int(g)])


def test_sector_neutrality_after_capping(res):
    """Le plafonnement preserve la neutralite sectorielle (pas de saturation)."""
    delta = (res["sector_t0"] - res["sector_bench"]).abs()
    assert delta.max() < 1e-9, f"ecart sectoriel residuel : {delta.max()}"


def test_financials_stay_at_benchmark_weight(res):
    """Les financiers conservent strictement leur poids d'indice."""
    p = res["portfolio"]
    fin = p[p["sector"] == mo.FINANCIALS_SECTOR]
    assert not fin.empty
    assert np.allclose(fin["w_final"], fin["w_bench"])


def test_cap_constraint(res):
    """Aucun poids ne depasse min(cap_abs, w_EGUS + cap_rel)."""
    p = res["portfolio"]
    cap = np.minimum(res["cap_abs"], p["w_bench"] + res["cap_rel"])
    assert (p["w_final"] <= cap + 1e-9).all()


def test_weights_sum_to_one(res):
    assert res["w_final"].sum() == pytest.approx(1.0, abs=1e-9)
    assert (res["w_final"] >= -1e-12).all()


def test_held_names_have_prices_and_returns(res):
    """Tout titre detenu possede des cotations : rendement non NaN."""
    p = res["portfolio"]
    held = p[p["w_final"] > 0]
    assert not held.empty
    assert held["rendement_6m"].notna().all()
    assert res["n_held"] == len(held)


def test_no_lookahead_score_invariance(ann):
    """Le score est invariant a la troncature future des fondamentaux."""
    uni = mo.load_universe_at(T0)
    s_full, _ = mo.compute_gscores(ann, uni, T0)
    ann_pit = ann[ann["pub_date"] <= T0].copy()
    s_pit, _ = mo.compute_gscores(ann_pit, uni, T0)
    a = s_full["G_SCORE"].reindex(uni["ticker"])
    b = s_pit["G_SCORE"].reindex(uni["ticker"])
    assert a.equals(b) or np.allclose(a.fillna(-1), b.fillna(-1))


def test_determinism(res):
    """Deux executions identiques fournissent la meme NAV et les memes poids."""
    again = mo.run_mohanram(T0)
    assert np.allclose(res["nav"].values, again["nav"].values)
    assert np.allclose(res["w_final"].values, again["w_final"].values)


def test_level1_on_partial_base_data(ann):
    """Champ de base partiellement manquant (non financier) -> erreur critique."""
    uni = mo.load_universe_at(T0)
    sector = uni["gics_sector"]
    target = uni["ticker"][
        (sector != mo.FINANCIALS_SECTOR)
        & (uni["w_bench"] > 0.002)
    ].iloc[0]
    broken = ann.copy()
    broken.loc[broken["ticker"] == target, "cfo"] = np.nan
    with pytest.raises(mo.MohanramDataError):
        mo.compute_gscores(broken, uni, T0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
