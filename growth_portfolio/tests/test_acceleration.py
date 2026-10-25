"""Tests de la strategie d'acceleration MSCI et de l'univers PIT."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import acceleration
from src import msci_universe as universe

SECTORS = ["Information Technology", "Health Care", "Financials"]


def _synthetic_fund(n_tickers: int = 60, n_quarters: int = 12) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2018-01-01")
    for i in range(n_tickers):
        ticker = f"T{i:02d}"
        for q in range(n_quarters):
            period_end = start + pd.DateOffset(months=3 * q)
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "pub_date": period_end + pd.Timedelta(days=45),
                    "revenue_q": 100.0 * (1.0 + 0.03 * i) * (1.0 + 0.04 * q),
                    "operating_income_q": 10.0 * (1.0 + 0.02 * i) * (1.0 + 0.06 * q),
                    "net_income_q": 8.0 * (1.0 + 0.01 * i) * (1.0 + 0.05 * q),
                    "total_assets": 200.0 * (1.0 + 0.01 * i) * (1.0 + 0.01 * q),
                }
            )
    return pd.DataFrame(rows)


def _sector_series(tickers) -> pd.Series:
    return pd.Series(
        [SECTORS[i % len(SECTORS)] for i in range(len(tickers))],
        index=list(tickers),
    )


def test_acceleration_is_pit_invariant():
    fund = _synthetic_fund()
    tickers = sorted(fund["ticker"].unique())
    sector = _sector_series(tickers)
    date = pd.Timestamp("2020-12-31")

    before = acceleration.compute_acceleration_scores(fund, date, tickers, sector)

    future = fund[fund["ticker"] == tickers[0]].tail(1).copy()
    future["period_end"] = date + pd.Timedelta(days=30)
    future["pub_date"] = date + pd.Timedelta(days=60)
    future[["revenue_q", "operating_income_q", "net_income_q", "total_assets"]] *= 1000
    after = acceleration.compute_acceleration_scores(
        pd.concat([fund, future], ignore_index=True), date, tickers, sector
    )

    pd.testing.assert_series_equal(
        before["Z_Acceleration"], after["Z_Acceleration"], check_names=False
    )


def test_select_exactly_50_and_weights_sum_to_one():
    tickers = [f"T{i:02d}" for i in range(60)]
    scores = pd.DataFrame(
        {"Z_Acceleration": np.linspace(-1.0, 2.0, 60)}, index=tickers
    )
    weights = pd.Series(np.linspace(0.001, 0.02, 60), index=tickers)

    selected = acceleration.select_and_weight(scores, weights, n=50, lam=0.5)

    assert len(selected) == 50
    assert selected["target_weight"].gt(0).all()
    assert selected["target_weight"].sum() == pytest.approx(1.0)
    assert selected["target_weight"].max() <= 1.0
    assert set(selected["ticker"]).issubset(set(tickers))


def test_exponential_multiplier_ratio():
    tickers = ["A", "B"]
    scores = pd.DataFrame({"Z_Acceleration": [2.0, 0.0]}, index=tickers)
    weights = pd.Series([0.01, 0.01], index=tickers)

    selected = acceleration.select_and_weight(scores, weights, n=2, lam=0.5)
    selected = selected.set_index("ticker")
    ratio = selected.loc["A", "target_weight"] / selected.loc["B", "target_weight"]

    assert ratio == pytest.approx(np.exp(0.5 * 2.0), rel=1e-12)


def test_select_raises_when_too_few_names():
    scores = pd.DataFrame({"Z_Acceleration": [1.0, 2.0]}, index=["A", "B"])
    weights = pd.Series([0.5, 0.5], index=["A", "B"])

    with pytest.raises(ValueError, match="titres scores"):
        acceleration.select_and_weight(scores, weights, n=50)


def test_pit_sector_map_covers_494_cusips():
    sectors = universe.build_pit_sector_map()
    assert sectors["cusip"].nunique() == 494
    assert sectors["sector"].nunique() == 11
    assert not sectors.duplicated(["date", "cusip", "sector"]).any()
    assert sectors["weight"].gt(0).all()


def test_msci_universe_shape_and_coverage():
    uni = universe.build_universe()
    summary = universe.universe_summary(uni)

    assert summary["n_tickers"] == 450
    assert summary["n_dates"] == 21
    assert summary["n_eligible_min"] >= 50
    assert summary["coverage_weight_min"] > 0.99
    assert summary["missing_sector_count"] == 0

def test_project_bounded_simplex_respects_bounds():
    index = ["A", "B", "C"]
    weights = pd.Series([0.8, 0.1, 0.1], index=index)
    lower = pd.Series([0.1, 0.1, 0.1], index=index)
    upper = pd.Series([0.6, 0.3, 0.3], index=index)

    projected = acceleration.project_bounded_simplex(weights, lower, upper)

    assert projected.sum() == pytest.approx(1.0, abs=1e-10)
    assert (projected >= lower - 1e-12).all()
    assert (projected <= upper + 1e-12).all()
    assert (projected >= 0.0).all()


def test_project_bounded_simplex_rejects_infeasible_bounds():
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    lower = pd.Series([0.6, 0.6], index=["A", "B"])
    upper = pd.Series([1.0, 1.0], index=["A", "B"])

    with pytest.raises(ValueError, match="incompatibles"):
        acceleration.project_bounded_simplex(weights, lower, upper)


def test_anchor_selection_uses_largest_weights_and_score_picks():
    tickers = [f"T{i:02d}" for i in range(70)]
    weights = pd.Series(np.linspace(0.001, 0.05, 70), index=tickers)
    z = pd.Series(np.linspace(-2.0, 3.0, 70), index=tickers)

    selected = acceleration.build_anchor_score_weights(
        pd.DataFrame({"Z_Acceleration": z}),
        weights,
        n_anchors=10,
        n_total=60,
    )

    anchors = selected[selected["is_anchor"]]
    picks = selected[~selected["is_anchor"]]
    assert len(selected) == 60
    assert selected.index.is_unique
    assert len(anchors) == 10
    assert len(picks) == 50
    assert set(anchors.index) == set(weights.nlargest(10).index)
    assert not set(anchors.index) & set(picks.index)
    assert set(picks.index).issubset(set(tickers) - set(anchors.index))
    assert selected["target_weight"].sum() == pytest.approx(1.0, abs=1e-10)
    assert selected["target_weight"].gt(0).all()


def test_anchor_weights_respect_rebased_bands_and_missing_score():
    tickers = [f"T{i:02d}" for i in range(70)]
    weights = pd.Series(np.linspace(0.001, 0.05, 70), index=tickers)
    z = pd.Series(np.linspace(-2.0, 3.0, 70), index=tickers)
    z.loc[weights.idxmax()] = np.nan

    selected = acceleration.build_anchor_score_weights(
        pd.DataFrame({"Z_Acceleration": z}),
        weights,
        n_anchors=10,
        n_total=60,
    )

    assert selected.loc[weights.idxmax(), "is_anchor"]
    assert selected.loc[weights.idxmax(), "z_filled"] == pytest.approx(0.0)
    assert selected["target_weight"].sum() == pytest.approx(1.0, abs=1e-10)
    assert (selected["target_weight"] >= selected["lower_weight"] - 1e-12).all()
    assert (selected["target_weight"] <= selected["upper_weight"] + 1e-12).all()
    assert (selected["target_weight"] >= 0.0).all()
    assert np.allclose(
        selected["target_weight"],
        selected["target_weight"].clip(selected["lower_weight"], selected["upper_weight"]),
    )


def _fake_eligible(tickers, weights, sectors):
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "weight": list(weights),
            "sector": sectors.reindex(tickers).values,
        }
    )


def test_build_anchor_portfolios_smoke(monkeypatch):
    fund = _synthetic_fund()
    tickers = sorted(fund["ticker"].unique())
    sectors = _sector_series(tickers)
    weights = np.linspace(0.001, 0.03, len(tickers))
    elig = _fake_eligible(tickers, weights, sectors)
    monkeypatch.setattr(universe, "eligible_universe", lambda _uni, _date: elig)
    date = pd.Timestamp("2020-12-31")

    result = acceleration.build_anchor_portfolios(
        {"dates": [date]}, fund, n_anchors=10, n_total=60
    )

    summary = result["summary"].iloc[0]
    assert summary["n_anchors"] == 10
    assert summary["n_score_picks"] == 50
    assert summary["n_selected"] == 60
    assert result["weights"][date].sum() == pytest.approx(1.0, abs=1e-10)
    assert result["weights"][date].gt(0).sum() == 60


def test_anchor_portfolio_is_pit_invariant(monkeypatch):
    fund = _synthetic_fund()
    tickers = sorted(fund["ticker"].unique())
    sectors = _sector_series(tickers)
    weights = np.linspace(0.001, 0.03, len(tickers))
    elig = _fake_eligible(tickers, weights, sectors)
    monkeypatch.setattr(universe, "eligible_universe", lambda _uni, _date: elig)
    date = pd.Timestamp("2020-12-31")

    before = acceleration.build_anchor_portfolios(
        {"dates": [date]}, fund, n_anchors=10, n_total=60
    )
    future = fund[fund["ticker"] == tickers[0]].tail(1).copy()
    future["period_end"] = date + pd.Timedelta(days=30)
    future["pub_date"] = date + pd.Timedelta(days=60)
    future[["revenue_q", "operating_income_q", "net_income_q", "total_assets"]] *= 1000
    after = acceleration.build_anchor_portfolios(
        {"dates": [date]},
        pd.concat([fund, future], ignore_index=True),
        n_anchors=10,
        n_total=60,
    )

    pd.testing.assert_series_equal(
        before["weights"][date].sort_index(),
        after["weights"][date].sort_index(),
        check_names=False,
    )

def test_anchor_variant_specs_include_capped_subvariants():
    specs = acceleration.anchor_variant_specs()
    assert set(specs) == {
        "top10_plus_50",
        "top20_plus_40",
        "top10_plus_50_cap15",
        "top20_plus_40_cap15",
        "top20_plus_40_cap12_5",
        "top20_plus_40_cap10",
        "top10_plus_50_ex_nvda",
        "top10_plus_50_cap15_ex_nvda",
        "top20_plus_40_ex_nvda",
        "top20_plus_40_cap15_ex_nvda",
    }
    assert specs["top10_plus_50_cap15"]["n_anchors"] == 10
    assert specs["top20_plus_40_cap15"]["n_anchors"] == 20
    assert specs["top20_plus_40_cap12_5"]["n_anchors"] == 20
    assert specs["top20_plus_40_cap10"]["n_anchors"] == 20
    assert specs["top10_plus_50_cap15"]["max_weight"] == pytest.approx(0.15)
    assert specs["top20_plus_40_cap15"]["max_weight"] == pytest.approx(0.15)
    assert specs["top20_plus_40_cap12_5"]["max_weight"] == pytest.approx(0.125)
    assert specs["top20_plus_40_cap10"]["max_weight"] == pytest.approx(0.10)
    assert specs["top10_plus_50"]["max_weight"] is None
    assert specs["top10_plus_50_ex_nvda"]["exclude_tickers"] == ("NVDA",)
    assert specs["top20_plus_40_cap15_ex_nvda"]["max_weight"] == pytest.approx(0.15)
    assert specs["top20_plus_40_cap15_ex_nvda"]["exclude_tickers"] == ("NVDA",)


def test_capped_variant_respects_15_percent_cap():
    rng = np.random.default_rng(7)
    tickers = [f"T{i:02d}" for i in range(70)]
    weights = pd.Series(rng.dirichlet(np.full(70, 0.2)), index=tickers)
    weights.iloc[0] = 0.25
    weights.iloc[1] = 0.20
    weights = weights / weights.sum()
    z = pd.Series(rng.normal(size=70), index=tickers)
    z.iloc[0] = 3.0
    z.iloc[1] = 3.0

    selected = acceleration.build_anchor_score_weights(
        pd.DataFrame({"Z_Acceleration": z}),
        weights,
        n_anchors=10,
        n_total=60,
        max_weight=0.15,
    )

    assert selected["target_weight"].max() <= 0.15 + 1e-12
    assert selected["target_weight"].sum() == pytest.approx(1.0, abs=1e-10)
    assert selected["cap_binding"].any()
    assert (selected["target_weight"] >= selected["lower_weight"] - 1e-12).all()
    assert (selected["target_weight"] <= selected["upper_weight"] + 1e-12).all()


def test_capped_summary_counts_cap_bindings(monkeypatch):
    fund = _synthetic_fund()
    tickers = sorted(fund["ticker"].unique())
    sectors = _sector_series(tickers)
    weights = np.linspace(0.001, 0.03, len(tickers))
    weights[-1] = 0.20
    weights = weights / weights.sum()
    elig = _fake_eligible(tickers, weights, sectors)
    monkeypatch.setattr(universe, "eligible_universe", lambda _uni, _date: elig)
    date = pd.Timestamp("2020-12-31")

    result = acceleration.build_anchor_portfolios(
        {"dates": [date]}, fund, n_anchors=10, n_total=60, max_weight=0.15
    )

    row = result["summary"].iloc[0]
    assert row["n_selected"] == 60
    assert row["max_weight"] <= 0.15 + 1e-12
    assert row["cap_bindings"] >= 1
    assert row["max_weight_cap"] == pytest.approx(0.15)

def test_anchor_selection_excludes_nvda_and_replaces_anchor():
    tickers = ["NVDA"] + [f"T{i:02d}" for i in range(69)]
    weights = pd.Series(np.linspace(0.05, 0.001, len(tickers)), index=tickers)
    z = pd.Series(np.linspace(2.0, -2.0, len(tickers)), index=tickers)

    selected = acceleration.build_anchor_score_weights(
        pd.DataFrame({"Z_Acceleration": z}),
        weights,
        n_anchors=10,
        n_total=60,
        exclude_tickers=("NVDA",),
    )

    assert "NVDA" not in selected.index
    assert len(selected) == 60
    assert selected["is_anchor"].sum() == 10
    assert selected["target_weight"].sum() == pytest.approx(1.0, abs=1e-10)
    expected_anchors = set(weights.drop("NVDA").nlargest(10).index)
    assert set(selected.index[selected["is_anchor"]]) == expected_anchors


def test_ex_nvda_build_has_zero_nvda_weight(monkeypatch):
    tickers = ["NVDA"] + [f"T{i:02d}" for i in range(69)]
    rows = []
    for i, ticker in enumerate(tickers):
        for q in range(12):
            period_end = pd.Timestamp("2018-01-01") + pd.DateOffset(months=3 * q)
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "pub_date": period_end + pd.Timedelta(days=45),
                    "revenue_q": 100.0 * (1.0 + 0.03 * i) * (1.0 + 0.04 * q),
                    "operating_income_q": 10.0 * (1.0 + 0.02 * i) * (1.0 + 0.06 * q),
                    "net_income_q": 8.0 * (1.0 + 0.01 * i) * (1.0 + 0.05 * q),
                    "total_assets": 200.0 * (1.0 + 0.01 * i) * (1.0 + 0.01 * q),
                }
            )
    fund = pd.DataFrame(rows)
    sectors = _sector_series(tickers)
    weights = np.linspace(0.001, 0.03, len(tickers))
    weights[0] = 0.20
    weights = weights / weights.sum()
    elig = _fake_eligible(tickers, weights, sectors)
    monkeypatch.setattr(universe, "eligible_universe", lambda _uni, _date: elig)
    date = pd.Timestamp("2020-12-31")

    result = acceleration.build_anchor_portfolios(
        {"dates": [date]},
        fund,
        n_anchors=10,
        n_total=60,
        exclude_tickers=("NVDA",),
    )

    weights_out = result["weights"][date]
    assert "NVDA" not in weights_out.index
    assert weights_out.sum() == pytest.approx(1.0, abs=1e-10)
    assert result["summary"].iloc[0]["n_excluded_present"] == 1

def test_build_benchmark_excluding_removes_excluded_return():
    dates = [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-03-31")]
    days = pd.bdate_range("2020-12-31", "2021-04-01")
    prices = pd.DataFrame(100.0, index=days, columns=["NVDA", "A", "B"])
    nvda_path = np.linspace(100.0, 200.0, len(days))
    prices["NVDA"] = nvda_path
    weights = pd.DataFrame(
        [
            {"date": dates[0], "ticker": "NVDA", "weight": 0.5},
            {"date": dates[0], "ticker": "A", "weight": 0.3},
            {"date": dates[0], "ticker": "B", "weight": 0.2},
            {"date": dates[1], "ticker": "NVDA", "weight": 0.5},
            {"date": dates[1], "ticker": "A", "weight": 0.3},
            {"date": dates[1], "ticker": "B", "weight": 0.2},
        ]
    )
    benchmark = pd.Series(100.0, index=days)
    uni = {
        "weights": weights,
        "prices": prices,
        "dates": dates,
        "benchmark": benchmark,
    }

    nav = universe.build_benchmark_excluding(uni, ["NVDA"], start=dates[0], end=dates[-1])

    assert nav.iloc[0] == pytest.approx(100.0)
    assert nav.iloc[-1] == pytest.approx(100.0, abs=1e-8)
