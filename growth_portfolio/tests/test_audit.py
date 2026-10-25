"""Audit quantitatif du moteur Growth USA (agent: quant_auditor).

Baseline auditee (**v8**) :

    - univers = inventaire **N-PORT SEC** de l'ETF **EGUS** consolide dans
      `data/universe_quarterly.csv` ; panier **Point-in-Time trimestriel**
      (11 `filing_date` 2024-2026, 81 a 126 titres) ; actions US `EC` uniquement
      (STIV/DE exclus), poids renormalises a 1 par trimestre ;
    - **strategie UNIQUE** a trois piliers statiques PIT :

          Z_Composite = 0.25*Z_Acceleration + 0.45*Z_Momentum + 0.30*Z_Quality_GARP

      - Accumulation  : `Zagg(blend(S_SalesAccel, S_MarginExp, S_AssetEff))` ;
      - Momentum      : `Zagg(0.65*S_RAM + 0.35*S_Trend)` (RAM = 6M-1M / vol 6M) ;
      - Qualite/GARP  : `Zagg(0.40*S_ROE + 0.35*S_FCFYield + 0.25*S_GARP)` ;
    - `factors.build_factor_panel()` -> 11 dates ; chaque coupe transversale
      n'utilise que les donnees disponibles a `t` (prix <= t, `pub_date <= t`) ;
    - `portfolio.build_all_portfolios` -> 11 dates, N = 40, Core `w_idx >= 2.5 %`,
      `max_top10=None`, cible sectorielle = poids GICS de l'indice du trimestre
      (bande +/- 4 %) ;
    - `backtest.run_backtest` : NAV Base 100 au 2024-01-25, derive buy&hold
      intra-trimestre, turnover par rebalancement (11 evenements) ;
    - benchmark unique **EGUS** ; **aucune dependance macro** (`src.macro`
      supprime).

La suite verifie, independamment du code de production :

    1. l'absence de look-ahead (invariance par troncature / perturbation du futur
       a chaque date, PIT des fondamentaux et des poids d'univers, coupe
       transversale pure de la winsorisation et des Z-scores) ;
    2. la chronologie des poids (poids fixes en `t` appliques a `t+1`, rendement du
       jour de publication non compte, derive buy&hold, turnover vs derive) ;
    3. les formulations quantitatives des trois piliers (replication bit a bit) ;
    4. les contraintes Core+Satellites sur les 11 dates ;
    5. la renormalisation PIT de `data/universe_quarterly.csv`.

Execution :  python -m pytest tests/test_audit.py -q
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtest as bt
from src import config
from src import data_loader as dl
from src import factors
from src import metrics as mt
from src import portfolio as pf

from conftest import STRATEGY_NAME

# ---------------------------------------------------------------------------
# Constantes globales de l'audit
# ---------------------------------------------------------------------------
TOL = 1e-8
STRICT = 1e-10
AUDIT_DATES = list(config.REBALANCE_DATES)
N_DATES = 11
Z_COLS = ["Z_Acceleration", "Z_Momentum", "Z_Quality_GARP", "Z_Composite"]
SUB_COLS = ["S_SalesAccel", "S_MarginExp", "S_AssetEff",
            "S_RAM", "S_Trend", "S_ROE", "S_FCFYield", "S_GARP"]
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
# Mega-caps ancres de la poche Core (tickers du panier N-PORT). GOOGL a ete
# retire de l'univers (ISIN non resolu) : on controle GOOG (Classe C).
CORE_MEGACAPS = ["NVDA", "AAPL", "AMZN", "GOOG"]


# ===========================================================================
# Helpers
# ===========================================================================
def _as_ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _universe_for_date(date, index_weights, sector_map) -> pd.DataFrame:
    """Coupe (gics_sector, weight) de l'univers PIT du trimestre `date`."""
    d = _as_ts(date)
    w = index_weights.xs(d, level="date")["weight"]
    tickers = pd.Index(w.index)
    return pd.DataFrame(
        {"gics_sector": sector_map.reindex(tickers).to_numpy(),
         "weight": w.to_numpy()},
        index=tickers,
    )


def _fund_slice(date, fund, tickers) -> pd.DataFrame:
    """Panel fondamental PIT (pub_date <= date) restreint aux tickers."""
    f = fund.copy()
    f["pub_date"] = pd.to_datetime(f["pub_date"])
    f = f[f["pub_date"] <= _as_ts(date)]
    f = f[f["ticker"].isin(tickers)]
    return f.drop_duplicates(subset=["ticker", "period_end"])


def _assert_close(a, b, tol=STRICT, msg=""):
    """Egalite stricte (NaN == NaN) entre deux Series/arrays alignees."""
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    assert av.shape == bv.shape, msg
    diff = np.abs(av - bv)
    finite = diff[np.isfinite(diff)]
    worst = float(finite.max()) if finite.size else 0.0
    assert np.allclose(av, bv, atol=tol, rtol=0.0, equal_nan=True), (msg, worst)


def _Z(m, tickers, sector) -> pd.Series:
    """Z-score intra-sectoriel (comme le moteur)."""
    return factors.zscore(pd.Series(m).reindex(tickers), by_sector=True,
                          sector=sector)


def _replicate_accel(date, fund, index_weights, sector_map) -> dict:
    """Reproduit le pilier Acceleration (S_* et Z_Acceleration), centrage inclus."""
    u = _universe_for_date(date, index_weights, sector_map)
    tickers = pd.Index(u.index)
    sector = u["gics_sector"]
    f = _fund_slice(date, fund, tickers)

    rev = dl.annual_flow_points(f, "revenue_q", tickers)
    oi = dl.annual_flow_points(f, "operating_income_q", tickers)
    ni = dl.annual_flow_points(f, "net_income_q", tickers)
    ta = dl.annual_stock_points(f, "total_assets", tickers)
    is_fin = (sector == factors.FINANCIALS_SECTOR).reindex(tickers).fillna(False)

    sales_accel = (rev["N"] / rev["N1"] - 1.0) - (rev["N1"] / rev["N2"] - 1.0)
    margin_exp = oi["N"] / rev["N"] - oi["N1"] / rev["N1"]
    asset_eff = rev["N"] / ta["N"] - rev["N1"] / ta["N1"]
    roa_delta = ni["N"] / ta["N"] - ni["N1"] / ta["N1"]
    asset_eff = asset_eff.where(~is_fin, roa_delta)

    S_sa = _Z(sales_accel, tickers, sector)
    S_me = _Z(margin_exp, tickers, sector)
    S_ae = _Z(asset_eff, tickers, sector)
    z_acc = _Z(factors._blend([(S_sa, 1.0), (S_me, 1.0), (S_ae, 1.0)]),
               tickers, sector)
    return {
        "S_SalesAccel": S_sa - S_sa.mean(),
        "S_MarginExp": S_me - S_me.mean(),
        "S_AssetEff": S_ae - S_ae.mean(),
        "Z_Acceleration": z_acc - z_acc.mean(),
    }


def _replicate_momentum(date, prices, index_weights, sector_map) -> dict:
    """Reproduit le pilier Momentum : RAM 6M-1M / vol 6M, tendance SMA200."""
    u = _universe_for_date(date, index_weights, sector_map)
    tickers = pd.Index(u.index)
    sector = u["gics_sector"]
    px = prices.loc[prices.index <= _as_ts(date)]
    pxf = px.ffill()

    def price_at(offset):
        if len(pxf) >= offset:
            return pxf.iloc[-offset].reindex(tickers)
        return pd.Series(np.nan, index=tickers)

    r6_1 = price_at(factors.MOM_1M) / price_at(factors.MOM_6M) - 1.0
    rets = px.pct_change(fill_method=None)
    vol_window = rets.tail(factors.VOL_WINDOW_6M)
    n_obs = vol_window.notna().sum()
    sigma_6m = vol_window.std() * np.sqrt(252.0)
    sigma_6m = sigma_6m.where(n_obs >= factors.MIN_VOL_OBS)
    ram = r6_1 / sigma_6m.reindex(tickers)

    last_price = (pxf.iloc[-1].reindex(tickers) if len(pxf)
                  else pd.Series(np.nan, index=tickers))
    sma200 = (pxf.tail(factors.SMA_TREND).mean().reindex(tickers) if len(pxf)
              else pd.Series(np.nan, index=tickers))
    trend = last_price / sma200 - 1.0

    S_ram = _Z(ram, tickers, sector)
    S_trd = _Z(trend, tickers, sector)
    z_mom = _Z(factors._blend([(S_ram, 0.65), (S_trd, 0.35)]), tickers, sector)
    return {
        "S_RAM": S_ram - S_ram.mean(),
        "S_Trend": S_trd - S_trd.mean(),
        "Z_Momentum": z_mom - z_mom.mean(),
    }


def _replicate_quality(date, prices, fund, index_weights, sector_map) -> dict:
    """Reproduit le pilier Qualite/GARP (ROE, FCF Yield, GARP, variante Financials)."""
    u = _universe_for_date(date, index_weights, sector_map)
    tickers = pd.Index(u.index)
    sector = u["gics_sector"]
    is_fin = (sector == factors.FINANCIALS_SECTOR).reindex(tickers).fillna(False)
    f = _fund_slice(date, fund, tickers)

    ni = dl.annual_flow_points(f, "net_income_q", tickers)
    ocf = dl.annual_flow_points(f, "ocf_q", tickers)
    capex = dl.annual_flow_points(f, "capex_q", tickers)
    eq = dl.annual_stock_points(f, "stockholders_equity", tickers)
    shares = dl.annual_stock_points(f, "shares", tickers)

    px = prices.loc[prices.index <= _as_ts(date)]
    pxf = px.ffill()
    last_price = (pxf.iloc[-1].reindex(tickers) if len(pxf)
                  else pd.Series(np.nan, index=tickers))

    equity = eq["N"]
    roe = (ni["N"] / equity).where(equity > 0)
    ocf_n = ocf["N"]
    capex_n = capex["N"].abs()
    mktcap = shares["N"] * last_price
    fcf_yield = (ocf_n - capex_n) / mktcap.where(mktcap > 0)

    eps_ttm = ni["N"] / shares["N"]
    eps_3y = ni["N3"] / shares["N"]
    pe = last_price / eps_ttm
    valid_eps = (eps_ttm > 0) & (eps_3y > 0)
    eps_growth_3y = pd.Series(np.nan, index=tickers, dtype=float)
    eps_growth_3y.loc[valid_eps] = (
        (eps_ttm.loc[valid_eps] / eps_3y.loc[valid_eps])
        ** (1.0 / factors.EPS_GROWTH_YEARS) - 1.0
    )
    garp = (eps_growth_3y / pe).where(pe > 0)

    S_roe = _Z(roe, tickers, sector)
    S_fcf = _Z(fcf_yield, tickers, sector)
    S_garp = _Z(garp, tickers, sector)
    z_std = factors._blend([(S_roe, 0.40), (S_fcf, 0.35), (S_garp, 0.25)])
    z_fin = factors._blend([(S_roe, 0.60), (S_garp, 0.40)])
    z_qual = _Z(z_std.where(~is_fin, z_fin), tickers, sector)
    return {
        "S_ROE": S_roe - S_roe.mean(),
        "S_FCFYield": S_fcf - S_fcf.mean(),
        "S_GARP": S_garp - S_garp.mean(),
        "Z_Quality_GARP": z_qual - z_qual.mean(),
    }


def _toy_market():
    """Prix jouets deterministes dans la fenetre de performance [PERF_START, SIM_END]."""
    idx = pd.bdate_range("2026-01-05", "2026-04-30")
    rets = pd.DataFrame(index=idx)
    rets["A"] = 0.010
    rets["B"] = 0.004
    rets["C"] = -0.006
    px = 100.0 * (1.0 + rets).cumprod()
    bench = pd.Series(100.0 * (1.0 + 0.003) ** np.arange(len(idx)),
                      index=idx, name="Benchmark")
    return idx, px, bench


def _info_attr_hits(path: Path):
    """Acces `.<attr>` nommes 'info' (hors `logger.info` / `logging.info`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "info":
            val = node.value
            if isinstance(val, ast.Name) and val.id in {"logger", "logging"}:
                continue
            hits.append((node.lineno, ast.unparse(node)))
    return hits


def _max_abs_diff(a, b) -> float:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    fin = d[np.isfinite(d)]
    return float(fin.max()) if fin.size else 0.0


# ===========================================================================
# 1) ANTI LOOK-AHEAD BIAS
# ===========================================================================
def test_no_info_access_in_quant_modules():
    """Aucun acces a `Ticker.info` dans les modules quantitatifs (fuite temporelle)."""
    offenders = []
    for module in ("factors.py", "portfolio.py", "backtest.py", "metrics.py",
                   "data_loader.py", "gscore.py"):
        for lineno, expr in _info_attr_hits(SRC_DIR / module):
            offenders.append((module, lineno, expr))
    assert not offenders, f"Acces .info detecte: {offenders}"


def test_no_info_subscript_access():
    """Aucune lecture `['info']` / `get('info')` dans les modules quantitatifs."""
    pat = re.compile(r"""\[\s*['"]info['"]\s*\]|\.get\(\s*['"]info['"]""")
    offenders = []
    for module in ("factors.py", "portfolio.py", "backtest.py", "metrics.py",
                   "data_loader.py", "build_universe.py"):
        for i, line in enumerate((SRC_DIR / module).read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                offenders.append((module, i, line.strip()))
    assert not offenders, f"Reference a ['info'] detectee: {offenders}"


def test_build_universe_info_only_for_identification():
    """`build_universe.py` n'utilise `.info` que pour ISIN->ticker / secteur."""
    src = (SRC_DIR / "build_universe.py").read_text(encoding="utf-8")
    hits = _info_attr_hits(SRC_DIR / "build_universe.py")
    assert hits, "aucun acces .info : test non discriminant"
    # Aucune metrique fondamentale extraite de .info.
    forbidden = ["marketCap", "trailingPE", "forwardPE", "returnOnEquity",
                 "grossMargins", "totalRevenue", "priceToBook", "dividendYield",
                 "profitMargins", "revenue"]
    for key in forbidden:
        assert key not in src, f"metrique fondamentale '{key}' extraite de .info"
    # Seules les cles d'identification sont lues.
    assert "symbol" in src and "sector" in src


def test_scores_invariant_to_future_prices_all_dates(prices, fund, bench_ret,
                                                     index_weights, sector_map):
    """Δ = 0 pour chaque date : troncature (<=t) et perturbation du futur (>t)."""
    worst = 0.0
    for d in AUDIT_DATES:
        u = _universe_for_date(d, index_weights, sector_map)
        ref = factors.compute_strategy_scores(d, prices, fund, bench_ret, u)

        trunc = prices.loc[prices.index <= d]
        a = factors.compute_strategy_scores(d, trunc, fund, bench_ret, u)

        pert = prices.copy()
        future = pert.index > d
        pert.loc[future] = pert.loc[future] * 1e6 + 12345.0
        b = factors.compute_strategy_scores(d, pert, fund, bench_ret, u)

        for col in Z_COLS + SUB_COLS:
            diff_a = np.abs(ref[col].to_numpy(float) - a[col].to_numpy(float))
            diff_b = np.abs(ref[col].to_numpy(float) - b[col].to_numpy(float))
            for diff in (diff_a, diff_b):
                fin = diff[np.isfinite(diff)]
                if fin.size:
                    worst = max(worst, float(fin.max()))
                assert np.allclose(ref[col].to_numpy(float), a[col].to_numpy(float),
                                   atol=STRICT, rtol=0.0, equal_nan=True), (d, col)
                assert np.allclose(ref[col].to_numpy(float), b[col].to_numpy(float),
                                   atol=STRICT, rtol=0.0, equal_nan=True), (d, col)
        assert bool((ref["eligible"].astype(bool) == a["eligible"].astype(bool)).all())
        assert bool((ref["eligible"].astype(bool) == b["eligible"].astype(bool)).all())
    assert worst == 0.0


def test_scores_invariant_to_future_fundamentals(prices, fund, bench_ret,
                                                 index_weights, sector_map):
    """Une ligne de fondamentaux publiee APRES t ne modifie pas les scores a t."""
    d = AUDIT_DATES[-1]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = factors.compute_strategy_scores(d, prices, fund, bench_ret, u)

    injected = fund.copy()
    row = injected.iloc[[0]].copy()
    row["period_end"] = d + pd.Timedelta(days=60)
    row["pub_date"] = d + pd.Timedelta(days=60)
    for c in injected.columns:
        if c not in ("ticker", "period_end", "pub_date") and \
                pd.api.types.is_numeric_dtype(injected[c]):
            row[c] = float(row[c].iloc[0]) * 100.0 + 1e6
    injected = pd.concat([injected, row], ignore_index=True)

    new = factors.compute_strategy_scores(d, prices, injected, bench_ret, u)
    for col in Z_COLS:
        _assert_close(ref[col], new[col], msg=col)


def test_scores_invariant_to_fund_truncation(prices, fund, bench_ret,
                                             index_weights, sector_map):
    """Passer un panel deja tronque (pub_date <= t) donne les memes scores."""
    d = AUDIT_DATES[-1]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = factors.compute_strategy_scores(d, prices, fund, bench_ret, u)
    fund_trunc = fund[pd.to_datetime(fund["pub_date"]) <= d]
    new = factors.compute_strategy_scores(d, prices, fund_trunc, bench_ret, u)
    for col in Z_COLS:
        _assert_close(ref[col], new[col], msg=col)


def test_pit_publication_dates(fund):
    """Pour chaque date : toute ligne utilisee a `pub_date <= t` ET `pub_date >= period_end`."""
    pe = pd.to_datetime(fund["period_end"])
    pub = pd.to_datetime(fund["pub_date"])
    assert (pub >= pe).all(), "pub_date anterieure a period_end"
    for d in AUDIT_DATES:
        asof = dl.fundamentals_asof(fund, d)
        if len(asof) == 0:
            continue
        assert (pd.to_datetime(asof["pub_date"]) <= d).all(), d


def test_fundamentals_used_have_publication_before_date(fund, index_weights):
    """Les fondamentaux effectivement injectes a t proviennent tous de pub_date <= t."""
    for d in AUDIT_DATES:
        w = index_weights.xs(d, level="date")["weight"]
        used = _fund_slice(d, fund, pd.Index(w.index))
        if len(used) == 0:
            continue
        assert (pd.to_datetime(used["pub_date"]) <= d).all(), d


def test_no_weight_before_filing_date(index_weights):
    """Aucun poids d'univers avant sa `filing_date` (regle PIT)."""
    assert set(index_weights.index.get_level_values("date").unique()) == set(AUDIT_DATES)
    asof = pd.Timestamp("2024-06-30")
    u = dl.get_universe(asof=asof)
    assert (u["rebalance_date"] <= asof).all()
    assert u["rebalance_date"].nunique() == 1
    last_known = max(d for d in AUDIT_DATES if d <= asof)
    assert u["rebalance_date"].iloc[0] == last_known


def test_index_weights_pit_and_renormalized(index_weights):
    """Poids PIT = univers_quarterly renormalise ; Σ=1 par date ; paires valides."""
    uni = dl.get_universe()
    allowed = set(map(tuple, uni[["rebalance_date", "ticker"]].values))
    pairs = set(map(tuple, index_weights.index.to_frame(index=False).values))
    assert pairs.issubset(allowed)
    sums = index_weights.groupby(level="date")["weight"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-9)
    assert (index_weights["weight"].values >= -1e-12).all()


def test_panel_matches_fresh_computation(panel, prices, fund, bench_ret,
                                         index_weights, sector_map):
    """Le panel (possiblement en cache) est identique a un recalcul frais."""
    for d in (AUDIT_DATES[0], AUDIT_DATES[5], AUDIT_DATES[-1]):
        fresh = factors.compute_strategy_scores(
            d, prices, fund, bench_ret, _universe_for_date(d, index_weights, sector_map))
        cached = panel.loc[d].reindex(fresh.index)
        for col in factors.FACTOR_COLUMNS:
            if col == "eligible":
                assert bool((fresh[col].astype(bool) == cached[col].astype(bool)).all())
            else:
                _assert_close(fresh[col], cached[col], msg=(d, col))


def test_winsorize_is_cross_sectional_per_date(monkeypatch, prices, fund, bench_ret,
                                               index_weights, sector_map):
    """La winsorisation porte sur la coupe du trimestre, jamais sur la matrice complete."""
    lengths: list = []
    orig = factors.winsorize

    def spy(s, *a, **k):
        lengths.append(len(s))
        return orig(s, *a, **k)

    monkeypatch.setattr(factors, "winsorize", spy)
    for d in (AUDIT_DATES[0], AUDIT_DATES[-1]):
        u = _universe_for_date(d, index_weights, sector_map)
        factors.compute_strategy_scores(d, prices, fund, bench_ret, u)
    assert lengths, "winsorize n'a pas ete appele"
    max_basket = int(index_weights.groupby(level="date").size().max())
    total_rows = len(index_weights)
    assert max(lengths) <= max_basket + 5, f"coupe={max(lengths)} > panier={max_basket}"
    assert max(lengths) < total_rows  # jamais la matrice temporelle complete


def test_winsorize_clips_at_5_95_quantiles():
    """Winsorisation bien bornee par les 5e / 95e centiles."""
    s = pd.Series(np.arange(100, dtype=float))
    w = factors.winsorize(s)
    assert abs(float(w.min()) - float(s.quantile(0.05))) < 1e-12
    assert abs(float(w.max()) - float(s.quantile(0.95))) < 1e-12


def test_winsorize_per_date_differs_from_global():
    """Preuve numerique : winsoriser par date != winsoriser globalement."""
    s = pd.Series([0.0, 1.0, 2.0, 100.0, 0.5, 1.5, 2.5, 3.0, 4.0, 5.0],
                  index=pd.MultiIndex.from_product(
                      [["d1", "d2"], list("abcde")]))
    per_date = s.groupby(level=0).apply(factors.winsorize).droplevel(0)
    global_w = factors.winsorize(s)
    assert not np.allclose(per_date.to_numpy(), global_w.to_numpy())


# ===========================================================================
# 2) FORMULES QUANTITATIVES
# ===========================================================================
def test_momentum_window_constants():
    """Fenetres : 21 / 126 / 252 jours, vol 6M (126), SMA de tendance 200."""
    assert (factors.MOM_1M, factors.MOM_6M, factors.MOM_12M) == (21, 126, 252)
    assert factors.VOL_WINDOW_6M == 126
    assert factors.SMA_TREND == 200
    assert factors.MIN_VOL_OBS == 60
    # L'ancien moteur VOL_WINDOW_3Y = 756 a ete retire.
    assert not hasattr(factors, "VOL_WINDOW_3Y")


def test_momentum_excludes_last_month(prices):
    """Perturber les 20 dernieres seances ne change pas le numerateur 6M-1M."""
    d = AUDIT_DATES[-1]
    px = prices.loc[prices.index <= d]
    pxf = px.ffill()
    base_6 = (pxf.iloc[-factors.MOM_1M] / pxf.iloc[-factors.MOM_6M] - 1.0)

    pert = prices.copy()
    last_month = (pert.index <= d) & (pert.index > pxf.index[-factors.MOM_1M])
    pert.loc[last_month] = pert.loc[last_month] * 3.0
    pxp = pert.loc[pert.index <= d].ffill()
    new_6 = (pxp.iloc[-factors.MOM_1M] / pxp.iloc[-factors.MOM_6M] - 1.0)

    assert np.allclose(base_6.dropna(), new_6.reindex(base_6.index).dropna(),
                       atol=1e-12, equal_nan=True)


def test_acceleration_formula_replication(panel, fund, index_weights, sector_map):
    """S_SalesAccel / S_MarginExp / S_AssetEff et Z_Acceleration recalcules == moteur."""
    for d in AUDIT_DATES:
        rep = _replicate_accel(d, fund, index_weights, sector_map)
        got = panel.loc[d]
        for col, ser in rep.items():
            _assert_close(ser.reindex(got.index), got[col], msg=(d, col))


def test_momentum_formula_replication(panel, prices, index_weights, sector_map):
    """S_RAM (6M-1M / vol 6M) et Z_Momentum recalcules == valeur du moteur."""
    for d in AUDIT_DATES:
        rep = _replicate_momentum(d, prices, index_weights, sector_map)
        got = panel.loc[d]
        for col, ser in rep.items():
            _assert_close(ser.reindex(got.index), got[col], msg=(d, col))


def test_quality_formula_replication(panel, prices, fund, index_weights, sector_map):
    """S_ROE / S_FCFYield / S_GARP et Z_Quality_GARP recalcules == moteur."""
    for d in AUDIT_DATES:
        rep = _replicate_quality(d, prices, fund, index_weights, sector_map)
        got = panel.loc[d]
        for col, ser in rep.items():
            _assert_close(ser.reindex(got.index), got[col], msg=(d, col))


def test_composite_formula_replication(panel):
    """Z_Composite = composite_weights_apply(panel, 0.25, 0.45, 0.30) a chaque date."""
    for d, sub in panel.groupby(level="date"):
        rep = factors.composite_weights_apply(
            sub, config.DEFAULT_W_ACCEL, config.DEFAULT_W_MOMENTUM,
            config.DEFAULT_W_QUALITY)
        _assert_close(rep.reindex(sub.index), sub["Z_Composite"], msg=d)


def test_quality_contains_roe_fcf_garp_and_purge_bricks():
    """Les briques cles du pilier Qualite/GARP et de la purge sont bien presentes."""
    src = (SRC_DIR / "factors.py").read_text(encoding="utf-8")
    # Pilier Qualite/GARP (Norme GARP) :
    assert 'roe = (ni["N"] / equity)' in src                  # ROE = NI_TTM / Equity
    assert 'fcf_yield = (ocf_n - capex_n) / mktcap' in src    # FCF Yield
    assert 'garp = (eps_growth_3y / pe)' in src               # GARP = croissance / P/E
    assert 'eps_ttm = ni["N"] / shares["N"]' in src           # EPS TTM
    # Purge Junk Growth (Novy-Marx GP/A + test d'accruals OCF < NI + levier D/E) :
    assert 'gpa = gp_ttm / ta_latest' in src                  # Novy-Marx : GP / Assets
    assert 'd2e = (debt_latest / eq_latest)' in src           # endettement D/E
    assert '(ocf.reindex(cand) < ni.reindex(cand))' in src    # accruals Mohanram
    # Variante Financials (FCF omis) :
    assert '(S_ROE, 0.60)' in src and '(S_GARP, 0.40)' in src


def test_quality_independent_of_price_momentum(prices, fund, bench_ret,
                                               index_weights, sector_map):
    """Perturber les prix de la fenetre momentum ne change PAS Z_Quality_GARP."""
    d = AUDIT_DATES[-1]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = factors.compute_strategy_scores(d, prices, fund, bench_ret, u)

    pert = prices.copy()
    px = prices.loc[prices.index <= d]
    win = (pert.index <= px.index[-factors.MOM_1M]) & \
          (pert.index >= px.index[-factors.MOM_6M])
    pert.loc[win] = pert.loc[win] * 1.7 + 5.0
    new = factors.compute_strategy_scores(d, pert, fund, bench_ret, u)
    _assert_close(ref["Z_Quality_GARP"], new["Z_Quality_GARP"], msg="Z_Quality_GARP")


def test_quality_reacts_to_profitability(prices, fund, bench_ret, index_weights,
                                         sector_map):
    """Augmenter le resultat net augmente S_ROE de ce titre (professionnalite)."""
    d = AUDIT_DATES[-1]
    u = _universe_for_date(d, index_weights, sector_map)
    ref = factors.compute_strategy_scores(d, prices, fund, bench_ret, u)

    eq = fund.groupby("ticker")["stockholders_equity"].max()
    cand = ref["S_ROE"].dropna()
    cand = cand[[t for t in cand.index if float(eq.get(t, 0.0)) > 0]]
    ticker = cand.idxmin()
    pert = fund.copy()
    mask = pert["ticker"] == ticker
    pert.loc[mask, "net_income_q"] = pert.loc[mask, "total_assets"].abs() * 1000.0
    new = factors.compute_strategy_scores(d, prices, pert, bench_ret, u)
    assert float(new.loc[ticker, "S_ROE"]) > float(ref.loc[ticker, "S_ROE"])


# ===========================================================================
# 3) CHRONOLOGIE DES POIDS ET DES RENDEMENTS (t -> t+1)
# ===========================================================================
def test_toy_new_weights_not_applied_on_rebalance_day():
    """Le rendement du jour de rebalancement utilise les ANCIENS poids derives."""
    idx, px, bench = _toy_market()
    d2 = pd.Timestamp("2026-02-16")
    weights = {
        pd.Timestamp("2026-01-05"): pd.Series({"A": 0.6, "B": 0.3, "C": 0.1}),
        d2: pd.Series({"A": 0.1, "B": 0.3, "C": 0.6}),
    }
    res = bt.run_backtest({"Toy": weights}, px, bench, tx_cost_bps=0.0,
                          rebalance_dates=list(weights))
    blk = res["Toy"]
    wd, nav = blk["weights_daily"], blk["nav"]
    rets = px.reindex(wd.index).pct_change(fill_method=None).fillna(0.0)

    loc = wd.index.get_loc(d2)
    prev = wd.index[loc - 1]
    exp_old = 1.0 + float((wd.loc[prev] * rets.loc[d2]).sum())
    got = float(nav.loc[d2] / nav.loc[prev])
    assert abs(got - exp_old) < 1e-12
    # Le portefeuille avec les NOUVEAUX poids aurait donne un autre rendement.
    with_target = 1.0 + float((weights[d2].reindex(wd.columns) * rets.loc[d2]).sum())
    assert abs(got - with_target) > 1e-8


def test_toy_new_weights_applied_next_day():
    """r_port(t+1) = sum_i w_target_i * r_i(t+1)."""
    idx, px, bench = _toy_market()
    d2 = pd.Timestamp("2026-02-16")
    weights = {
        pd.Timestamp("2026-01-05"): pd.Series({"A": 0.6, "B": 0.3, "C": 0.1}),
        d2: pd.Series({"A": 0.1, "B": 0.3, "C": 0.6}),
    }
    res = bt.run_backtest({"Toy": weights}, px, bench, tx_cost_bps=0.0,
                          rebalance_dates=list(weights))
    blk = res["Toy"]
    wd, nav = blk["weights_daily"], blk["nav"]
    rets = px.reindex(wd.index).pct_change(fill_method=None).fillna(0.0)

    loc = wd.index.get_loc(d2)
    nxt = wd.index[loc + 1]
    exp = 1.0 + float((weights[d2].reindex(wd.columns) * rets.loc[nxt]).sum())
    got = float(nav.loc[nxt] / nav.loc[d2])
    assert abs(got - exp) < 1e-12
    # Les poids enregistres le jour t correspondent bien a la cible.
    assert np.allclose(wd.loc[d2].to_numpy(),
                       weights[d2].reindex(wd.columns).to_numpy())


def test_toy_rebalance_weekend_maps_to_prior_trading_day():
    """Un `filing_date` hors seance est applique au dernier jour de bourse precedent."""
    idx, px, bench = _toy_market()
    sunday = pd.Timestamp("2026-01-18")          # dimanche
    fri = pd.Timestamp("2026-01-16")
    mon = pd.Timestamp("2026-01-19")
    weights = {
        pd.Timestamp("2026-01-05"): pd.Series({"A": 0.6, "B": 0.3, "C": 0.1}),
        sunday: pd.Series({"A": 0.1, "B": 0.3, "C": 0.6}),
    }
    res = bt.run_backtest({"Toy": weights}, px, bench, tx_cost_bps=0.0,
                          rebalance_dates=list(weights))
    blk = res["Toy"]
    wd, nav = blk["weights_daily"], blk["nav"]
    rets = px.reindex(wd.index).pct_change(fill_method=None).fillna(0.0)

    # Le rendement du vendredi 16 utilise les anciens poids.
    loc = wd.index.get_loc(fri)
    prev = wd.index[loc - 1]
    exp_old = 1.0 + float((wd.loc[prev] * rets.loc[fri]).sum())
    assert abs(float(nav.loc[fri] / nav.loc[prev]) - exp_old) < 1e-12
    # Les nouveaux poids sont detenus le lundi 19.
    exp_new = 1.0 + float((weights[sunday].reindex(wd.columns) * rets.loc[mon]).sum())
    assert abs(float(nav.loc[mon] / nav.loc[fri]) - exp_new) < 1e-12


def test_real_backtest_return_alignment(composite_portfolios, prices, benchmark):
    """Sur tout jour (couts nuls) : r_nav(d) = sum_i w_{i,d-1} * r_i(d)."""
    res = bt.run_backtest({STRATEGY_NAME: composite_portfolios}, prices,
                          benchmark, tx_cost_bps=0.0)
    blk = res[STRATEGY_NAME]
    wd, nav = blk["weights_daily"], blk["nav"]
    rets = prices.reindex(wd.index).pct_change(fill_method=None).fillna(0.0)
    dates = list(wd.index)
    n_ok = 0
    for i in range(1, len(dates)):
        day, prev = dates[i], dates[i - 1]
        expected = 1.0 + float((wd.loc[prev] * rets.loc[day]).sum())
        got = float(nav.loc[day] / nav.loc[prev])
        assert abs(got - expected) < 1e-9, day
        n_ok += 1
    assert n_ok > 600


def test_weight_drift_buyhold(backtest_result, prices):
    """Entre 2 rebalancements : w_i,1 ∝ w_i,0 * (1 + r_i) (buy & hold)."""
    wd = backtest_result[STRATEGY_NAME]["weights_daily"]
    rets = prices.reindex(wd.index).pct_change(fill_method=None).fillna(0.0)
    rb_days = set()
    for d in AUDIT_DATES:
        pos = prices.index.searchsorted(d, side="right") - 1
        if pos >= 0:
            rb_days.add(prices.index[pos].normalize())
    dates = list(wd.index)
    checked = 0
    for i in range(1, len(dates)):
        day, prev = dates[i], dates[i - 1]
        if day in rb_days:
            continue
        w0 = wd.loc[prev]
        r = rets.loc[day]
        denom = 1.0 + float((w0 * r).sum())
        if abs(denom) < 1e-9:
            continue
        exp = (w0 * (1.0 + r)) / denom
        assert float((wd.loc[day] - exp).abs().max()) < 1e-9, day
        checked += 1
    assert checked > 600


def test_turnover_matches_derived_weights(backtest_result, prices, index_weights):
    """Turnover_t = 0.5 * sum |w_cible - w_derive_fin_de_trimestre| (11 evenements)."""
    blk = backtest_result[STRATEGY_NAME]
    wd, turnover = blk["weights_daily"], blk["turnover"]
    rets = prices.reindex(wd.index).pct_change(fill_method=None).fillna(0.0)
    checked = 0
    for d in AUDIT_DATES:
        pos = prices.index.searchsorted(d, side="right") - 1
        if pos < 0:
            continue
        bd = prices.index[pos].normalize()
        if bd not in wd.index:
            continue
        loc = wd.index.get_loc(bd)
        if loc == 0:
            continue
        prev = wd.index[loc - 1]
        w_prev = wd.loc[prev]
        r = rets.loc[bd]
        denom = 1.0 + float((w_prev * r).sum())
        drifted = (w_prev * (1.0 + r)) / denom
        target = wd.loc[bd]
        expected = 0.5 * float((target - drifted).abs().sum())
        assert abs(float(turnover.loc[d]) - expected) < 1e-9, d
        checked += 1
    assert checked >= 10


# ===========================================================================
# 4) CONTRAINTES PAR DATE (strategie unique x 11 dates)
# ===========================================================================
def test_constraints_sum_nonneg_finite(composite_portfolios):
    """Σw = 1 (±1e-8), w >= 0, poids finis, pour chaque date."""
    assert len(composite_portfolios) == N_DATES == len(AUDIT_DATES)
    for d, w in composite_portfolios.items():
        assert abs(float(w.sum()) - 1.0) <= 1e-8, (d, float(w.sum()))
        assert (w >= -1e-12).all(), (d, float(w.min()))
        assert np.isfinite(w.to_numpy()).all(), d


def test_constraints_n_titles(composite_portfolios):
    """Le nombre de titres respecte N ∈ [40, 50] (defaut 40)."""
    for d, w in composite_portfolios.items():
        n = int((w > 0).sum())
        assert config.N_MIN <= n <= config.N_MAX, (d, n)
        assert n == config.DEFAULT_N, (d, n)


def test_constraints_active_bounds_and_cap(composite_portfolios, index_weights):
    """Bornes actives [dw_min, dw_max] (±1e-6) et cap absolu w <= w_max."""
    for d, w in composite_portfolios.items():
        wi = index_weights.xs(d, level="date")["weight"].reindex(w.index).fillna(0.0)
        dw = (w - wi)
        assert float(dw.min()) >= config.DEFAULT_DW_MIN - 1e-6, (d, float(dw.min()))
        assert float(dw.max()) <= config.DEFAULT_DW_MAX + 1e-6, (d, float(dw.max()))
        assert float(w.max()) <= config.DEFAULT_W_MAX + 1e-9, (d, float(w.max()))


def test_constraints_core_present(composite_portfolios, panel, index_weights):
    """Toute Mega-Cap eligibles (w_idx >= 2.5%) est presente (Core)."""
    for d, w in composite_portfolios.items():
        p = panel.loc[d]
        elig = p.index[p["eligible"].astype(bool)]
        wi = index_weights.xs(d, level="date")["weight"]
        core = [t for t in elig if float(wi.get(t, 0.0)) >= pf.CORE_THRESHOLD]
        assert core, d
        missing = [t for t in core if t not in w.index]
        assert not missing, (d, missing)


def test_constraints_core_megacaps_are_anchors(composite_portfolios):
    """Nvidia, Apple, Amazon, Alphabet (GOOG) sont des ancres Core (>= 10/11 dates)."""
    for t in CORE_MEGACAPS:
        n_present = sum(1 for w in composite_portfolios.values() if t in w.index)
        assert n_present >= 10, (t, n_present)


def test_constraints_sector_band(composite_portfolios, sector_map, index_sector_w):
    """Deviation sectorielle <= bande (+/- 4%) autour de l'indice du trimestre."""
    band = config.DEFAULT_SECTOR_BAND
    for d, w in composite_portfolios.items():
        sw = (w.groupby(sector_map.reindex(w.index)).sum()
              .reindex(config.GICS_SECTORS).fillna(0.0))
        iw = index_sector_w.loc[d].reindex(config.GICS_SECTORS).fillna(0.0)
        assert float((sw - iw).abs().max()) <= band + 1e-6, \
            (d, float((sw - iw).abs().max()))


# ===========================================================================
# 5) PROJECTION SUR LE SIMPLEXE BORNE (unitaire)
# ===========================================================================
def test_project_weights_box_and_sum():
    """project_weights : sum=1, w>=0, lo_i <= w_i <= hi_i (avec/sans secteur)."""
    idx = ["A", "B", "C", "D"]
    w_idx = pd.Series([0.30, 0.30, 0.20, 0.20], index=idx)
    raw = pd.Series([0.40, 0.10, 0.30, 0.20], index=idx)
    smap = pd.Series(["Tech", "Tech", "Health", "Health"], index=idx)
    st = pd.Series({"Tech": 0.5, "Health": 0.5})

    for target in (None, st):
        out = pf.project_weights(raw, w_idx, target, smap, dw_min=-0.05,
                                 dw_max=0.05, w_max=0.5, max_top10=None)
        assert abs(float(out.sum()) - 1.0) < TOL
        assert (out >= -1e-12).all()
        lo = np.maximum(0.0, w_idx.to_numpy() - 0.05)
        hi = np.maximum(np.minimum(0.5, w_idx.to_numpy() + 0.05), lo)
        assert (out.to_numpy() >= lo - 1e-9).all()
        assert (out.to_numpy() <= hi + 1e-9).all()


def test_project_weights_sector_target_respected():
    """Avec pilotage sectoriel, les sommes sectorielles suivent la cible faisable."""
    idx = ["A", "B", "C", "D"]
    w_idx = pd.Series([0.30, 0.30, 0.20, 0.20], index=idx)
    raw = pd.Series([0.40, 0.10, 0.30, 0.20], index=idx)
    smap = pd.Series(["Tech", "Tech", "Health", "Health"], index=idx)
    st = pd.Series({"Tech": 0.5, "Health": 0.5})
    out = pf.project_weights(raw, w_idx, st, smap, dw_min=-0.05, dw_max=0.05,
                             w_max=0.5, max_top10=None)
    sec = out.groupby(smap).sum()
    assert abs(float(sec["Tech"]) - 0.5) < 1e-6
    assert abs(float(sec["Health"]) - 0.5) < 1e-6


# ===========================================================================
# 6) RENORMALISATION PIT DE `data/universe_quarterly.csv`
# ===========================================================================
def test_universe_quarterly_renormalized(index_weights):
    """Les 11 trimestres PIT renormalisent leurs poids a 1 ; pas de doublon."""
    uni = pd.read_csv(config.UNIVERSE_QUARTERLY_CSV)
    uni["rebalance_date"] = pd.to_datetime(uni["rebalance_date"]).dt.normalize()
    cfg_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in config.REBALANCE_DATES])
    # Le fichier source couvre tout l'historique N-PORT (>= 11 trimestres) ;
    # le moteur n'en retient que les 11 dates de config.REBALANCE_DATES.
    assert uni["rebalance_date"].nunique() >= N_DATES
    assert set(cfg_dates).issubset(set(uni["rebalance_date"].unique()))

    cfg = uni[uni["rebalance_date"].isin(cfg_dates)]
    sums = cfg.groupby("rebalance_date")["weight"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-9), sums
    assert (cfg["weight"].values >= 0).all()
    assert cfg.duplicated(["rebalance_date", "ticker"]).sum() == 0
    assert uni["ticker"].nunique() >= 200
    # Les poids utilises par le moteur portent bien sur les 11 dates.
    assert index_weights.index.get_level_values("date").nunique() == N_DATES


def test_universe_quarterly_excludes_stiv_de_and_foreign():
    """Aucun ISIN non `EC`/`US` (STIV/DE, cotes etrangeres) dans l'univers PIT."""
    inv = pd.read_csv(config.INVENTORY_CSV)
    uni = pd.read_csv(config.UNIVERSE_QUARTERLY_CSV)
    non_eq = set(inv.loc[(inv["asset_cat"] != "EC") |
                         (inv["inv_country"] != "US"), "isin"])
    assert non_eq, "inventaire sans STIV/DE : test non discriminant"
    assert not (set(uni["isin"]) & non_eq), "STIV/DE/etranger present dans l'univers"
    # Tous les ISIN retenus proviennent bien de lignes EC/US de l'inventaire.
    ecus = set(inv.loc[(inv["asset_cat"] == "EC") &
                       (inv["inv_country"] == "US"), "isin"])
    assert set(uni["isin"]).issubset(ecus)


# ===========================================================================
# 7) BACKTEST GLOBAL (fixtures partagees)
# ===========================================================================
def test_strategy_stats_vs_egus(composite_portfolios, prices, benchmarks,
                                index_weights):
    """La strategie composite est evaluee vs EGUS ; NAV finie et Base 100."""
    res = bt.run_backtest({STRATEGY_NAME: composite_portfolios}, prices,
                          benchmarks, tx_cost_bps=10.0)
    assert "EGUS" in res
    assert STRATEGY_NAME in res
    blk = res[STRATEGY_NAME]
    for k in ("nav", "returns", "turnover", "weights_daily"):
        assert k in blk
    nav = blk["nav"]
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()
    assert abs(float(nav.iloc[0]) - 100.0) < 1e-8
    stats = mt.performance_stats(nav, res["EGUS"]["nav"],
                                 weights_daily=blk["weights_daily"],
                                 index_weights=index_weights,
                                 turnover=blk["turnover"], rf=0.0)
    assert np.isfinite(stats["alpha_ann"])
    assert np.isfinite(stats["te_ann"]) and stats["te_ann"] > 0
    assert stats["cum_return"] > -1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
