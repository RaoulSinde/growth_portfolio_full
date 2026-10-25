"""Moteur de backtest avec derive buy&hold et couts de transaction.

Le backtest simule l'evolution quotidienne de la valeur liquidative (Base 100 au
premier jour ouvre de la fenetre ``[PERF_START, SIM_END]``) pour le benchmark et
chaque strategie :

- dans le cas *buy & hold strict* (une seule date de construction, univers EGUS
  PIT 2025-12-31), les poids sont appliques **des le premier jour ouvre de 2026**,
  derivent en buy&hold jusqu'au 30/09/2026 sans rebalancement intermediaire et le
  turnover est *one-shot* (construction initiale) ;
- dans le cas general (plusieurs dates), les poids d'un rebalancement date ``t``
  s'appliquent a partir du jour ouvre **strictement** suivant ``t`` ;
- entre deux rebalancements les poids derivent en buy&hold
  ``w_{i,d} = w_{i,d-1}(1 + r_{i,d}) / (1 + r_{port,d})`` ;
- au jour de bourse d'un rebalancement interne a la fenetre, le portefeuille
  derive sur la seance, puis le cout ``tx_cost_bps / 1e4 * 0.5 * sum |w_cible -
  w_derive|`` est applique a la NAV et les poids sont remis a la cible pour le
  lendemain. Le cout de construction initial n'est pas impute a la NAV afin de
  conserver une Base 100 exacte au premier jour de la fenetre (le capital de
  depart est post-construction).

``bench`` accepte une ``pandas.Series`` (retro-compatibilite : la cle
``"Benchmark"`` est utilisee) **ou** un ``Mapping[str, Series]`` (par ex.
``data_loader.get_benchmarks()`` -> cle ``"EGUS"``).
Chaque benchmark expose son propre bloc ``{"nav", "returns", "turnover",
"weights_daily"}``.

SECURITE : aucune donnee issue de l'attribut ``info`` d'un Ticker.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import config

__all__ = ["run_backtest"]

# Tolerance pour eviter une division par zero sur le rendement du portefeuille.
_EPS = 1e-12


def _to_ts(date) -> pd.Timestamp:
    """Normalise une date en Timestamp timezone-naive a minuit."""
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _rb_trading_day(rb_date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """Dernier jour de bourse ``<= rb_date`` (le 'jour de bourse' du rebalancement)."""
    pos = int(trading_days.searchsorted(rb_date, side="right")) - 1
    if pos < 0:
        return None
    return trading_days[pos]


def _normalize_weights(w: pd.Series, columns: pd.Index) -> pd.Series:
    """Aligne/normalise un vecteur de poids sur les colonnes de prix."""
    s = pd.Series(w).astype(float)
    s.index = [str(t) for t in s.index]
    cols = pd.Index([str(c) for c in columns])
    out = s.reindex(cols).fillna(0.0)
    total = float(out.sum())
    if total > 0:
        out = out / total
    return pd.Series(out.to_numpy(), index=columns)


def _benchmark_nav(ser: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """NAV (Base 100 au premier jour de la fenetre) d'un benchmark (prix)."""
    s = pd.Series(ser).astype(float).copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    s = s.reindex(trading_days).ffill()
    if s.notna().sum() and float(s.dropna().iloc[0]) != 0:
        first_valid = float(s.dropna().iloc[0])
        nav = s / first_valid * 100.0
    else:  # pragma: no cover - benchmark vide
        nav = pd.Series(100.0, index=trading_days)
    return nav.ffill().fillna(100.0)


def run_backtest(strategies: Mapping[str, Mapping[pd.Timestamp, pd.Series]],
                 prices: pd.DataFrame,
                 bench: Union[pd.Series, Mapping[str, pd.Series]],
                 tx_cost_bps: float = config.DEFAULT_TX_COST_BPS,
                 rebalance_dates: Optional[Sequence] = None,
                 start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None) -> Dict[str, dict]:
    """Simule benchmark(s) + strategies sur ``[start, end]``.

    ``start`` / ``end`` sont optionnels et retombent sur ``config.PERF_START`` /
    ``config.SIM_END``. ``strategies`` = ``{nom: {date_rebalancement:
    Series(poids, index=ticker)}}``. ``bench`` = Series (cle ``"Benchmark"``) ou
    Mapping nom -> Series de prix. Retourne, pour chaque benchmark et chaque
    strategie, un dict ``{"nav", "returns", "turnover", "weights_daily"}``.
    """
    dates = [_to_ts(d) for d in (rebalance_dates if rebalance_dates is not None
                                 else config.REBALANCE_DATES)]
    dates = sorted(dict.fromkeys(dates))

    prices = prices.sort_index()
    full_days = pd.DatetimeIndex(prices.index).normalize()
    perf_start = _to_ts(start if start is not None else config.PERF_START)
    perf_end = _to_ts(end if end is not None else config.SIM_END)

    px = prices.loc[(prices.index >= perf_start) & (prices.index <= perf_end)]
    trading_days = pd.DatetimeIndex(px.index)
    if len(trading_days) == 0:
        raise ValueError("Aucun jour de bourse dans la fenetre de performance.")

    # Rendements quotidiens (les trous -> 0 pour ne pas biaiser la derive).
    rets_full = prices.pct_change(fill_method=None)
    rets = rets_full.reindex(trading_days).fillna(0.0)

    # --- Benchmarks (Series OU Mapping) ------------------------------------
    if isinstance(bench, Mapping):
        bench_map: Dict[str, pd.Series] = {
            str(name): pd.Series(ser) for name, ser in bench.items()
        }
    else:
        bench_map = {"Benchmark": pd.Series(bench)}

    result: Dict[str, dict] = {}

    def _zero_turnover() -> pd.Series:
        return pd.Series(0.0, index=pd.DatetimeIndex(dates, name="date"))

    for name, ser in bench_map.items():
        bench_nav = _benchmark_nav(ser, trading_days)
        result[name] = {
            "nav": bench_nav,
            "returns": bench_nav.pct_change().fillna(0.0),
            "turnover": _zero_turnover(),
            "weights_daily": None,
        }

    # --- Mapping rebalancement -> jour de bourse ---------------------------
    rb_day_of = {d: _rb_trading_day(d, full_days) for d in dates}
    events: Dict[pd.Timestamp, pd.Timestamp] = {}
    for d in dates:
        bd = rb_day_of[d]
        if bd is not None:
            events[bd] = d  # dernier rebalancement applique ce jour-la

    turnover_index = pd.DatetimeIndex(dates, name="date")
    cols = prices.columns

    for name, w_by_date in strategies.items():
        w_norm = {_to_ts(k): v for k, v in w_by_date.items()}
        first_day = trading_days[0]

        # Poids initiaux : dernier rebalancement dont le jour de bourse precede
        # le premier jour de bourse de la fenetre (cas buy & hold).
        prior = [d for d in dates if rb_day_of[d] is not None
                 and rb_day_of[d] < first_day]
        if prior:
            init_rb = max(prior)
            w = _normalize_weights(w_norm[init_rb], cols)
        else:
            # Repli : premier rebalancement disponible.
            avail = [d for d in dates if d in w_norm]
            if avail:
                init_rb = avail[0]
                w = _normalize_weights(w_norm[init_rb], cols)
            else:  # pragma: no cover - aucune date fournie
                init_rb = dates[0] if dates else None
                w = pd.Series(1.0 / len(cols), index=cols)

        # Turnover one-shot de la construction initiale (poids depuis le cash).
        turnover = _zero_turnover()
        if init_rb is not None and init_rb in turnover.index:
            turnover.loc[init_rb] = float(0.5 * np.abs(w).sum())

        nav = 100.0
        navs: list = [nav]
        weights_daily = pd.DataFrame(index=trading_days, columns=cols, dtype=float)

        for i, day in enumerate(trading_days):
            if i == 0:
                weights_daily.loc[day] = w.to_numpy()
                continue

            # 1) Derive buy&hold sur la seance.
            r = rets.loc[day].reindex(cols).fillna(0.0)
            r_port = float((w * r).sum())
            denom = 1.0 + r_port
            nav *= denom
            if abs(denom) > _EPS:
                w = (w * (1.0 + r)) / denom
            else:  # pragma: no cover - rendement pathologique
                w = w * (1.0 + r)

            # 2) Rebalancement eventuel (derive effectuee, cout, puis cible).
            if day in events:
                rb = events[day]
                tgt = _normalize_weights(w_norm.get(rb, w), cols)
                to = float(0.5 * np.abs(tgt - w).sum())
                turnover.loc[rb] = to
                cost = float(tx_cost_bps) / 1e4 * to
                nav *= (1.0 - cost)
                w = tgt

            navs.append(nav)
            weights_daily.loc[day] = w.to_numpy()

        nav_ser = pd.Series(navs, index=trading_days, name=name, dtype=float)
        result[name] = {
            "nav": nav_ser,
            "returns": nav_ser.pct_change().fillna(0.0),
            "turnover": turnover,
            "weights_daily": weights_daily,
        }
    return result
