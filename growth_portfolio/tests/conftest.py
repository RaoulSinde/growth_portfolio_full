"""Fixtures partagees (scope session) pour les suites de tests.

La baseline courante est une **strategie UNIQUE a trois piliers PIT** (2024-2026,
11 dates de rebalancement) :

    Z_Composite = 0.25*Z_Acceleration + 0.45*Z_Momentum + 0.30*Z_Quality_GARP

Les constructions lourdes (panel factoriel, portefeuilles Core + Satellites via
projection IPF, backtest) sont mises en commun entre `test_factors.py`,
`test_portfolio.py`, `test_audit.py` et `test_audit_bias.py` afin de ne les
calculer qu'une seule fois par session.

Ces fixtures remplacent l'ancienne API 3 strategies (Quality / Momentum /
Combinee) x 14 dates : une seule strategie composite est desormais construite et
backtestee, exposee sous la cle ``STRATEGY_NAME = "Growth Composite"``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import backtest as bt
from src import config
from src import data_loader as dl
from src import factors
from src import gscore
from src import portfolio as pf

# Nom de la strategie unique (cle du backtest) et ponderations par defaut.
STRATEGY_NAME = "Growth Composite"
COMPOSITE_WEIGHTS = {
    "acceleration": config.DEFAULT_W_ACCEL,   # 0.25
    "momentum": config.DEFAULT_W_MOMENTUM,    # 0.45
    "quality": config.DEFAULT_W_QUALITY,      # 0.30
}


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    """Prix de cloture ajustes (index date, colonnes tickers, 2023-01 -> 2026-09)."""
    return dl.get_prices()


@pytest.fixture(scope="session")
def benchmarks() -> dict:
    """Benchmark unique : ``{"EGUS": Series de prix}``."""
    return dl.get_benchmarks()


@pytest.fixture(scope="session")
def bench(benchmarks) -> pd.Series:
    """Serie de prix du benchmark EGUS."""
    return benchmarks["EGUS"]


@pytest.fixture(scope="session")
def benchmark(benchmarks) -> pd.Series:
    """Alias explicite du benchmark unique EGUS."""
    return benchmarks["EGUS"]


@pytest.fixture(scope="session")
def bench_ret(bench) -> pd.Series:
    """Rendements quotidiens du benchmark (signature ``compute_strategy_scores``)."""
    return bench.pct_change(fill_method=None)


@pytest.fixture(scope="session")
def fund() -> pd.DataFrame:
    """Panel fondamental PIT (une ligne par ticker x period_end, ``pub_date``)."""
    return dl.get_fundamentals()


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """Panel factoriel PIT (11 dates x panier N-PORT du trimestre)."""
    return factors.build_factor_panel()


@pytest.fixture(scope="session")
def gscore_panel() -> pd.DataFrame:
    """Panel G-Score de Mohanram PIT (11 dates x panier N-PORT du trimestre)."""
    return gscore.build_gscore_panel()


@pytest.fixture(scope="session")
def index_weights() -> pd.DataFrame:
    """Poids PIT de l'indice (MultiIndex date, ticker), somme = 1 par date."""
    return dl.get_index_weights()


@pytest.fixture(scope="session")
def index_sector_w() -> pd.DataFrame:
    """Poids sectoriels GICS PIT de l'indice (index = dates, colonnes = secteurs)."""
    return dl.get_index_sector_weights()


@pytest.fixture(scope="session")
def sector_map() -> pd.Series:
    """Mapping ticker -> secteur GICS (univers PIT complet)."""
    return pf._get_sector_map()


@pytest.fixture(scope="session")
def composite_portfolios(panel, index_weights, index_sector_w) -> dict:
    """Strategie unique : ``{date: Series(poids, index=ticker)}`` sur 11 dates.

    Construite via ``pf.build_all_portfolios`` avec les ponderations de piliers
    par defaut (acceleration 0.25 / momentum 0.45 / quality 0.30).
    """
    return pf.build_all_portfolios(
        params={"weights": dict(COMPOSITE_WEIGHTS)},
        panel=panel, index_weights=index_weights,
        index_sector_w=index_sector_w,
    )


@pytest.fixture(scope="session")
def backtest_result(composite_portfolios, prices, benchmarks) -> dict:
    """Backtest commun (couts 10 bps) de la strategie unique + benchmark EGUS."""
    return bt.run_backtest({STRATEGY_NAME: composite_portfolios}, prices,
                           benchmarks, tx_cost_bps=10.0)
