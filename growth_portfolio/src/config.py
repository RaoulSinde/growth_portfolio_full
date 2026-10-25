"""Parametres globaux du moteur quant Growth USA."""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
UNIVERSE_CSV = os.path.join(DATA_DIR, "universe.csv")
# Inventaire N-PORT SEC consolide (source des poids PIT trimestriels).
INVENTORY_CSV = os.path.join(DATA_DIR, "inventaire_consolide_S000079196.csv")
UNIVERSE_QUARTERLY_CSV = os.path.join(DATA_DIR, "universe_quarterly.csv")
SECTOR_MAP_CSV = os.path.join(DATA_DIR, "sector_map.csv")

os.makedirs(CACHE_DIR, exist_ok=True)

BENCHMARK_TICKER = "EGUS"
# Historique de prix charge AVANT PERF_START pour couvrir le Momentum 6M-1M,
# la SMA200 et la volatilite 6M a la premiere date de rebalancement.
PRICE_START = "2023-01-01"
PRICE_END = "2026-09-30"
PERF_START = "2024-01-25"
PERF_END = "2026-09-30"

# Rebalancement TRIMESTRIEL : une date par `filing_date` de l'inventaire
# N-PORT (date de connaissance publique des paniers/poids).
# Fenetre de test : 2024-2026 (11 dates ; la 1re = 2024-01-25 = PERF_START).
REBALANCE_DATES = [
    pd.Timestamp("2024-01-25"),
    pd.Timestamp("2024-04-24"),
    pd.Timestamp("2024-07-25"),
    pd.Timestamp("2024-10-28"),
    pd.Timestamp("2025-01-23"),
    pd.Timestamp("2025-04-23"),
    pd.Timestamp("2025-07-24"),
    pd.Timestamp("2025-10-27"),
    pd.Timestamp("2026-01-28"),
    pd.Timestamp("2026-04-23"),
    pd.Timestamp("2026-07-27"),
]
# Date de fin de simulation (fin septembre 2026)
SIM_END = pd.Timestamp("2026-09-30")

# Benchmark de reference : EGUS (iShares ESG Aware MSCI USA Growth ETF).
BENCHMARKS = {
    "EGUS": "EGUS",
}

GICS_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

# Poids par defaut de la strategie unique (3 piliers statiques PIT).
# Z_Composite = w_accel*Z_Acceleration + w_mom*Z_Momentum + w_qual*Z_Quality_GARP
DEFAULT_W_ACCEL = 0.25      # Pilier 1 : Acceleration fondamentale (25 %)
DEFAULT_W_MOMENTUM = 0.45   # Pilier 2 : Momentum de prix ajuste du risque (45 %)
DEFAULT_W_QUALITY = 0.30    # Pilier 3 : Qualite intrinsèque & GARP (30 %)

DEFAULT_N = 40
N_MIN = 40
N_MAX = 50
DEFAULT_SECTOR_BAND = 0.04
DEFAULT_DW_MIN = -0.035
DEFAULT_DW_MAX = 0.025
DEFAULT_W_MAX = 0.125
DEFAULT_TX_COST_BPS = 10.0
