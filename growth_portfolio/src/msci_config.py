"""Configuration des donnees MSCI USA Growth pour le backtest d'acceleration."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = PROJECT_ROOT / "reports"

MSCI_WEIGHTS = PROCESSED_DIR / "index_weights.parquet"
MSCI_PRICES = PROCESSED_DIR / "index_prices.parquet"
RECONSTRUCTED = PROCESSED_DIR / "reconstructed_index.parquet"
TICKER_MAP = CACHE_DIR / "index_ticker_map.parquet"
COMPONENT_PRICES = CACHE_DIR / "component_prices.parquet"
COMPONENT_PRICES_ADJ = CACHE_DIR / "component_prices_adj.parquet"
CONSTITUENTS_XLSX = INDEX_DIR / "msci_usa_growth_constituents.xlsx"

SEC_TICKERS_CACHE = PROJECT_ROOT / "growth_portfolio" / "data" / "cache" / "sec_company_tickers.json"
SEC_COMPANYFACTS_DIR = PROJECT_ROOT / "growth_portfolio" / "data" / "cache" / "sec_companyfacts"

FUNDAMENTALS_MSCI = PROCESSED_DIR / "fundamentals_sec_msci.parquet"
SCORES_FILE = PROCESSED_DIR / "acceleration_scores.parquet"
WEIGHTS_FILE = PROCESSED_DIR / "acceleration_weights.parquet"
REPORT_CSV = REPORTS_DIR / "acceleration_backtest.csv"
REPORT_HTML = REPORTS_DIR / "acceleration_report.html"

START = pd.Timestamp("2020-12-31")
END = pd.Timestamp("2025-12-31")

N_HOLDINGS = 50
LAMBDA = 0.50
Z_CLIP = 3.0
TX_COST_BPS = 10.0

N_ANCHORS = 10
N_TOTAL_HOLDINGS = 60
BAND_REL = 0.50
BAND_ABS_CAP = 0.04
ANCHOR_VARIANTS = {
    "top10_plus_50": 10,
    "top20_plus_40": 20,
}
CAP_MAX_WEIGHT = 0.15
CAP_VARIANTS = {
    "top10_plus_50_cap15": 10,
    "top20_plus_40_cap15": 20,
    "top20_plus_40_cap12_5": 20,
    "top20_plus_40_cap10": 20,
}
CAP_VARIANT_CAPS = {
    "top10_plus_50_cap15": 0.15,
    "top20_plus_40_cap15": 0.15,
    "top20_plus_40_cap12_5": 0.125,
    "top20_plus_40_cap10": 0.10,
}
EXCLUDED_TICKERS = ("NVDA",)
EX_NVDA_VARIANTS = {
    "top10_plus_50_ex_nvda": {"n_anchors": 10, "max_weight": None},
    "top10_plus_50_cap15_ex_nvda": {"n_anchors": 10, "max_weight": CAP_MAX_WEIGHT},
    "top20_plus_40_ex_nvda": {"n_anchors": 20, "max_weight": None},
    "top20_plus_40_cap15_ex_nvda": {"n_anchors": 20, "max_weight": CAP_MAX_WEIGHT},
}

MIN_END = date(2017, 1, 1)
TRADING_DAYS = 252
