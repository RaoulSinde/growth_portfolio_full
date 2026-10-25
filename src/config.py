from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT / "reports"

CONSTITUENTS_FILE = INDEX_DIR / "msci_usa_growth_constituents.xlsx"
INDEX_PRICES_FILE = INDEX_DIR / "msci_usa_growth_prices.xlsx"

WEIGHTS_FILE = PROCESSED_DIR / "index_weights.parquet"
INDEX_PRICES_FILE_OUT = PROCESSED_DIR / "index_prices.parquet"
TICKER_MAP_FILE = CACHE_DIR / "index_ticker_map.parquet"
COMPONENT_PRICES_FILE = CACHE_DIR / "component_prices.parquet"
COMPONENT_PRICES_ADJ_FILE = CACHE_DIR / "component_prices_adj.parquet"
RECONSTRUCTED_FILE = PROCESSED_DIR / "reconstructed_index.parquet"
PERIOD_RETURNS_FILE = PROCESSED_DIR / "index_period_returns.parquet"
COMPARISON_FILE = PROCESSED_DIR / "index_comparison.parquet"
REPORT_CSV = REPORTS_DIR / "index_vs_components.csv"
REPORT_PNG = REPORTS_DIR / "index_vs_components.png"

START = "2020-12-31"
END = "2026-10-20"
PRICE_HISTORY_START = "2019-12-01"
PRICE_DOWNLOAD_START = PRICE_HISTORY_START
PRICE_DOWNLOAD_END = "2026-10-22"
INDEX_SECURITY_ID = "105825"
FFILL_LIMIT = 5
TRADING_DAYS = 252

for directory in (PROCESSED_DIR, CACHE_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
