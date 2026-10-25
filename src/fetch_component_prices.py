import time

import pandas as pd
import yfinance as yf

from src.config import (
    COMPONENT_PRICES_ADJ_FILE,
    COMPONENT_PRICES_FILE,
    CACHE_DIR,
    FFILL_LIMIT,
    INDEX_PRICES_FILE_OUT,
    PRICE_DOWNLOAD_END,
    PRICE_DOWNLOAD_START,
    TICKER_MAP_FILE,
)

CHUNK_SIZE = 40
LONG_FILE = CACHE_DIR / "component_prices_long.parquet"


def _strip_timezone(index):
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _extract(data, tickers):
    close = {}
    adj_close = {}
    if data is None or len(data) == 0:
        return close, adj_close
    columns = data.columns
    multi = isinstance(columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            if multi:
                if ("Close", ticker) not in columns:
                    continue
                close_series = data[("Close", ticker)]
                adj_series = data[("Adj Close", ticker)]
            else:
                if "Close" not in columns:
                    continue
                close_series = data["Close"]
                adj_series = data["Adj Close"]
        except KeyError:
            continue
        close_series = close_series.dropna()
        adj_series = adj_series.dropna()
        if len(close_series):
            close_series.index = _strip_timezone(close_series.index)
            close_series = close_series[~close_series.index.duplicated(keep="last")]
            close[ticker] = close_series
        if len(adj_series):
            adj_series.index = _strip_timezone(adj_series.index)
            adj_series = adj_series[~adj_series.index.duplicated(keep="last")]
            adj_close[ticker] = adj_series
    return close, adj_close


def _download(tickers):
    for attempt in range(4):
        try:
            data = yf.download(
                tickers,
                start=PRICE_DOWNLOAD_START,
                end=PRICE_DOWNLOAD_END,
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if data is not None and len(data):
                return data
        except Exception as exc:
            print(f"Download retry {attempt + 1}: {exc}")
        time.sleep(3 * (attempt + 1))
    return None


def download_prices(tickers):
    close_frames = {}
    adj_frames = {}
    total = len(tickers)
    for start in range(0, total, CHUNK_SIZE):
        chunk = tickers[start : start + CHUNK_SIZE]
        data = _download(chunk)
        close, adj_close = _extract(data, chunk)
        if data is None or len(data) < 100:
            for ticker in chunk:
                if ticker in close and len(close[ticker]) >= 100:
                    continue
                single = _download([ticker])
                single_close, single_adj = _extract(single, [ticker])
                close.update(single_close)
                adj_close.update(single_adj)
        close_frames.update(close)
        adj_frames.update(adj_close)
        print(f"Prices {min(start + CHUNK_SIZE, total)}/{total}")
        time.sleep(1.0)
    return close_frames, adj_frames


def build_matrices(close_frames, adj_frames):
    index_calendar = pd.DatetimeIndex(
        pd.read_parquet(INDEX_PRICES_FILE_OUT)["date"].sort_values().reset_index(drop=True)
    )
    close = pd.DataFrame(close_frames).sort_index()
    adj = pd.DataFrame(adj_frames).sort_index()
    calendar = close.index.union(index_calendar).sort_values()

    close = close.reindex(calendar)
    adj = adj.reindex(calendar).reindex(columns=close.columns)
    close = close.ffill(limit=FFILL_LIMIT)
    adj = adj.ffill(limit=FFILL_LIMIT)
    return close, adj


def main():
    mapping = pd.read_parquet(TICKER_MAP_FILE)
    tickers = sorted(mapping["ticker"].dropna().astype(str).unique())
    print(f"Tickers to download: {len(tickers)}")

    close_frames, adj_frames = download_prices(tickers)
    close, adj = build_matrices(close_frames, adj_frames)
    close = close.sort_index(axis=1)
    adj = adj.reindex(columns=close.columns).sort_index(axis=1)

    close.to_parquet(COMPONENT_PRICES_FILE)
    adj.to_parquet(COMPONENT_PRICES_ADJ_FILE)

    long = (
        pd.concat(
            [
                close.stack(dropna=False).rename("close"),
                adj.stack(dropna=False).rename("adj_close"),
            ],
            axis=1,
        )
        .rename_axis(["date", "ticker"])
        .reset_index()
    )
    long.to_parquet(LONG_FILE, index=False)

    counts = close.notna().sum().sort_values()
    print(f"Price matrix: {close.shape[0]} dates x {close.shape[1]} tickers")
    print(f"Tickers with no close: {(counts == 0).sum()}")
    print(f"Tickers with < 100 observations: {(counts < 100).sum()}")
    if (counts == 0).any():
        print("No close data:")
        print(counts[counts == 0].to_string())


if __name__ == "__main__":
    main()
